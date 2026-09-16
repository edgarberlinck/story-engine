"""Audiobook exporter: turn compiled chapters into a final, mixed audio file.

High-quality assembly (plan §6 "Audio Assembly"):

1. Per scene, the speech track (narration + dialogue) is laid out on a
   timeline with natural gaps between lines.
2. Sound/music cues are NOT inserted sequentially: they are rendered as
   background beds that start at their position in the scene and play UNDER
   the following speech:
   - each bed is faded in/out and attenuated,
   - beds are side-chain compressed (ducked) against the speech, so the
     music dips whenever someone talks — like a real audiobook mix.
3. Scenes/chapters are joined with breathing gaps, and the final master is
   loudness-normalized to the audiobook standard (EBU R128, I=-16 LUFS,
   TP=-1.5 dB) before encoding (.m4a/.mp3/.wav by extension). All audio
   processing/joining is done with ffmpeg.

Voice segments and cues are content-hash cached, so re-exporting after
editing one line regenerates only that line.
"""

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import soundfile as sf

from core.audio_preview import generate_segment_audio
from generators.sound_engine import generate_cue, DEFAULT_SFX_SECONDS
from services.audio_scene_service import audio_scene_service
from services.database.manuscript_service import manuscript_service

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
SEGMENT_GAP_S = 0.5
SPEAKER_CHANGE_GAP_S = 0.85   # slightly longer pause when the voice changes
SCENE_GAP_S = 1.2
CHAPTER_GAP_S = 2.0

# Micro fade applied to each speech segment so voice/tone changes never
# click or jump — every line eases in and out of its neighbors.
SPEECH_FADE_S = 0.06

# Background bed mixing
SFX_VOLUME = 0.45        # one-shot effects sit slightly above the music bed
MUSIC_VOLUME = 0.30      # music stays a bed under the narration
BED_FADE_IN_S = 0.8
BED_FADE_OUT_S = 1.8
MUSIC_MAX_S = 30.0       # longest generated music bed per cue
MUSIC_MIN_S = 4.0

# Ducking: compress the bed using the speech as side-chain key.
DUCK_ARGS = "threshold=0.02:ratio=8:attack=50:release=600:makeup=1"

# Audiobook loudness target (EBU R128).
LOUDNORM = "loudnorm=I=-16:TP=-1.5:LRA=11"


@dataclass
class ExportResult:
    output_path: Optional[Path] = None
    segments_rendered: int = 0
    cues_rendered: int = 0
    skipped: List[str] = field(default_factory=list)
    error: Optional[str] = None


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg not found on PATH; install it to export audio")
    return path


def _run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-600:]}")


def _normalize(src: Path, dst: Path, fade: bool = False) -> Path:
    """Re-encode any clip to mono 24 kHz 16-bit PCM.

    With ``fade=True`` a micro fade-in/out is applied so consecutive speech
    segments (tone or character changes) transition smoothly without clicks.
    """
    af = f"aresample={SAMPLE_RATE}"
    if fade:
        duration = sf.info(str(src)).duration
        fade_out_start = max(0.0, duration - SPEECH_FADE_S)
        af += (
            f",afade=t=in:d={SPEECH_FADE_S}"
            f",afade=t=out:st={fade_out_start:.3f}:d={SPEECH_FADE_S}"
        )
    _run([
        _ffmpeg(), "-y", "-i", str(src),
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-af", af,
        "-c:a", "pcm_s16le", str(dst),
    ])
    return dst


def _silence(seconds: float, dst: Path) -> Path:
    _run([
        _ffmpeg(), "-y",
        "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=mono",
        "-t", f"{seconds:.3f}", "-c:a", "pcm_s16le", str(dst),
    ])
    return dst


def _duration(path: Path) -> float:
    return sf.info(str(path)).duration


def _concat(parts: List[Path], dst: Path) -> Path:
    list_file = dst.parent / f".{dst.stem}_concat.txt"
    list_file.write_text(
        "".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8"
    )
    try:
        _run([
            _ffmpeg(), "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file), "-c", "copy", str(dst),
        ])
    finally:
        list_file.unlink(missing_ok=True)
    return dst


@dataclass
class _Cue:
    path: Path
    offset_s: float       # where the bed starts on the scene timeline
    volume: float
    duration_s: float


def _mix_scene(
    speech: Path,
    cues: List[_Cue],
    dst: Path,
) -> Path:
    """Mix the speech track with background beds, ducked under the voice.

    Filter graph:
      each cue  -> delay to its offset, fades, attenuation      -> bed_i
      beds      -> amix                                          -> beds
      beds      -> sidechaincompress keyed by the speech (duck)  -> ducked
      speech + ducked -> amix -> scene mix
    """
    if not cues:
        shutil.copy(str(speech), str(dst))
        return dst

    cmd = [_ffmpeg(), "-y", "-i", str(speech)]
    for cue in cues:
        cmd += ["-i", str(cue.path)]

    filters = []
    bed_labels = []
    for i, cue in enumerate(cues, start=1):
        delay_ms = int(cue.offset_s * 1000)
        fade_out_start = max(0.0, cue.duration_s - BED_FADE_OUT_S)
        filters.append(
            f"[{i}:a]afade=t=in:d={BED_FADE_IN_S},"
            f"afade=t=out:st={fade_out_start:.3f}:d={BED_FADE_OUT_S},"
            f"volume={cue.volume},"
            f"adelay={delay_ms}:all=1[bed{i}]"
        )
        bed_labels.append(f"[bed{i}]")

    if len(bed_labels) == 1:
        filters.append(f"{bed_labels[0]}anull[beds]")
    else:
        filters.append(
            f"{''.join(bed_labels)}amix=inputs={len(bed_labels)}:"
            "duration=longest:normalize=0[beds]"
        )

    # Duck the beds with the speech as side-chain key, then mix.
    filters.append("[0:a]asplit=2[voice][key]")
    filters.append(f"[beds][key]sidechaincompress={DUCK_ARGS}[ducked]")
    filters.append(
        "[voice][ducked]amix=inputs=2:duration=longest:normalize=0[mix]"
    )

    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "[mix]", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le", str(dst),
    ]
    _run(cmd)
    return dst


def _unload_generation_models() -> None:
    """Free the heavyweight generation models once rendering is done."""
    try:
        from core.voice_engine import voice_engine
        voice_engine.unload()
    except Exception:  # noqa: BLE001
        logger.debug("Voice engine unload skipped", exc_info=True)
    try:
        from generators.sound_engine import sound_engine
        sound_engine.unload()
    except Exception:  # noqa: BLE001
        logger.debug("Sound engine unload skipped", exc_info=True)


def _encode_master(src: Path, output_path: Path) -> None:
    """Loudness-normalize the master and encode to the requested format."""
    suffix = output_path.suffix.lower()
    if suffix == ".wav":
        codec = ["-c:a", "pcm_s16le"]
    elif suffix == ".mp3":
        codec = ["-c:a", "libmp3lame", "-q:a", "1"]
    else:  # .m4a and anything else -> AAC
        codec = ["-c:a", "aac", "-b:a", "192k"]
    _run([
        _ffmpeg(), "-y", "-i", str(src),
        "-af", LOUDNORM, "-ar", str(SAMPLE_RATE), "-ac", "1",
        *codec, str(output_path),
    ])


def _cue_seconds(segment, remaining_hint: Optional[float] = None) -> float:
    """Cue length: explicit timing wins; music beds size to the scene."""
    timing = segment.timing or {}
    length = (timing.get("end") or 0) - (timing.get("start") or 0)
    if length > 0:
        return min(MUSIC_MAX_S, float(length))
    if segment.segment_type == "music":
        if remaining_hint is not None:
            return max(MUSIC_MIN_S, min(MUSIC_MAX_S, remaining_hint))
        return MUSIC_MAX_S / 2
    return DEFAULT_SFX_SECONDS


def _render_scene(
    project: str,
    scene_number: int,
    rep,
    tmp_dir: Path,
    include_cues: bool,
    result: ExportResult,
    report: Callable[[str], None],
) -> Optional[Path]:
    """Render one scene into a fully mixed WAV. Returns None if empty."""
    scene_tag = f"s{scene_number:04d}"

    # Pass 1: lay the speech track out on a timeline.
    speech_parts: List[Path] = []
    cursor = 0.0
    pending_cues: List[Tuple[int, object, float]] = []  # (index, segment, offset)
    n = 0
    prev_voice: Optional[str] = None
    for seg_idx, segment in enumerate(rep.segments):
        if segment.segment_type in ("sound_effect", "music"):
            if include_cues:
                # The bed starts where the timeline currently is.
                pending_cues.append((seg_idx, segment, cursor))
            continue
        wav = generate_segment_audio(project, scene_number, seg_idx, segment)
        if wav is None:
            continue
        voice_key = (
            (segment.voice or segment.speaker or "narrator").strip().lower()
            or "narrator"
        )
        if speech_parts:
            # A change of voice (character or narrator hand-off) breathes a
            # little longer than a same-speaker pause, so the transition
            # between timbres/tones feels natural instead of abrupt.
            gap = SEGMENT_GAP_S if voice_key == prev_voice else SPEAKER_CHANGE_GAP_S
            n += 1
            speech_parts.append(
                _silence(gap, tmp_dir / f"{scene_tag}_gap{n}.wav")
            )
            cursor += gap
        prev_voice = voice_key
        n += 1
        normalized = _normalize(wav, tmp_dir / f"{scene_tag}_seg{n}.wav", fade=True)
        speech_parts.append(normalized)
        cursor += _duration(normalized)
        result.segments_rendered += 1
        report(f"    {segment.speaker or 'narrator'}: {segment.text[:40]}")

    scene_len = cursor
    if not speech_parts and not pending_cues:
        return None

    if speech_parts:
        speech = _concat(speech_parts, tmp_dir / f"{scene_tag}_speech.wav")
    else:
        speech = _silence(max(scene_len, 1.0), tmp_dir / f"{scene_tag}_speech.wav")

    # Pass 2: render the cues as beds sized against the scene.
    cues: List[_Cue] = []
    for seg_idx, segment, offset in pending_cues:
        remaining = max(MUSIC_MIN_S, scene_len - offset)
        seconds = _cue_seconds(segment, remaining_hint=remaining)
        cue_wav = generate_cue(project, segment.text, seconds)
        if cue_wav is None:
            result.skipped.append(
                f"Scene {scene_number} cue skipped: {segment.text[:40]}"
            )
            continue
        normalized = _normalize(
            cue_wav, tmp_dir / f"{scene_tag}_cue{seg_idx}.wav"
        )
        volume = MUSIC_VOLUME if segment.segment_type == "music" else SFX_VOLUME
        cues.append(_Cue(
            path=normalized, offset_s=offset,
            volume=volume, duration_s=_duration(normalized),
        ))
        result.cues_rendered += 1
        report(f"    bed [{segment.segment_type}]: {segment.text[:40]}")

    return _mix_scene(speech, cues, tmp_dir / f"{scene_tag}_mix.wav")


def export_chapters(
    project: str,
    chapter_ids: List[int],
    output_path: str,
    locale: str = "en",
    include_cues: bool = True,
    progress: Optional[Callable[[str], None]] = None,
) -> ExportResult:
    """Export the selected chapters into one mixed, mastered audio file."""
    result = ExportResult()
    report = progress or (lambda msg: logger.info("%s", msg))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    order = {c["id"]: i for i, c in enumerate(manuscript_service.list_chapters(project))}
    unknown = [cid for cid in chapter_ids if cid not in order]
    if unknown:
        result.error = f"Unknown chapter id(s): {unknown}"
        return result
    chapters = sorted(chapter_ids, key=lambda cid: order[cid])

    try:
        _ffmpeg()
    except RuntimeError as e:
        result.error = str(e)
        return result

    with tempfile.TemporaryDirectory(prefix="audiobook_") as tmp:
        tmp_dir = Path(tmp)
        timeline: List[Path] = []
        gap_n = 0

        def add_gap(seconds: float):
            nonlocal gap_n
            gap_n += 1
            timeline.append(_silence(seconds, tmp_dir / f"gap_{gap_n:03d}.wav"))

        for c_idx, chapter_id in enumerate(chapters):
            chapter = manuscript_service.get_chapter(chapter_id)
            scene_numbers = chapter["compiled_scenes"].get(locale, [])
            if not scene_numbers:
                result.skipped.append(
                    f"Chapter '{chapter['title']}' has no compiled {locale} scenes"
                )
                continue
            if timeline:
                add_gap(CHAPTER_GAP_S)
            report(f"Chapter: {chapter['title']}")

            first_scene_in_chapter = True
            for scene_number in scene_numbers:
                rep = audio_scene_service.get_representation(project, scene_number)
                if rep is None:
                    result.skipped.append(f"Scene {scene_number} not found")
                    continue
                report(f"  Scene {scene_number}: {rep.title}")
                mixed = _render_scene(
                    project, scene_number, rep, tmp_dir,
                    include_cues, result, report,
                )
                if mixed is None:
                    continue
                if not first_scene_in_chapter:
                    add_gap(SCENE_GAP_S)
                first_scene_in_chapter = False
                timeline.append(mixed)

        # All generation is done; the rest is pure ffmpeg. Release the TTS
        # and MusicGen models (several GB of RAM/VRAM) before mastering.
        _unload_generation_models()

        if not timeline:
            result.error = "Nothing to export (no compiled scenes with audio)"
            return result

        report("Joining scenes...")
        master = _concat(timeline, tmp_dir / "master.wav")
        report("Mastering (loudness normalization) and encoding...")
        _encode_master(master, out)

    result.output_path = out
    report(f"Exported: {out}")
    return result
