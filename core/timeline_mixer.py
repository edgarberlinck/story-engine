"""Mix a scene timeline (tracks/clips with offsets) into one WAV via ffmpeg.

Used by the interactive timeline view (honoring per-track mute/solo/volume)
and by the exporter/finalizer when segments carry manual ``start_offset``
overrides, so what you see on the timeline is what you get in the final mix.
"""

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

from core.scene_timeline_model import TimelineModel, compute_timeline

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000


@dataclass
class TrackState:
    mute: bool = False
    solo: bool = False
    volume: float = 1.0


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg not found on PATH; install it to mix audio")
    return path


def _run(cmd) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-600:]}")


def _silence(seconds: float, dst: Path) -> Path:
    _run(
        [
            _ffmpeg(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={SAMPLE_RATE}:cl=mono",
            "-t",
            f"{max(seconds, 0.5):.3f}",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
    )
    return dst


def _audible_tracks(model: TimelineModel, track_states) -> list:
    states = track_states or {}
    any_solo = any(getattr(s, "solo", False) for s in states.values())
    audible = []
    for track in model.tracks:
        state = states.get(track.name, TrackState())
        if any_solo and not state.solo:
            continue
        if not any_solo and state.mute:
            continue
        if state.volume <= 0.0:
            continue
        audible.append((track, state))
    return audible


def mix_timeline(
    model: TimelineModel,
    track_states: Optional[Dict[str, TrackState]],
    out_path,
    start_at: float = 0.0,
) -> Path:
    """Mix all audible clips at their timeline offsets into ``out_path``.

    ``track_states`` maps track name -> TrackState (mute/solo/volume). Solo
    on any track hides all non-solo tracks; otherwise muted tracks are
    skipped. ``start_at`` trims the head of the mix (playhead seek).
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    inputs = []  # (path, volume, delay_ms)
    for track, state in _audible_tracks(model, track_states):
        for clip in track.clips:
            if clip.missing or clip.path is None or not Path(clip.path).exists():
                continue
            inputs.append(
                (
                    Path(clip.path),
                    max(0.0, float(state.volume)),
                    int(max(0.0, clip.start) * 1000),
                )
            )

    if not inputs:
        return _silence(model.duration or 1.0, out)

    cmd = [_ffmpeg(), "-y"]
    for path, _v, _d in inputs:
        cmd += ["-i", str(path)]

    filters = []
    labels = []
    for i, (_path, volume, delay_ms) in enumerate(inputs):
        filters.append(
            f"[{i}:a]aresample={SAMPLE_RATE},"
            f"aformat=channel_layouts=mono,"
            f"volume={volume:.4f},"
            f"adelay={delay_ms}:all=1[c{i}]"
        )
        labels.append(f"[c{i}]")

    if len(labels) == 1:
        filters.append(f"{labels[0]}anull[mix]")
    else:
        filters.append(
            f"{''.join(labels)}amix=inputs={len(labels)}:"
            "duration=longest:normalize=0[mix]"
        )

    final_label = "[mix]"
    if start_at > 0.0:
        filters.append(f"[mix]atrim=start={start_at:.3f},asetpts=PTS-STARTPTS[cut]")
        final_label = "[cut]"

    cmd += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        final_label,
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(out),
    ]
    _run(cmd)
    return out


def render_scene_with_offsets(
    project: str,
    scene_number: int,
    representation,
    dst,
    report: Optional[Callable[[str], None]] = None,
) -> Path:
    """Render a scene honoring manual start_offset overrides (WYSIWYG).

    Generates any missing fragment audio (speech via the preview pipeline,
    beds at their timeline-computed durations), then mixes every clip at its
    timeline position with all tracks audible at unity gain.
    """
    from core.scene_generation import generate_clip_audio

    say = report or (lambda msg: logger.info("%s", msg))

    model = compute_timeline(project, scene_number, representation)
    for track in model.tracks:
        for clip in track.clips:
            if not clip.missing:
                continue
            say(f"    generating: {clip.label}")
            generate_clip_audio(project, scene_number, representation, clip)

    # Recompute with real durations/paths after generation.
    model = compute_timeline(project, scene_number, representation)
    # Bed tracks keep the exporter's attenuation (no ducking in offset mode:
    # manual placement means the author controls speech/bed overlap).
    from core.audiobook_exporter import SFX_VOLUME, MUSIC_VOLUME

    states = {
        "Sound FX": TrackState(volume=SFX_VOLUME),
        "Music": TrackState(volume=MUSIC_VOLUME),
    }
    return mix_timeline(model, states, dst)
