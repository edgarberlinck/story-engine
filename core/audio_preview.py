"""Per-segment audio preview generation (plan §7 "Audio Preview").

Generates audio for individual segments of an audio-scene representation so
one changed line never forces regenerating the whole audiobook. Each segment
WAV is cached under the scene folder and keyed by a hash of the fields that
affect the audio (text, speaker, voice, emotion, tone, intensity, delivery),
so an unchanged segment is a cache hit and a changed one regenerates only
itself.

Narrator segments honor the project-level narrator configuration (plan §2):
- mode "dedicated": a designed voice from the configured voice prompt.
- mode "character": the chosen character's timbre/attributes narrate.
"""

import hashlib
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

from core.voice_engine import voice_engine, pick_speaker
from core.voice_resolver import resolve_character_voice
from services.audio_scene_service import AudioSceneSegment
from services.database.project_settings_service import (
    project_settings_service,
    NARRATOR_MODE_CHARACTER,
)
from utils.project_paths import scene_dir

logger = logging.getLogger(__name__)

# Qwen3-TTS output tends to read slower than a natural audiobook pace, so
# speech is nudged slightly faster by default. Authors can override per
# segment with [Speed=1.2] / <character speed="0.9"> in the markup.
DEFAULT_SPEECH_SPEED = 1.12

# Preview length for background sound/music cues (a taste, not the full bed).
PREVIEW_CUE_SECONDS = 8.0


def _effective_speed(segment: AudioSceneSegment) -> float:
    speed = getattr(segment, "speed", None)
    return float(speed) if speed else DEFAULT_SPEECH_SPEED


def _segment_cache_key(segment: AudioSceneSegment, voice_identity: str = "") -> str:
    """Hash of every field that affects the generated audio."""
    parts = "|".join(
        str(x)
        for x in (
            segment.segment_type,
            segment.speaker,
            segment.text,
            segment.emotion,
            segment.tone,
            segment.intensity,
            segment.delivery,
            segment.voice,
            _effective_speed(segment),
        )
    )
    if voice_identity:
        parts += f"|{voice_identity}"
    return hashlib.sha1(parts.encode("utf-8")).hexdigest()[:16]


def _voice_identity_for(project: str, segment: AudioSceneSegment) -> str:
    """Resolved voice identity for the cache key ("" when not applicable)."""
    if segment.segment_type in ("sound_effect", "music"):
        return ""
    speaker_name = (segment.speaker or "narrator").strip() or "narrator"
    voice_name = (segment.voice or "").strip()
    if speaker_name.lower() == "narrator" and not voice_name:
        return ""  # narrator config is project-level, unchanged behavior
    try:
        return resolve_character_voice(project, voice_name or speaker_name).identity
    except Exception:  # noqa: BLE001
        logger.warning("Voice identity resolution failed", exc_info=True)
        return ""


def segment_cache_key(project: str, segment: AudioSceneSegment) -> str:
    """Public cache key including the character's resolved voice identity."""
    return _segment_cache_key(segment, _voice_identity_for(project, segment))


def segment_audio_path(
    project: str, scene_number: int, segment_index: int, segment: AudioSceneSegment
) -> Path:
    """Cache location for one segment's audio."""
    audio_dir = scene_dir(scene_number, project) / "audio"
    key = segment_cache_key(project, segment)
    return audio_dir / f"segment_{segment_index:03d}_{key}.wav"


def _character_attributes(project: str, name: str) -> Dict[str, Any]:
    """Fetch a character's stored attributes (empty dict if unknown)."""
    try:
        from services.database.character_attribute_service import (
            character_attribute_service,
        )

        return character_attribute_service.get_attributes(project, name) or {}
    except Exception:  # noqa: BLE001
        logger.warning(
            "Attribute lookup failed for %s/%s; using defaults",
            project,
            name,
            exc_info=True,
        )
        return {}


def generate_segment_audio(
    project: str,
    scene_number: int,
    segment_index: int,
    segment: AudioSceneSegment,
    force: bool = False,
) -> Optional[Path]:
    """Generate (or reuse cached) audio for a single segment.

    Returns the WAV path, or None when the segment has no speakable text.
    ``force=True`` regenerates even on a cache hit.

    Sound/music segments are previewed as short background cues rendered by
    the sound engine (cached under the project's sfx folder).
    """
    text = (segment.text or "").strip()
    if not text:
        return None
    if segment.segment_type in ("sound_effect", "music"):
        from generators.sound_engine import generate_cue

        return generate_cue(project, text, PREVIEW_CUE_SECONDS, force=force)

    wav_path = segment_audio_path(project, scene_number, segment_index, segment)
    if wav_path.exists() and not force:
        logger.info("Segment audio cache hit: %s", wav_path)
        return wav_path
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    speaker_name = (segment.speaker or "narrator").strip() or "narrator"
    voice_name = (segment.voice or "").strip()

    if speaker_name.lower() == "narrator" and not voice_name:
        wav, sr = _generate_narrator_audio(project, text, segment)
    else:
        # Dialogue (or a narrator explicitly voiced by a character).
        # The timbre-defining base voice is resolved ONCE per character and
        # pinned (core.voice_resolver); emotion/tone/delivery only nudge the
        # per-fragment delivery via the instruct.
        character = voice_name or speaker_name
        resolved = resolve_character_voice(project, character)
        extras = ", ".join(
            str(p) for p in (segment.emotion, segment.tone, segment.delivery) if p
        )
        instruct = (
            f"{resolved.base_instruct}, {extras}" if extras else resolved.base_instruct
        )
        if resolved.kind == "designed" and voice_engine.design_model_available():
            wav, sr = voice_engine.generate_designed_voice(text, instruct)
        elif resolved.kind == "designed":
            # Designed voice unavailable: deterministic preset fallback with
            # the voice prompt kept as the timbre-defining instruct.
            speaker, _ = pick_speaker("person", {})
            wav, sr = voice_engine.generate_voice_line(text, speaker, instruct)
        else:
            wav, sr = voice_engine.generate_voice_line(text, resolved.speaker, instruct)

    import soundfile as sf

    sf.write(str(wav_path), wav, sr)
    _apply_speed(wav_path, _effective_speed(segment))
    logger.info("Segment audio written: %s", wav_path)
    return wav_path


def _apply_speed(wav_path: Path, speed: float) -> None:
    """Time-stretch the WAV in place with ffmpeg's atempo (pitch preserved).

    A no-op when the speed is ~1.0 or ffmpeg is unavailable.
    """
    if abs(speed - 1.0) < 0.01:
        return
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("ffmpeg not found; speech speed %.2f not applied", speed)
        return
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        proc = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(wav_path),
                "-af",
                f"atempo={max(0.5, min(2.0, speed)):.3f}",
                "-c:a",
                "pcm_s16le",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            shutil.move(str(tmp_path), str(wav_path))
        else:
            logger.warning("atempo failed: %s", proc.stderr[-300:])
    finally:
        tmp_path.unlink(missing_ok=True)


def _generate_narrator_audio(project: str, text: str, segment: AudioSceneSegment):
    """Speak narration with the project-configured narrator voice."""
    narrator = project_settings_service.get_narrator(project)

    if narrator.get("mode") == NARRATOR_MODE_CHARACTER and narrator.get("character"):
        attributes = _character_attributes(project, narrator["character"])
        speaker, instruct = pick_speaker(
            attributes.get("type", "person"),
            attributes,
            emotion=segment.emotion,
            tone=segment.tone,
            intensity=segment.intensity,
            delivery=segment.delivery,
        )
        return voice_engine.generate_voice_line(text, speaker, instruct)

    # Dedicated/designed narrator voice.
    voice_prompt = narrator.get("voice_prompt") or "A neutral storytelling voice."
    extras = ", ".join(
        str(p) for p in (segment.emotion, segment.tone, segment.delivery) if p
    )
    instruct = f"{voice_prompt} {extras}".strip().rstrip(",")
    if voice_engine.design_model_available():
        return voice_engine.generate_designed_voice(text, instruct)
    # Fallback: preset neutral narrator timbre with the prompt as instruct.
    speaker, _ = pick_speaker("person", {})
    return voice_engine.generate_voice_line(text, speaker, instruct)
