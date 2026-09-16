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
from pathlib import Path
from typing import Optional, Dict, Any

from core.voice_engine import voice_engine, pick_speaker
from services.audio_scene_service import AudioSceneSegment
from services.database.project_settings_service import (
    project_settings_service,
    NARRATOR_MODE_CHARACTER,
)
from utils.project_paths import scene_dir

logger = logging.getLogger(__name__)


def _segment_cache_key(segment: AudioSceneSegment) -> str:
    """Hash of every field that affects the generated audio."""
    parts = "|".join(
        str(x)
        for x in (
            segment.segment_type, segment.speaker, segment.text,
            segment.emotion, segment.tone, segment.intensity,
            segment.delivery, segment.voice,
        )
    )
    return hashlib.sha1(parts.encode("utf-8")).hexdigest()[:16]


def segment_audio_path(
    project: str, scene_number: int, segment_index: int, segment: AudioSceneSegment
) -> Path:
    """Cache location for one segment's audio."""
    audio_dir = scene_dir(scene_number, project) / "audio"
    return audio_dir / f"segment_{segment_index:03d}_{_segment_cache_key(segment)}.wav"


def _character_attributes(project: str, name: str) -> Dict[str, Any]:
    """Fetch a character's stored attributes (empty dict if unknown)."""
    try:
        from services.database.character_attribute_service import (
            character_attribute_service,
        )
        return character_attribute_service.get_attributes(project, name) or {}
    except Exception:  # noqa: BLE001
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
    """
    text = (segment.text or "").strip()
    if not text or segment.segment_type in ("sound_effect", "music"):
        return None

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
        character = voice_name or speaker_name
        attributes = _character_attributes(project, character)
        speaker, instruct = pick_speaker(
            attributes.get("type", "person"),
            attributes,
            emotion=segment.emotion,
            tone=segment.tone,
            intensity=segment.intensity,
            delivery=segment.delivery,
        )
        wav, sr = voice_engine.generate_voice_line(text, speaker, instruct)

    import soundfile as sf

    sf.write(str(wav_path), wav, sr)
    logger.info("Segment audio written: %s", wav_path)
    return wav_path


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
