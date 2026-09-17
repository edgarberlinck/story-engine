"""Stable per-character voice resolution.

Fixes voice drift: a character's timbre-defining voice (designed voice prompt
or preset speaker) is resolved ONCE per (project, character) and pinned in a
module-level cache, so every fragment of that character uses the same base
voice. Emotion/tone/delivery remain per-fragment *delivery* nudges appended to
the instruct — they must never flip the underlying timbre.

Resolution order:
1. If the character record has a ``voice_prompt`` (character_service), the
   voice is a designed voice using that prompt as the timbre-defining base.
2. Otherwise the preset speaker is picked once from the stored attributes via
   ``pick_speaker`` and pinned.
3. On any lookup failure a deterministic fallback preset is used — and cached,
   so a transient failure can never change the voice for a single fragment.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

from core.voice_engine import pick_speaker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedVoice:
    """One stable voice identity for a character."""

    kind: str  # "designed" | "preset"
    speaker: str = ""  # preset speaker name (kind == "preset")
    base_instruct: str = ""  # timbre-defining instruct base
    voice_prompt: str = ""  # designed voice prompt (kind == "designed")

    @property
    def identity(self) -> str:
        """Stable string identifying the voice (for audio cache keys)."""
        if self.kind == "designed":
            return f"designed:{self.voice_prompt}"
        return f"preset:{self.speaker}"


# Pinned resolutions: (project, character-lower) -> ResolvedVoice
_VOICE_CACHE: Dict[Tuple[str, str], ResolvedVoice] = {}


def clear_voice_cache() -> None:
    """Forget pinned voices (e.g. after editing a character's voice)."""
    _VOICE_CACHE.clear()


def resolve_character_voice(project: str, character_name: str) -> ResolvedVoice:
    """Resolve (and pin) ONE stable voice for a character in a project."""
    key = (project or "", (character_name or "").strip().lower())
    cached = _VOICE_CACHE.get(key)
    if cached is not None:
        return cached
    resolved = _resolve(project, character_name)
    _VOICE_CACHE[key] = resolved
    return resolved


def _resolve(project: str, name: str) -> ResolvedVoice:
    # 1. Designed voice from the persisted character record.
    record = None
    try:
        from services.database.character_service import character_service

        record = character_service.get_character(name, project)
    except Exception:  # noqa: BLE001
        logger.warning(
            "Character record lookup failed for %s/%s; "
            "falling back to attribute-based preset",
            project,
            name,
            exc_info=True,
        )
    if record:
        voice_prompt = (record.get("voice_prompt") or "").strip()
        if voice_prompt:
            return ResolvedVoice(
                kind="designed",
                voice_prompt=voice_prompt,
                base_instruct=voice_prompt,
            )

    # 2. Preset speaker picked ONCE from the stored attributes.
    attributes = {}
    try:
        from services.database.character_attribute_service import (
            character_attribute_service,
        )

        attributes = character_attribute_service.get_attributes(project, name) or {}
    except Exception:  # noqa: BLE001
        # 3. Deterministic fallback: log loudly, then resolve (and pin) the
        # default preset so a failure never flips the voice mid-scene.
        logger.warning(
            "Attribute lookup failed for %s/%s; using deterministic " "default preset",
            project,
            name,
            exc_info=True,
        )
        attributes = {}

    speaker, instruct = pick_speaker(attributes.get("type", "person"), attributes)
    return ResolvedVoice(kind="preset", speaker=speaker, base_instruct=instruct)
