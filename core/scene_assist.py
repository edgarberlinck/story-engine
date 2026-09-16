"""LLM assistance for audio-scene representations (plan §5, §7).

The LLM operates on the STRUCTURED scene representation — never on final
audio. The user reviews the result before it is applied (the UI is
responsible for the review step; this module only proposes a modified
representation and never persists anything itself).

Example requests:
- "Make Nikita sound more irritated."
- "Rewrite this dialogue so Roger sounds afraid but tries to hide it."
- "Make the narration more cinematic."
"""

import json
from typing import Optional

from core.scene_planner import _extract_json_blocks
from services.audio_scene_service import AudioSceneRepresentation

try:
    from generators.text_generator import generate_text_with_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


_ALLOWED_SEGMENT_KEYS = {
    "type", "speaker", "text", "emotion", "tone", "intensity",
    "delivery", "voice", "timing", "sound_effects", "music",
}


def _build_prompt(representation: AudioSceneRepresentation, user_request: str) -> str:
    repr_json = json.dumps(representation.to_dict(), indent=2)
    return f"""You edit a structured audio-scene representation for an audiobook.

CURRENT SCENE REPRESENTATION (JSON):
{repr_json}

USER REQUEST:
{user_request}

Rules:
- Modify ONLY what the request asks for; keep everything else identical.
- Keep the same JSON structure and keys (scene_id, title, timeline, segments,
  characters_present, objects_present, location).
- Each segment keeps its keys: type, speaker, text, emotion, tone, intensity,
  delivery, voice, timing, sound_effects, music.
- intensity is a number between 0.0 and 1.0.
- Do NOT invent new characters or change speakers unless asked.
- Return the FULL modified representation as valid JSON only, no commentary.

Return JSON:"""


def assist_scene(
    representation: AudioSceneRepresentation,
    user_request: str,
    max_new_tokens: int = 1200,
) -> Optional[AudioSceneRepresentation]:
    """Ask the LLM to modify the representation per the user's request.

    Returns the proposed new representation, or None if the LLM is
    unavailable or produced unusable output. The caller must let the user
    review the proposal before saving it (plan §5: user review comes first).
    """
    if not LLM_AVAILABLE:
        return None

    prompt = _build_prompt(representation, user_request)
    try:
        result = generate_text_with_llm(prompt, max_new_tokens=max_new_tokens)
        if not result:
            return None
        data = json.loads(_extract_json_blocks(result))
    except Exception as e:  # noqa: BLE001
        print(f"LLM scene assistance failed: {e}")
        return None

    if not isinstance(data, dict) or "segments" not in data:
        return None

    # Preserve information the LLM must not lose (plan §9): identity fields
    # come from the original representation, not from LLM guesses.
    data["scene_id"] = representation.scene_id
    data.setdefault("title", representation.title)
    data.setdefault("timeline", representation.timeline)
    data.setdefault("characters_present", representation.characters_present)
    data.setdefault("objects_present", representation.objects_present)
    data.setdefault("location", representation.location)

    # Drop unknown keys the model may have added to segments.
    cleaned_segments = []
    for seg in data.get("segments", []):
        if not isinstance(seg, dict):
            continue
        cleaned_segments.append({k: v for k, v in seg.items() if k in _ALLOWED_SEGMENT_KEYS})
    if not cleaned_segments:
        return None
    data["segments"] = cleaned_segments

    try:
        return AudioSceneRepresentation.from_dict(data)
    except Exception as e:  # noqa: BLE001
        print(f"LLM scene assistance produced invalid representation: {e}")
        return None
