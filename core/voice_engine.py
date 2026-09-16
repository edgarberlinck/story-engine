"""Character voice generation engine.

Generates a short spoken introduction for a character using the local Qwen3-TTS
models. Two checkpoints are used:

- ``Qwen3-TTS-12Hz-1.7B-VoiceDesign`` (``models/text_to_speech/qwen3_tts_voicedesign``):
  **designs a brand-new voice from a natural-language prompt**
  (``generate_voice_design``). This is the engine used when the user provides
  a voice prompt — the prompt creates the actual timbre.
- ``Qwen3-TTS-12Hz-1.7B-CustomVoice`` (``models/text_to_speech/qwen3_tts``):
  preset timbres + instruction-driven delivery (``generate_custom_voice``).
  Fallback for prompt-less generation and when VoiceDesign is not installed.

The flow:
    1. ``build_voice_line()`` turns the character's attributes into the line
       "Hi, my name is <name>, I'm a <description>" (matches the attribute
       vocabulary, e.g. "an elderly Mediterranean woman with a warm
       personality").
    2. With a prompt: ``VoiceEngine.generate_designed_voice()`` runs
       ``generate_voice_design(line, instruct=prompt, language)``.
       Without a prompt: ``pick_speaker()`` maps the character's
       type/gender/age to a preset timbre + delivery nudge.
    3. ``VoiceEngine.generate_character_voice()`` lazily loads the right model
       once and synthesizes the line to a WAV in the character's ``voice/``
       folder, returning the prompt used (persisted on the character record).

See ``docs/humans/voice-generation-implementation.md`` for the working path
and known failures.
"""

import logging
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from utils.project_paths import character_dir

logger = logging.getLogger(__name__)

# Local model checkpoints (downloaded by `make install`).
QWEN3_TTS_MODEL_DIR = Path("models/text_to_speech/qwen3_tts")
QWEN3_TTS_VOICEDESIGN_MODEL_DIR = Path("models/text_to_speech/qwen3_tts_voicedesign")

# Supported preset speakers in the 1.7B CustomVoice checkpoint.
# Genders are best-effort (the checkpoint README does not list per-speaker
# gender; inferred from usage in the model card examples).
FEMALE_SPEAKERS = ["serena", "vivian", "ono_anna", "sohee"]
MALE_SPEAKERS = ["ryan", "aiden", "eric", "dylan", "uncle_fu"]

# Natural-language delivery nudges used as `instruct` in generate_custom_voice.
# These steer emotion/timbre/pace — NOT speed (drive pacing via generation
# kwargs instead, per the model docs).
INSTRUCT_BY_PERSONALITY = {
    "Brave": "a brave, confident voice, steady and resolute",
    "Intelligent": "a thoughtful, articulate voice, clear and measured",
    "Curious": "a bright, curious voice, lively and inquisitive",
    "Serious": "a serious, composed voice, calm and matter-of-fact",
    "Funny": "a playful, cheerful voice, light and humorous",
    "Mysterious": "a low, mysterious voice, soft and a little secretive",
    "Friendly": "a warm, friendly voice, open and approachable",
    "Aggressive": "a sharp, forceful voice, blunt and intense",
    "Shy": "a soft, shy voice, quiet and gentle",
    "Confident": "a confident, assured voice, clear and steady",
    "Elegant": "an elegant, refined voice, smooth and polished",
    "Chaotic": "an energetic, excitable voice, fast and unpredictable",
}

# Default instruct when the personality is unknown.
_DEFAULT_INSTRUCT = "a natural, pleasant voice, clear and easy to understand"

# Emotion hints for the common expressions.
INSTRUCT_BY_EXPRESSION = {
    "Happy": "sounding happy and warm",
    "Serious": "sounding serious",
    "Calm": "sounding calm",
    "Friendly": "sounding friendly",
    "Confident": "sounding confident",
    "Mysterious": "sounding mysterious",
    "Sad": "sounding gentle and a little sad",
    "Angry": "sounding angry and forceful",
    "Fearful": "sounding shaky and breathless",
    "Stressed": "sounding breathless and urgent",
}

# Tone hints for the common delivery styles.
INSTRUCT_BY_TONE = {
    "confrontational": "confrontational and direct",
    "descriptive": "descriptive and clear",
    "friendly": "warm and approachable",
    "narrative": "storytelling and engaging",
    "cinematic": "cinematic and dramatic",
    "natural": "natural and conversational",
}

# Intensity levels mapped to instruct nudges.
INTENSITY_MAP = {
    0.1: "mild and gentle",
    0.3: "moderate and controlled",
    0.5: "balanced and natural",
    0.7: "strong and expressive",
    0.9: "intense and forceful",
}

# Ages that map to a "young" delivery.
_YOUNG_AGES = {"Child", "Teenager", "Young Adult"}


def _map_intensity(intensity: Optional[float]) -> str:
    """Map a numeric intensity (0.0-1.0) to an instruct nudge.

    Qwen3-TTS supports intensity control via the instruct parameter but does
    not expose speed/drive in the instruct; pacing is driven via generation
    kwargs instead. We therefore translate intensity into a descriptive
    adjective that the instruct can carry.

    ``None`` means "no explicit intensity requested" and yields no nudge.
    """
    if intensity is None:
        return ""
    t = max(0.0, min(1.0, float(intensity)))
    best = min(INTENSITY_MAP.keys(), key=lambda k: abs(k - t))
    return INTENSITY_MAP[best]


def _build_instruct_from_emo_tone_delivery(
    emotion: str,
    tone: str,
    intensity: float,
    delivery: str,
    personality: str = "",
    expression: str = "",
) -> str:
    """Build an instruct string from emotion, tone, intensity, and delivery.

    The Qwen3-TTS CustomVoice model accepts an `instruct` parameter that
    steers timbre, emotion, and delivery. We combine the available components
    into a natural-language prompt fragment.
    """
    parts = []

    # Emotion
    if emotion and emotion in INSTRUCT_BY_EXPRESSION:
        parts.append(INSTRUCT_BY_EXPRESSION[emotion])
    elif emotion:
        parts.append(f"sounding {emotion.lower()}")

    # Tone
    if tone and tone in INSTRUCT_BY_TONE:
        parts.append(INSTRUCT_BY_TONE[tone])

    # Intensity
    intensity_nudge = _map_intensity(intensity)
    if intensity_nudge:
        parts.append(intensity_nudge)

    # Delivery
    if delivery:
        parts.append(delivery)

    # Personality (from attributes, if provided)
    if personality and personality in INSTRUCT_BY_PERSONALITY:
        parts.append(INSTRUCT_BY_PERSONALITY[personality])

    # Default
    if not parts:
        parts.append(_DEFAULT_INSTRUCT)

    return ", ".join(parts)


# Keywords used to infer the desired timbre gender from a regeneration prompt.
# When the user writes "a warm female voice... she sounds...", the speaker
# must follow the prompt, NOT the (possibly empty) stored attributes.
_FEMALE_HINTS = (
    "female",
    "woman",
    "women",
    "she",
    "her",
    "feminine",
    "lady",
    "girl",
    "grandmother",
    "mother",
    "queen",
    "goddess",
    "princess",
)
_MALE_HINTS = (
    "male",
    "man",
    "men",
    "he",
    "his",
    "masculine",
    "gentleman",
    "boy",
    "grandfather",
    "father",
    "king",
    "god",
    "prince",
)


def infer_gender_from_prompt(prompt: str) -> Optional[str]:
    """Return "female" / "male" / None from free-text voice prompts.

    Case-insensitive keyword scan; the first decisive hint wins.
    """
    if not prompt:
        return None
    text = prompt.lower()
    for hint in _FEMALE_HINTS:
        if hint in text:
            return "female"
    for hint in _MALE_HINTS:
        if hint in text:
            return "male"
    return None


# Builder UI vocabulary → voice vocabulary. The character builder stores a
# flat dict (gender, age_range, mood, ...); the full attribute system stores
# the richer keys below (age, ethnicity, personality, expression).
_AGE_NORMALIZE = {
    "Child": "child",
    "Teen": "teenager",
    "Teenager": "teenager",
    "20-30": "young adult",
    "30-40": "adult",
    "40-50": "middle-aged",
    "50+": "elderly",
    "Young Adult": "young adult",
    "Adult": "adult",
    "Middle-aged": "middle-aged",
    "Elderly": "elderly",
}


def _attr(attributes: Dict[str, str], *keys: str) -> str:
    """First non-empty/non-skip value among the candidate keys."""
    for key in keys:
        value = attributes.get(key, "")
        if value and str(value).strip().lower() not in ("none", "other", "unknown", ""):
            return str(value).strip()
    return ""


def _describe_character(char_type: str, attributes: Dict[str, str]) -> str:
    """A short natural-language description from the attribute vocabulary.

    Mirrors ``build_character_prompt``'s subject logic but reads like speech
    ("an elderly Mediterranean woman"), so the TTS line sounds natural.
    """
    parts = []
    age_raw = _attr(attributes, "age", "age_range")
    if age_raw:
        age = _AGE_NORMALIZE.get(age_raw, age_raw.lower())
        parts.append(age)

    if char_type == "animal":
        species = _attr(attributes, "species")
        if species:
            parts.append(species.lower())
    else:
        ethnicity = _attr(attributes, "ethnicity")
        if ethnicity:
            parts.append(ethnicity.lower())
        noun = {"man": "man", "woman": "woman"}.get(char_type, "person")
        parts.append(noun)

    personality = _attr(attributes, "personality", "mood")
    if personality:
        parts.append(f"with a {personality.lower()} personality")
    return " ".join(parts) if parts else "a person"


def build_voice_line(
    name: str, char_type: str = "person", attributes: Optional[Dict[str, str]] = None
) -> str:
    """The introduction line spoken by the character.

    >>> build_voice_line("Leila", "woman", {"age": "Elderly", "ethnicity": "Mediterranean"})
    "Hi, my name is Leila, I'm an elderly Mediterranean woman."
    """
    attributes = attributes or {}
    description = _describe_character(char_type, attributes)

    # Choose "a" vs "an" from the leading sound of the description.
    first = description.split()[0].lower() if description else ""
    article = "an" if first[:1] in ("a", "e", "i", "o", "u") else "a"
    return f"Hi, my name is {name}, I'm {article} {description}."


def pick_speaker(
    char_type: str,
    attributes: Optional[Dict[str, str]] = None,
    emotion: Optional[str] = None,
    tone: Optional[str] = None,
    intensity: Optional[float] = None,
    delivery: Optional[str] = None,
) -> Tuple[str, str]:
    """Choose (speaker, instruct) from the character attributes.

    Gender/age pick the preset timbre; personality/expression/refine the
    delivery via ``instruct``.  Additional emotion/tone/intensity/delivery
    parameters can be passed to shape the voice performance.
    """
    attributes = attributes or {}

    # Gender detection: explicit "gender" attr (animals) or type.
    gender = _attr(attributes, "gender")
    if not gender and char_type in ("man", "woman"):
        gender = "Male" if char_type == "man" else "Female"

    if str(gender).lower().startswith("f") or char_type == "woman":
        speaker = FEMALE_SPEAKERS[0]
    else:
        speaker = MALE_SPEAKERS[0]

    # Age pushes toward young/old voices when we have matching presets.
    age_raw = _attr(attributes, "age", "age_range")
    age = _AGE_NORMALIZE.get(age_raw, age_raw.lower()) if age_raw else ""
    if age == "elderly" and speaker in FEMALE_SPEAKERS:
        speaker = "vivian"  # older-sounding female preset
    if age in ("child", "teenager", "young adult"):
        speaker = FEMALE_SPEAKERS[1] if speaker in FEMALE_SPEAKERS else MALE_SPEAKERS[1]

    # Delivery: personality first, then expression/mood.
    instruct_parts = []
    personality = _attr(attributes, "personality", "mood")
    if personality in INSTRUCT_BY_PERSONALITY:
        instruct_parts.append(INSTRUCT_BY_PERSONALITY[personality])
    expression = _attr(attributes, "expression", "mood")
    if expression in INSTRUCT_BY_EXPRESSION and not instruct_parts:
        instruct_parts.append(INSTRUCT_BY_EXPRESSION[expression])

    # Build from explicit emotion/tone/intensity/delivery parameters
    instruct = _build_instruct_from_emo_tone_delivery(
        emotion=emotion,
        tone=tone,
        intensity=intensity,
        delivery=delivery,
        personality=personality if personality else None,
        expression=expression if expression else None,
    )

    # If the _build_instruct function produced a result using the explicit
    # parameters, use it; otherwise fall back to the old instruct_parts logic.
    if instruct not in (None, _DEFAULT_INSTRUCT):
        return speaker, instruct

    instruct = ", ".join(instruct_parts) if instruct_parts else _DEFAULT_INSTRUCT
    return speaker, instruct


class VoiceEngine:
    """Lazy-loading wrapper around the local Qwen3-TTS models.

    Two checkpoints, loaded on demand:
    - ``design_model``: VoiceDesign 1.7B — prompt-driven voice creation.
    - ``model``: CustomVoice 1.7B — preset timbres (fallback).
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(
        self, model_dir: Optional[Path] = None, design_model_dir: Optional[Path] = None
    ):
        self.model_dir = Path(model_dir) if model_dir else QWEN3_TTS_MODEL_DIR
        self.design_model_dir = (
            Path(design_model_dir)
            if design_model_dir
            else QWEN3_TTS_VOICEDESIGN_MODEL_DIR
        )
        self._model = None
        self._design_model = None

    @classmethod
    def get_instance(cls) -> "VoiceEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def model(self):
        """CustomVoice 1.7B (preset timbres, delivery via instruct)."""
        if self._model is None:
            from qwen_tts import Qwen3TTSModel

            self._model = Qwen3TTSModel.from_pretrained(
                str(self.model_dir),
                device_map="mps",
                dtype="float16",
                attn_implementation="sdpa",
            )
            logger.info("Loaded Qwen3-TTS CustomVoice from %s", self.model_dir)
        return self._model

    @property
    def design_model(self):
        """VoiceDesign 1.7B (designs a voice from a natural-language prompt)."""
        if self._design_model is None:
            from qwen_tts import Qwen3TTSModel

            self._design_model = Qwen3TTSModel.from_pretrained(
                str(self.design_model_dir),
                device_map="mps",
                dtype="float16",
                attn_implementation="sdpa",
            )
            logger.info("Loaded Qwen3-TTS VoiceDesign from %s", self.design_model_dir)
        return self._design_model

    def design_model_available(self) -> bool:
        """True if the VoiceDesign checkpoint is present on disk."""
        return (
            self.design_model_dir.is_dir()
            and (self.design_model_dir / "model.safetensors").exists()
        )

    def unload(self) -> None:
        """Release the loaded TTS models and free accelerator memory.

        Each 1.7B checkpoint holds ~3.5 GB on MPS in float16; keeping both
        resident after a long export wastes RAM. Call this once generation
        is finished (the models lazily reload on the next use).
        """
        import gc

        released = self._model is not None or self._design_model is not None
        self._model = None
        self._design_model = None
        if not released:
            return
        gc.collect()
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        logger.info("Voice engine models unloaded")

    def generate_voice_line(
        self,
        line: str,
        speaker: str,
        instruct: str,
        language: str = "English",
    ) -> Tuple[np.ndarray, int]:
        """Synthesize a single line with CustomVoice (preset speaker + instruct)."""
        wavs, sr = self.model.generate_custom_voice(
            text=line,
            speaker=speaker,
            language=language,
            instruct=instruct,
            non_streaming_mode=True,
        )
        return wavs[0], sr

    def generate_designed_voice(
        self,
        line: str,
        instruct: str,
        language: str = "English",
    ) -> Tuple[np.ndarray, int]:
        """Design a brand-new voice from the prompt and speak the line.

        This is the model built for exactly this use case: the prompt
        describes the desired voice (timbre, emotion, pace) and the model
        creates it from scratch (``generate_voice_design``).
        """
        wavs, sr = self.design_model.generate_voice_design(
            text=line,
            instruct=instruct,
            language=language,
            non_streaming_mode=True,
        )
        return wavs[0], sr

    def generate_character_voice(
        self,
        name: str,
        project: str = "test_project",
        char_type: str = "person",
        attributes: Optional[Dict[str, str]] = None,
        output_dir: Optional[Path] = None,
        language: str = "English",
        emotion: Optional[str] = None,
        tone: Optional[str] = None,
        intensity: Optional[float] = None,
        delivery: Optional[str] = None,
        instruct: Optional[str] = None,
        force: bool = False,
    ) -> Tuple[Path, str]:
        """Generate (and cache) the character's introduction WAV.

        Returns (wav_path, instruct_used).

        - With a user ``instruct`` (regenerate-with-prompt): the VoiceDesign
          model creates a brand-new voice from the prompt.
        - Without ``instruct``: the CustomVoice model speaks with a preset
          timbre picked from the character's attributes.
        - ``force=True`` regenerates even if the WAV already exists.
        - ``emotion``, ``tone``, ``intensity``, ``delivery``: performance
          controls that shape the voice via the instruct string (within the
          capabilities of the selected TTS backend).
        """
        out_dir = Path(output_dir) if output_dir else character_dir(name, project)
        # Voice files live in a dedicated subfolder so the character folder
        # stays organized: reference image + versions + manifest at the top,
        # audio under voice/.
        voice_dir = out_dir / "voice"
        voice_dir.mkdir(parents=True, exist_ok=True)
        wav_path = voice_dir / "voice.wav"

        if wav_path.exists() and not force:
            logger.info("Voice already exists: %s", wav_path)
            return wav_path, ""

        line = build_voice_line(name, char_type, attributes)

        # Select speaker using attributes + explicit performance parameters.
        speaker, derived_instruct = pick_speaker(
            char_type,
            attributes,
            emotion=emotion,
            tone=tone,
            intensity=intensity,
            delivery=delivery,
        )
        # An explicit instruct (e.g. a user regeneration prompt) always wins
        # over the auto-derived performance instruct.
        explicit_instruct = (instruct or "").strip()
        effective_instruct = explicit_instruct or derived_instruct

        if explicit_instruct and self.design_model_available():
            # Prompt-driven voice design: the prompt creates the timbre.
            logger.info("Designing voice for %s from prompt", name)
            wav, sr = self.generate_designed_voice(
                line, effective_instruct, language=language
            )
            instruct_used = effective_instruct
        else:
            # Fallback: preset speaker + delivery nudge.
            logger.info("Generating voice for %s: %s (speaker=%s)", name, line, speaker)
            wav, sr = self.generate_voice_line(
                line,
                speaker,
                effective_instruct,
                language=language,
            )
            instruct_used = effective_instruct

        import soundfile as sf

        sf.write(str(wav_path), wav, sr)
        logger.info("Saved voice to %s", wav_path)
        return wav_path, instruct_used


voice_engine = VoiceEngine.get_instance()
