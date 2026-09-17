"""Sound engine: text-to-audio for sound effects and background music.

Wraps the locally downloaded MusicGen medium model (models.py:
``musicgen_medium``) to render the <sound prompt=".."/> and <music
prompt=".."/> segments of the story markup into actual audio.

Generated clips are cached by a hash of (prompt, duration) under the
project's ``sfx`` folder so an unchanged cue never regenerates.
"""

import hashlib
import logging
import threading
from pathlib import Path
from typing import Optional

from models import MODEL_PATHS
from utils.project_paths import project_dir

logger = logging.getLogger(__name__)

MUSICGEN_DIR = Path(MODEL_PATHS["music_generation"]) / "musicgen_medium"

# MusicGen generates ~50 audio-codec frames per second.
_FRAMES_PER_SECOND = 50

DEFAULT_SFX_SECONDS = 5.0
DEFAULT_MUSIC_SECONDS = 10.0


class SoundEngine:
    """Lazy-loading MusicGen wrapper (singleton, like VoiceEngine)."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir else MUSICGEN_DIR
        self._model = None
        self._processor = None

    @classmethod
    def get_instance(cls) -> "SoundEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def available(self) -> bool:
        """True when the MusicGen checkpoint is present on disk."""
        return self.model_dir.is_dir() and (self.model_dir / "config.json").exists()

    def unload(self) -> None:
        """Release the MusicGen model to free memory (lazily reloads on use)."""
        import gc

        if self._model is None and self._processor is None:
            return
        self._model = None
        self._processor = None
        gc.collect()
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        logger.info("Sound engine model unloaded")

    def _load(self):
        if self._model is None:
            from transformers import (
                AutoProcessor,
                MusicgenConfig,
                MusicgenForConditionalGeneration,
            )

            # Workaround for a transformers regression (seen in 4.57.x):
            # MusicgenForConditionalGeneration.config_class is wrongly the
            # decoder config, which breaks from_pretrained's isinstance check.
            if MusicgenForConditionalGeneration.config_class is not MusicgenConfig:
                MusicgenForConditionalGeneration.config_class = MusicgenConfig

            logger.info("Loading MusicGen from %s", self.model_dir)
            self._processor = AutoProcessor.from_pretrained(str(self.model_dir))
            self._model = MusicgenForConditionalGeneration.from_pretrained(
                str(self.model_dir)
            )
        return self._model, self._processor

    def generate(self, prompt: str, seconds: float = DEFAULT_SFX_SECONDS):
        """Render a prompt to audio. Returns (mono numpy array, sample_rate)."""
        model, processor = self._load()
        inputs = processor(text=[prompt], padding=True, return_tensors="pt")
        max_new_tokens = max(1, int(seconds * _FRAMES_PER_SECOND))
        audio = model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=3.0,
            max_new_tokens=max_new_tokens,
        )
        sr = model.config.audio_encoder.sampling_rate
        wav = audio[0, 0].cpu().numpy()
        return wav, sr


def _cue_cache_key(prompt: str, seconds: float) -> str:
    return hashlib.sha1(f"{prompt}|{seconds:.1f}".encode("utf-8")).hexdigest()[:16]


def cue_audio_path(project: str, prompt: str, seconds: float) -> Path:
    return project_dir(project) / "sfx" / f"cue_{_cue_cache_key(prompt, seconds)}.wav"


def generate_cue(
    project: str,
    prompt: str,
    seconds: float = DEFAULT_SFX_SECONDS,
    force: bool = False,
) -> Optional[Path]:
    """Generate (or reuse cached) audio for a sound/music cue.

    Returns None when the prompt is empty or MusicGen is not installed.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return None

    engine = SoundEngine.get_instance()
    if not engine.available():
        logger.warning("MusicGen not installed; skipping cue: %s", prompt)
        return None

    wav_path = cue_audio_path(project, prompt, seconds)
    if wav_path.exists() and not force:
        logger.info("Cue cache hit: %s", wav_path)
        return wav_path
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating cue (%.1fs): %s", seconds, prompt)
    wav, sr = engine.generate(prompt, seconds)

    import soundfile as sf

    sf.write(str(wav_path), wav, sr)
    return wav_path


sound_engine = SoundEngine.get_instance()
