"""Dedicated text-to-sound-effects generation using Stable Audio Open."""

import logging
import math
import threading
from pathlib import Path
from typing import Optional

from models import MODEL_PATHS, get_model_config

logger = logging.getLogger(__name__)

STABLE_AUDIO_DIR = Path(MODEL_PATHS["sound_effects"]) / "stable_audio_open"
DEFAULT_SFX_SECONDS = 5.0
MAX_SFX_SECONDS = 47.0


class SoundEffectsEngine:
    """Lazy-loading local Stable Audio Open pipeline."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir else STABLE_AUDIO_DIR
        self._pipeline = None

    @classmethod
    def get_instance(cls) -> "SoundEffectsEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def available(self) -> bool:
        """True when a local Diffusers checkpoint is present."""
        return (
            self.model_dir.is_dir() and (self.model_dir / "model_index.json").exists()
        )

    def _load(self):
        if self._pipeline is None:
            from diffusers import StableAudioPipeline

            device, dtype = get_model_config("sound_effects")
            logger.info("Loading Stable Audio Open from %s", self.model_dir)
            pipeline = StableAudioPipeline.from_pretrained(
                str(self.model_dir), torch_dtype=dtype, local_files_only=True
            )
            self._pipeline = pipeline.to(device)
        return self._pipeline

    def generate(self, prompt: str, seconds: float = DEFAULT_SFX_SECONDS):
        """Return (samples-by-channels numpy array, sample_rate)."""
        if not math.isfinite(seconds) or not 0 < seconds <= MAX_SFX_SECONDS:
            raise ValueError(
                f"Sound effects duration must be between 0 and {MAX_SFX_SECONDS}s"
            )
        if not (prompt or "").strip():
            raise ValueError("Sound effects prompt must not be empty")

        import torch

        pipeline = self._load()
        with torch.inference_mode():
            audio = pipeline(
                prompt=prompt.strip(),
                negative_prompt="Low quality.",
                audio_start_in_s=0.0,
                audio_end_in_s=seconds,
                num_inference_steps=100,
                guidance_scale=7.0,
                output_type="np",
            ).audios[0]
        sr = pipeline.vae.config.sampling_rate
        return audio.T[: int(seconds * sr)], sr

    def unload(self) -> None:
        """Release the pipeline; reload lazily on the next generation."""
        if self._pipeline is None:
            return

        import gc
        import torch

        self._pipeline = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Sound effects model unloaded")


sound_effects_engine = SoundEffectsEngine.get_instance()
