"""Dedicated local Stable Audio Open generation for environmental sound effects."""

import gc
import logging
import math
import threading
from pathlib import Path
from typing import Optional

from models import MODEL_PATHS, get_model_config

logger = logging.getLogger(__name__)

MODEL_NAME = "stable_audio_open"
MODEL_DIR = Path(MODEL_PATHS["sound_effects"]) / MODEL_NAME
DEFAULT_SFX_SECONDS = 5.0
MAX_SFX_SECONDS = 47.0


class SoundEffectsGenerator:
    """Lazy-loading singleton, independent of the MusicGen music engine."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self._pipeline = None

    @classmethod
    def get_instance(cls) -> "SoundEffectsGenerator":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def available(self) -> bool:
        return self.model_dir.is_dir() and (
            self.model_dir / "model_index.json"
        ).is_file()

    def _load(self):
        if self._pipeline is None:
            import torch
            from diffusers import StableAudioPipeline

            device, dtype = get_model_config("sound_effects")
            if device == "cuda":
                dtype = torch.float16
            logger.info("Loading Stable Audio Open from %s", self.model_dir)
            pipeline = StableAudioPipeline.from_pretrained(
                str(self.model_dir), torch_dtype=dtype, local_files_only=True
            )
            self._pipeline = pipeline.to(device)
        return self._pipeline

    def generate(self, prompt: str, seconds: float = DEFAULT_SFX_SECONDS):
        """Return (mono numpy waveform, sample rate) for a sound-effects prompt."""
        if not math.isfinite(seconds) or not 0 < seconds <= MAX_SFX_SECONDS:
            raise ValueError(f"Sound effects duration must be in (0, {MAX_SFX_SECONDS}]")
        if not (prompt or "").strip():
            raise ValueError("Sound effects prompt must not be empty")

        pipeline = self._load()
        audio = pipeline(
            prompt=prompt,
            negative_prompt="Low quality, music.",
            num_inference_steps=100,
            audio_start_in_s=0.0,
            audio_end_in_s=seconds,
            num_waveforms_per_prompt=1,
        ).audios[0]
        wav = audio.float().mean(dim=0).cpu().numpy()
        return wav, pipeline.vae.sampling_rate

    def unload(self) -> None:
        if self._pipeline is None:
            return
        self._pipeline = None
        gc.collect()
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Sound effects model unloaded")


sound_effects_generator = SoundEffectsGenerator.get_instance()
