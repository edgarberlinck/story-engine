"""Dedicated text-to-sound-effects generation using Stable Audio Open."""

import logging
import math
import threading
from pathlib import Path
from typing import Optional

from models import MODEL_PATHS, SOUND_EFFECTS_MODELS, get_model_config

logger = logging.getLogger(__name__)

MODEL_NAME = "stable_audio_open"
MODEL_ID = SOUND_EFFECTS_MODELS[MODEL_NAME]
MODEL_DIR = (
    Path(__file__).resolve().parent.parent / MODEL_PATHS["sound_effects"] / MODEL_NAME
)
DEFAULT_SFX_SECONDS = 5.0
MAX_SFX_SECONDS = 47.0
NEGATIVE_PROMPT = "Music, speech, low quality, distortion."
INFERENCE_STEPS = 100


class SoundEffectsGenerator:
    """Lazy-loading singleton, independent of the MusicGen music engine."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir is not None else MODEL_DIR
        self._pipeline = None

    @classmethod
    def get_instance(cls) -> "SoundEffectsGenerator":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def available(self) -> bool:
        return (self.model_dir / "model_index.json").is_file()

    def _load(self):
        if self._pipeline is None:
            import torch
            from diffusers import StableAudioPipeline

            device, dtype = get_model_config("sound_effects")
            # Full precision avoids half-precision instability on CPU/MPS.
            if device != "cuda":
                dtype = torch.float32
            logger.info("Loading Stable Audio Open from %s", self.model_dir)
            pipeline = StableAudioPipeline.from_pretrained(
                str(self.model_dir),
                torch_dtype=dtype,
                local_files_only=True,
                use_safetensors=True,
            )
            self._pipeline = pipeline.to(device)
        return self._pipeline

    def generate(self, prompt: str, seconds: float = DEFAULT_SFX_SECONDS):
        """Return (mono numpy waveform, sample rate), matching the music API."""
        seconds = float(seconds)
        if not math.isfinite(seconds) or not 0 < seconds <= MAX_SFX_SECONDS:
            raise ValueError(f"Sound-effect duration must be in (0, {MAX_SFX_SECONDS}]")
        pipeline = self._load()
        audio = pipeline(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            audio_end_in_s=seconds,
            num_inference_steps=INFERENCE_STEPS,
            num_waveforms_per_prompt=1,
            output_type="pt",
        ).audios[0]
        sample_rate = pipeline.vae.sampling_rate
        wav = audio.detach().float().mean(dim=0).cpu().numpy()
        return wav[: int(seconds * sample_rate)], sample_rate

    def unload(self) -> None:
        """Release the sound-effects pipeline before mastering the audiobook."""
        import gc

        if self._pipeline is None:
            return
        self._pipeline = None
        gc.collect()
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        logger.info("Sound-effects generator unloaded")


sound_effects_generator = SoundEffectsGenerator.get_instance()
