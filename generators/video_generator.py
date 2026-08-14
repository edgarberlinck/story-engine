#!/usr/bin/env python3
"""
Image-to-video generation across all supported i2v models.

Supported models (see models.py IMAGE_TO_VIDEO_MODELS):
  - wan22_i2v (default)  : Wan-AI/Wan2.2-I2V-A14B
  - hunyuan_video_i2v    : tencent/HunyuanVideo-I2V

Each model is invoked with its own correct pipeline class and parameters.
"""

import json
import os
import sys
import time
from pathlib import Path

# Avoid HuggingFace tokenizers spawning fork-based parallelism (leaks
# semaphores on macOS and triggers resource_tracker warnings at shutdown).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from PIL import Image

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generators.image_generator import cleanup_pipeline
from models import IMAGE_TO_VIDEO_MODELS, MODEL_PATHS, get_model_config
from utils.model_metrics import get_memory_usage

AVAILABLE_VIDEO_MODELS = dict(IMAGE_TO_VIDEO_MODELS)
DEFAULT_VIDEO_MODEL = "wan22_i2v"

# Per-model generation parameters. Each model has different native
# resolutions, frame counts, fps and guidance requirements.
MODEL_GENERATION_PARAMS = {
    "wan22_i2v": {
        "width": 832,
        "height": 480,
        "num_frames": 81,
        "fps": 16,
        "guidance_scale": 3.5,
        "num_inference_steps": 40,
        "negative_prompt": (
            "bright colors, overexposed, static, blurred details, subtitles, "
            "worst quality, low quality, deformed, disfigured, extra limbs, "
            "fused fingers, still frame, messy background"
        ),
    },
    "hunyuan_video_i2v": {
        "width": 720,
        "height": 480,
        "num_frames": 61,
        "fps": 15,
        "guidance_scale": 6.5,
        "num_inference_steps": 30,
        "negative_prompt": None,  # HunyuanVideo I2V does not use CFG negatives
    },
}


def resolve_video_model_path(model_name: str) -> str:
    """Resolve an i2v model to its local path, falling back to the hub id."""
    hub_id = AVAILABLE_VIDEO_MODELS[model_name]
    local_path = os.path.join(
        str(project_root), MODEL_PATHS["image_to_video"], model_name
    )
    if os.path.isdir(local_path) and os.listdir(local_path):
        return local_path
    print(
        f"Warning: {model_name} not found locally at {local_path}. "
        f"Run 'make install' first. Falling back to hub download: {hub_id}"
    )
    return hub_id


def _load_pipeline(model_name: str, model_path: str, device: str, torch_dtype):
    """Load the correct diffusers pipeline for the given i2v model."""
    if model_name == "wan22_i2v":
        from diffusers import WanImageToVideoPipeline

        pipe = WanImageToVideoPipeline.from_pretrained(
            model_path, torch_dtype=torch_dtype
        )
    elif model_name == "hunyuan_video_i2v":
        from diffusers import HunyuanVideoImageToVideoPipeline

        pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(
            model_path, torch_dtype=torch_dtype
        )
    else:
        raise ValueError(f"Unsupported image-to-video model: {model_name}")

    # These models are large; offload when CUDA is available, otherwise
    # move to the resolved device (mps/cpu).
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    return pipe


def _prepare_image(image_path: str, width: int, height: int) -> Image.Image:
    """Load and resize the conditioning image to the model's resolution."""
    image = Image.open(image_path).convert("RGB")
    return image.resize((width, height), Image.LANCZOS)


def generate_video(
    image_path: str,
    prompt: str,
    model_name: str = DEFAULT_VIDEO_MODEL,
    output_dir: str = "outputs",
    output_basename: str = None,
    seed: int = 42,
    **overrides,
):
    """Generate a video from an image using the specified i2v model.

    Args:
        image_path: Path to the conditioning image (the scene).
        prompt: Motion/scene description guiding the animation.
        model_name: One of AVAILABLE_VIDEO_MODELS (default: wan22_i2v).
        output_dir: Directory where the video and metrics are written.
        output_basename: Base filename (defaults to <image stem>_<model>).
        seed: Random seed for reproducibility.
        **overrides: Override any per-model generation parameter
                     (width, height, num_frames, fps, guidance_scale, ...).

    Returns:
        dict with 'video_path', 'metrics_path' and the metrics themselves.
    """
    if model_name not in AVAILABLE_VIDEO_MODELS:
        raise ValueError(
            f"Unsupported image-to-video model: {model_name}. "
            f"Available: {', '.join(AVAILABLE_VIDEO_MODELS)}"
        )
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Conditioning image not found: {image_path}")

    from diffusers.utils import export_to_video

    params = {**MODEL_GENERATION_PARAMS[model_name], **overrides}
    fps = params.pop("fps")
    negative_prompt = params.pop("negative_prompt")

    model_path = resolve_video_model_path(model_name)
    device, torch_dtype = get_model_config("image_to_video")
    print(f"Using image-to-video model: {model_name} ({model_path})")
    print(f"Device: {device}, dtype: {torch_dtype}")

    pipe = _load_pipeline(model_name, model_path, device, torch_dtype)
    try:
        image = _prepare_image(image_path, params["width"], params["height"])
        generator = torch.Generator(device="cpu").manual_seed(seed)

        call_kwargs = dict(
            image=image,
            prompt=prompt,
            generator=generator,
            **params,
        )
        if negative_prompt is not None:
            call_kwargs["negative_prompt"] = negative_prompt

        start_time = time.time()
        start_memory = get_memory_usage()

        result = pipe(**call_kwargs)
        frames = result.frames[0]

        duration_ms = int((time.time() - start_time) * 1000)
        peak_memory_mb = int(get_memory_usage() - start_memory)
    finally:
        cleanup_pipeline(pipe)
        pipe = None

    os.makedirs(output_dir, exist_ok=True)
    if output_basename is None:
        output_basename = f"{Path(image_path).stem}_{model_name}"
    video_path = os.path.join(output_dir, f"{output_basename}.mp4")
    export_to_video(frames, video_path, fps=fps)
    print(f"Video saved to: {video_path}")

    metrics = {
        "model": model_name,
        "prompt": prompt,
        "image": image_path,
        "seed": seed,
        "fps": fps,
        "duration_ms": duration_ms,
        "peak_memory_mb": peak_memory_mb,
        "output": video_path,
        **{k: v for k, v in params.items()},
    }
    metrics_path = os.path.join(
        output_dir, f"{output_basename}_benchmark_metrics.json"
    )
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    return {"video_path": video_path, "metrics_path": metrics_path, "metrics": metrics}
