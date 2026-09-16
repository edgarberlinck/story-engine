#!/usr/bin/env python3
"""
Image-to-video generation across all supported i2v models.

Supported models (see models.py IMAGE_TO_VIDEO_MODELS):
   - ltx_video_095_i2v (default): Lightricks/LTX-Video-0.9.5
     (LTXImageToVideoPipeline, compact ~3.6 GB transformer — runs on MPS).

Benchmark decision (2026-08): LTX-Video 0.9.5 is the surviving i2v model.
Wan 2.2 I2V A14B was DROPPED: a dual 14B-expert MoE stored F32 on disk
(106 GB of transformers) that OOMs (~23 GB free) during model load on a
64 GB Apple-Silicon Mac, independent of resolution/frames. HunyuanVideo-I2V
is also dropped.

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

# No i2v models available; video generation archived due to hardware
# constraints. Set to None to indicate no model is available until a machine
# with sufficient GPU memory (>= 32 GB VRAM for LTX-Video or >= 80 GB unified
# memory for larger models) is available.
DEFAULT_VIDEO_MODEL = None

# Per-model generation parameters. Each model has different native
# resolutions, frame counts, fps and guidance requirements.
# No models are currently available (archived see models.py).
# Parameters will be populated when a video model is re-introduced.
MODEL_GENERATION_PARAMS = {}


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
    """Load the correct diffusers pipeline for the given i2v model.
    New models add an entry here plus one in MODEL_GENERATION_PARAMS."""
    if model_name == "ltx_video_095_i2v":
        from diffusers import LTXImageToVideoPipeline

        pipe = LTXImageToVideoPipeline.from_pretrained(
            model_path, torch_dtype=torch_dtype
        )
    else:
        raise ValueError(f"Unsupported image-to-video model: {model_name}")

    # The LTX-Video text encoder is large (~17 GB); offloading it to CPU one
    # component at a time keeps the footprint small enough for an accelerator
    # with limited free memory (e.g. a 64 GB Apple-Silicon Mac on MPS).
    # Without an accelerator (cpu), load it resident on CPU.
    if device in ("cuda", "mps"):
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    return pipe


def _prepare_image(
    image_path: str, width: int, height: int, aspect_tolerance: float = 0.02
) -> Image.Image:
    """Load and prepare the conditioning image for the video model.

    The model samples at its own ``width x height``. Naively calling
    ``image.resize((width, height))`` when the source aspect ratio differs from
    the target aspect ratio STRETCHES the frame (e.g. a square 1024x1024 scene
    squished into a 576x1024 video) -- that is the "aspect ratio looks off" /
    "movements distort the video" symptom. To avoid it we:

    1. If the source aspect ratio already matches the target within
     ``aspect_tolerance``, do a straight resize (no distortion).
    2. Otherwise "contain-fit" the source into the target box and pad the
     leftover edges by REPLICATING the image border (edge extension), so the
     frame is never distorted. When a scene is generated at the model's native
     resolution this branch never triggers and the frame is a clean resize.
    """
    image = Image.open(image_path).convert("RGB")

    src_w, src_h = image.size
    target_ar = width / float(height)
    src_ar = src_w / float(src_h)

    # Branch 1: matching aspect ratio -> a clean, undistorted resize.
    if abs(src_ar - target_ar) <= aspect_tolerance:
        if (src_w, src_h) != (width, height):
            image = image.resize((width, height), Image.LANCZOS)
        return image

    # Branch 2: differing aspect ratio -> fit and pad (never distort).
    scale = min(width / float(src_w), height / float(src_h))
    fit_w = max(1, int(round(src_w * scale)))
    fit_h = max(1, int(round(src_h * scale)))
    fitted = image.resize((fit_w, fit_h), Image.LANCZOS)

    left = (width - fit_w) // 2
    top = (height - fit_h) // 2

    # Edge-extend the frame into the target box (replicate the border pixels).
    # PIL's resize padding is only 1px, so we replicate the 1px edges of the
    # fitted frame to fill each border strip, then paste the fitted frame in the
    # centre. No hard dependency on OpenCV.
    canvas = Image.new("RGB", (width, height))

    # Top / bottom strips mirror the fitted frame's top / bottom row.
    top_row = fitted.crop((0, 0, fit_w, 1))
    for y in range(0, top):
        canvas.paste(top_row, (left, y))
    for y in range(top + fit_h, height):
        canvas.paste(fitted.crop((0, fit_h - 1, fit_w, fit_h)), (left, y))

    # Left / right strips mirror the fitted frame's left / right column
    # (only within the non-overlapping vertical band).
    left_col = fitted.crop((0, 0, 1, fit_h))
    for x in range(0, left):
        canvas.paste(left_col, (x, top))
    for x in range(left + fit_w, width):
        canvas.paste(fitted.crop((fit_w - 1, 0, fit_w, fit_h)), (x, top))

    canvas.paste(fitted, (left, top))
    return canvas


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
        model_name: One of AVAILABLE_VIDEO_MODELS
          (default: ltx_video_095_i2v).
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
        # The generator must be on the same device as the pipeline's inference
        # device (e.g. "mps" on Apple Silicon via enable_model_cpu_offload); a
        # cpu generator on an accelerated pipeline fails at sampling with
        # "Expected a '<device>' device type for generator but found 'cpu'".
        generator = torch.Generator(device=device).manual_seed(seed)

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
