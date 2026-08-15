#!/usr/bin/env python3
"""Reference-conditioned holistic scene generation.

Generates a complete scene in a single diffusion invocation while conditioning on
character reference images as FLUX.2 reference latents. This replaces the
"per-character asset -> segmentation -> compositing" path for reference-capable
models (currently ``flux_klein`` via :class:`Flux2KleinKVPipeline`), mirroring how
the reference project (fairytale-generator) conditions on character references.

No segmentation or compositing is performed here; the model renders the characters
in-context.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Avoid HuggingFace tokenizers spawning fork-based parallelism (leaks semaphores
# on macOS and triggers resource_tracker warnings at shutdown).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from PIL import Image
from diffusers import Flux2KleinKVPipeline
import torch

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models import DIFFUSION_MODELS, get_model_config
from generators.image_generator import resolve_model_path, cleanup_pipeline

# Reference images are downscaled to this maximum long side before conditioning to
# limit MPS memory pressure (multiple references inflate attention/KV and VAE
# latent cost). Must be a multiple of 16 after rounding.
_MAX_REFERENCE_SIDE = 512
_MIN_REFERENCE_SIDE = 64


def _resize_reference(image: Image.Image) -> Image.Image:
    """Downscale a reference image so its long side is <= MAX_REFERENCE_SIDE.

    Preserves aspect ratio, never upscales, and rounds dimensions down to a
    multiple of 16 (with a minimum of MIN_REFERENCE_SIDE per side).
    """
    width, height = image.size
    long_side = max(width, height)
    if long_side <= _MAX_REFERENCE_SIDE:
        new_w, new_h = width, height
    else:
        scale = _MAX_REFERENCE_SIDE / float(long_side)
        new_w = int(width * scale)
        new_h = int(height * scale)

    # Round down to a multiple of 16, enforcing a minimum.
    new_w = max((new_w // 16) * 16, _MIN_REFERENCE_SIDE)
    new_h = max((new_h // 16) * 16, _MIN_REFERENCE_SIDE)

    if (new_w, new_h) != (width, height):
        image = image.resize((new_w, new_h), Image.LANCZOS)
    return image


def _prepare_references(reference_image_paths: list[str]) -> list[Image.Image]:
    """Load + resize reference images to RGB PIL images.

    Unreadable paths are skipped with a warning. At least one valid reference is
    required for reference-conditioned generation.
    """
    prepared: list[Image.Image] = []
    for path in reference_image_paths:
        try:
            with Image.open(path) as img:
                rgb = img.convert("RGB")
            prepared.append(_resize_reference(rgb))
        except Exception as exc:  # noqa: BLE001 - tolerate unreadable references
            print(f"WARNING: skipping unreadable reference image '{path}': {exc}")
    return prepared


def load_reference_scene_pipeline(model_path: str, torch_dtype):
    """Load the FLUX.2 Klein KV pipeline for reference conditioning.

    A module-level seam so tests can patch it without constructing diffusers
    pipelines (consistent with how the codebase patches ``hf_pipeline`` /
    ``generate_images``).
    """
    return Flux2KleinKVPipeline.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
    )


def generate_reference_conditioned_scene(
    prompt: str,
    reference_image_paths: list[str],
    *,
    model_name: str = "flux_klein",
    seed: int = 42,
    steps: int = 4,
    width: int = 1024,
    height: int = 1024,
    num_images: int = 1,
    task_name: str | None = None,
) -> list[str]:
    """Generate a complete scene conditioning on character reference images.

    Args:
        prompt: Holistic scene prompt (may describe all characters in context).
        reference_image_paths: Ordered paths to character reference images.
        model_name: Reference-capable diffusion model (default ``flux_klein``).
        seed: Random seed for deterministic generation.
        steps: Number of denoising steps (default 4 for the distilled KV pipeline).
        width/height: Output dimensions.
        num_images: Number of images to generate.
        task_name: Optional filename base.

    Returns:
        list[str] of generated PNG paths.

    Raises:
        ValueError: If no reference images are provided.
    """
    if not reference_image_paths:
        raise ValueError("reference_image_paths must be non-empty for reference-conditioned generation")

    references = _prepare_references(reference_image_paths)
    if not references:
        raise ValueError("No usable reference images after preprocessing")

    if model_name not in DIFFUSION_MODELS:
        raise ValueError(f"Unsupported diffusion model: {model_name}")
    model_path = resolve_model_path("diffusion", model_name, DIFFUSION_MODELS[model_name])
    print(f"Using reference-conditioned diffusion model: {model_name}")
    print(f"Model path: {model_path}")

    device, torch_dtype = get_model_config("diffusion")

    pipe = None
    try:
        pipe = load_reference_scene_pipeline(model_path, torch_dtype)
        pipe = pipe.to(device)
        generator = torch.Generator(device=device).manual_seed(seed)

        result = pipe(
            image=references,
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            num_images_per_prompt=num_images,
            generator=generator,
            max_sequence_length=512,
        )
    finally:
        cleanup_pipeline(pipe)
        pipe = None

    generated_images: list[str] = []
    filename_base = task_name if task_name else "reference_conditioned_scene"
    os.makedirs("outputs", exist_ok=True)
    for i, image in enumerate(result.images):
        if num_images > 1:
            output_path = f"outputs/{filename_base}_{i + 1:03d}.png"
        else:
            output_path = f"outputs/{filename_base}.png"
        image.save(output_path)
        generated_images.append(output_path)
        print(f"Image {i + 1} saved to: {output_path}")

    return generated_images


__all__ = [
    "generate_reference_conditioned_scene",
    "load_reference_scene_pipeline",
]
