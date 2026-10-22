"""
Low-strength img2img refinement pass for composed scenes.

The asset-composition pipeline produces deterministically correct but
visually "pasted" composites (mismatched lighting, no depth integration).
This module runs a LOW-strength img2img pass over the finished composite so
the diffusion model re-harmonizes lighting/shadows/grain while preserving
composition and character placement.

Design constraints (see multi-character scene design doc):
- strength is the key knob: 0.15-0.35. Never exceed 0.4 without
  `allow_high_strength=True` (higher values drift composition/identity).
- The refinement prompt must be scene-level and neutral (no character
  names/identity attributes) so identity does not drift.
- Never crash: on any failure, log and return the original composite path.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from generators.image_generator import resolve_model_path, cleanup_pipeline
from models import DIFFUSION_MODELS, get_model_config

DEFAULT_REFINEMENT_PROMPT = (
    "Photorealistic scene, natural stage lighting, integrated environment, "
    "cinematic, coherent lighting across all subjects, natural shadows, "
    "photographic quality"
)

MAX_SAFE_STRENGTH = 0.4


def generate_image_from_reference(
    reference_image: str,
    prompt: str,
    model_name: str = "sdxl",
    strength: float = 0.3,
    seed: int = 42,
    output_path: Optional[str] = None,
    steps: int = 30,
    guidance_scale: float = 7.5,
) -> str:
    """Generate a new character image while preserving the reference pose."""
    if model_name not in {"sdxl", "flux_dev"}:
        raise ValueError(
            f"Reference-image generation is not supported for model: {model_name}. "
            "Choose SDXL or FLUX.1 Dev."
        )
    if not Path(reference_image).is_file():
        raise FileNotFoundError(f"Reference image not found: {reference_image}")

    if strength > MAX_SAFE_STRENGTH:
        strength = MAX_SAFE_STRENGTH

    temporary_output = None
    if output_path is None:
        fd, output_path = tempfile.mkstemp(
            prefix="story_engine_character_", suffix=".png"
        )
        os.close(fd)
        temporary_output = output_path

    pipe = None
    try:
        import torch
        from PIL import Image

        model_path = resolve_model_path(
            "diffusion", model_name, DIFFUSION_MODELS[model_name]
        )
        device, torch_dtype = get_model_config("diffusion")
        if model_name == "flux_dev":
            from diffusers import FluxImg2ImgPipeline

            pipe = FluxImg2ImgPipeline.from_pretrained(
                model_path, torch_dtype=torch_dtype
            )
        else:
            from diffusers import StableDiffusionXLImg2ImgPipeline

            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_path, torch_dtype=torch_dtype, safety_checker=None
            )
        pipe = pipe.to(device)

        with Image.open(reference_image) as source:
            init_image = source.convert("RGB")
            kwargs = {
                "prompt": (
                    f"{prompt.rstrip().rstrip('.,')}. Preserve the same pose, "
                    "body position, camera angle, and framing as the reference image."
                ),
                "image": init_image,
                "strength": strength,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "generator": torch.Generator(device=device).manual_seed(seed),
            }
            if model_name == "flux_dev":
                kwargs["max_sequence_length"] = 512

            result = pipe(**kwargs)
            result.images[0].save(output_path)
        return output_path
    except Exception:
        if temporary_output and os.path.exists(temporary_output):
            os.unlink(temporary_output)
        raise
    finally:
        cleanup_pipeline(pipe)


def refine_composite(
    composite_image_path: str,
    prompt: str,
    model_name: str = "sdxl",
    strength: float = 0.25,
    seed: int = 42,
    output_path: Optional[str] = None,
    steps: int = 30,
    guidance_scale: float = 7.5,
    allow_high_strength: bool = False,
) -> str:
    """Run a low-strength img2img pass on a composite image to improve visual
    integration while preserving composition and character placement.

    Returns path to the refined image (or the original composite path if
    refinement fails for any reason — this function never raises).
    """
    pipe = None
    try:
        import torch
        from PIL import Image

        if not prompt:
            prompt = DEFAULT_REFINEMENT_PROMPT

        if strength > MAX_SAFE_STRENGTH and not allow_high_strength:
            print(
                f"[img2img_engine] strength {strength} exceeds safe max "
                f"{MAX_SAFE_STRENGTH}; clamping (pass allow_high_strength=True to override)."
            )
            strength = MAX_SAFE_STRENGTH

        if model_name not in DIFFUSION_MODELS:
            raise ValueError(f"Unsupported img2img model: {model_name}")

        model_path = resolve_model_path(
            "diffusion", model_name, DIFFUSION_MODELS[model_name]
        )
        device, torch_dtype = get_model_config("diffusion")

        print(
            f"[img2img_engine] Refinement pass: model={model_name}, "
            f"strength={strength}, seed={seed}, steps={steps}"
        )
        print(f"[img2img_engine] Refinement prompt: {prompt}")

        init_image = Image.open(composite_image_path).convert("RGB")

        if model_name == "flux_dev":
            from diffusers import FluxImg2ImgPipeline

            pipe = FluxImg2ImgPipeline.from_pretrained(
                model_path, torch_dtype=torch_dtype
            )
        else:
            from diffusers import StableDiffusionXLImg2ImgPipeline

            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_path, torch_dtype=torch_dtype, safety_checker=None
            )
        pipe = pipe.to(device)

        generator = torch.Generator(device=device).manual_seed(seed)

        kwargs = dict(
            prompt=prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        if model_name == "flux_dev":
            kwargs["max_sequence_length"] = 512

        result = pipe(**kwargs)
        refined = result.images[0]

        if output_path is None:
            src = Path(composite_image_path)
            output_path = str(src.with_name("scene_refined.png"))
        refined.save(output_path)
        print(f"[img2img_engine] Refined image saved to: {output_path}")
        return output_path

    except Exception as e:
        print(
            f"[img2img_engine] Refinement failed ({e}); returning original composite."
        )
        return composite_image_path
    finally:
        cleanup_pipeline(pipe)
