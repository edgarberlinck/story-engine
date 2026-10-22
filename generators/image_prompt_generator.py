"""Generate editable character prompts from reference images."""

from pathlib import Path


IMAGE_CAPTIONING_MODEL = "Salesforce/blip-image-captioning-base"


def generate_prompt_from_image(image_path: str) -> str:
    """Describe a reference image and turn it into a pose-aware image prompt."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Reference image not found: {image_path}")

    generator = None
    try:
        from PIL import Image
        from transformers import pipeline

        from models import get_model_config

        device, _ = get_model_config("text_generation")
        generator = pipeline(
            "image-to-text", model=IMAGE_CAPTIONING_MODEL, device=device
        )
        with Image.open(image_path) as source:
            result = generator(source.convert("RGB"), max_new_tokens=80)

        caption = result[0].get("generated_text", "").strip()
        if not caption:
            raise ValueError("The image model did not return a description.")

        return (
            f"Character appearance and pose: {caption}. "
            "Preserve the same pose, body position, camera angle, and framing "
            "as the reference image."
        )
    finally:
        from generators.image_generator import cleanup_pipeline

        cleanup_pipeline(generator)
