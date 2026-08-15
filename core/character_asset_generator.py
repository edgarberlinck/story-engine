"""
Character asset generation for scene composition.

Generates scene-specific, per-character images against a plain/neutral
background so they can be reliably segmented (see `core/scene_compositor.py`)
and deterministically composed into a scene (see `core/scene_pipeline.py`).

This module deliberately reuses existing infrastructure:
- `generators/image_engine.py::generate_images` for the actual text-to-image
  call (no new diffusion pipeline classes).
- `core/scene_planner.py::ResolvedCharacter` for identity/scene-presentation
  data (Stage A output).
- `utils/token_budget.py` for CLIP token budget fitting on the per-character
  prompt.

No image-to-image / reference conditioning is used anywhere here — this is
still plain text-to-image, just invoked once per character with a prompt
engineered for easy background removal.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Any
from pathlib import Path

from utils.token_budget import TokenBudgetManager


# Prompt fragments that make the background trivial to segment out.
PLAIN_BACKGROUND_SUFFIX = (
    "plain gray background, studio lighting, centered subject, "
    "full body, entire body visible, photorealistic detail on face and hands"
)

# Generation-time artifacts that must never leak into asset prompts, even if
# an upstream (LLM) stage copied them from the stored character prompt.
_FORBIDDEN_ARTIFACTS = [
    r"\bfantasy clothing\b",
    r"\bstanding upright\b",
    r"\bneutral background\b",
    r"\bnot a portrait\b",
    r"\bnot a close-?up\b",
    r"\bfeet touching the ground\b",
    r"\bcamera far from subject\b",
    r"\bfull length wide shot\b",
]


def _sanitize_fragment(text: str) -> str:
    import re
    cleaned = text
    for pattern in _FORBIDDEN_ARTIFACTS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",\s*,+", ",", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" ,")


@dataclass
class CharacterAsset:
    """Result of generating one character's scene-specific asset."""
    name: str
    image_path: str
    prompt_used: str
    seed: int
    mask_path: Optional[str] = None
    cutout_path: Optional[str] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    segmentation_ok: bool = False
    segmentation_method: Optional[str] = None
    validation: Optional[Dict[str, Any]] = None


def character_seed(name: str, base_seed: int = 42) -> int:
    """Deterministic per-character seed, derived from the character's name
    (not the scene number), so the same character starts from a similar
    noise basin across different scenes. This is an approximation of
    identity consistency, not a guarantee (see design doc §3.4).
    """
    # Simple, stable string hash -> small positive offset.
    digest = sum(ord(c) for c in name.lower())
    return base_seed + (digest % 1000)


def build_character_asset_prompt(
    resolved_character: Any,
    project_style: Optional[str] = None,
    token_limit: int = 77,
) -> str:
    """Build the plain-background asset prompt for one resolved character.

    Accepts a `ResolvedCharacter` (from `core/scene_planner.py`) or any
    object/dict exposing `name`, `identity`, `scene_presentation`, and
    optionally `scene_pose` / `scene_action`.
    """

    def _get(attr, default=None):
        if isinstance(resolved_character, dict):
            return resolved_character.get(attr, default)
        return getattr(resolved_character, attr, default)

    name = _get("name", "")
    identity = _get("identity", []) or []
    scene_presentation = _get("scene_presentation", []) or []
    scene_pose = _get("scene_pose", None)
    scene_action = _get("scene_action", None)

    parts: List[str] = []
    parts.append(name)
    if identity:
        parts.append(_sanitize_fragment(", ".join(identity)))
    if scene_presentation:
        parts.append(_sanitize_fragment(", ".join(scene_presentation)))
    if scene_pose:
        parts.append(_sanitize_fragment(scene_pose))
    if scene_action:
        parts.append(_sanitize_fragment(scene_action))
    if project_style:
        parts.append(project_style)
    parts.append(PLAIN_BACKGROUND_SUFFIX)

    prompt = ", ".join(p for p in parts if p)

    # Fit to CLIP token budget (per-character prompt, own 77-token budget).
    manager = TokenBudgetManager()
    if manager.count_tokens(prompt) > token_limit:
        prompt = manager.truncate_to_tokens(prompt, token_limit)

    return prompt


def generate_character_asset(
    resolved_character: Any,
    project: str,
    project_style: Optional[str] = None,
    model: str = "sdxl",
    base_seed: int = 42,
    task_name_prefix: str = "asset",
    output_dir: Optional[Path] = None,
) -> CharacterAsset:
    """Generate a single scene-specific character asset with a plain
    background, suitable for later segmentation and composition.
    """
    from generators.image_engine import generate_images

    def _get(attr, default=None):
        if isinstance(resolved_character, dict):
            return resolved_character.get(attr, default)
        return getattr(resolved_character, attr, default)

    name = _get("name", "character")
    prompt = build_character_asset_prompt(resolved_character, project_style)
    seed = character_seed(name, base_seed)

    print(f"[character_asset_generator] Generating asset for '{name}' (seed={seed})")
    print(f"  Prompt: {prompt}")

    files = generate_images(
        prompt=prompt,
        model_name=model,
        seed=seed,
        task_name=f"{task_name_prefix}_{name}",
    )
    image_path = files[0]

    if output_dir is not None:
        import shutil
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"asset_{name}.png"
        shutil.copy(image_path, target)
        image_path = str(target)

    return CharacterAsset(
        name=name,
        image_path=image_path,
        prompt_used=prompt,
        seed=seed,
    )


def generate_character_assets(
    resolved_characters: List[Any],
    project: str,
    project_style: Optional[str] = None,
    model: str = "sdxl",
    base_seed: int = 42,
    output_dir: Optional[Path] = None,
) -> List[CharacterAsset]:
    """Generate scene-specific assets for every resolved character."""
    assets = []
    for rc in resolved_characters:
        asset = generate_character_asset(
            rc,
            project=project,
            project_style=project_style,
            model=model,
            base_seed=base_seed,
            output_dir=output_dir,
        )
        assets.append(asset)
    return assets
