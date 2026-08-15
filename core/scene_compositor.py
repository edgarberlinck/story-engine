"""
Deterministic scene composition: segmentation + placement/blending.

Pure image operations (PIL + numpy), no diffusion calls. This is the
asset-composition **fallback** path (see
`docs/story-engine-reference-conditioned-scene-design.md` §2), used when a
model lacks reference-conditioning support:

1. Segment each character asset (DETR panoptic primary, chroma-key fallback).
2. Composite background + character cutouts using explicit anchor points,
   scale, z-order, soft shadow, and edge feathering.

This module never crashes the pipeline: every failure degrades to a cheaper
fallback and logs a warning (matching the project's existing defensive
coding style, see `utils/face_check.py` / `scene_planner.py`).
"""

from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import warnings

import numpy as np
from PIL import Image, ImageFilter, ImageDraw

try:
    from models import SEGMENTATION_MODELS, MODEL_PATHS
except Exception:
    SEGMENTATION_MODELS = {"detr_resnet_50_panoptic": "facebook/detr-resnet-50-panoptic"}
    MODEL_PATHS = {"segmentation": "models/segmentation"}


_PANOPTIC_PIPELINE = None
_PANOPTIC_AVAILABLE: Optional[bool] = None

# Mask coverage sanity bounds. DETR panoptic can return a "person" mask that
# spans essentially the whole frame (a segmentation failure that turns the
# cutout into an opaque rectangle) — the classic "sticker" artifact. We reject
# masks that are implausibly small (captured almost nothing) or implausibly
# large (background not removed), and fall through to a different method.
MIN_MASK_COVERAGE = 0.05   # at least 5% of the frame must be the character
MAX_MASK_COVERAGE = 0.70   # at most 70% — anything more is background leakage


def _mask_coverage(mask: np.ndarray) -> float:
    """Fraction of the frame the mask covers (0.0-1.0)."""
    return float(mask.sum() / mask.size) if mask.size else 0.0


def _mask_coverage_plausible(mask: np.ndarray, name_hint: str = "character") -> bool:
    """Reject masks whose coverage is outside the plausible band. This catches
    both "segmentation found nothing" and "segmentation kept everything". Logs
    a warning so the failure is inspectable."""
    cov = _mask_coverage(mask)
    if cov < MIN_MASK_COVERAGE:
        warnings.warn(
            f"Segmentation for '{name_hint}' captured only {cov:.1%} of the "
            "image area; treating as failed segmentation."
        )
        return False
    if cov > MAX_MASK_COVERAGE:
        warnings.warn(
            f"Segmentation for '{name_hint}' left {cov:.1%} of the frame opaque "
            "(background not removed); treating as failed segmentation."
        )
        return False
    return True


def _resolve_segmentation_model_path() -> str:
    """Resolve the local path for the panoptic segmentation model, mirroring
    `resolve_model_path()` conventions used elsewhere in the project."""
    try:
        from generators.image_generator import resolve_model_path
        return resolve_model_path(
            "segmentation",
            "detr_resnet_50_panoptic",
            SEGMENTATION_MODELS["detr_resnet_50_panoptic"],
        )
    except Exception:
        local = Path(MODEL_PATHS["segmentation"]) / "detr_resnet_50_panoptic"
        if local.exists():
            return str(local)
        return SEGMENTATION_MODELS["detr_resnet_50_panoptic"]


def _get_panoptic_pipeline():
    """Lazily load the DETR panoptic segmentation pipeline. Returns None if
    the model/dependencies are unavailable (fallback path is then used)."""
    global _PANOPTIC_PIPELINE, _PANOPTIC_AVAILABLE

    if _PANOPTIC_AVAILABLE is False:
        return None
    if _PANOPTIC_PIPELINE is not None:
        return _PANOPTIC_PIPELINE

    try:
        from transformers import pipeline as hf_pipeline

        model_path = _resolve_segmentation_model_path()
        _PANOPTIC_PIPELINE = hf_pipeline(
            "image-segmentation",
            model=model_path,
            subtask="panoptic",
        )
        _PANOPTIC_AVAILABLE = True
        return _PANOPTIC_PIPELINE
    except Exception as e:
        warnings.warn(f"DETR panoptic segmentation unavailable, will use chroma-key fallback: {e}")
        _PANOPTIC_AVAILABLE = False
        return None


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _segment_with_detr(image: Image.Image) -> Optional[Tuple[np.ndarray, str]]:
    """Return (bool mask, method_name) for the largest "person" segment, or
    None if no person segment is found / model unavailable."""
    pipe = _get_panoptic_pipeline()
    if pipe is None:
        return None

    try:
        results = pipe(image)
    except Exception as e:
        warnings.warn(f"DETR panoptic inference failed: {e}")
        return None

    person_masks = []
    for r in results:
        label = str(r.get("label", "")).lower()
        if "person" in label:
            mask_img = r.get("mask")
            if mask_img is None:
                continue
            mask_arr = np.array(mask_img.convert("L")) > 127
            person_masks.append(mask_arr)

    if not person_masks:
        return None

    # Largest person mask by area (main subject).
    best = max(person_masks, key=lambda m: m.sum())
    if best.sum() == 0:
        return None
    return best, "detr_panoptic"


def _border_flood_background(
    arr: np.ndarray,
    tolerance: float = 26.0,
    max_iters: int = 120,
) -> np.ndarray:
    """Grow a background mask from the image borders via color-connected
    flood fill (pure numpy, no scipy/cv2).

    The generated "plain background" assets are rarely perfectly uniform — they
    carry lighting gradients/vignetting (e.g. brighter toward the bottom). A
    single global color threshold (corner-sampled) fails on such gradients,
    leaving a border of background attached to the character (the rectangular
    artifact). Flood-filling from the borders is local: each new pixel is
    compared against its already-background neighbours' colour, so gradients
    are removed while interior character content is preserved.

    Returns a boolean mask of "background" pixels (True = remove).
    """
    h, w = arr.shape[:2]
    f = arr.astype(np.float32)
    bg = np.zeros((h, w), dtype=bool)
    bg[0, :] = bg[-1, :] = bg[:, 0] = bg[:, -1] = True

    for _ in range(max_iters):
        bgf = bg.astype(np.float32)
        # Number of 4-neighbours that are already background, per pixel.
        n = (
            np.roll(bgf, 1, 0) + np.roll(bgf, -1, 0)
            + np.roll(bgf, 1, 1) + np.roll(bgf, -1, 1)
        )
        # Sum of neighbour colours weighted by whether that neighbour is bg.
        s = (
            np.roll(f, 1, 0) * np.roll(bgf, 1, 0)[..., None]
            + np.roll(f, -1, 0) * np.roll(bgf, -1, 0)[..., None]
            + np.roll(f, 1, 1) * np.roll(bgf, 1, 1)[..., None]
            + np.roll(f, -1, 1) * np.roll(bgf, -1, 1)[..., None]
        )
        mean = s / (n[..., None] + 1e-9)
        dist = np.sqrt(((f - mean) ** 2).sum(axis=2))
        cand = (n > 0) & (~bg)
        accept = cand & (dist < tolerance)
        if not accept.any():
            break
        bg[accept] = True

    return bg


def _segment_with_chroma_key(
    image: Image.Image, corner_sample: int = 12, aggressive: bool = False
) -> Tuple[np.ndarray, str]:
    """Fallback segmentation: remove the plain background via a border-
    connected flood fill (robust to lighting gradients), then fall back to a
    corner-sampled global threshold if the flood fill captures too little.

    `aggressive=True` widens the tolerance (removes more of a noisy/uneven
    background) at the cost of possibly eating into soft character edges —
    used only as a validation retry, never the first attempt.
    """
    arr = np.array(image.convert("RGB")).astype(np.float32)
    h, w, _ = arr.shape

    tol = 26.0 if not aggressive else 40.0
    bg = _border_flood_background(arr, tolerance=tol)

    # If the flood fill barely grew (e.g. uniform bg where it should have
    # removed most of the frame), fall back to the corner-threshold estimate.
    bg_frac = bg.sum() / bg.size
    if bg_frac < 0.30:
        corners = np.concatenate([
            arr[:corner_sample, :corner_sample].reshape(-1, 3),
            arr[:corner_sample, -corner_sample:].reshape(-1, 3),
            arr[-corner_sample:, :corner_sample].reshape(-1, 3),
            arr[-corner_sample:, -corner_sample:].reshape(-1, 3),
        ], axis=0)
        bg_color = np.median(corners, axis=0)
        dist = np.sqrt(((arr - bg_color) ** 2).sum(axis=2))
        threshold = max(20.0, float(np.std(dist)) * 0.25 + 12.0) if aggressive \
            else max(30.0, float(np.std(dist)) * 0.5 + 20.0)
        bg = dist > threshold

    mask = ~bg

    # Clean up small holes/specks with a simple morphological-ish pass via PIL.
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.filter(ImageFilter.MedianFilter(size=5))
    mask = np.array(mask_img) > 127

    return mask, "chroma_key"


def segment_character(
    image_path: str,
    output_dir: Optional[Path] = None,
    name_hint: str = "character",
    retry: bool = False,
) -> Tuple[Optional[str], Optional[str], Optional[Tuple[int, int, int, int]], str]:
    """Segment a character out of a plain-background asset image.

    Tries each available segmentation method (DETR panoptic, then chroma-key)
    and accepts the first mask whose coverage is within the plausible band
    (i.e. it actually isolates the character rather than returning a full-frame
    rectangle or an empty sliver). Returns None (failed) if no method yields a
    usable mask — the caller should fall back to the raw asset image.

    `retry=True` is an "alternative processing" retry (per the asset-validation
    loop): it widens the accepted coverage band and uses an aggressive
    chroma-key threshold to remove more of a noisy background. It is never the
    first attempt.

    Returns:
        (mask_path, cutout_path, bbox, method) - any of the first three may
        be None if segmentation failed entirely.
    """
    image = Image.open(image_path).convert("RGB")

    # Candidate masks from the available methods, in preference order.
    candidates: List[Tuple[np.ndarray, str]] = []
    det = _segment_with_detr(image)
    if det is not None:
        candidates.append(det)
    candidates.append(_segment_with_chroma_key(image, aggressive=retry))

    mask = None
    method = None
    min_cov = MIN_MASK_COVERAGE * 0.5 if retry else MIN_MASK_COVERAGE
    max_cov = MAX_MASK_COVERAGE + 0.15 if retry else MAX_MASK_COVERAGE
    for cand_mask, cand_method in candidates:
        cov = _mask_coverage(cand_mask)
        if min_cov <= cov <= max_cov:
            mask, method = cand_mask, cand_method
            break

    if mask is None:
        return None, None, None, method or "none"

    bbox = _mask_bbox(mask)

    # Feather mask edges slightly to avoid hard cutout edges.
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=2))

    rgba = image.convert("RGBA")
    rgba.putalpha(mask_img)

    mask_path = None
    cutout_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_path = str(output_dir / f"mask_{name_hint}.png")
        cutout_path = str(output_dir / f"cutout_{name_hint}.png")
        mask_img.save(mask_path)
        rgba.save(cutout_path)

    return mask_path, cutout_path, bbox, method


def _draw_shadow(canvas: Image.Image, anchor_px: Tuple[int, int], width_px: int) -> Image.Image:
    """Draw a soft, blurred dark ellipse under the character's anchor point
    to approximate a contact shadow. Purely deterministic, not AI.
    """
    shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow_layer)
    cx, cy = anchor_px
    ew = int(width_px * 0.9)
    eh = max(6, int(width_px * 0.18))
    draw.ellipse(
        [cx - ew // 2, cy - eh // 2, cx + ew // 2, cy + eh // 2],
        fill=(0, 0, 0, 90),
    )
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=eh / 2))
    return Image.alpha_composite(canvas.convert("RGBA"), shadow_layer)


def validate_cutout(
    cutout_path: str,
    name_hint: str = "character",
    max_opaque_frac: float = 0.80,
    max_bbox_cover: float = 0.85,
) -> Dict[str, Any]:
    """Deterministic post-segmentation validation of an RGBA character cutout.

    Segmentation can "succeed" (return a mask) while still shipping a broken
    asset — most commonly a full-frame opaque rectangle. These cheap checks
    catch that class of failure before the asset is composited:

    - RGBA with actual transparency (some fully/pseudo transparent pixels).
    - Opaque coverage within a plausible band (character, not a rectangle).
    - The opaque bbox does not hug the full frame (indicates background left).
    - The cutout is not degenerate (empty or fully transparent).

    Returns a dict: {"valid": bool, "issues": [str, ...], "metrics": {...}}.
    This is intentionally deterministic and model-free; a semantic/vision
    validator can be layered on top where a vision model is available.
    """
    issues: List[str] = []
    metrics: Dict[str, Any] = {}

    try:
        img = Image.open(cutout_path)
    except Exception as e:
        return {"valid": False, "issues": [f"cannot open cutout: {e}"], "metrics": {}}

    if img.mode != "RGBA":
        img = img.convert("RGBA")

    arr = np.array(img)
    alpha = arr[:, :, 3]
    h, w = alpha.shape
    total = float(h * w)
    if total == 0:
        return {"valid": False, "issues": ["empty cutout"], "metrics": metrics}

    opaque = float((alpha > 200).sum())
    any_alpha = float((alpha > 0).sum())
    metrics["opaque_fraction"] = round(opaque / total, 4)
    metrics["any_alpha_fraction"] = round(any_alpha / total, 4)

    # 1. Must actually have transparency.
    if any_alpha / total > 0.999:
        issues.append("no transparency detected (opaque rectangle)")
    # 2. Opaque coverage must be within a plausible band (not a full rectangle).
    opaque_frac = opaque / total
    if opaque_frac > max_opaque_frac:
        issues.append(f"cutout is {opaque_frac:.0%} opaque; background likely remains")
    # 3. Opaque bbox must not hug the frame borders.
    opaque_mask = alpha > 200
    if opaque_mask.any():
        ys, xs = np.where(opaque_mask)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        cov = ((x1 - x0 + 1) / w) * ((y1 - y0 + 1) / h)
        metrics["bbox_cover"] = round(float(cov), 4)
        border_touch = (x0 <= 2 or y0 <= 2 or x1 >= w - 3 or y1 >= h - 3)
        if cov > max_bbox_cover and border_touch:
            issues.append("opaque region spans the full frame (background not removed)")
    # 4. Must not be empty / fully transparent.
    if opaque_frac < 0.01:
        issues.append("cutout is empty or fully transparent")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "metrics": metrics,
        "name": name_hint,
    }


def validate_character_asset(
    cutout_path: str,
    name_hint: str = "character",
) -> Dict[str, Any]:
    """Public asset-validation entry point (Phase 2). Currently wraps the
    deterministic RGBA checks; structured so a semantic/vision validator can
    be added later without changing call sites. Returns {"valid", "issues",
    "metrics"}."""
    return validate_cutout(cutout_path, name_hint=name_hint)


def compose_scene(
    background_path: str,
    placements: List[Dict[str, Any]],
    canvas_width: int = 1024,
    canvas_height: int = 1024,
    output_path: Optional[str] = None,
) -> Image.Image:
    """Composite a background with one or more character cutouts.

    Args:
        background_path: path to the background/environment image.
        placements: list of dicts, each with:
            - cutout_path (str): RGBA cutout image path (or raw asset image
              if segmentation failed — will be pasted opaquely).
            - anchor (tuple[float, float]): normalized (x, y) in [0, 1] for
              the character's bottom-center point (e.g. feet position).
            - scale (float): fraction of canvas_height the character's
              bounding-box height should occupy.
            - z (int): stacking order, lower first.
        canvas_width / canvas_height: target output size.
        output_path: optional path to save the final composite.

    Returns:
        Composed PIL Image (RGBA).
    """
    background = Image.open(background_path).convert("RGBA")
    if background.size != (canvas_width, canvas_height):
        background = background.resize((canvas_width, canvas_height), Image.LANCZOS)

    canvas = background.copy()

    ordered = sorted(placements, key=lambda p: p.get("z", 1))

    for placement in ordered:
        cutout_path = placement["cutout_path"]
        anchor = placement.get("anchor", (0.5, 0.8))
        scale = placement.get("scale", 0.4)

        cutout = Image.open(cutout_path)
        if cutout.mode != "RGBA":
            cutout = cutout.convert("RGBA")

        target_height = int(canvas_height * scale)
        aspect = cutout.width / cutout.height if cutout.height else 1.0
        target_width = max(1, int(target_height * aspect))
        resized = cutout.resize((target_width, target_height), Image.LANCZOS)

        anchor_x_px = int(anchor[0] * canvas_width)
        anchor_y_px = int(anchor[1] * canvas_height)

        # Contact shadow under the anchor point, before pasting the character.
        canvas = _draw_shadow(canvas, (anchor_x_px, anchor_y_px), target_width)

        paste_x = anchor_x_px - target_width // 2
        paste_y = anchor_y_px - target_height

        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        layer.paste(resized, (paste_x, paste_y), resized)
        canvas = Image.alpha_composite(canvas, layer)

    if output_path:
        canvas.convert("RGB").save(output_path)

    return canvas


def default_canvas_layout(character_names: List[str], width: int = 1024, height: int = 1024) -> Dict[str, Any]:
    """Cheap deterministic fallback layout when the LLM plan doesn't supply
    `canvas_layout` (evenly spaced across the frame, bottom-anchored)."""
    n = max(1, len(character_names))
    placements = []
    for i, name in enumerate(character_names):
        if n == 2:
            # Stage-left / stage-right placement for duos (per design doc §4.3).
            x = 0.28 if i == 0 else 0.72
        else:
            x = (i + 1) / (n + 1)
        placements.append({
            "name": name,
            "anchor": [x, 0.85],
            "scale": 0.5,
            "z": i + 1,
        })
    return {"width": width, "height": height, "placements": placements}
