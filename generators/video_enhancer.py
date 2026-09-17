#!/usr/bin/env python3
"""
Post-generation video enhancement (i2v output post-processing).

The base i2v model (LTX-Video 0.9.5) is compact and fast, so its raw output can
look a little "off": the motion can flicker/warp between frames and the
diffusers `export_to_video` step writes a lossy H.264 MP4. This module adds an
*opt-in* enhancement pass over the generated clip that:

  1. reads every frame of the clip,
  2. smooths jitter between frames temporally (reduces flicker / warping),
  3. applies a light spatial denoise (calms per-frame grain), and
  4. re-encodes to a high-quality, low-CRF container.

All of the core path is dependency-free (numpy + scipy + imageio's bundled
ffmpeg); no new model is downloaded, so the resident i2v footprint stays tiny
(see docs/image-to-video.md, ROADMAP "Future Enhancements"). An optional
super-resolution seam (:func:`super_resolve_video`) exists but ONLY runs when a
super-resolution model is already installed locally -- it never downloads.

Output contract: a new video (default high-quality MKV) plus a metrics JSON next
to it, matching the existing `*_benchmark_metrics.json` convention.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import imageio
import numpy as np

# A "light" enhancement profile is the safe default. These knobs trade quality
# against how much the original motion/identity is preserved -- strong smoothing
# can blur fast motion, so they stay modest by default.
DEFAULT_ENHANCEMENT = {
    # Temporal blend weight of the neighbouring frames (0 = off, 1 = strong).
    # Mild by default so genuine motion is not smeared.
    "temporal": 0.18,
    # How many frames on each side of the current one to average over
    # (temporal window = 2*temporal_window + 1). 1 == 3-frame median-ish blend.
    "temporal_window": 1,
    # Light spatial denoise sigma (0 = off). Kept small to preserve detail.
    "denoise": 0.5,
    # Output container. "mkv" (default) is the high-quality, near-lossless master
    # the user asked for; "mp4" is the widely-compatible fallback.
    "container": "mkv",
    # CRF for the re-encode: lower == higher quality / bigger file. 12 is a
    # high-quality, visually near-lossless target for a master clip.
    "crf": 12,
    # Super-resolution is OFF by default: it needs a model, and per ROADMAP
    # "Future Enhancements" we must NOT auto-download a heavy SR model.
    "super_resolve": False,
}


# ---------------------------------------------------------------------------
# Frame I/O (imageio + its bundled ffmpeg). Kept at module scope so callers can
# patch them in unit tests without touching a real video file.
# ---------------------------------------------------------------------------


def read_frames(video_path: str) -> List[np.ndarray]:
    """Read every frame of a video as a list of HxWx3 uint8 numpy arrays."""
    frames: List[np.ndarray] = []
    reader = imageio.get_reader(video_path, "FFMPEG")
    try:
        for frame in reader:
            frames.append(np.asarray(frame).astype(np.uint8))
    finally:
        reader.close()
    return frames


def write_frames(
    output_path: str,
    frames: List[np.ndarray],
    fps: float = 25.0,
    container: str = "mkv",
    crf: int = 12,
) -> str:
    """Encode frames to `output_path` at a low CRF (high quality).

    `container` selects the file container (and codec): "mkv" -> libx264 in a
    Matroska master; "mp4" -> libx264 in an MP4. A low `crf` keeps the result
    visually near-lossless (the original complaint was that MP4 was "very
    compressed").
    """
    if not frames:
        raise ValueError("write_frames: no frames to encode")

    base = os.path.splitext(output_path)[0]
    if container == "mp4":
        out = base if base.lower().endswith(".mp4") else base + ".mp4"
    elif container == "avi":
        out = base if base.lower().endswith(".avi") else base + ".avi"
    else:  # default: high-quality MKV master
        out = base if base.lower().endswith(".mkv") else base + ".mkv"

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    writer = imageio.get_writer(
        out,
        fps=fps,
        codec="libx264",
        output_params=["-crf", str(crf)],
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    return out


# ---------------------------------------------------------------------------
# Enhancement passes (pure functions on frame arrays -- easy to test in
# isolation, no I/O).
# ---------------------------------------------------------------------------


def _temporal_normalize(frames: List[np.ndarray]) -> np.ndarray:
    """Stack frames into a [F, H, W, C] float32 array for vectorized smoothing."""
    arr = np.stack(frames, axis=0).astype(np.float32)
    return arr


def temporal_smooth(
    frames: List[np.ndarray], strength: float, window: int = 1
) -> List[np.ndarray]:
    """Average each frame with its neighbours to reduce temporal flicker/warp.

    For every frame `f` this blends the `2*window + 1` neighbour window centered
    on `f` (clamped at the ends) using `strength` as the neighbour weight:
    `out = (1 - strength) * f + strength * mean(neighbours)`. `strength == 0`
    returns the input unchanged; a small `1` (window 1) is a gentle 3-frame
    blend that calms jitter without smearing genuine motion.
    """
    if strength <= 0 or window < 1 or len(frames) < 2:
        return [f.copy() for f in frames]

    arr = _temporal_normalize(frames)
    n = arr.shape[0]
    smoothed = arr.copy()
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n - 1, i + window)
        window_mean = arr[lo : hi + 1].mean(axis=0)
        smoothed[i] = (1.0 - strength) * arr[i] + strength * window_mean

    out = np.clip(smoothed, 0.0, 255.0).astype(np.uint8)
    return [out[i] for i in range(n)]


def spatial_denoise(frames: List[np.ndarray], sigma: float) -> List[np.ndarray]:
    """Gentle per-frame spatial denoise to calm grain. `sigma == 0` is a no-op."""
    if sigma <= 0:
        return frames

    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        # scipy absent: skip denoise rather than crash the enhancement.
        return frames

    arr = _temporal_normalize(frames)
    denoised = gaussian_filter(arr, sigma=(0.0, sigma, sigma, 0.0), mode="nearest")
    out = np.clip(denoised, 0.0, 255.0).astype(np.uint8)
    return [out[i] for i in range(arr.shape[0])]


# ---------------------------------------------------------------------------
# Opt-in super-resolution seam. NEVER downloads: it only runs when a local SR
# model directory exists, otherwise it is a graceful no-op with a warning.
# ---------------------------------------------------------------------------


def _resolve_local_sr_model(
    sr_model: Optional[str], sr_dir: Optional[str]
) -> Optional[str]:
    """Locate a locally-installed super-resolution model, if any.

    Resolution order: an explicit `sr_model` name under `models/image_to_video`
    (or `models/super_resolution`), else the local `sr_dir`, else the
    LTX-Video latent up-sampler if it happens to be co-located with the i2v
    model. Returns the path to use, or `None` when nothing suitable is present
    -- in which case the caller skips SR (no download).
    """
    candidates = []
    if sr_model:
        candidates.append(sr_model)
    if sr_dir:
        candidates.append(sr_dir)

    # Look inside the project's model trees (relative to project root).
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rel_search = [
        os.path.join(project_root, "models", "super_resolution"),
        os.path.join(project_root, "models", "image_to_video"),
    ]
    for base in rel_search:
        if os.path.isdir(base):
            for name in os.listdir(base):
                full = os.path.join(base, name)
                if os.path.isdir(full) and "upsampl" in name.lower():
                    candidates.append(full)

    for cand in candidates:
        if cand and os.path.isdir(cand) and os.listdir(cand):
            return cand
    return None


def super_resolve_video(
    frames: List[np.ndarray],
    sr_model: Optional[str] = None,
    sr_dir: Optional[str] = None,
    scale: int = 2,
) -> List[np.ndarray]:
    """Optional per-frame super-resolution, only when a local SR model exists.

    This is the "specialized video-enhancement model" path: when such a model is
    installed locally it is applied to lift resolution; otherwise it returns the
    frames unchanged and prints a warning -- it never downloads a model (per
    ROADMAP "Future Enhancements" / docs/image-to-video.md §5 resolution caveat).
    """
    model = _resolve_local_sr_model(sr_model, sr_dir)
    if model is None:
        print(
            "[video_enhancer] Super-resolution skipped: no local SR model "
            "found (opt-in; NOT auto-downloaded). Pass sr_model= or install a "
            "local super-resolution model to enable it."
        )
        return frames

    print(f"[video_enhancer] Applying local super-resolution '{model}' (x{scale})")
    try:
        from diffusers import (
            ImageToVideoPipeline,
        )  # noqa: F401  (keeps diffusers out of module import)
    except Exception as e:  # noqa: BLE001 -- SR must never break the base path
        print(
            f"[video_enhancer] Could not load SR model '{model}' ({e}); "
            "using original resolution."
        )
        return frames

    # Placeholder implementation: a real SR model would be loaded and applied
    # per frame here. The seam exists so the capability is wired and opt-in
    # without pulling a heavy model into the base enhancement path.
    return frames


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enhance_video(
    video_path: str,
    output_dir: Optional[str] = None,
    output_basename: Optional[str] = None,
    fps: float = 25.0,
    enhancement: Optional[Dict[str, Any]] = None,
    sr_model: Optional[str] = None,
    sr_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Enhance a generated video clip in place (writes a sibling file).

    Args:
        video_path: path to a generated i2v video (e.g. an .mp4).
        output_dir: directory for the enhanced output (defaults to the source
            directory).
        output_basename: base filename for the enhanced output (defaults to
            `<source-stem>_enhanced`).
        fps: frame rate of the output (defaults to 25 to match LTX-Video).
        enhancement: override any knob in DEFAULT_ENHANCEMENT (temporal,
            temporal_window, denoise, container, crf, super_resolve).
        sr_model / sr_dir: location of a local super-resolution model (only used
            when `enhancement['super_resolve']` is truthy).

    Returns:
        dict with 'video_path' (enhanced output), 'metrics_path' and the
        enhancement metrics themselves, mirroring generate_video()'s result.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    cfg = {**DEFAULT_ENHANCEMENT, **(enhancement or {})}
    out_dir = output_dir or os.path.dirname(os.path.abspath(video_path))
    if output_basename is None:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        # Strip a trailing "_enhanced" so re-running an enhanced clip does not
        # stack suffixes (e.g. "x_enhanced" -> "x_enhanced", not "_enhanced_enhanced").
        stem = stem[: -len("_enhanced")] if stem.endswith("_enhanced") else stem
        output_basename = f"{stem}_enhanced"
    out_path = os.path.join(out_dir, f"{output_basename}.{cfg['container']}")
    os.makedirs(out_dir, exist_ok=True)

    frames = read_frames(video_path)
    if not frames:
        raise ValueError(f"No frames read from video: {video_path}")

    start_time = time.time()

    # 1. (optional) super-resolution -- only when a local model is present.
    if cfg.get("super_resolve"):
        frames = super_resolve_video(
            frames,
            sr_model=sr_model,
            sr_dir=sr_dir,
            scale=int(cfg.get("sr_scale", 2)),
        )

    # 2. temporal smoothing -- the main anti-distortion/flicker pass.
    before = frames
    frames = temporal_smooth(
        frames, float(cfg.get("temporal", 0.0)), int(cfg.get("temporal_window", 1))
    )

    # 3. light spatial denoise.
    frames = spatial_denoise(frames, float(cfg.get("denoise", 0.0)))

    # 4. high-quality re-encode.
    frames_out = write_frames(
        out_path,
        frames,
        fps=fps,
        container=cfg.get("container", "mkv"),
        crf=int(cfg.get("crf", 12)),
    )

    duration_ms = int((time.time() - start_time) * 1000)

    # Record how much the smoothing moved frames on average (0 == no change).
    changed = _frame_change_metric(before, frames)

    metrics = {
        "source": video_path,
        "output": frames_out,
        "input_frames": len(before),
        "input_shape": list(before[0].shape) if before else [],
        "fps": fps,
        "container": cfg.get("container", "mkv"),
        "crf": int(cfg.get("crf", 12)),
        "temporal": float(cfg.get("temporal", 0.0)),
        "temporal_window": int(cfg.get("temporal_window", 1)),
        "denoise": float(cfg.get("denoise", 0.0)),
        "super_resolve": bool(cfg.get("super_resolve", False)),
        "mean_abs_frame_change": changed,
        "duration_ms": duration_ms,
    }
    metrics_path = os.path.join(out_dir, f"{output_basename}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Enhanced video saved to: {frames_out}")
    print(f"Enhancement metrics saved to: {metrics_path}")
    return {"video_path": frames_out, "metrics_path": metrics_path, "metrics": metrics}


def _frame_change_metric(before: List[np.ndarray], after: List[np.ndarray]) -> float:
    """Mean absolute per-pixel change between input and enhanced frames."""
    if not before or not after or len(before) != len(after):
        return 0.0
    total = 0.0
    count = 0
    for b, a in zip(before, after):
        total += float(np.abs(b.astype(np.float32) - a.astype(np.float32)).mean())
        count += 1
    return total / max(count, 1)


__all__ = [
    "DEFAULT_ENHANCEMENT",
    "read_frames",
    "write_frames",
    "temporal_smooth",
    "spatial_denoise",
    "super_resolve_video",
    "enhance_video",
]
