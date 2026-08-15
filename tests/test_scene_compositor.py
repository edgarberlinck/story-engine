"""
Focused unit tests for the scene-composition asset pipeline:

- Mask coverage sanity bounds (rejects full-frame rectangles AND empty slivers).
- `segment_character` validation + aggressive retry fallback.
- `validate_cutout` / `validate_character_asset` deterministic RGBA checks.
- `_border_flood_background` gradient-robust background removal.

These use synthetic images and mock DETR so they run without model weights.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import scene_compositor as sc


def _make_gradient_bg_asset() -> Image.Image:
    """A 'plain background' asset with a vertical lighting gradient and a
    solid character blob in the middle — the gradient is what breaks a naive
    corner-threshold chroma-key."""
    h, w = 256, 256
    # Vertical gradient background (dark top -> lighter bottom).
    bg = np.linspace(60, 160, h, dtype=np.float32)[:, None] * np.ones((1, w), dtype=np.float32)
    arr = np.stack([bg, bg, bg], axis=2).astype(np.uint8)
    # Character blob (distinct colour) in the centre, not touching borders.
    img = Image.fromarray(arr).convert("RGB")
    draw = Image.new("RGB", (w, h), (0, 0, 0))
    # Paint a filled ellipse for the "character".
    arr[:, :, :] = arr[:, :, :]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    blob = ((xx - w // 2) ** 2 / (0.18 * w) ** 2
            + (yy - h // 2) ** 2 / (0.35 * h) ** 2) <= 1
    # A red character on the gradient background.
    return Image.fromarray(np.where(blob[..., None], (200, 40, 40), arr).astype(np.uint8))


class TestMaskCoverage(unittest.TestCase):
    def test_coverage_calculation(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[:5, :5] = True  # 25/100 = 0.25
        self.assertAlmostEqual(sc._mask_coverage(mask), 0.25)

    def test_plausible_band(self):
        ok = np.zeros((100, 100), dtype=bool)
        ok[20:80, 20:80] = True  # 0.36 coverage
        self.assertTrue(sc._mask_coverage_plausible(ok, "t"))

    def test_full_frame_rejected(self):
        mask = np.ones((100, 100), dtype=bool)  # 1.0 coverage
        self.assertFalse(sc._mask_coverage_plausible(mask, "t"))

    def test_empty_rejected(self):
        mask = np.zeros((100, 100), dtype=bool)
        self.assertFalse(sc._mask_coverage_plausible(mask, "t"))


class TestSegmentCharacterValidation(unittest.TestCase):
    def test_full_frame_detr_mask_falls_back(self):
        """If DETR returns a full-frame person mask, segmentation must not
        accept it — it should fall through to chroma-key."""
        asset = _make_gradient_bg_asset()
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "asset.png")
            asset.save(src)

            # DETR returns a full-frame mask (the classic failure).
            full = np.ones((256, 256), dtype=bool)
            with patch.object(sc, "_segment_with_detr", return_value=(full, "detr_panoptic")):
                mask_path, cutout_path, bbox, method = sc.segment_character(
                    src, output_dir=Path(d), name_hint="t"
                )
            # Must have fallen back to chroma-key (not the full-frame DETR mask).
            self.assertIsNotNone(cutout_path)
            self.assertEqual(method, "chroma_key")
            cut = np.array(Image.open(cutout_path))
            alpha = cut[:, :, 3]
            # Border pixels must be transparent now (background removed).
            self.assertEqual((alpha[:5, :] > 200).sum(), 0)
            self.assertEqual((alpha[-5:, :] > 200).sum(), 0)

    def test_rejects_empty_detr_mask(self):
        asset = _make_gradient_bg_asset()
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "asset.png")
            asset.save(src)
            empty = np.zeros((256, 256), dtype=bool)
            with patch.object(sc, "_segment_with_detr", return_value=(empty, "detr_panoptic")):
                mask_path, cutout_path, bbox, method = sc.segment_character(
                    src, output_dir=Path(d), name_hint="t"
                )
            # Falls back to chroma-key which should find the blob.
            self.assertIsNotNone(cutout_path)
            self.assertEqual(method, "chroma_key")

    def test_segmentation_none_when_nothing_usable(self):
        # A solid single-colour image -> chroma-key removes everything -> none.
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "asset.png")
            Image.new("RGB", (64, 64), (120, 120, 120)).save(src)
            with patch.object(sc, "_segment_with_detr", return_value=None):
                _, cutout_path, _, _ = sc.segment_character(src, name_hint="t")
            self.assertIsNone(cutout_path)


class TestBorderFloodBackground(unittest.TestCase):
    def test_removes_gradient_background(self):
        """Flood fill from borders must remove a gradient background that is
        connected to the edges while preserving the interior character."""
        asset = _make_gradient_bg_asset()
        arr = np.array(asset.convert("RGB"))
        bg = sc._border_flood_background(arr.astype(np.float32), tolerance=26.0)
        # Background (non-character) should be mostly flagged.
        self.assertGreater(bg.sum() / bg.size, 0.5)
        # The central character should NOT be flagged as background.
        h, w = arr.shape[:2]
        cx, cy = w // 2, h // 2
        self.assertFalse(bg[cy, cx])


class TestValidateCutout(unittest.TestCase):
    def test_opaque_rectangle_detected(self):
        # A cutout with zero transparency = the classic broken asset.
        img = Image.new("RGBA", (100, 100), (200, 200, 200, 255))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "cut.png")
            img.save(p)
            res = sc.validate_cutout(p)
        self.assertFalse(res["valid"])
        self.assertTrue(any("transparency" in i or "opaque" in i for i in res["issues"]))

    def test_clean_cutout_passes(self):
        # RGBA with a central character and transparent borders.
        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        arr[20:80, 20:80, :3] = 200
        arr[20:80, 20:80, 3] = 255
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "cut.png")
            Image.fromarray(arr).save(p)
            res = sc.validate_cutout(p)
        self.assertTrue(res["valid"], res.get("issues"))

    def test_full_frame_bbox_detected(self):
        # Opaque region spans to the border => background not removed.
        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        arr[:, :, :3] = 200
        arr[:, :, 3] = 255  # fully opaque full frame
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "cut.png")
            Image.fromarray(arr).save(p)
            res = sc.validate_cutout(p)
        self.assertFalse(res["valid"])
        self.assertTrue(any("full frame" in i or "opaque" in i for i in res["issues"]))

    def test_missing_file_reports_invalid(self):
        res = sc.validate_cutout("/nonexistent/does_not_exist.png")
        self.assertFalse(res["valid"])
        self.assertTrue(res["issues"])


if __name__ == "__main__":
    unittest.main()
