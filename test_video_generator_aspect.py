#!/usr/bin/env python3
"""Aspect-ratio tests for generators.video_generator._prepare_image."""

import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from generators.video_generator import _prepare_image


class PrepareImageAspectTest(unittest.TestCase):
    def _write_png(self, tmp, name, arr):
        path = os.path.join(tmp, name)
        Image.fromarray(arr.astype(np.uint8), "RGB").save(path)
        return path

    def test_matching_aspect_clean_resize(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr = np.full((36, 64, 3), [12, 34, 56], dtype=np.uint8)
            path = self._write_png(tmp, "wide.png", arr)
            out = _prepare_image(path, 128, 72)

            self.assertEqual(out.size, (128, 72))
            expected = Image.open(path).convert("RGB").resize((128, 72), Image.LANCZOS)
            self.assertTrue(np.array_equal(np.asarray(out), np.asarray(expected)))

    def test_square_to_wide_preserves_gradient_and_edge_replicates_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            gradient = np.linspace(0, 255, 64, dtype=np.uint8)[:, None]
            arr = np.repeat(gradient, 64, axis=1)
            rgb = np.stack([arr, arr, arr], axis=2)
            path = self._write_png(tmp, "square.png", rgb)

            out = _prepare_image(path, 128, 72)
            data = np.asarray(out)

            self.assertEqual(out.size, (128, 72))
            center_column = data[:, data.shape[1] // 2, 0]
            self.assertLessEqual(center_column.min(), 3)
            self.assertGreaterEqual(center_column.max(), 252)
            self.assertNotEqual(int(center_column[0]), int(center_column[-1]))

            # For a square source contained in a wider target, the fitted image
            # uses full target height and pads the left/right edges by repeating
            # the fitted border columns.
            self.assertTrue(np.array_equal(data[:, 0, :], data[:, 1, :]))
            self.assertTrue(np.array_equal(data[:, -1, :], data[:, -2, :]))

    def test_aspect_tolerance_matching_and_padding_branches(self):
        with tempfile.TemporaryDirectory() as tmp:
            almost = np.full((576, 1024, 3), 77, dtype=np.uint8)
            almost_path = self._write_png(tmp, "almost.png", almost)
            matched = _prepare_image(almost_path, 1024, 576, aspect_tolerance=0.0006)
            self.assertEqual(matched.size, (1024, 576))
            self.assertTrue(np.all(np.asarray(matched) == 77))

            square = np.full((64, 64, 3), 99, dtype=np.uint8)
            square[:, 0, :] = 10
            square[:, -1, :] = 200
            square_path = self._write_png(tmp, "diff.png", square)
            padded = _prepare_image(square_path, 128, 72, aspect_tolerance=0.02)
            data = np.asarray(padded)
            self.assertEqual(padded.size, (128, 72))
            self.assertTrue(np.array_equal(data[:, 0, :], data[:, 1, :]))
            self.assertTrue(np.array_equal(data[:, -1, :], data[:, -2, :]))


if __name__ == "__main__":
    unittest.main()
