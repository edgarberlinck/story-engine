#!/usr/bin/env python3
"""Unit tests for generators.video_enhancer."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from generators import video_enhancer as ve


class VideoEnhancerPureFunctionTest(unittest.TestCase):
    def test_default_enhancement_keys_and_values(self):
        expected = {
            "temporal": 0.18,
            "temporal_window": 1,
            "denoise": 0.5,
            "container": "mkv",
            "crf": 12,
            "super_resolve": False,
        }
        self.assertEqual(ve.DEFAULT_ENHANCEMENT, expected)

    def test_temporal_smooth_noop_paths_return_unchanged_copies(self):
        frames = [
            np.full((4, 4, 3), 10, dtype=np.uint8),
            np.full((4, 4, 3), 20, dtype=np.uint8),
        ]

        for result in (
            ve.temporal_smooth(frames, strength=0.0, window=1),
            ve.temporal_smooth(frames, strength=0.5, window=0),
            ve.temporal_smooth([frames[0]], strength=0.5, window=1),
        ):
            expected = frames if len(result) == 2 else [frames[0]]
            self.assertEqual(len(result), len(expected))
            for got, want in zip(result, expected):
                self.assertTrue(np.array_equal(got, want))

        noop = ve.temporal_smooth(frames, strength=0.0, window=1)
        self.assertIsNot(noop[0], frames[0])

    def test_temporal_smooth_changes_middle_frame_toward_neighbours(self):
        frames = [
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.full((2, 2, 3), 255, dtype=np.uint8),
            np.zeros((2, 2, 3), dtype=np.uint8),
        ]

        out = ve.temporal_smooth(frames, strength=0.75, window=1)

        self.assertLess(out[1].mean(), frames[1].mean())
        self.assertGreater(out[1].mean(), 0)
        before_diff = np.abs(frames[1].astype(np.int16) - frames[0].astype(np.int16)).mean()
        after_diff = np.abs(out[1].astype(np.int16) - out[0].astype(np.int16)).mean()
        self.assertLess(after_diff, before_diff)

    def test_spatial_denoise_noop_changed_and_scipy_absent_paths(self):
        frame = np.zeros((9, 9, 3), dtype=np.uint8)
        frame[4, 4, :] = 255
        frames = [frame]

        self.assertIs(ve.spatial_denoise(frames, sigma=0.0), frames)

        denoised = ve.spatial_denoise(frames, sigma=1.0)
        self.assertEqual(denoised[0].shape, frame.shape)
        self.assertFalse(np.array_equal(denoised[0], frame))

        with patch.dict("sys.modules", {"scipy.ndimage": None}):
            self.assertIs(ve.spatial_denoise(frames, sigma=1.0), frames)

    def test_super_resolve_without_local_model_is_noop(self):
        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
        with patch.object(ve, "_resolve_local_sr_model", return_value=None), \
             patch("builtins.print") as printed:
            out = ve.super_resolve_video(frames)

        self.assertIs(out, frames)
        self.assertTrue(printed.called)


class VideoEnhancerIOTest(unittest.TestCase):
    def _frames(self, count=3, shape=(16, 16, 3)):
        rng = np.random.default_rng(1234)
        return [rng.integers(0, 256, size=shape, dtype=np.uint8) for _ in range(count)]

    def test_write_and_read_frames_round_trip_and_extensions(self):
        frames = self._frames()
        with tempfile.TemporaryDirectory() as tmp:
            mkv = ve.write_frames(os.path.join(tmp, "clip"), frames, fps=5.0, container="mkv", crf=18)
            self.assertTrue(mkv.endswith(".mkv"))
            self.assertTrue(os.path.isfile(mkv))

            read_back = ve.read_frames(mkv)
            self.assertEqual(len(read_back), len(frames))
            self.assertEqual(read_back[0].shape, frames[0].shape)
            self.assertEqual(read_back[0].dtype, np.uint8)

            mp4 = ve.write_frames(os.path.join(tmp, "movie"), frames, fps=5.0, container="mp4", crf=18)
            avi = ve.write_frames(os.path.join(tmp, "movie"), frames, fps=5.0, container="avi", crf=18)
            self.assertTrue(mp4.endswith(".mp4"))
            self.assertTrue(avi.endswith(".avi"))

            with self.assertRaises(ValueError):
                ve.write_frames(os.path.join(tmp, "empty"), [], fps=5.0)

    def test_enhance_video_end_to_end_metrics_and_enhanced_stem(self):
        frames = self._frames(count=4)
        with tempfile.TemporaryDirectory() as tmp:
            src = ve.write_frames(os.path.join(tmp, "sample_enhanced"), frames, fps=5.0, container="mp4", crf=18)
            out_dir = os.path.join(tmp, "out")
            result = ve.enhance_video(
                src,
                output_dir=out_dir,
                fps=5.0,
                enhancement={"temporal": 0.2, "denoise": 0.2, "container": "mkv", "crf": 14},
            )

            self.assertEqual(set(result), {"video_path", "metrics_path", "metrics"})
            self.assertTrue(os.path.isfile(result["video_path"]))
            self.assertTrue(os.path.isfile(result["metrics_path"]))
            self.assertTrue(os.path.basename(result["video_path"]).startswith("sample_enhanced."))
            self.assertNotIn("enhanced_enhanced", os.path.basename(result["video_path"]))

            with open(result["metrics_path"]) as f:
                metrics = json.load(f)
            self.assertIn("mean_abs_frame_change", metrics)
            self.assertEqual(metrics["container"], "mkv")
            self.assertEqual(metrics["crf"], 14)
            self.assertEqual(result["metrics"]["container"], "mkv")

            with self.assertRaises(FileNotFoundError):
                ve.enhance_video(os.path.join(tmp, "missing.mp4"), output_dir=out_dir)


class VideoEnhancerSuperResolutionTest(unittest.TestCase):
    def test_resolve_returns_none_when_nothing_present(self):
        with patch("os.path.isdir", return_value=False), patch("os.listdir", return_value=[]):
            self.assertIsNone(ve._resolve_local_sr_model(None, None))

    def test_resolve_finds_explicit_model_on_disk(self):
        with patch("os.path.isdir", return_value=True), patch("os.listdir", return_value=["weights"]):
            self.assertEqual(ve._resolve_local_sr_model("some/path", None), "some/path")

    def test_super_resolve_with_local_model_is_noop_no_download(self):
        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
        with patch.object(ve, "_resolve_local_sr_model", return_value="fake-sr-model"), \
             patch("builtins.print"):
            out = ve.super_resolve_video(frames)

        self.assertIs(out, frames)

    def test_super_resolve_load_failure_degrades_gracefully(self):
        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
        with patch.object(ve, "_resolve_local_sr_model", return_value="fake-sr-model"), \
             patch.dict("sys.modules", {"diffusers": None, "diffusers.ImageToVideoPipeline": None}), \
             patch("builtins.print"):
            out = ve.super_resolve_video(frames)

        self.assertIs(out, frames)

    def test_enhance_video_runs_super_resolve_when_enabled(self):
        rng = np.random.default_rng(1)
        frames = [rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8) for _ in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            src = ve.write_frames(os.path.join(tmp, "sample"), frames, fps=5.0, container="mp4", crf=18)
            with patch.object(ve, "_resolve_local_sr_model", return_value="fake-sr-model"), \
                 patch("builtins.print"):
                result = ve.enhance_video(
                    src,
                    output_dir=tmp,
                    fps=5.0,
                    enhancement={
                        "super_resolve": True,
                        "temporal": 0.0,
                        "denoise": 0.0,
                        "container": "mkv",
                        "crf": 18,
                    },
                    sr_model="fake",
                )

            self.assertEqual(set(result), {"video_path", "metrics_path", "metrics"})
            self.assertTrue(os.path.isfile(result["video_path"]))

    def test_enhance_video_empty_input_raises_valueerror(self):
        with patch("os.path.isfile", return_value=True), patch.object(ve, "read_frames", return_value=[]):
            with self.assertRaises(ValueError):
                ve.enhance_video("anything.mp4")

    def test_frame_change_metric_mismatched_lengths_returns_zero(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        self.assertEqual(ve._frame_change_metric([], []), 0.0)
        self.assertEqual(ve._frame_change_metric([frame], []), 0.0)


if __name__ == "__main__":
    unittest.main()
