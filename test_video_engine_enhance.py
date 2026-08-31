#!/usr/bin/env python3
"""Tests for video_engine.animate_scene enhancement wiring."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from generators import video_engine
from generators import video_enhancer


class AnimateSceneEnhanceTest(unittest.TestCase):
    def _tiny_video(self, tmp, name="raw"):
        frames = [
            np.full((16, 16, 3), value, dtype=np.uint8)
            for value in (0, 90, 180)
        ]
        return video_enhancer.write_frames(os.path.join(tmp, name), frames, fps=5.0, container="mp4", crf=18)

    def _raw_result(self, path):
        return {"video_path": path, "metrics_path": path + ".json", "metrics": {"fps": 5.0}}

    def test_enhance_true_adds_enhanced_keys(self):
        scene = {"scene_number": 1, "image_path": "x.png", "prompt": "p"}
        with tempfile.TemporaryDirectory() as tmp:
            raw = self._tiny_video(tmp)
            out_dir = Path(os.path.join(tmp, "out"))
            with patch.object(video_engine, "scene_out_dir", return_value=out_dir), \
                 patch.object(video_engine, "generate_video", return_value=self._raw_result(raw)) as gen:
                result = video_engine.animate_scene(
                    scene,
                    project="proj",
                    enhance=True,
                    enhancement={"temporal": 0.1, "denoise": 0.0, "container": "mkv", "crf": 16},
                )

            gen.assert_called_once()
            self.assertIn("enhanced_video_path", result)
            self.assertIn("enhanced_metrics_path", result)
            self.assertIn("enhanced_metrics", result)
            self.assertNotEqual(result["enhanced_video_path"], raw)
            self.assertTrue(os.path.isfile(result["enhanced_video_path"]))
            self.assertTrue(os.path.isfile(result["enhanced_metrics_path"]))

    def test_enhance_false_does_not_add_enhancement_keys(self):
        scene = {"scene_number": 1, "image_path": "x.png", "prompt": "p"}
        with tempfile.TemporaryDirectory() as tmp:
            raw = self._tiny_video(tmp)
            with patch.object(video_engine, "scene_out_dir", return_value=Path(tmp)), \
                 patch.object(video_engine, "generate_video", return_value=self._raw_result(raw)) as gen:
                result = video_engine.animate_scene(scene, enhance=False)

            gen.assert_called_once()
            self.assertNotIn("enhanced_video_path", result)
            self.assertNotIn("enhanced_metrics_path", result)
            self.assertNotIn("enhanced_metrics", result)

    def test_enhancement_failure_degrades_to_raw_video(self):
        scene = {"scene_number": 1, "image_path": "x.png", "prompt": "p"}
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "missing.mp4")
            with patch.object(video_engine, "scene_out_dir", return_value=Path(tmp)), \
                 patch.object(video_engine, "generate_video", return_value=self._raw_result(missing)), \
                 patch("builtins.print"):
                result = video_engine.animate_scene(scene, enhance=True)

            self.assertEqual(result["enhanced_video_path"], missing)
            self.assertNotIn("enhanced_metrics_path", result)
            self.assertNotIn("enhanced_metrics", result)


if __name__ == "__main__":
    unittest.main()
