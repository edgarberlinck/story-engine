"""Unit tests for the reference-conditioned scene generator backend.

These mock the pipeline loader / model config so they run without model
weights or MPS inference. They verify:
- preprocessing scales references while preserving aspect ratio,
- the pipeline is called with PIL reference images and the right kwargs,
- output paths are unique and the empty-references guard raises.
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

from generators import reference_scene_generator as rsg


def _make_image(path: str, size=(512, 512)) -> str:
    Image.fromarray(np.zeros((size[1], size[0], 3), dtype=np.uint8)).save(path)
    return path


class _FakeResult:
    def __init__(self, images):
        self.images = images


class _FakePipe:
    def __init__(self, images=None):
        self._images = images or [Image.new("RGB", (16, 16))]
        self.last_kwargs = None

    def to(self, device):
        self.device = device
        return self

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResult(self._images)


class ReferenceGeneratorTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def _tmp_ref(self, name="ref.png", size=(512, 512)) -> str:
        return _make_image(os.path.join(self._tmp, name), size)

    def test_preprocessing_scales_long_side_to_512_preserving_ratio(self):
        # np array (1200 rows, 800 cols) -> PIL size (800, 1200): long side 1200 -> 512.
        img = Image.fromarray(np.zeros((1200, 800, 3), dtype=np.uint8))
        in_w, in_h = img.size
        out = rsg._resize_reference(img)
        self.assertEqual(max(out.size), 512)
        # Aspect ratio preserved within rounding to multiple of 16.
        self.assertAlmostEqual(out.size[0] / out.size[1], in_w / in_h, delta=0.1)
        self.assertEqual(out.size[0] % 16, 0)
        self.assertEqual(out.size[1] % 16, 0)

    def test_preprocessing_does_not_upscale_small_images(self):
        img = Image.new("RGB", (100, 200))
        out = rsg._resize_reference(img)
        # Long side must not grow past the original long side.
        self.assertLessEqual(max(out.size), 200)

    def test_prepare_references_skips_unreadable(self):
        ok = self._tmp_ref("ok.png")
        bad = os.path.join(self._tmp, "missing.png")
        prepared = rsg._prepare_references([bad, ok])
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0].mode, "RGB")

    def test_load_reference_scene_pipeline_uses_kv_pipeline(self):
        with patch("diffusers.Flux2KleinKVPipeline.from_pretrained") as mock_kv, \
             patch("diffusers.Flux2KleinPipeline.from_pretrained") as mock_plain:
            rsg.load_reference_scene_pipeline("local-model", "float16")
            mock_kv.assert_called_once()
            mock_plain.assert_not_called()

    def test_generate_passes_pil_reference_list_and_kwargs(self):
        ref = self._tmp_ref()
        fake = _FakePipe()
        with patch.object(rsg, "load_reference_scene_pipeline", return_value=fake), \
             patch.object(rsg, "get_model_config", return_value=("cpu", "float16")), \
             patch.object(rsg, "cleanup_pipeline"), \
             patch.object(rsg, "resolve_model_path", return_value="m"):
            out = rsg.generate_reference_conditioned_scene(
                "A stage scene.", [ref], steps=4, task_name="test_scene"
            )
        kw = fake.last_kwargs
        self.assertIsInstance(kw["image"], list)
        self.assertEqual(kw["image"][0].mode, "RGB")
        self.assertEqual(kw["prompt"], "A stage scene.")
        self.assertEqual(kw["num_inference_steps"], 4)
        self.assertEqual(kw["max_sequence_length"], 512)
        self.assertEqual(kw["num_images_per_prompt"], 1)
        self.assertNotIn("guidance_scale", kw)
        self.assertEqual(len(out), 1)
        self.assertTrue(Path(out[0]).exists())

    def test_generate_returns_unique_paths_for_multiple_images(self):
        ref = self._tmp_ref()
        fake = _FakePipe(images=[Image.new("RGB", (16, 16)), Image.new("RGB", (16, 16))])
        with patch.object(rsg, "load_reference_scene_pipeline", return_value=fake), \
             patch.object(rsg, "get_model_config", return_value=("cpu", "float16")), \
             patch.object(rsg, "cleanup_pipeline"), \
             patch.object(rsg, "resolve_model_path", return_value="m"):
            out = rsg.generate_reference_conditioned_scene(
                "prompt", [ref], num_images=2, task_name="multi"
            )
        self.assertEqual(len(out), 2)
        self.assertNotEqual(out[0], out[1])
        self.assertTrue(Path(out[0]).exists())
        self.assertTrue(Path(out[1]).exists())

    def test_generate_rejects_empty_references(self):
        with self.assertRaises(ValueError):
            rsg.generate_reference_conditioned_scene("prompt", [])

    def test_generate_rejects_all_unreadable_references(self):
        with patch.object(rsg, "cleanup_pipeline"):
            with self.assertRaises(ValueError):
                rsg.generate_reference_conditioned_scene(
                    "prompt", [os.path.join(self._tmp, "nope.png")]
                )


if __name__ == "__main__":
    unittest.main()
