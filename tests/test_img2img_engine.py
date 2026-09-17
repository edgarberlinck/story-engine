import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from generators import img2img_engine as ie


class _FakeImage:
    def __init__(self):
        self.saved = []

    def convert(self, mode):
        self.mode = mode
        return self

    def save(self, path):
        self.saved.append(path)


class _FakePipe:
    def __init__(self):
        self.calls = []

    def to(self, device):
        self.device = device
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=[_FakeImage()])


class Img2ImgEngineTest(unittest.TestCase):
    def test_refine_composite_success_clamps_strength(self):
        pipe = _FakePipe()
        opened = _FakeImage()
        with patch("PIL.Image.open", return_value=opened) as image_open, patch(
            "diffusers.StableDiffusionXLImg2ImgPipeline.from_pretrained",
            return_value=pipe,
        ) as load, patch.object(
            ie, "resolve_model_path", return_value="local/sdxl"
        ), patch.object(
            ie, "get_model_config", return_value=("cpu", "float16")
        ), patch.object(
            ie, "cleanup_pipeline"
        ) as cleanup:
            out = ie.refine_composite(
                "/tmp/composite.png", "", strength=0.9, output_path="/tmp/out.png"
            )

        self.assertEqual(out, "/tmp/out.png")
        image_open.assert_called_once_with("/tmp/composite.png")
        load.assert_called_once_with(
            "local/sdxl", torch_dtype="float16", safety_checker=None
        )
        self.assertEqual(pipe.calls[0]["prompt"], ie.DEFAULT_REFINEMENT_PROMPT)
        self.assertEqual(pipe.calls[0]["strength"], ie.MAX_SAFE_STRENGTH)
        cleanup.assert_called_once_with(pipe)

    def test_refine_composite_flux_allows_high_strength_and_default_output(self):
        pipe = _FakePipe()
        with patch("PIL.Image.open", return_value=_FakeImage()), patch(
            "diffusers.FluxImg2ImgPipeline.from_pretrained", return_value=pipe
        ), patch.object(
            ie, "resolve_model_path", return_value="local/flux"
        ), patch.object(
            ie, "get_model_config", return_value=("cpu", "float16")
        ), patch.object(
            ie, "cleanup_pipeline"
        ):
            out = ie.refine_composite(
                "/tmp/scene.png",
                "prompt",
                model_name="flux_dev",
                strength=0.8,
                allow_high_strength=True,
            )

        self.assertEqual(out, "/tmp/scene_refined.png")
        self.assertEqual(pipe.calls[0]["strength"], 0.8)
        self.assertIn("max_sequence_length", pipe.calls[0])

    def test_refine_composite_returns_original_on_failure(self):
        with patch.object(ie, "cleanup_pipeline") as cleanup:
            out = ie.refine_composite(
                "/tmp/original.png", "prompt", model_name="missing"
            )
        self.assertEqual(out, "/tmp/original.png")
        cleanup.assert_called_once_with(None)


if __name__ == "__main__":
    unittest.main()
