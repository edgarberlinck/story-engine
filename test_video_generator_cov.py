import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from generators import video_generator as vg


class _FakeImage:
    LANCZOS = 1

    def __init__(self):
        self.resized = None
        self.size = (640, 480)

    def convert(self, mode):
        self.mode = mode
        return self

    def resize(self, size, resample):
        self.resized = (size, resample)
        return self


class _FakePipe:
    def __init__(self):
        self.calls = []
        self.offloaded = False

    def enable_model_cpu_offload(self):
        self.offloaded = True

    def to(self, device):
        self.device = device
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(frames=[["frame1", "frame2"]])


# All real i2v models are archived (models.py); tests exercise the pipeline
# machinery with a fake registry entry.
_FAKE_MODELS = {"ltx_video_095_i2v": "Lightricks/LTX-Video-0.9.5"}
_FAKE_PARAMS = {
    "ltx_video_095_i2v": {
        "width": 704,
        "height": 480,
        "num_frames": 9,
        "fps": 25,
        "guidance_scale": 3.0,
        "num_inference_steps": 2,
        "negative_prompt": "worst quality",
    }
}


class VideoGeneratorCoverageTest(unittest.TestCase):
    def test_resolve_video_model_path_local_and_hub(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            vg, "project_root", tmp
        ), patch.object(vg, "AVAILABLE_VIDEO_MODELS", _FAKE_MODELS), patch.object(
            vg, "MODEL_PATHS", {"image_to_video": "models/i2v"}
        ):
            local = os.path.join(tmp, "models/i2v/ltx_video_095_i2v")
            os.makedirs(local)
            self.assertEqual(
                vg.resolve_video_model_path("ltx_video_095_i2v"),
                vg.AVAILABLE_VIDEO_MODELS["ltx_video_095_i2v"],
            )
            open(os.path.join(local, "config"), "w").close()
            self.assertEqual(vg.resolve_video_model_path("ltx_video_095_i2v"), local)

    def test_load_pipeline_device_branches_and_unsupported(self):
        pipe = _FakePipe()
        with patch(
            "diffusers.LTXImageToVideoPipeline.from_pretrained", return_value=pipe
        ) as load:
            self.assertIs(
                vg._load_pipeline("ltx_video_095_i2v", "local", "mps", "dtype"), pipe
            )
        load.assert_called_once_with("local", torch_dtype="dtype")
        self.assertTrue(pipe.offloaded)

        pipe2 = _FakePipe()
        with patch(
            "diffusers.LTXImageToVideoPipeline.from_pretrained", return_value=pipe2
        ):
            self.assertIs(
                vg._load_pipeline("ltx_video_095_i2v", "local", "cpu", "dtype"), pipe2
            )
        self.assertEqual(pipe2.device, "cpu")
        with self.assertRaises(ValueError):
            vg._load_pipeline("bad", "local", "cpu", "dtype")

    def test_prepare_image_converts_and_resizes(self):
        image = _FakeImage()
        with patch.object(vg.Image, "open", return_value=image), patch.object(
            vg.Image, "LANCZOS", 99, create=True
        ):
            out = vg._prepare_image("in.png", 320, 240)
        self.assertIs(out, image)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.resized, ((320, 240), 99))

    def test_generate_video_success_with_overrides_and_metrics(self):
        pipe = _FakePipe()
        with tempfile.TemporaryDirectory() as tmp:
            img = os.path.join(tmp, "scene.png")
            open(img, "w").close()
            out_dir = os.path.join(tmp, "out")
            with patch.object(vg, "AVAILABLE_VIDEO_MODELS", _FAKE_MODELS), patch.object(
                vg, "MODEL_GENERATION_PARAMS", _FAKE_PARAMS
            ), patch.object(
                vg, "resolve_video_model_path", return_value="local/model"
            ), patch.object(
                vg, "get_model_config", return_value=("cpu", "float16")
            ), patch.object(
                vg, "_load_pipeline", return_value=pipe
            ) as load, patch.object(
                vg, "_prepare_image", return_value="imageobj"
            ) as prep, patch.object(
                vg, "get_memory_usage", side_effect=[100, 118]
            ), patch.object(
                vg, "cleanup_pipeline"
            ) as cleanup, patch(
                "diffusers.utils.export_to_video"
            ) as export:
                result = vg.generate_video(
                    img,
                    "motion",
                    model_name="ltx_video_095_i2v",
                    output_dir=out_dir,
                    output_basename="movie",
                    seed=12,
                    num_frames=3,
                    fps=8,
                    negative_prompt=None,
                )

            self.assertTrue(result["video_path"].endswith("movie.mp4"))
            self.assertTrue(os.path.isfile(result["metrics_path"]))
            with open(result["metrics_path"]) as f:
                metrics = json.load(f)
            self.assertEqual(metrics["num_frames"], 3)
            self.assertEqual(metrics["fps"], 8)
            self.assertEqual(metrics["peak_memory_mb"], 18)
            self.assertNotIn("negative_prompt", pipe.calls[0])
            prep.assert_called_once()
            load.assert_called_once()
            cleanup.assert_called_once_with(pipe)
            export.assert_called_once_with(
                ["frame1", "frame2"], result["video_path"], fps=8
            )

    def test_generate_video_uses_model_device_for_generator(self):
        # Regression: the torch.Generator must be created on the model's
        # inference device so an accelerated pipeline (e.g. "mps" on Apple
        # Silicon) does not fail at sampling with
        # "Expected a 'mps' device type for generator but found 'cpu'".
        pipe = _FakePipe()
        captured = {}  # record the device torch.Generator was built with

        class _GenCls:
            def __init__(self, device="cpu"):
                captured["device"] = device

            def manual_seed(self, seed):
                captured["seed"] = seed
                return self

        with tempfile.TemporaryDirectory() as tmp:
            img = os.path.join(tmp, "scene.png")
            open(img, "w").close()
            out_dir = os.path.join(tmp, "out")
            with patch.object(vg, "AVAILABLE_VIDEO_MODELS", _FAKE_MODELS), patch.object(
                vg, "MODEL_GENERATION_PARAMS", _FAKE_PARAMS
            ), patch.object(
                vg, "resolve_video_model_path", return_value="local/model"
            ), patch.object(
                vg, "get_model_config", return_value=("mps", "float16")
            ), patch.object(
                vg, "_load_pipeline", return_value=pipe
            ), patch.object(
                vg, "_prepare_image", return_value="imageobj"
            ), patch.object(
                vg, "get_memory_usage", side_effect=[100, 118]
            ), patch.object(
                vg, "cleanup_pipeline"
            ), patch(
                "diffusers.utils.export_to_video"
            ), patch.object(
                vg.torch, "Generator", _GenCls
            ):
                vg.generate_video(
                    img,
                    "motion",
                    model_name="ltx_video_095_i2v",
                    output_dir=out_dir,
                    seed=42,
                    num_frames=3,
                    fps=25,
                    negative_prompt=None,
                )

            # The generator is built on the model's device ("mps"), not "cpu".
            self.assertEqual(captured["device"], "mps")
            self.assertEqual(captured["seed"], 42)


if __name__ == "__main__":
    unittest.main()
