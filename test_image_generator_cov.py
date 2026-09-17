import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

from generators import image_generator as ig


class _FakeImage:
    def __init__(self):
        self.saved = []

    def save(self, path):
        self.saved.append(path)


class _FakeResult:
    def __init__(self, n=1):
        self.images = [_FakeImage() for _ in range(n)]


class _FakePipe:
    def __init__(self, result=None, exc=None):
        self.result = result or _FakeResult()
        self.exc = exc
        self.calls = []

    def to(self, device):
        self.device = device
        return self

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.exc:
            raise self.exc
        return self.result


class _FakeMetrics:
    def start_timer(self):
        pass

    def end_timer(self):
        pass

    def record_generation(self, *args):
        self.recorded = args

    def save_metrics(self, base, model):
        return f"{base}_{model}.json"


class ImageGeneratorCoverageTest(unittest.TestCase):
    def _patch_common(self):
        return [
            patch.object(ig, "setup_model_directories"),
            patch.object(ig, "resolve_model_path", return_value="local/model"),
            patch.object(ig, "get_model_config", return_value=("cpu", "float16")),
            patch.object(ig, "cleanup_pipeline"),
            patch.object(ig, "ModelMetrics", _FakeMetrics),
        ]

    def test_resolve_model_path_prefers_local_nonempty(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(ig, "project_root", tmp), patch.object(
                ig, "MODEL_PATHS", {"diffusion": "models/diffusion"}
            ):
                local = os.path.join(tmp, "models/diffusion/sdxl")
                os.makedirs(local)
                self.assertEqual(
                    ig.resolve_model_path("diffusion", "sdxl", "hub"), "hub"
                )
                open(os.path.join(local, "file"), "w").close()
                self.assertEqual(
                    ig.resolve_model_path("diffusion", "sdxl", "hub"), local
                )

    def test_cleanup_pipeline_clears_cuda_and_mps_branches(self):
        with patch.object(ig.gc, "collect") as collect, patch.object(
            ig.torch.backends.mps, "is_available", return_value=True
        ), patch.object(ig.torch.mps, "empty_cache") as mps_empty:
            ig.cleanup_pipeline(object())
        collect.assert_called_once()
        mps_empty.assert_called_once()

        with patch.object(ig.gc, "collect"), patch.object(
            ig.torch.backends.mps, "is_available", return_value=False
        ), patch.object(ig.torch.cuda, "is_available", return_value=True), patch.object(
            ig.torch.cuda, "empty_cache"
        ) as cuda_empty:
            ig.cleanup_pipeline(None)
        cuda_empty.assert_called_once()

    def test_generate_images_sdxl_and_flux_dev_call_expected_kwargs(self):
        sdxl_pipe = _FakePipe(_FakeResult(2))
        patches = self._patch_common()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patch.object(
            ig.StableDiffusionXLPipeline, "from_pretrained", return_value=sdxl_pipe
        ):
            out = ig.generate_images(
                "blue bird", num_images=2, model_name="sdxl", task_name="task"
            )
        self.assertEqual(
            out, ["outputs/task_sdxl_benchmark.png", "outputs/task_sdxl_benchmark.png"]
        )
        self.assertEqual(sdxl_pipe.calls[0][1]["num_images"], 2)

        flux_pipe = _FakePipe(_FakeResult(1))
        patches = self._patch_common()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patch.object(
            ig.FluxPipeline, "from_pretrained", return_value=flux_pipe
        ):
            ig.generate_images("blue bird", model_name="flux_dev", steps=4, cfg=1.5)
        self.assertEqual(flux_pipe.calls[0][1]["max_sequence_length"], 512)
        self.assertEqual(flux_pipe.calls[0][1]["num_inference_steps"], 4)

    def test_generate_images_invalid_and_exception_cleanup(self):
        with self.assertRaises(ValueError):
            ig.generate_images("p", model_name="not-real")
        bad_pipe = _FakePipe(exc=RuntimeError("generate failed"))
        patches = self._patch_common()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patch.object(
            ig.FluxPipeline, "from_pretrained", return_value=bad_pipe
        ):
            with self.assertRaises(RuntimeError):
                ig.generate_images("p", model_name="flux_dev")

    def test_generate_image_success_multiple_default_and_fallback_generation_failure(
        self,
    ):
        pipe = _FakePipe(_FakeResult(1))
        with patch.object(
            ig, "get_model_config", return_value=("cpu", "float16")
        ), patch.object(
            ig.StableDiffusionXLPipeline, "from_pretrained", return_value=pipe
        ), patch.object(
            ig, "cleanup_pipeline"
        ):
            out = ig.generate_image(
                "hello world", amount=2, output_filename="bad name!"
            )
        self.assertEqual(out, ["outputs/badname_001.png", "outputs/badname_002.png"])
        self.assertEqual(pipe.calls[0][0][0], "hello world")

        first_loader = MagicMock(
            side_effect=[
                RuntimeError("primary failed"),
                _FakePipe(exc=RuntimeError("gen fail")),
            ]
        )
        with patch.object(
            ig, "get_model_config", return_value=("cpu", "float16")
        ), patch.object(
            ig.StableDiffusionXLPipeline, "from_pretrained", first_loader
        ), patch.object(
            ig, "cleanup_pipeline"
        ), patch(
            "builtins.open", mock_open()
        ) as mopen:
            out = ig.generate_image("%%", model_name="unknown", output_filename="!!!")
        self.assertEqual(out, ["outputs/generated_image.png"])
        mopen.assert_called_once_with("outputs/generated_image.png", "w")

    def test_generate_image_fallback_failure_raises_value_error(self):
        with patch.object(
            ig, "get_model_config", return_value=("cpu", "float16")
        ), patch.object(
            ig.StableDiffusionXLPipeline,
            "from_pretrained",
            side_effect=RuntimeError("load failed"),
        ):
            with self.assertRaises(ValueError):
                ig.generate_image("p")


if __name__ == "__main__":
    unittest.main()
