"""Regression tests for the flux_klein model registration and loader branch.

Verifies that registering `flux_klein` in DIFFUSION_MODELS and the
`generate_images` loader branch load the FLUX.2 Klein pipeline rather than
falling back to SDXL. All diffusion loading is mocked.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DIFFUSION_MODELS
from generators import image_generator as ig


class _FakeMetricTracker:
    def start_timer(self):
        pass

    def end_timer(self):
        pass

    def record_generation(self, *a, **k):
        pass

    def save_metrics(self, *a, **k):
        return ""


class _FakeImages:
    def __init__(self, n=1):
        self.images = [_FakeImage() for _ in range(n)]


class _FakeImage:
    def save(self, path):
        pass


def _fake_pipe():
    pipe = MagicMock()
    pipe.to.return_value = pipe
    pipe.return_value = _FakeImages()
    return pipe


class FluxKleinRegistrationTest(unittest.TestCase):
    def test_flux_klein_is_registered(self):
        self.assertIn("flux_klein", DIFFUSION_MODELS)
        self.assertEqual(DIFFUSION_MODELS["flux_klein"], "black-forest-labs/FLUX.2-klein-4B")

    def test_generate_images_flux_klein_loads_klein_not_sdxl(self):
        with patch.object(ig, "setup_model_directories"), \
             patch.object(ig, "resolve_model_path", return_value="local/klein"), \
             patch.object(ig, "get_model_config", return_value=("cpu", "float16")), \
             patch.object(ig.Flux2KleinPipeline, "from_pretrained",
                          return_value=_fake_pipe()) as mock_klein, \
             patch.object(ig.StableDiffusionXLPipeline, "from_pretrained") as mock_sdxl, \
             patch.object(ig, "cleanup_pipeline"), \
             patch("generators.image_generator.ModelMetrics", _FakeMetricTracker):
            out = ig.generate_images("A test scene", model_name="flux_klein", task_name="t")

        mock_klein.assert_called_once()
        mock_sdxl.assert_not_called()
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
