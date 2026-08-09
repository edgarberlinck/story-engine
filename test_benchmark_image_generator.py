#!/usr/bin/env python3
"""Tests for the benchmark image generator module."""

import unittest
from unittest.mock import patch
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from generators.benchmark_image_generator import benchmark_models
from generators.image_generator import AVAILABLE_DIFFUSION_MODELS


class TestBenchmarkImageGenerator(unittest.TestCase):
    """Test the benchmark module without running actual generation."""

    @patch("generators.benchmark_image_generator.generate_images")
    def test_benchmark_runs_all_models(self, mock_generate):
        """Benchmark should call generate_images once per available model."""
        mock_generate.return_value = ["outputs/fake.png"]

        results = benchmark_models("test prompt", num_images=1, task_name="t", seed=7)

        self.assertEqual(set(results.keys()), set(AVAILABLE_DIFFUSION_MODELS.keys()))
        self.assertEqual(mock_generate.call_count, len(AVAILABLE_DIFFUSION_MODELS))
        for files in results.values():
            self.assertEqual(files, ["outputs/fake.png"])

    @patch("generators.benchmark_image_generator.generate_images")
    def test_benchmark_passes_seed_and_task_name(self, mock_generate):
        """Seed and task_name must be forwarded for fair, comparable runs."""
        mock_generate.return_value = []

        benchmark_models("prompt", num_images=2, task_name="char_base", seed=42)

        for call in mock_generate.call_args_list:
            self.assertEqual(call.kwargs.get("seed"), 42)
            self.assertEqual(call.kwargs.get("task_name"), "char_base")
            self.assertEqual(call.args[1], 2)  # num_images

    @patch("generators.benchmark_image_generator.generate_images")
    def test_benchmark_continues_on_model_failure(self, mock_generate):
        """A failing model must not abort the benchmark; it yields an empty list."""
        mock_generate.side_effect = RuntimeError("model exploded")

        results = benchmark_models("prompt")

        self.assertEqual(set(results.keys()), set(AVAILABLE_DIFFUSION_MODELS.keys()))
        for files in results.values():
            self.assertEqual(files, [])

    def test_deprecated_import_still_works(self):
        """Old import path should keep working (backward compatibility)."""
        from generators.image_generator import benchmark_models as legacy_bm
        self.assertTrue(callable(legacy_bm))


if __name__ == "__main__":
    unittest.main()
