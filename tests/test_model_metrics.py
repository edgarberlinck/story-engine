"""
Tests for utils/model_metrics.

save_metrics writes into "outputs/..." relative to the CWD, so tests chdir
into a temporary directory. psutil is used for real (memory numbers are
only checked for type/shape, not exact values).
"""

import os
import json
import tempfile
import unittest

from utils import model_metrics
from utils.model_metrics import ModelMetrics, get_memory_usage


class TestGetMemoryUsage(unittest.TestCase):
    def test_returns_positive_float(self):
        usage = get_memory_usage()
        self.assertIsInstance(usage, float)
        self.assertGreater(usage, 0.0)


class TestModelMetrics(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._tmpdir = self._tmp.name
        self._orig_cwd = os.getcwd()
        os.chdir(self._tmpdir)
        # save_metrics writes to a CWD-relative "outputs/..." path, so make
        # sure the target subdir exists in the temp dir.
        os.makedirs("outputs", exist_ok=True)
        self.metrics = ModelMetrics()
        # start_timer records start_time/start_memory; end_timer derives the
        # duration/peak fields that record_generation reads.
        self.metrics.start_timer()
        self.metrics.end_timer()

    def tearDown(self):
        os.chdir(self._orig_cwd)
        self._tmp.cleanup()

    def test_timer_sets_fields(self):
        self.metrics.start_timer()
        self.metrics.end_timer()
        self.assertGreaterEqual(self.metrics.duration_ms, 0)
        # peak_memory_mb is an int delta (can be 0 on a quiet process).
        self.assertIsInstance(self.metrics.peak_memory_mb, int)

    def test_record_generation_stores_all_fields(self):
        self.metrics.record_generation(
            model_name="flux_dev",
            prompt="a cat",
            seed=42,
            steps=20,
            cfg=7.5,
            width=1024,
            height=1024,
            output_path="out.png",
        )
        m = self.metrics.metrics
        self.assertEqual(m["model"], "flux_dev")
        self.assertEqual(m["prompt"], "a cat")
        self.assertEqual(m["seed"], 42)
        self.assertEqual(m["steps"], 20)
        self.assertEqual(m["cfg"], 7.5)
        self.assertEqual(m["width"], 1024)
        self.assertEqual(m["height"], 1024)
        self.assertEqual(m["output"], "out.png")
        self.assertIn("duration_ms", m)
        self.assertIn("peak_memory_mb", m)

    def test_save_metrics_writes_json_file(self):
        self.metrics.record_generation(
            model_name="flux_dev",
            prompt="a cat",
            seed=42,
            steps=20,
            cfg=7.5,
            width=512,
            height=512,
            output_path="out.png",
        )
        filename = self.metrics.save_metrics("my_run", "flux_dev")
        self.assertEqual(filename, "outputs/my_run_flux_dev_benchmark_metrics.json")
        self.assertTrue(os.path.isfile(filename))
        with open(filename) as f:
            data = json.load(f)
        self.assertEqual(data["model"], "flux_dev")
        self.assertEqual(data["seed"], 42)


if __name__ == "__main__":
    unittest.main()
