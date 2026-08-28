"""
Tests for utils/configuration_manager.setup_model_directories.

The function creates directories relative to the current working directory
on the real filesystem, so each test chdir's into a temporary directory and
restores the original CWD afterwards.
"""

import os
import tempfile
import unittest

from utils.configuration_manager import setup_model_directories


class TestSetupModelDirectories(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._tmpdir = self._tmp.name
        self._orig_cwd = os.getcwd()
        os.chdir(self._tmpdir)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        self._tmp.cleanup()

    def test_creates_model_directories_and_outputs(self):
        setup_model_directories()
        self.assertTrue(os.path.isdir("models/diffusion"))
        self.assertTrue(os.path.isdir("models/segmentation"))
        self.assertTrue(os.path.isdir("models/text_generation"))
        self.assertTrue(os.path.isdir("outputs"))

    def test_is_idempotent_when_directories_exist(self):
        # Running twice must not raise (exist_ok=True).
        setup_model_directories()
        setup_model_directories()
        self.assertTrue(os.path.isdir("models/diffusion"))
        self.assertTrue(os.path.isdir("outputs"))


if __name__ == "__main__":
    unittest.main()
