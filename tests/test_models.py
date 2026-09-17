"""
Tests for models.py get_model_config dtype/device resolution.
"""

import unittest
from unittest.mock import patch

import torch

from models import get_model_config, DEFAULT_MODEL_CONFIGS


class TestGetModelConfig(unittest.TestCase):
    def test_mps_device_maps_each_dtype(self):
        # mps available -> device "mps" for every configured model type.
        with patch("torch.backends.mps.is_available", return_value=True), patch(
            "torch.cuda.is_available", return_value=False
        ):
            for model_type, expected_torch_dtype in {
                "diffusion": torch.float16,
                "segmentation": torch.float32,
                "text_generation": torch.float16,
                "image_to_video": torch.bfloat16,
                "text_to_speech": torch.float16,
                "lip_sync": torch.float16,
                "music_generation": torch.float16,
            }.items():
                device, dtype = get_model_config(model_type)
                self.assertEqual(device, "mps")
                self.assertEqual(dtype, expected_torch_dtype)

    def test_cuda_device_when_mps_unavailable(self):
        with patch("torch.backends.mps.is_available", return_value=False), patch(
            "torch.cuda.is_available", return_value=True
        ):
            device, dtype = get_model_config("segmentation")
            self.assertEqual(device, "cuda")
            self.assertEqual(dtype, torch.float32)

    def test_cpu_device_when_no_accelerator(self):
        with patch("torch.backends.mps.is_available", return_value=False), patch(
            "torch.cuda.is_available", return_value=False
        ):
            device, dtype = get_model_config("image_to_video")
            self.assertEqual(device, "cpu")
            self.assertEqual(dtype, torch.bfloat16)

    def test_unknown_model_type_defaults_to_float32(self):
        # Unknown type -> empty config -> dtype falls back to float32.
        with patch("torch.backends.mps.is_available", return_value=False), patch(
            "torch.cuda.is_available", return_value=False
        ):
            device, dtype = get_model_config("does_not_exist")
            self.assertEqual(device, "cpu")
            self.assertEqual(dtype, torch.float32)

    def test_float16_and_bfloat16_mapping(self):
        # Explicitly exercise the float16 / bfloat16 / float32 string mapping
        # under the cpu branch (no accelerators).
        with patch("torch.backends.mps.is_available", return_value=False), patch(
            "torch.cuda.is_available", return_value=False
        ):
            _, d16 = get_model_config("diffusion")
            self.assertEqual(d16, torch.float16)
            _, db = get_model_config("image_to_video")
            self.assertEqual(db, torch.bfloat16)
            _, d32 = get_model_config("segmentation")
            self.assertEqual(d32, torch.float32)

    def test_configs_present_for_all_types(self):
        # Sanity: DEFAULT_MODEL_CONFIGS has an entry per known model type.
        for model_type in [
            "diffusion",
            "segmentation",
            "text_generation",
            "image_to_video",
            "text_to_speech",
            "lip_sync",
            "music_generation",
        ]:
            self.assertIn(model_type, DEFAULT_MODEL_CONFIGS)


if __name__ == "__main__":
    unittest.main()
