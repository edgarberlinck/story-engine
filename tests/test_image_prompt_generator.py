import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from generators.image_prompt_generator import generate_prompt_from_image


class ImagePromptGeneratorTest(unittest.TestCase):
    def test_generates_pose_aware_prompt_from_image_caption(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "reference.png"
            Image.new("RGB", (16, 16), color="white").save(image_path)
            fake_pipeline = lambda *args, **kwargs: [
                {"generated_text": "a woman wearing a blue jacket"}
            ]

            with patch("transformers.pipeline", return_value=fake_pipeline) as pipeline, patch(
                "models.get_model_config", return_value=("cpu", "float32")
            ), patch("generators.image_generator.cleanup_pipeline") as cleanup:
                prompt = generate_prompt_from_image(str(image_path))

        self.assertIn("a woman wearing a blue jacket", prompt)
        self.assertIn("same pose", prompt)
        pipeline.assert_called_once_with(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base",
            device="cpu",
        )
        cleanup.assert_called_once_with(fake_pipeline)

    def test_rejects_missing_image(self):
        with self.assertRaises(FileNotFoundError):
            generate_prompt_from_image("/missing/reference.png")


if __name__ == "__main__":
    unittest.main()
