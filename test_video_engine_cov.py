import unittest
from unittest.mock import patch

from generators import video_engine as ve


class VideoEngineCoverageTest(unittest.TestCase):
    def test_create_validated_scene_no_characters(self):
        scene = {"scene_number": 5, "image_path": "scene.png", "prompt": "p"}
        with patch.object(ve, "generate_scene", return_value=scene) as gen:
            out = ve.create_validated_scene("p", project="proj", seed=10)
        self.assertIsNone(out["character_verified"])
        gen.assert_called_once_with("p", project="proj", scene_number=None, seed=10)

    def test_create_validated_scene_combines_single_and_list_names(self):
        chars = {
            "A": {"name": "A", "reference_image": "a.png"},
            "B": {"name": "B", "reference_image": "b.png"},
        }
        scene = {"scene_number": 1, "image_path": "scene.png", "prompt": "p"}
        with patch.object(ve, "get_character", side_effect=lambda name, project: chars[name]) as getc, \
             patch.object(ve, "generate_scene", return_value=scene), \
             patch.object(ve, "verify_character_in_scene", return_value=True) as verify:
            out = ve.create_validated_scene("p", character_name="A", character_names=["B"], project="proj")
        self.assertTrue(out["character_verified"])
        self.assertEqual([c.args[0] for c in getc.call_args_list], ["B", "A"])
        self.assertEqual(verify.call_count, 2)

    def test_create_validated_scene_false_when_not_required_after_attempts(self):
        char = {"name": "A", "reference_image": "a.png"}
        scene = {"scene_number": 1, "image_path": "scene.png", "prompt": "p"}
        with patch.object(ve, "get_character", return_value=char), \
             patch.object(ve, "generate_scene", return_value=scene) as gen, \
             patch.object(ve, "verify_character_in_scene", return_value=False):
            out = ve.create_validated_scene("p", character_name="A", max_attempts=2, require_verification=False)
        self.assertFalse(out["character_verified"])
        self.assertEqual(gen.call_count, 2)

    def test_animate_scene_uses_enriched_prompt_and_scene_out_dir(self):
        scene = {"scene_number": 7, "image_path": "image.png", "prompt": "raw", "enriched_prompt": "rich"}
        with patch.object(ve, "scene_out_dir", return_value="/tmp/out") as out_dir, \
             patch.object(ve, "generate_video", return_value={"video_path": "v.mp4"}) as gen:
            result = ve.animate_scene(scene, model_name="model", project="proj", seed=99, fps=5)
        self.assertEqual(result["video_path"], "v.mp4")
        out_dir.assert_called_once_with(7, "proj")
        gen.assert_called_once_with(image_path="image.png", prompt="rich", model_name="model", output_dir="/tmp/out", output_basename="scene_7_model", seed=99, fps=5)


if __name__ == "__main__":
    unittest.main()
