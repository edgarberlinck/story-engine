import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from generators import image_engine as ie


class ImageEngineCoverageTest(unittest.TestCase):
    def test_reference_store_base_and_null_methods(self):
        store = ie.CharacterReferenceStore()
        with self.assertRaises(NotImplementedError):
            store.has_reference("c")
        with self.assertRaises(NotImplementedError):
            store.get_reference("c")
        self.assertIsNone(ie.NullCharacterReferenceStore().get_reference("c"))

    def test_generate_character_moves_reference_and_saves_record(self):
        service = MagicMock()
        service.get_character.return_value = {
            "name": "Alice",
            "reference_image": "ref.png",
        }
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            ie, "generate_images", return_value=["/tmp/generated.png"]
        ) as gen, patch.object(
            ie, "character_dir", return_value=Path(tmp)
        ), patch.object(
            ie.shutil, "move"
        ) as move, patch.object(
            ie, "character_service", service
        ):
            result = ie.generate_character(
                "Alice!", "portrait", project="proj", seed=9, steps=3
            )

        self.assertEqual(result["name"], "Alice")
        gen.assert_called_once()
        self.assertEqual(gen.call_args.kwargs["task_name"], "character_Alice")
        move.assert_called_once_with(
            "/tmp/generated.png", str(Path(tmp) / "reference.png")
        )
        service.save_character.assert_called_once()

    def test_get_character_missing_stale_and_present(self):
        service = MagicMock()
        service.get_character.side_effect = [None]
        with patch.object(ie, "character_service", service):
            self.assertIsNone(ie.get_character("Ghost", project="p"))

        stale = {
            "name": "A",
            "prompt": "p",
            "seed": 7,
            "model": "sdxl",
            "reference_image": "/no/file",
        }
        service = MagicMock()
        service.get_character.return_value = stale
        with patch.object(ie, "character_service", service), patch.object(
            ie, "generate_character", return_value={"regen": True}
        ) as regen:
            self.assertEqual(ie.get_character("A", project="p"), {"regen": True})
        regen.assert_called_once_with("A", "p", model="sdxl", project="p", seed=7)

        with tempfile.NamedTemporaryFile() as f:
            char = {"name": "A", "reference_image": f.name}
            service = MagicMock()
            service.get_character.return_value = char
            with patch.object(ie, "character_service", service):
                self.assertIs(ie.get_character("A"), char)

    def test_enrich_prompt_branches_and_conflict_messages(self):
        chars = [{"name": "Ann", "prompt": "red hair", "attributes": {"hair": "red"}}]
        service = MagicMock()
        service.find_characters_in_text.return_value = chars
        with patch.object(ie, "character_service", service), patch(
            "utils.token_budget.build_token_aware_scene_prompt",
            return_value=(
                "aware",
                {
                    "total_tokens_estimated": 10,
                    "max_tokens": 77,
                    "items_dropped": 0,
                    "dropped_details": [],
                },
            ),
        ), patch("utils.token_budget.count_tokens", return_value=10):
            enriched = ie._enrich_scene_prompt("Ann waves", "p")
        self.assertIn("Ann appearance: red hair", enriched)

        chars2 = [{"name": "Ann", "prompt": "red"}, {"name": "Bob", "prompt": "blue"}]
        service.find_characters_in_text.return_value = chars2
        with patch.object(ie, "character_service", service), patch(
            "utils.token_budget.build_token_aware_scene_prompt",
            return_value=(
                "aware two",
                {
                    "total_tokens_estimated": 10,
                    "max_tokens": 77,
                    "items_dropped": 1,
                    "dropped_details": [
                        {"category": "x", "priority": 1, "text": "drop"}
                    ],
                },
            ),
        ), patch("utils.token_budget.count_tokens", return_value=5):
            self.assertEqual(ie._enrich_scene_prompt("Ann and Bob", "p"), "aware two")

        conflict = MagicMock()
        conflict.message.return_value = "conflict!"
        with patch(
            "core.style_conflict.detect_scene_style_conflicts", return_value=[conflict]
        ):
            self.assertEqual(
                ie.detect_scene_style_conflicts("p", "proj"), ["conflict!"]
            )

    def test_enrich_prompt_appearance_only_uses_attributes(self):
        chars = [
            {"name": "Ann", "prompt": "stored style", "attributes": {"hair": "red"}}
        ]
        service = MagicMock()
        service.find_characters_in_text.return_value = chars
        with patch.object(ie, "character_service", service), patch(
            "utils.token_budget.build_token_aware_scene_prompt",
            return_value=(
                "aware",
                {
                    "total_tokens_estimated": 10,
                    "max_tokens": 77,
                    "items_dropped": 0,
                    "dropped_details": [],
                },
            ),
        ), patch("utils.token_budget.count_tokens", return_value=10), patch(
            "core.prompt_decomposer.extract_appearance_from_stored_prompt",
            return_value="extracted",
        ), patch(
            "core.prompt_decomposer.build_appearance_prompt",
            return_value="attribute appearance",
        ) as build:
            enriched = ie._enrich_scene_prompt(
                "Ann waves", "p", use_appearance_only=True
            )
        self.assertIn("Ann appearance: attribute appearance", enriched)
        build.assert_called_once_with("man", {"hair": "red"})

    def test_generate_scene_asset_pipeline_and_regular_result(self):
        chars = [{"name": "A", "prompt": "p"}, {"name": "B", "prompt": "q"}]
        service = MagicMock()
        service.find_characters_in_text.return_value = chars
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            ie, "character_service", service
        ), patch.object(
            ie, "detect_scene_style_conflicts", return_value=["warn"]
        ), patch.object(
            ie, "scene_dir", return_value=Path(tmp)
        ), patch(
            "core.scene_pipeline.generate_scene_pipeline",
            return_value={"scene_number": 3, "image_path": "x"},
        ):
            result = ie.generate_scene("A and B", project="p", scene_number=3)
        self.assertEqual(result["style_warnings"], ["warn"])

        service.find_characters_in_text.return_value = []
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            ie, "character_service", service
        ), patch.object(
            ie, "detect_scene_style_conflicts", return_value=[]
        ), patch.object(
            ie, "scene_dir", return_value=Path(tmp)
        ), patch.object(
            ie, "generate_images", return_value=["/tmp/generated.png"]
        ), patch.object(
            ie.shutil, "move"
        ) as move, patch(
            "utils.token_budget.count_tokens", return_value=11
        ):
            result = ie.generate_scene(
                "empty", project="p", scene_number=4, model="sdxl", seed=5
            )
        self.assertEqual(result["scene_number"], 4)
        self.assertEqual(result["token_count"], 11)
        move.assert_called_once_with("/tmp/generated.png", str(Path(tmp) / "scene.png"))

    def test_generate_scene_advanced_prompting_token_budget_and_optional_fields(self):
        chars = [{"name": "A", "prompt": "anime"}, {"name": "B", "prompt": "noir"}]
        service = MagicMock()
        service.find_characters_in_text.return_value = chars
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            ie, "character_service", service
        ), patch.object(ie, "next_scene_number", return_value=9), patch.object(
            ie, "scene_dir", return_value=Path(tmp)
        ), patch.object(
            ie, "detect_scene_style_conflicts", return_value=["style warning"]
        ), patch(
            "core.prompt_decomposer.should_use_scene_style_override", return_value=True
        ), patch(
            "core.advanced_prompting.build_advanced_scene_prompt",
            return_value=("advanced prompt", "negative things"),
        ), patch(
            "core.advanced_prompting.AdvancedPromptingEngine.recommend_model",
            return_value="flux_klein",
        ), patch(
            "core.style_conflict.detect_character_style", side_effect=["anime", "noir"]
        ), patch.object(
            ie, "_enrich_scene_prompt", return_value="appearance-only prompt"
        ) as enrich, patch(
            "utils.token_budget.build_token_aware_scene_prompt",
            return_value=(
                "short prompt",
                {
                    "total_tokens_estimated": 99,
                    "max_tokens": 77,
                    "items_dropped": 0,
                    "dropped_details": [],
                },
            ),
        ), patch(
            "utils.token_budget.count_tokens", side_effect=[50, 20, 20]
        ), patch.object(
            ie, "generate_images", return_value=["/tmp/generated.png"]
        ), patch.object(
            ie.shutil, "move"
        ):
            result = ie.generate_scene(
                "A and B",
                project="proj",
                scene_number=None,
                model="sdxl",
                use_asset_pipeline=False,
            )

        enrich.assert_called_once_with("A and B", "proj", True)
        self.assertIsInstance(result["scene_number"], int)
        self.assertEqual(result["enriched_prompt"], "short prompt")
        self.assertEqual(result["style_warnings"], ["style warning"])
        self.assertEqual(result["negative_prompt"], "negative things")
        self.assertEqual(result["model_recommendation"]["recommended"], "flux_klein")
        self.assertTrue(result["model_recommendation"]["use_recommended"])

    def test_verify_character_in_scene(self):
        self.assertIsNone(ie.verify_character_in_scene({"name": "A"}, "scene.png"))
        with patch.object(ie, "character_appears_in_image", return_value=True) as check:
            self.assertTrue(
                ie.verify_character_in_scene(
                    {"reference_image": "ref.png"}, "scene.png"
                )
            )
        check.assert_called_once_with("ref.png", "scene.png")


if __name__ == "__main__":
    unittest.main()
