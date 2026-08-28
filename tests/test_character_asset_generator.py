"""
Unit tests for core.character_asset_generator.

Run with stdlib unittest.  File I/O into a tmp dir is exercised via
``shutil.copy`` from a tmp source; the heavy ``generate_images`` call is
patched so no real diffusion model is loaded.
"""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile

from core.character_asset_generator import (
    PLAIN_BACKGROUND_SUFFIX,
    _FORBIDDEN_ARTIFACTS,
    character_seed,
    build_character_asset_prompt,
    generate_character_asset,
    generate_character_assets,
    CharacterAsset,
)


class _FakeResolvedCharacter:
    """Mimics a ResolvedCharacter with the attributes the builder reads."""

    def __init__(self, name, identity=None, scene_presentation=None,
                 scene_pose=None, scene_action=None):
        self.name = name
        self.identity = identity or []
        self.scene_presentation = scene_presentation or []
        self.scene_pose = scene_pose
        self.scene_action = scene_action


class TestCharacterSeed(unittest.TestCase):

    def test_deterministic(self):
        self.assertEqual(character_seed("Alex"), character_seed("Alex"))

    def test_different_names_differ(self):
        # Different characters start from (almost) different basins.
        self.assertNotEqual(character_seed("Alex"), character_seed("Bob"))

    def test_case_insensitive(self):
        self.assertEqual(character_seed("Alex"), character_seed("aleX"))

    def test_base_seed_offset(self):
        self.assertEqual(
             character_seed("Alex", base_seed=0),
             character_seed("Alex", base_seed=100) - 100)

    def test_default_base_seed(self):
        self.assertEqual(character_seed("Alex"), character_seed("Alex", 42))

    def test_positive_offset_within_range(self):
        seed = character_seed("WhateverName", base_seed=42)
        self.assertGreaterEqual(seed, 42)
        self.assertLess(seed, 42 + 1000)


class TestSanitizeArtifacts(unittest.TestCase):

    def setUp(self):
        # Helper is module-private; grab it directly.
        import core.character_asset_generator as m
        self._sanitize = m._sanitize_fragment

    def test_forbidden_list_nonempty(self):
        self.assertGreaterEqual(len(_FORBIDDEN_ARTIFACTS), 3)

    def test_fantasy_clothing_stripped(self):
        out = self._sanitize("fantasy clothing, standing upright, brave")
        self.assertNotIn("fantasy clothing", out.lower())
        self.assertNotIn("standing upright", out.lower())
        self.assertIn("brave", out)

    def test_neutral_background_stripped(self):
        out = self._sanitize("a hero, neutral background, brave")
        self.assertNotIn("neutral background", out.lower())
        self.assertIn("brave", out)

    def test_not_a_portrait_stripped(self):
        out = self._sanitize("not a portrait, not a close-up")
        self.assertEqual(out.strip(), "")

    def test_double_commas_cleaned(self):
        out = self._sanitize("hero, , brave")
        self.assertNotIn(",,", out)
        self.assertIn("hero, brave", out)

    def test_surrounding_comma_stripped(self):
        out = self._sanitize("fantasy clothing, brave")
        self.assertFalse(out.startswith(","))

    def test_plain_text_unchanged(self):
        out = self._sanitize("a brave knight with a sword")
        self.assertEqual(out, "a brave knight with a sword")


class TestBuildCharacterAssetPrompt(unittest.TestCase):

    def test_object_input(self):
        rc = _FakeResolvedCharacter(
             "Nikita",
             identity=["young woman", "long red hair"],
             scene_presentation=["fantasy clothing", "standing upright"],
             scene_pose="neutral background",
             scene_action="not a close-up",
        )
        prompt = build_character_asset_prompt(rc)
        self.assertIn("Nikita", prompt)
        self.assertIn("young woman", prompt)
        self.assertIn(PLAIN_BACKGROUND_SUFFIX, prompt)
         # Forbidden artifacts must be stripped from scene_presentation/pose.
        self.assertNotIn("fantasy clothing", prompt.lower())
        self.assertNotIn("neutral background", prompt.lower())

    def test_dict_input(self):
        rc = {
             "name": "Roger",
             "identity": ["muscular man"],
             "scene_presentation": ["casual clothes"],
        }
        prompt = build_character_asset_prompt(rc)
        self.assertIn("Roger", prompt)
        self.assertIn("muscular man", prompt)
        self.assertIn(PLAIN_BACKGROUND_SUFFIX, prompt)

    def test_project_style_appended(self):
        rc = _FakeResolvedCharacter("Alex")
        prompt = build_character_asset_prompt(rc, project_style="cinematic")
        self.assertIn("cinematic", prompt)

    def test_object_missing_attrs_use_getattr_default(self):
         # Attribute present but None -> treated as empty.
        rc = _FakeResolvedCharacter("Alex")
        prompt = build_character_asset_prompt(rc)
        self.assertIn("Alex", prompt)
        self.assertNotIn("None", prompt)

    def test_token_budget_truncation(self):
           # A very long prompt with a tiny budget must be truncated to fit.
        rc = _FakeResolvedCharacter(
               "VeryLongNameHere",
             identity=[f"word {i}" for i in range(40)],
             scene_presentation=[f"pose {i}" for i in range(40)],
          )
        prompt = build_character_asset_prompt(rc, token_limit=8)
        # The full prompt would be far longer than the truncated result.
        full = build_character_asset_prompt(rc, token_limit=100000)
        self.assertLess(len(prompt.split()), len(full.split()))

    def test_empty_lists_skipped(self):
        rc = _FakeResolvedCharacter("Alex", identity=[], scene_presentation=[])
        prompt = build_character_asset_prompt(rc)
        self.assertIn("Alex", prompt)
        self.assertIn(PLAIN_BACKGROUND_SUFFIX, prompt)


class TestGenerateCharacterAsset(unittest.TestCase):

    def test_generate_single_mocked(self):
        rc = _FakeResolvedCharacter("Alex")
         # generate_images is imported *inside* the function from
         # generators.image_engine, so patch it at its source.
        with patch("generators.image_engine.generate_images") as gen:
            fake_path = "/tmp/fake_image.png"
            gen.return_value = [fake_path]
            asset = generate_character_asset(rc, project="proj", model="sdxl")

        self.assertIsInstance(asset, CharacterAsset)
        self.assertEqual(asset.name, "Alex")
        self.assertEqual(asset.image_path, fake_path)
        self.assertEqual(asset.seed, character_seed("Alex"))
        self.assertIn("Alex", asset.prompt_used)
        gen.assert_called_once()
         # Verify the prompt/seed flow through to generate_images.
        _, kwargs = gen.call_args
        self.assertEqual(kwargs["seed"], character_seed("Alex"))
        self.assertEqual(kwargs["model_name"], "sdxl")
        self.assertEqual(kwargs["task_name"], "asset_Alex")

    def test_output_dir_copies_file(self):
        rc = _FakeResolvedCharacter("Alex")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "source.png"
            src.write_text("fakepng")
            out_dir = Path(tmp) / "assets"

            with patch("generators.image_engine.generate_images") as gen:
                gen.return_value = [str(src)]
                asset = generate_character_asset(
                     rc, project="proj", model="sdxl", output_dir=out_dir)

            target = out_dir / "asset_Alex.png"
            self.assertTrue(target.exists())
            self.assertEqual(asset.image_path, str(target))
            self.assertEqual(target.read_text(), "fakepng")

    def test_dict_input_default_name(self):
         # A dict without a name key falls back to "character".
        rc = {}
        with patch("generators.image_engine.generate_images") as gen:
            gen.return_value = ["/tmp/x.png"]
            asset = generate_character_asset(rc, project="p")
        self.assertEqual(asset.name, "character")
        self.assertEqual(asset.seed, character_seed("character"))


class TestGenerateCharacterAssets(unittest.TestCase):

    def test_generate_multiple(self):
        rc1 = _FakeResolvedCharacter("Alice")
        rc2 = _FakeResolvedCharacter("Bob")
        with patch("generators.image_engine.generate_images") as gen:
            gen.return_value = ["/tmp/img.png"]
            assets = generate_character_assets(
                 [rc1, rc2], project="proj", model="sdxl",
                 base_seed=7)

        self.assertEqual(len(assets), 2)
        self.assertEqual([a.name for a in assets], ["Alice", "Bob"])
        self.assertEqual(gen.call_count, 2)
         # Seeds use the per-character name + shared base_seed.
        self.assertEqual(assets[0].seed, character_seed("Alice", 7))
        self.assertEqual(assets[1].seed, character_seed("Bob", 7))

    def test_generate_empty_list(self):
        with patch("generators.image_engine.generate_images") as gen:
            assets = generate_character_assets([], project="proj")
        self.assertEqual(assets, [])
        gen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
