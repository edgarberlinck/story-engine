"""
Extra coverage tests for core/character_manager.py.

These complement tests/core/test_character_manager.py and exercise the
versioning, voice, and deletion paths of CharacterManager plus the
ensure_full_body helper. All database singletons are rebound to a
temporary SQLite file the same way the sibling test does, and the heavy
image/voice generation entry points are mocked so no real model loads.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from core.character_manager import CharacterManager, ensure_full_body, FULL_BODY_SUFFIX


class TestEnsureFullBody(unittest.TestCase):
    def test_returns_unchanged_when_already_full_body(self):
        prompt = "a person. full length wide shot of a standing figure"
        self.assertEqual(ensure_full_body(prompt), prompt)

    def test_appends_suffix_when_missing(self):
        result = ensure_full_body("a tall man")
        self.assertIn(FULL_BODY_SUFFIX, result)
        self.assertTrue(result.startswith("a tall man."))

    def test_strips_trailing_punctuation_before_suffix(self):
        # Trailing comma and period are stripped so the suffix reads cleanly.
        self.assertEqual(
            ensure_full_body("a tall man,"), "a tall man. " + FULL_BODY_SUFFIX
        )
        self.assertEqual(
            ensure_full_body("a tall man."), "a tall man. " + FULL_BODY_SUFFIX
        )

    def test_case_insensitive_detection(self):
        prompt = "Full Length Wide Shot already here"
        self.assertEqual(ensure_full_body(prompt), prompt)


class TestCharacterManagerCoverage(unittest.TestCase):
    def setUp(self):
        # Temporary DB shared by every service singleton so no real
        # story_engine.db is touched and the schemas are created on it.
        self.tmpdb = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.tmpdb.name
        self.tmpdb.close()

        from services.database import database_service
        from services.database.character_service import character_service
        from services.database.character_version_service import (
            character_version_service,
        )

        self.original_db_path = database_service.db_service.db_path
        self.original_char_path = character_service.db_path
        self.original_version_path = character_version_service.db_path

        database_service.db_service.db_path = self.db_path
        character_service.db_path = self.db_path
        character_version_service.db_path = self.db_path

        database_service.db_service.init_database()
        character_service._init_table()
        character_version_service._init_table()

        self.project = "test_project"

    def tearDown(self):
        from services.database import database_service
        from services.database.character_service import character_service
        from services.database.character_version_service import (
            character_version_service,
        )

        database_service.db_service.db_path = self.original_db_path
        character_service.db_path = self.original_char_path
        character_version_service.db_path = self.original_version_path
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    # -- simple forwarding methods ---------------------------------------

    def test_list_characters_empty(self):
        cm = CharacterManager()
        chars = cm.list_characters("nonexistent_project")
        self.assertIsInstance(chars, list)
        self.assertEqual(chars, [])

    def test_get_character_missing_returns_none(self):
        cm = CharacterManager()
        self.assertIsNone(cm.get_character("nobody", self.project))

    def test_get_voice_path_missing_returns_none(self):
        cm = CharacterManager()
        self.assertIsNone(cm.get_voice_path(self.project, "nobody"))

    def test_get_voice_path_present(self):
        cm = CharacterManager()
        cm.char_service.save_character(
            name="Hero",
            prompt="a hero",
            seed=1,
            model="flux_dev",
            reference_image="/tmp/x.png",
            project=self.project,
            voice_path="/tmp/hero.wav",
        )
        self.assertEqual(cm.get_voice_path(self.project, "Hero"), "/tmp/hero.wav")

    def test_list_versions_empty(self):
        cm = CharacterManager()
        self.assertEqual(cm.list_versions(self.project, "nobody"), [])

    # -- voice generation -------------------------------------------------

    def test_generate_voice_persists_path(self):
        cm = CharacterManager()
        cm.char_service.save_character(
            name="Hero",
            prompt="a hero",
            seed=1,
            model="flux_dev",
            reference_image="/tmp/x.png",
            project=self.project,
        )
        fake_wav = Path("/tmp/hero_generated.wav")
        with patch("core.voice_engine.voice_engine") as fake_engine:
            fake_engine.generate_character_voice.return_value = (
                fake_wav,
                "warm delivery",
            )
            result = cm.generate_voice(
                self.project,
                "Hero",
                char_type="man",
                attributes={"personality": "Brave"},
                instruct="warm delivery",
                force=True,
            )
        self.assertEqual(result, str(fake_wav))
        fake_engine.generate_character_voice.assert_called_once()
        self.assertGreaterEqual(fake_engine.generate_character_voice.call_count, 1)
        # Voice path persisted on the character record.
        self.assertEqual(cm.get_voice_path(self.project, "Hero"), str(fake_wav))

    # -- deletion ---------------------------------------------------------

    def test_delete_character_removes_files_and_versions(self):
        cm = CharacterManager()
        # Persist a character and a version row up front.
        cm.char_service.save_character(
            name="Hero",
            prompt="a hero",
            seed=1,
            model="flux_dev",
            reference_image="/tmp/x.png",
            project=self.project,
        )
        created = cm.version_service.add_version(
            project=self.project,
            character_name="Hero",
            prompt="a hero",
            seed=1,
            model="flux_dev",
            image_path="/tmp/hero_v1.png",
        )
        self.assertIsNotNone(created["version"])

        # The on-disk character directory to be removed by the manager.
        char_dir = Path(tempfile.mkdtemp())
        marker = char_dir / "reference.png"
        marker.write_text("x")
        try:
            with patch("core.character_manager.character_dir", return_value=char_dir):
                ok = cm.delete_character(self.project, "Hero")
            self.assertTrue(ok)
            # Character row gone.
            self.assertIsNone(cm.get_character("Hero", self.project))
            # Version rows gone.
            self.assertEqual(cm.list_versions(self.project, "Hero"), [])
            # On-disk folder removed.
            self.assertFalse(marker.exists())
        finally:
            shutil.rmtree(char_dir, ignore_errors=True)

    # -- versioning -------------------------------------------------------

    def _run_generate_versions(self, cm, num_versions=2, tmp_dir=None):
        with patch(
            "core.character_manager.character_dir",
            return_value=Path(tmp_dir or tempfile.mkdtemp()),
        ), patch("generators.image_generator.generate_images", return_value=None):
            return cm.generate_versions(
                self.project, "Hero", "a hero", num_versions=num_versions
            )

    def test_generate_versions_with_placeholder(self):
        cm = CharacterManager()
        cm.char_service.save_character(
            name="Hero",
            prompt="a hero",
            seed=1,
            model="flux_dev",
            reference_image="/tmp/x.png",
            project=self.project,
        )
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            versions = self._run_generate_versions(
                cm, num_versions=2, tmp_dir=str(tmp_dir)
            )
            self.assertEqual(len(versions), 2)
            self.assertEqual(versions[0]["version"], 1)
            self.assertEqual(versions[1]["version"], 2)
            # Manifest written next to the version dir.
            manifest = tmp_dir / "manifest.json"
            self.assertTrue(manifest.exists())
            # First version is the default.
            default = cm.version_service.get_default(self.project, "Hero")
            self.assertEqual(default["version"], 1)
            # Placeholder version files created on disk.
            self.assertTrue((tmp_dir / "versions" / "v_1.png").exists())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_generate_versions_uses_generated_files(self):
        cm = CharacterManager()
        cm.char_service.save_character(
            name="Hero",
            prompt="a hero",
            seed=1,
            model="flux_dev",
            reference_image="/tmp/x.png",
            project=self.project,
        )
        tmp_dir = Path(tempfile.mkdtemp())
        src = tmp_dir / "src.png"
        src.write_text("imgdata")
        with patch(
            "core.character_manager.character_dir", return_value=tmp_dir / "char_home"
        ), patch("generators.image_generator.generate_images", return_value=[str(src)]):
            versions = cm.generate_versions(
                self.project, "Hero", "a hero", num_versions=1
            )
        self.assertEqual(len(versions), 1)
        self.assertTrue((tmp_dir / "char_home" / "versions" / "v_1.png").exists())
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_set_default_version_copies_reference(self):
        cm = CharacterManager()
        cm.char_service.save_character(
            name="Hero",
            prompt="a hero",
            seed=1,
            model="flux_dev",
            reference_image="/tmp/x.png",
            project=self.project,
        )
        # A real source image to copy into the reference location.
        src = Path(tempfile.mkdtemp()) / "v1.png"
        src.write_text("imgdata")
        home = Path(tempfile.mkdtemp())
        try:
            created = cm.version_service.add_version(
                project=self.project,
                character_name="Hero",
                prompt="a hero",
                seed=7,
                model="flux_dev",
                image_path=str(src),
            )
            with patch("core.character_manager.character_dir", return_value=home):
                ok = cm.set_default_version(self.project, "Hero", created["version"])
            self.assertTrue(ok)
            self.assertTrue((home / "reference.png").exists())
        finally:
            shutil.rmtree(src.parent, ignore_errors=True)
            shutil.rmtree(home, ignore_errors=True)

    def test_set_default_version_invalid_returns_false(self):
        cm = CharacterManager()
        cm.char_service.save_character(
            name="Hero",
            prompt="a hero",
            seed=1,
            model="flux_dev",
            reference_image="/tmp/x.png",
            project=self.project,
        )
        ok = cm.set_default_version(self.project, "Hero", 999)
        self.assertFalse(ok)

    def test_generate_versions_forces_full_body_prompt(self):
        cm = CharacterManager()
        captured = {}

        def fake_add_version(**kwargs):
            captured["prompt"] = kwargs["prompt"]
            return {
                "version": 1,
                "seed": kwargs["seed"],
                "image_path": kwargs["image_path"],
                "prompt": kwargs["prompt"],
                "model": kwargs["model"],
            }

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            with patch(
                "core.character_manager.character_dir", return_value=tmp_dir
            ), patch("generators.image_generator.generate_images"), patch.object(
                cm.version_service, "add_version", side_effect=fake_add_version
            ), patch(
                "core.character_manager.character_version_service.get_default",
                return_value=None,
            ):
                cm.generate_versions(self.project, "Hero", "a hero", num_versions=1)
            self.assertIn(FULL_BODY_SUFFIX, captured["prompt"])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
