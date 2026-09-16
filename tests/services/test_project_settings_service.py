"""
Tests for the project settings service and narrator configuration (plan §2).
"""

import unittest
import tempfile
import os
from services.database.project_settings_service import (
    ProjectSettingsService,
    NARRATOR_MODE_DEDICATED,
    NARRATOR_MODE_CHARACTER,
    DEFAULT_NARRATOR,
)


class TestProjectSettingsService(unittest.TestCase):
    def setUp(self):
        self.tmpdb = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.tmpdb.name
        self.tmpdb.close()
        self.service = ProjectSettingsService(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_get_setting_default(self):
        self.assertIsNone(self.service.get_setting("p", "missing"))
        self.assertEqual(self.service.get_setting("p", "missing", 42), 42)

    def test_set_and_get_setting_roundtrip(self):
        self.service.set_setting("p", "k", {"a": [1, 2]})
        self.assertEqual(self.service.get_setting("p", "k"), {"a": [1, 2]})
        self.service.set_setting("p", "k", "replaced")
        self.assertEqual(self.service.get_setting("p", "k"), "replaced")

    def test_settings_are_project_scoped(self):
        self.service.set_setting("a", "k", 1)
        self.assertIsNone(self.service.get_setting("b", "k"))

    def test_default_narrator(self):
        config = self.service.get_narrator("p")
        self.assertEqual(config["mode"], NARRATOR_MODE_DEDICATED)
        self.assertEqual(config["voice_prompt"], DEFAULT_NARRATOR["voice_prompt"])
        self.assertIsNone(config["character"])

    def test_set_dedicated_narrator(self):
        self.service.set_narrator("p", NARRATOR_MODE_DEDICATED, voice_prompt="Deep calm voice")
        config = self.service.get_narrator("p")
        self.assertEqual(config["mode"], NARRATOR_MODE_DEDICATED)
        self.assertEqual(config["voice_prompt"], "Deep calm voice")

    def test_set_character_narrator(self):
        self.service.set_narrator("p", NARRATOR_MODE_CHARACTER, character="Nikita")
        config = self.service.get_narrator("p")
        self.assertEqual(config["mode"], NARRATOR_MODE_CHARACTER)
        self.assertEqual(config["character"], "Nikita")

    def test_character_mode_requires_character(self):
        with self.assertRaises(ValueError):
            self.service.set_narrator("p", NARRATOR_MODE_CHARACTER)

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            self.service.set_narrator("p", "robot")


if __name__ == "__main__":
    unittest.main()
