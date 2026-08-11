"""
Tests for character version service.
"""

import unittest
import tempfile
import os
from services.database.character_version_service import CharacterVersionService


class TestCharacterVersionService(unittest.TestCase):
    def setUp(self):
        self.tmpdb = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.tmpdb.name
        self.tmpdb.close()
        self.service = CharacterVersionService(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_add_and_list_versions(self):
        v1 = self.service.add_version("proj", "char", "prompt1", 1, "flux", "/img1.png")
        v2 = self.service.add_version("proj", "char", "prompt2", 2, "flux", "/img2.png")
        versions = self.service.list_versions("proj", "char")
        self.assertEqual(len(versions), 2)
        self.assertEqual(versions[0]["version"], 2)

    def test_set_default(self):
        self.service.add_version("proj", "char", "prompt", 1, "flux", "/img.png")
        self.service.add_version("proj", "char", "prompt", 2, "flux", "/img2.png")
        self.assertTrue(self.service.set_default("proj", "char", 2))
        default = self.service.get_default("proj", "char")
        self.assertEqual(default["version"], 2)


if __name__ == "__main__":
    unittest.main()
