"""
Tests for character manager core.
"""

import unittest
import tempfile
import os
from core.character_manager import CharacterManager


class TestCharacterManager(unittest.TestCase):
    def setUp(self):
        self.tmpdb = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.tmpdb.name
        self.tmpdb.close()

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_list_characters_empty(self):
        cm = CharacterManager()
        # This will use default DB, but just test interface
        chars = cm.list_characters("nonexistent_project")
        self.assertIsInstance(chars, list)


if __name__ == "__main__":
    unittest.main()
