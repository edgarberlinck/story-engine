"""
Tests for scene service.
"""

import unittest
import tempfile
import os
from services.database.scene_service import SceneService


class TestSceneService(unittest.TestCase):
    def setUp(self):
        self.tmpdb = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.tmpdb.name
        self.tmpdb.close()
        self.service = SceneService(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_save_and_list_scenes(self):
        scene = self.service.save_scene("proj", 1, "prompt", "/img.png", 42, "flux")
        scenes = self.service.list_scenes("proj")
        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0]["scene_number"], 1)


if __name__ == "__main__":
    unittest.main()
