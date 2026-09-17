"""
Tests for core/scene_manager.SceneManager.

SceneManager wraps SceneService (filesystem sync + DB metadata) and
generate_scene (image generation). The image backend is patched out so no
real diffusion run happens. The DB-backed service is pointed at a throwaway
sqlite database (via SceneManager.service = SceneService(db_path)).

Filesystem layout is faked under a temporary directory by patching
utils.project_paths.OUTPUTS_ROOT, which SceneManager uses via
`from utils.project_paths import OUTPUTS_ROOT` (re-import to pick up the patch).
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from services.database.scene_service import SceneService
from core.scene_manager import SceneManager, scene_manager


class TestSceneManager(unittest.TestCase):
    def setUp(self):
        self.tmpdb = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.tmpdb.name
        self.tmpdb.close()

        self.tmp_dir = tempfile.TemporaryDirectory()
        self.out_root = self.tmp_dir.name

        # A SceneManager whose service talks to the temp DB.
        self.sm = SceneManager()
        self.sm.service = SceneService(self.db_path)

    def tearDown(self):
        self.tmp_dir.cleanup()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    # -- list_scenes ------------------------------------------------------
    def test_list_scenes_no_directory(self):
        # No scenes dir on disk -> sync no-ops, empty list returned.
        result = self.sm.list_scenes("ghost_project")
        self.assertEqual(result, [])

    def test_list_scenes_imports_from_filesystem(self):
        # A scene dir with scene.png that is NOT in the DB should be
        # imported into the DB and then returned by list_scenes.
        from pathlib import Path

        scene_dir = Path(self.out_root) / "proj" / "scenes" / "scene_1"
        scene_dir.mkdir(parents=True)
        (scene_dir / "scene.png").write_text("fake png bytes")

        with patch("core.scene_manager.OUTPUTS_ROOT", Path(self.out_root)):
            result = self.sm.list_scenes("proj")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["scene_number"], 1)
        self.assertEqual(result[0]["prompt"], "(imported from filesystem)")
        self.assertTrue(result[0]["image_path"].endswith("scene.png"))

    def test_list_scenes_skips_non_scene_and_known(self):
        from pathlib import Path

        scenes = Path(self.out_root) / "proj" / "scenes"
        # A non-matching directory and an empty "scene_2" with no image.
        (scenes / "random_folder").mkdir(parents=True)
        empty = scenes / "scene_2"
        empty.mkdir(parents=True)

        # scene_3 has a differently-named png -> should be picked up from glob.
        s3 = scenes / "scene_3"
        s3.mkdir(parents=True)
        (s3 / "render.png").write_text("x")

        with patch("core.scene_manager.OUTPUTS_ROOT", Path(self.out_root)):
            result = self.sm.list_scenes("proj")

        numbers = sorted(s["scene_number"] for s in result)
        # Only scene_3 (render.png) imported; random_folder skipped, scene_2
        # (no image) skipped.
        self.assertEqual(numbers, [3])
        self.assertTrue(result[0]["image_path"].endswith("render.png"))

    def test_list_scenes_already_known_is_idempotent(self):
        from pathlib import Path

        scene_dir = Path(self.out_root) / "proj" / "scenes" / "scene_1"
        scene_dir.mkdir(parents=True)
        (scene_dir / "scene.png").write_text("data")

        with patch("core.scene_manager.OUTPUTS_ROOT", Path(self.out_root)):
            # First call imports it.
            self.sm.list_scenes("proj")
            # Second call must not duplicate.
            result = self.sm.list_scenes("proj")

        self.assertEqual(len(result), 1)

    # -- create_scene -----------------------------------------------------
    def test_create_scene_saves_metadata(self):
        fake_result = {
            "scene_number": 7,
            "prompt": "a serene lake",
            "image_path": "/some/scene.png",
        }
        with patch(
            "core.scene_manager.generate_scene", return_value=fake_result
        ) as mock_gen:
            saved = self.sm.create_scene(
                "proj", "a serene lake", seed=99, model="flux_klein"
            )

        # generate_scene was called with the forwarded kwargs.
        mock_gen.assert_called_once()
        kwargs = mock_gen.call_args.kwargs
        self.assertEqual(kwargs["prompt"], "a serene lake")
        self.assertEqual(kwargs["project"], "proj")
        self.assertEqual(kwargs["seed"], 99)

        # Metadata persisted via the service.
        self.assertEqual(saved["scene_number"], 7)
        self.assertEqual(saved["prompt"], "a serene lake")
        self.assertEqual(saved["image_path"], "/some/scene.png")
        self.assertEqual(saved["seed"], 99)
        self.assertEqual(saved["model"], "flux_klein")

        got = self.sm.get_scene("proj", 7)
        self.assertIsNotNone(got)
        self.assertEqual(got["prompt"], "a serene lake")

    def test_create_scene_with_explicit_number(self):
        fake_result = {
            "scene_number": 3,
            "prompt": "city",
            "image_path": "/c.png",
        }
        with patch("core.scene_manager.generate_scene", return_value=fake_result):
            saved = self.sm.create_scene(
                "proj", "city", scene_number=3, model="sdxl", seed=1
            )
        self.assertEqual(saved["scene_number"], 3)
        self.assertEqual(saved["model"], "sdxl")

    # -- get_scene --------------------------------------------------------
    def test_get_scene_returns_none_when_missing(self):
        self.assertIsNone(self.sm.get_scene("ghost", 123))

    def test_module_singleton_is_scene_manager(self):
        self.assertIsInstance(scene_manager, SceneManager)
        self.assertIsNotNone(scene_manager.service)


if __name__ == "__main__":
    unittest.main()
