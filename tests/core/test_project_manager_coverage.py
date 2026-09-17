"""
Extra coverage tests for core/project_manager.py.

Complements tests/core/test_project_manager.py by exercising update, delete
(including the output-folder cleanup branch), get_project and the
space-to-underscore folder mapping. The database singleton is rebound to a
temporary SQLite file as in the sibling test, and OUTPUTS_ROOT is patched to a
temp directory so the real outputs/ folder is never touched (and folder cleanup
is mocked instead of performed for real).
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from core.project_manager import ProjectManager


class TestProjectManagerCoverage(unittest.TestCase):
    def setUp(self):
        self.tmpdb = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.tmpdb.name
        self.tmpdb.close()

        from services.database import database_service

        self.original_db_path = database_service.db_service.db_path
        database_service.db_service.db_path = self.db_path
        database_service.db_service.init_database()

        self.project = None

    def tearDown(self):
        from services.database import database_service

        database_service.db_service.db_path = self.original_db_path
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def _create(self, name="My Project", description="desc"):
        return ProjectManager().create_project(name, description)

    # -- create / list ---------------------------------------------------

    def test_create_and_list_projects(self):
        pm = ProjectManager()
        pid = self._create("Test", "Desc")
        projects = pm.list_projects()
        self.assertEqual(len(projects), 1)
        self.assertEqual(projects[0]["name"], "Test")

    def test_create_project_underscore_mapping(self):
        pm = ProjectManager()
        pid = self._create("My Project")
        self.assertTrue(pid.startswith("project_"))
        # get_project round-trips the record.
        self.assertEqual(pm.get_project(pid)["name"], "My Project")

    def test_list_projects_empty(self):
        pm = ProjectManager()
        self.assertEqual(pm.list_projects(), [])

    # -- update ----------------------------------------------------------

    def test_update_project_name_and_description(self):
        pm = ProjectManager()
        pid = self._create("Old", "old")
        self.assertTrue(pm.update_project(pid, name="New", description="new"))
        record = pm.get_project(pid)
        self.assertEqual(record["name"], "New")
        self.assertEqual(record["description"], "new")

    def test_update_project_partial(self):
        pm = ProjectManager()
        pid = self._create("Keep", "desc")
        self.assertTrue(pm.update_project(pid, name="Renamed"))
        record = pm.get_project(pid)
        self.assertEqual(record["name"], "Renamed")
        self.assertEqual(record["description"], "desc")

    def test_update_missing_returns_false(self):
        pm = ProjectManager()
        self.assertFalse(pm.update_project("project_nope"))

    # -- get -------------------------------------------------------------

    def test_get_project_missing_returns_none(self):
        pm = ProjectManager()
        self.assertIsNone(pm.get_project("project_nope"))

    # -- delete ----------------------------------------------------------

    def test_delete_project_missing_returns_false(self):
        pm = ProjectManager()
        self.assertFalse(pm.delete_project("project_nope"))

    def test_delete_project_with_output_folder_cleanup(self):
        pm = ProjectManager()
        pid = self._create("Cool Project")
        # Point OUTPUTS_ROOT at a temp root and create the on-disk folder the
        # manager would otherwise try to remove.
        root = Path(tempfile.mkdtemp())
        project_dir = root / "Cool_Project"
        project_dir.mkdir(parents=True)
        (project_dir / "file.txt").write_text("x")
        try:
            with patch("core.project_manager.OUTPUTS_ROOT", root), patch(
                "core.project_manager.shutil.rmtree"
            ) as rmtree:
                ok = pm.delete_project(pid)
            self.assertTrue(ok)
            rmtree.assert_called_once_with(project_dir)
            # The project is gone from the DB.
            self.assertIsNone(pm.get_project(pid))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_delete_project_no_output_folder(self):
        pm = ProjectManager()
        pid = self._create("Ephemeral")
        # Point OUTPUTS_ROOT at an empty temp root so the folder does not
        # exist; the cleanup branch is skipped but deletion still succeeds.
        root = Path(tempfile.mkdtemp())
        with patch("core.project_manager.OUTPUTS_ROOT", root), patch(
            "core.project_manager.shutil.rmtree"
        ) as rmtree:
            ok = pm.delete_project(pid)
        self.assertTrue(ok)
        rmtree.assert_not_called()
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
