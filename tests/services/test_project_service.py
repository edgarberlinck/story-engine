"""
Tests for services/database/project_service.ProjectService.

ProjectService is a thin wrapper over DatabaseService. We re-point the
db_service singleton's db_path at a throwaway sqlite database (the same
technique used in tests/core/test_project_manager.py) so the wrapper's
CRUD calls operate on a temp DB.
"""

import time
import unittest
import tempfile
import os

from services.database.project_service import ProjectService, project_service


class TestProjectService(unittest.TestCase):
    def setUp(self):
        self.tmpdb = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.tmpdb.name
        self.tmpdb.close()

        from services.database import database_service
        self._module = database_service
        self._singleton = database_service.db_service
        self._original_path = self._singleton.db_path
        self._singleton.db_path = self.db_path
         # Build the schema against the patched temp DB.
        self._singleton.init_database()

        # A fresh ProjectService bound to the patched db singleton.
        self.service = ProjectService()

    def tearDown(self):
        self._singleton.db_path = self._original_path
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_create_project(self):
        pid = self.service.create_project("Proj", "Desc")
        self.assertTrue(pid.startswith("project_"))

    def test_get_created_project(self):
        pid = self.service.create_project("Proj", "Desc")
        got = self.service.get_project(pid)
        self.assertIsNotNone(got)
        self.assertEqual(got["name"], "Proj")
        self.assertEqual(got["description"], "Desc")

    def test_get_missing_project_returns_none(self):
        self.assertIsNone(self.service.get_project("project_nope"))

    def test_update_project(self):
        pid = self.service.create_project("Old", "OldDesc")
        ok = self.service.update_project(pid, name="New", description="NewDesc")
        self.assertTrue(ok)
        got = self.service.get_project(pid)
        self.assertEqual(got["name"], "New")
        self.assertEqual(got["description"], "NewDesc")

    def test_update_partial(self):
        pid = self.service.create_project("K", "KDesc")
        ok = self.service.update_project(pid, name="KRenamed")
        self.assertTrue(ok)
        got = self.service.get_project(pid)
        self.assertEqual(got["name"], "KRenamed")
        self.assertEqual(got["description"], "KDesc")

    def test_update_missing_returns_false(self):
        ok = self.service.update_project("project_nope", name="X")
        self.assertFalse(ok)

    def test_create_without_description(self):
        pid = self.service.create_project("NoDesc")
        got = self.service.get_project(pid)
        self.assertIsNone(got["description"])

    def test_list_projects(self):
        self.service.create_project("A")
        time.sleep(1.1)
        self.service.create_project("B")
        projects = self.service.list_projects()
        self.assertEqual(len(projects), 2)
        names = {p["name"] for p in projects}
        self.assertEqual(names, {"A", "B"})

    def test_search_projects(self):
        self.service.create_project("FindMe", "desc")
        time.sleep(1.1)
        self.service.create_project("Other", "desc")
        results = self.service.search_projects("FindMe")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "FindMe")

    def test_search_projects_no_match(self):
        self.service.create_project("Here", "desc")
        results = self.service.search_projects("NoSuchThing")
        self.assertEqual(results, [])

    def test_delete_project(self):
        pid = self.service.create_project("Del", "desc")
        ok = self.service.delete_project(pid)
        self.assertTrue(ok)
        self.assertIsNone(self.service.get_project(pid))

    def test_delete_missing_returns_false(self):
        ok = self.service.delete_project("project_nope")
        self.assertFalse(ok)

    def test_singleton_is_project_service(self):
        self.assertIsInstance(project_service, ProjectService)
        self.assertIs(project_service.db, self._singleton)


if __name__ == "__main__":
    unittest.main()
