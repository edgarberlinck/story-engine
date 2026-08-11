"""
Tests for project manager core.
"""

import unittest
import tempfile
import os
from pathlib import Path
from core.project_manager import ProjectManager


class TestProjectManager(unittest.TestCase):
    def setUp(self):
        self.tmpdb = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.tmpdb.name
        self.tmpdb.close()
        # Monkey patch db path
        from services.database import database_service
        self.original_path = database_service.db_service.db_path
        database_service.db_service.db_path = self.db_path

    def tearDown(self):
        from services.database import database_service
        database_service.db_service.db_path = self.original_path
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_create_and_list_projects(self):
        pm = ProjectManager()
        pid = pm.create_project("Test", "Desc")
        projects = pm.list_projects()
        self.assertEqual(len(projects), 1)
        self.assertEqual(projects[0]["name"], "Test")


if __name__ == "__main__":
    unittest.main()
