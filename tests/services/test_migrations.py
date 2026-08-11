"""
Tests for database migrations and services.
"""

import unittest
import tempfile
import os
from services.database.migrations import migrate_database, seed_character_versions_from_existing
from services.database.database_service import DatabaseService
from services.database.character_version_service import CharacterVersionService
from services.database.scene_service import SceneService


class TestMigrations(unittest.TestCase):
    def setUp(self):
        self.tmpdb = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.tmpdb.name
        self.tmpdb.close()

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_migrate_creates_tables(self):
        migrate_database(self.db_path)
        conn = __import__("sqlite3").connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]
        self.assertIn("character_versions", tables)
        self.assertIn("scenes", tables)
        conn.close()

    def test_database_service_delete_project(self):
        db = DatabaseService(self.db_path)
        pid = db.create_project("Test Project", "desc")
        # create related data
        cv = CharacterVersionService(self.db_path)
        cv.add_version("test_proj", "char1", "prompt", 1, "flux", "/path/img.png")
        # delete project
        self.assertTrue(db.delete_project(pid))
        proj = db.get_project(pid)
        self.assertIsNone(proj)


if __name__ == "__main__":
    unittest.main()
