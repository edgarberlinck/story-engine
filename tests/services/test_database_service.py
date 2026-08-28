"""
Tests for services/database/database_service.DatabaseService.

Each test spins up a throwaway sqlite database via tempfile so the real
story_engine.db is never touched.
"""

import time
import unittest
import tempfile
import os

from services.database.database_service import DatabaseService


def create_with_distinct_id(db, name, desc=None):
    """Create a project with a unique id.

    DatabaseService.create_project derives the id from int(timestamp) and a
    *failed* create leaves an unclosed connection that write-locks the db
    file, so retrying in a tight loop deadlocks.  The reliable way to get a
    distinct id is to wait for the wall-clock integer second to change."""
    time.sleep(1.1)
    return db.create_project(name, desc)


class TestDatabaseService(unittest.TestCase):
    def setUp(self):
        self.tmpdb = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.tmpdb.name
        self.tmpdb.close()
        self.db = DatabaseService(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_create_and_get_project(self):
        pid = self.db.create_project("My Project", "A cool project")
        self.assertTrue(pid.startswith("project_"))
        fetched = self.db.get_project(pid)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["name"], "My Project")
        self.assertEqual(fetched["description"], "A cool project")
        self.assertIn("id", fetched)
        self.assertIn("created_at", fetched)
        self.assertIn("updated_at", fetched)

    def test_get_missing_project_returns_none(self):
        self.assertIsNone(self.db.get_project("project_nonexistent"))

    def test_update_project_name_and_description(self):
        pid = self.db.create_project("Old Name", "Old Desc")
        ok = self.db.update_project(pid, name="New Name", description="New Desc")
        self.assertTrue(ok)
        fetched = self.db.get_project(pid)
        self.assertEqual(fetched["name"], "New Name")
        self.assertEqual(fetched["description"], "New Desc")

    def test_update_project_partial_keeps_existing(self):
        pid = self.db.create_project("Keep Name", "Keep Desc")
         # Only update the name; description should be preserved.
        ok = self.db.update_project(pid, name="Renamed")
        self.assertTrue(ok)
        fetched = self.db.get_project(pid)
        self.assertEqual(fetched["name"], "Renamed")
        self.assertEqual(fetched["description"], "Keep Desc")

    def test_update_missing_project_returns_false(self):
        ok = self.db.update_project("project_nonexistent", name="X")
        self.assertFalse(ok)
        ok2 = self.db.update_project("project_nonexistent", description="Y")
        self.assertFalse(ok2)

    def test_create_without_description(self):
        pid = self.db.create_project("No Desc")
        fetched = self.db.get_project(pid)
        self.assertEqual(fetched["name"], "No Desc")
        self.assertIsNone(fetched["description"])

    def test_list_projects_sorted_newest_first(self):
        pid1 = create_with_distinct_id(self.db, "First")
        create_with_distinct_id(self.db, "Second")
        pid3 = create_with_distinct_id(self.db, "Third")
        projects = self.db.list_projects()
        self.assertEqual(len(projects), 3)
        names = [p["name"] for p in projects]
          # Newest first -> "Third" should appear before "First".
        self.assertLess(names.index("Third"), names.index("First"))
        self.assertEqual(names[0], "Third")

    def test_search_projects_by_name(self):
        create_with_distinct_id(self.db, "Alpha Project", "desc alpha")
        create_with_distinct_id(self.db, "Beta Thing", "desc beta")
        results = self.db.search_projects("Alpha")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Alpha Project")

    def test_search_projects_by_description(self):
        create_with_distinct_id(self.db, "Gamma", "matching description here")
        create_with_distinct_id(self.db, "Delta", "unrelated description")
        results = self.db.search_projects("matching description")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Gamma")

    def test_search_projects_no_match(self):
        self.db.create_project("Something", "else")
        results = self.db.search_projects("zzz_no_match_zzz")
        self.assertEqual(results, [])

    def test_delete_existing_project(self):
        pid = self.db.create_project("To Delete", "desc")
        ok = self.db.delete_project(pid)
        self.assertTrue(ok)
        self.assertIsNone(self.db.get_project(pid))
        remaining = self.db.list_projects()
        self.assertEqual(len(remaining), 0)

    def test_delete_missing_project_returns_false(self):
        ok = self.db.delete_project("project_nonexistent")
        self.assertFalse(ok)

    def test_delete_project_cascades_related_data(self):
         # Create a project plus related character_version + scene rows that
        # share the same project id, then delete and confirm cascade.
        import sqlite3
        pid = self.db.create_project("Cascade", "desc")

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO character_versions "
            "(project, character_name, version, prompt, model, image_path) "
            "VALUES (?, ?, 1, ?, ?, ?)",
            (pid, "char", "prompt", "flux", "/img.png"),
         )
        conn.execute(
            "INSERT INTO scenes "
            "(project, scene_number, prompt, image_path) "
            "VALUES (?, 1, ?, ?)",
            (pid, "scene", "/scene.png"),
         )
        conn.commit()

        self.assertTrue(self.db.delete_project(pid))

        conn = sqlite3.connect(self.db_path)
        cv = conn.execute(
            "SELECT COUNT(*) FROM character_versions WHERE project = ?",
             (pid,),
        ).fetchone()[0]
        sc = conn.execute(
            "SELECT COUNT(*) FROM scenes WHERE project = ?", (pid,)
        ).fetchone()[0]
        conn.close()
        self.assertEqual(cv, 0)
        self.assertEqual(sc, 0)


if __name__ == "__main__":
    unittest.main()
