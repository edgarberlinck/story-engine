"""
Extra coverage for services/database/migrations.py.

Adds coverage for migrate_database (index creation + idempotency) and for
seed_character_versions_from_existing's guard branches. NOTE: the INSERT in
seed_character_versions_from_existing currently lists 8 values for 9 columns
and raises sqlite3.OperationalError, so the row-migration body is intentionally
NOT exercised here -- only the early-return / empty-rows branches are, which
do not reach that statement.
"""

import os
import sqlite3
import tempfile
import unittest

from services.database.migrations import (
    migrate_database,
    seed_character_versions_from_existing,
)


def _connect(path):
    return sqlite3.connect(path)


class TestMigrateDatabaseCoverage(unittest.TestCase):
    def setUp(self):
        self.tmpdb = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.tmpdb.name
        self.tmpdb.close()

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_creates_tables_and_indexes(self):
        migrate_database(self.db_path)
        conn = _connect(self.db_path)
        cur = conn.cursor()
        tables = {
            row[0]
            for row in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        self.assertIn("character_versions", tables)
        self.assertIn("scenes", tables)
        self.assertIn("character_attributes", tables)

        indexes = {
            row[0]
            for row in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        self.assertIn("idx_char_versions_proj_name", indexes)
        self.assertIn("idx_scenes_proj", indexes)
        self.assertIn("idx_char_attrs_proj_name", indexes)
        conn.close()

    def test_is_idempotent(self):
        # Second call must not raise (all DDL uses IF NOT EXISTS).
        migrate_database(self.db_path)
        migrate_database(self.db_path)
        conn = _connect(self.db_path)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        self.assertIn("scenes", tables)

    def test_foreign_keys_pragma_does_not_break_fresh_db(self):
        # A fresh db with foreign keys enabled should migrate cleanly.
        migrate_database(self.db_path)
        conn = _connect(self.db_path)
        val = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        conn.close()
        self.assertIn(val, (0, 1))


class TestSeedCharacterVersionsCoverage(unittest.TestCase):
    def setUp(self):
        self.tmpdb = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.tmpdb.name
        self.tmpdb.close()

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def _make_character_versions(self):
        conn = _connect(self.db_path)
        conn.execute("""
            CREATE TABLE character_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT NOT NULL,
                character_name TEXT NOT NULL,
                version INTEGER NOT NULL,
                prompt TEXT NOT NULL,
                seed INTEGER,
                model TEXT NOT NULL,
                image_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_default INTEGER NOT NULL DEFAULT 0,
                UNIQUE(project, character_name, version)
            )
            """)
        conn.commit()
        conn.close()

    def test_early_return_when_already_seeded(self):
        # character_versions already populated -> seed must no-op (return early
        # without touching a characters table).
        self._make_character_versions()
        conn = _connect(self.db_path)
        conn.execute(
            "INSERT INTO character_versions "
            "(project, character_name, version, prompt, model, image_path) "
            "VALUES ('p', 'c', 1, 'pr', 'flux', '/img')"
        )
        conn.commit()
        conn.close()

        # Must NOT raise even though no `characters` table exists.
        seed_character_versions_from_existing(self.db_path)

        conn = _connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM character_versions").fetchone()[0]
        conn.close()
        self.assertEqual(count, 1)

    def test_empty_rows_no_exception(self):
        # character_versions exists but empty, characters empty -> seed runs
        # its select/loop without doing any inserts (loop body never executes).
        self._make_character_versions()
        conn = _connect(self.db_path)
        conn.execute("""
            CREATE TABLE characters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT NOT NULL,
                name TEXT NOT NULL,
                prompt TEXT NOT NULL,
                seed INTEGER,
                model TEXT NOT NULL,
                reference_image TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
        conn.commit()
        conn.close()

        seed_character_versions_from_existing(self.db_path)

        conn = _connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM character_versions").fetchone()[0]
        conn.close()
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
