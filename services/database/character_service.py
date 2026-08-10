"""Character service: persists character references (prompt, seed, model,
reference image path) so scenes can reference characters by name only.
"""

import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any, List


class CharacterService:
    def __init__(self, db_path: str = "story_engine.db"):
        self.db_path = db_path
        self._init_table()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_table(self):
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS characters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT NOT NULL DEFAULT 'test_project',
                name TEXT NOT NULL,
                prompt TEXT NOT NULL,
                seed INTEGER,
                model TEXT NOT NULL,
                reference_image TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(project, name)
            )
            """
        )
        conn.commit()
        conn.close()

    def save_character(
        self,
        name: str,
        prompt: str,
        seed: int,
        model: str,
        reference_image: str,
        project: str = "test_project",
    ) -> int:
        """Insert or update a character reference. Returns the row id."""
        conn = self._connect()
        cursor = conn.execute(
            """
            INSERT INTO characters
                (project, name, prompt, seed, model, reference_image, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(project, name) DO UPDATE SET
                prompt = excluded.prompt,
                seed = excluded.seed,
                model = excluded.model,
                reference_image = excluded.reference_image
            """,
            (project, name, prompt, seed, model, reference_image, datetime.now()),
        )
        conn.commit()
        row_id = cursor.lastrowid
        conn.close()
        return row_id

    def get_character(
        self, name: str, project: str = "test_project"
    ) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM characters WHERE project = ? AND name = ?",
            (project, name),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def list_characters(self, project: str = "test_project") -> List[Dict[str, Any]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM characters WHERE project = ? ORDER BY name", (project,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def find_characters_in_text(
        self, text: str, project: str = "test_project"
    ) -> List[Dict[str, Any]]:
        """Return all known characters whose name appears in the given text."""
        return [
            c for c in self.list_characters(project) if c["name"].lower() in text.lower()
        ]


# Singleton instance
character_service = CharacterService()
