"""
Character version service for managing multiple variants per character.
"""

import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Any


class CharacterVersionService:
    def __init__(self, db_path: str = "story_engine.db"):
        self.db_path = db_path
        self._init_table()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_table(self):
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS character_versions (
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
            """
        )
        conn.commit()
        conn.close()

    def list_versions(self, project: str, character_name: str) -> List[Dict[str, Any]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM character_versions 
            WHERE project = ? AND character_name = ?
            ORDER BY version DESC
        """, (project, character_name)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def add_version(self, project: str, character_name: str, prompt: str,
                    seed: int, model: str, image_path: str) -> Dict[str, Any]:
        conn = self._connect()
        # Get next version number
        cur = conn.execute("""
            SELECT COALESCE(MAX(version), 0) + 1 as next_version
            FROM character_versions
            WHERE project = ? AND character_name = ?
        """, (project, character_name))
        next_version = cur.fetchone()[0]

        cur = conn.execute("""
            INSERT INTO character_versions
            (project, character_name, version, prompt, seed, model, image_path, created_at, is_default)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (project, character_name, next_version, prompt, seed, model, image_path, datetime.now()))
        conn.commit()
        version_id = cur.lastrowid
        conn.close()

        return self.get_version(project, character_name, next_version)

    def get_version(self, project: str, character_name: str, version: int) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        row = conn.execute("""
            SELECT * FROM character_versions
            WHERE project = ? AND character_name = ? AND version = ?
        """, (project, character_name, version)).fetchone()
        conn.close()
        return dict(row) if row else None

    def set_default(self, project: str, character_name: str, version: int) -> bool:
        conn = self._connect()
        # Clear existing defaults
        conn.execute("""
            UPDATE character_versions SET is_default = 0
            WHERE project = ? AND character_name = ?
        """, (project, character_name))
        # Set new default
        cur = conn.execute("""
            UPDATE character_versions SET is_default = 1
            WHERE project = ? AND character_name = ? AND version = ?
        """, (project, character_name, version))
        conn.commit()
        success = cur.rowcount > 0
        conn.close()
        return success

    def get_default(self, project: str, character_name: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        row = conn.execute("""
            SELECT * FROM character_versions
            WHERE project = ? AND character_name = ? AND is_default = 1
        """, (project, character_name)).fetchone()
        conn.close()
        return dict(row) if row else None


character_version_service = CharacterVersionService()
