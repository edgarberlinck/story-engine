"""Object/Artifact service: persists object references (name, type, description,
owner, visual identity, properties) so scenes can reference objects by name only.
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional, Dict, Any, List


class ObjectService:
    def __init__(self, db_path: str = "story_engine.db"):
        self.db_path = db_path
        self._init_table()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_table(self):
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT NOT NULL DEFAULT 'test_project',
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                description TEXT,
                owner TEXT,
                visual_identity TEXT,
                properties_json TEXT NOT NULL DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(project, name)
            )
            """)
        conn.commit()
        conn.close()

    def save_object(
        self,
        name: str,
        obj_type: str,
        description: str,
        owner: Optional[str] = None,
        visual_identity: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        project: str = "test_project",
    ) -> int:
        """Insert or update an object reference. Returns the row id."""
        conn = self._connect()
        props = json.dumps(properties) if properties else "{}"
        cursor = conn.execute(
            """
            INSERT INTO objects
                (project, name, type, description, owner, visual_identity, properties_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(project, name) DO UPDATE SET
                type = excluded.type,
                description = excluded.description,
                owner = COALESCE(excluded.owner, objects.owner),
                visual_identity = COALESCE(excluded.visual_identity, objects.visual_identity),
                properties_json = COALESCE(excluded.properties_json, objects.properties_json)
            """,
            (
                project,
                name,
                obj_type,
                description,
                owner,
                visual_identity,
                props,
                datetime.now(),
            ),
        )
        conn.commit()
        row_id = cursor.lastrowid
        conn.close()
        return row_id

    def get_object(
        self, name: str, project: str = "test_project"
    ) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM objects WHERE project = ? AND name = ?",
            (project, name),
        ).fetchone()
        if row:
            obj = dict(row)
            obj["properties"] = json.loads(obj["properties_json"])
        conn.close()
        return obj if row else None

    def list_objects(self, project: str = "test_project") -> List[Dict[str, Any]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM objects WHERE project = ? ORDER BY name", (project,)
        ).fetchall()
        conn.close()
        result = []
        for r in rows:
            obj = dict(r)
            obj["properties"] = json.loads(obj.get("properties_json") or "{}")
            result.append(obj)
        return result

    def delete_object(self, name: str, project: str = "test_project") -> bool:
        conn = self._connect()
        cur = conn.execute(
            "DELETE FROM objects WHERE project = ? AND name = ?",
            (project, name),
        )
        conn.commit()
        success = cur.rowcount > 0
        conn.close()
        return success

    def find_objects_in_text(
        self, text: str, project: str = "test_project"
    ) -> List[Dict[str, Any]]:
        """Return all known objects whose name appears in the given text."""
        import re

        found = []
        text_lower = text.lower()
        for o in self.list_objects(project):
            name_lower = o["name"].lower()
            pattern = r"\b" + re.escape(name_lower) + r"\b"
            if re.search(pattern, text_lower):
                found.append(o)
        return found


# Singleton instance
object_service = ObjectService()
