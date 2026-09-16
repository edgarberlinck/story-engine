"""Location service: persists location references (name, type, description,
visual identity, parent/child hierarchy, state tracking) so scenes can reference
locations by name only.
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional, Dict, Any, List


class LocationService:
    def __init__(self, db_path: str = "story_engine.db"):
        self.db_path = db_path
        self._init_table()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_table(self):
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS locations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT NOT NULL DEFAULT 'test_project',
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                description TEXT,
                parent_location TEXT,
                visual_identity TEXT,
                properties_json TEXT NOT NULL DEFAULT '{}',
                state TEXT NOT NULL DEFAULT 'initial',
                state_history_json TEXT NOT NULL DEFAULT '[]',
                objects_present_json TEXT NOT NULL DEFAULT '[]',
                characters_associated_json TEXT NOT NULL DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(project, name)
            )
            """
        )
        conn.commit()
        conn.close()

    def save_location(
        self,
        name: str,
        loc_type: str,
        description: str,
        parent_location: Optional[str] = None,
        visual_identity: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        project: str = "test_project",
    ) -> int:
        """Insert or update a location reference. Returns the row id."""
        conn = self._connect()
        props = json.dumps(properties) if properties else "{}"
        state_history = json.dumps(["initial"])
        objects_present = json.dumps([])
        characters_associated = json.dumps([])
        conn.execute(
            """
            INSERT INTO locations
                (project, name, type, description, parent_location, visual_identity, properties_json, state, state_history_json, objects_present_json, characters_associated_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(project, name) DO UPDATE SET
                type = excluded.type,
                description = excluded.description,
                parent_location = COALESCE(excluded.parent_location, locations.parent_location),
                visual_identity = COALESCE(excluded.visual_identity, locations.visual_identity),
                properties_json = COALESCE(excluded.properties_json, locations.properties_json),
                state = COALESCE(excluded.state, locations.state),
                state_history_json = COALESCE(excluded.state_history_json, locations.state_history_json),
                objects_present_json = COALESCE(excluded.objects_present_json, locations.objects_present_json),
                characters_associated_json = COALESCE(excluded.characters_associated_json, locations.characters_associated_json)
            """,
            (project, name, loc_type, description, parent_location, visual_identity, props, "initial", state_history, objects_present, characters_associated, datetime.now()),
        )
        conn.commit()
        # Get the row ID - need to re-query since INSERT OR REPLACE may not give lastrowid
        row = conn.execute("SELECT id FROM locations WHERE project = ? AND name = ?", (project, name)).fetchone()
        row_id = row[0] if row else conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()
        return row_id

    def get_location(self, name: str, project: str = "test_project") -> Optional[Dict[str, Any]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM locations WHERE project = ? AND name = ?",
            (project, name),
        ).fetchone()
        if row:
            loc = dict(row)
            loc["properties"] = json.loads(loc["properties_json"])
            loc["state_history"] = json.loads(loc["state_history_json"])
            loc["objects_present"] = json.loads(loc["objects_present_json"])
            loc["characters_associated"] = json.loads(loc["characters_associated_json"])
        conn.close()
        return loc if row else None

    def list_locations(self, project: str = "test_project") -> List[Dict[str, Any]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM locations WHERE project = ? ORDER BY name", (project,)
        ).fetchall()
        conn.close()
        result = []
        for r in rows:
            loc = dict(r)
            loc["properties"] = json.loads(loc.get("properties_json") or "{}")
            loc["state_history"] = json.loads(loc.get("state_history_json") or "[]")
            loc["objects_present"] = json.loads(loc.get("objects_present_json") or "[]")
            loc["characters_associated"] = json.loads(loc.get("characters_associated_json") or "[]")
            result.append(loc)
        return result

    def delete_location(self, name: str, project: str = "test_project") -> bool:
        conn = self._connect()
        cur = conn.execute(
            "DELETE FROM locations WHERE project = ? AND name = ?",
            (project, name),
        )
        conn.commit()
        success = cur.rowcount > 0
        conn.close()
        return success

    def find_locations_in_text(
        self, text: str, project: str = "test_project"
    ) -> List[Dict[str, Any]]:
        """Return all known locations whose name appears in the given text."""
        import re
        found = []
        text_lower = text.lower()
        for loc in self.list_locations(project):
            name_lower = loc["name"].lower()
            pattern = r'\b' + re.escape(name_lower) + r'\b'
            if re.search(pattern, text_lower):
                found.append(loc)
        return found

    def update_state(
        self,
        name: str,
        project: str,
        new_state: str,
        objects_present: Optional[List[str]] = None,
        characters_associated: Optional[List[str]] = None,
    ) -> bool:
        """Update location state and optional present objects/characters."""
        conn = self._connect()
        # Get current state history
        loc_row = conn.execute(
            "SELECT state_history_json FROM locations WHERE project = ? AND name = ?",
            (project, name),
        ).fetchone()
        state_history = json.loads(loc_row[0]) if loc_row else ["initial"]
        state_history.append(new_state)

        objects_present_json = json.dumps(objects_present if objects_present else [])
        characters_associated_json = json.dumps(characters_associated if characters_associated else [])

        conn.execute(
            """
            UPDATE locations
            SET state = ?,
                state_history_json = ?,
                objects_present_json = ?,
                characters_associated_json = ?
            WHERE project = ? AND name = ?
            """,
            (new_state, json.dumps(state_history), objects_present_json, characters_associated_json, project, name),
        )
        conn.commit()
        success = conn.total_changes > 0
        conn.close()
        return success


# Singleton instance
location_service = LocationService()
