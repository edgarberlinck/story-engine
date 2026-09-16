"""Timeline service: explicit, user-directed timeline data (plan §4).

The user is the authority over the timeline. This service only stores what
the user defines (chapter/scene order, day, time, duration, events); it never
reorders or rewrites entries on its own.
"""

import sqlite3
import json
from typing import Optional, Dict, Any, List


class TimelineService:
    def __init__(self, db_path: str = "story_engine.db"):
        self.db_path = db_path
        self._init_table()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_table(self):
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS timeline (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT NOT NULL,
                chapter_id TEXT NOT NULL,
                chapter_number INTEGER NOT NULL,
                scene_id INTEGER,
                scene_number INTEGER NOT NULL,
                day INTEGER NOT NULL,
                time_of_day TEXT NOT NULL,
                duration_minutes INTEGER,
                events_json TEXT NOT NULL DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(project, chapter_id, scene_number)
            )
            """
        )
        conn.commit()
        conn.close()

    def save_entry(
        self,
        project: str,
        chapter_number: int,
        scene_number: int,
        day: int,
        time_of_day: str,
        duration_minutes: Optional[int] = None,
        events: Optional[List[str]] = None,
        scene_id: Optional[int] = None,
        chapter_id: Optional[str] = None,
    ) -> int:
        """Insert or update a timeline entry. Returns the row id."""
        chapter_id = chapter_id or f"chapter_{chapter_number}"
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO timeline
                (project, chapter_id, chapter_number, scene_id, scene_number,
                 day, time_of_day, duration_minutes, events_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(project, chapter_id, scene_number) DO UPDATE SET
                chapter_number = excluded.chapter_number,
                scene_id = excluded.scene_id,
                day = excluded.day,
                time_of_day = excluded.time_of_day,
                duration_minutes = excluded.duration_minutes,
                events_json = excluded.events_json
            """,
            (
                project, chapter_id, chapter_number, scene_id, scene_number,
                day, time_of_day, duration_minutes,
                json.dumps(events or []),
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id FROM timeline WHERE project = ? AND chapter_id = ? AND scene_number = ?",
            (project, chapter_id, scene_number),
        ).fetchone()
        conn.close()
        return row[0] if row else -1

    def list_entries(self, project: str) -> List[Dict[str, Any]]:
        """All timeline entries in user-defined order (chapter, then scene)."""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT * FROM timeline WHERE project = ?
            ORDER BY chapter_number, scene_number
            """,
            (project,),
        ).fetchall()
        conn.close()
        result = []
        for r in rows:
            entry = dict(r)
            entry["events"] = json.loads(entry.get("events_json") or "[]")
            result.append(entry)
        return result

    def delete_entry(self, project: str, chapter_number: int, scene_number: int) -> bool:
        conn = self._connect()
        cur = conn.execute(
            "DELETE FROM timeline WHERE project = ? AND chapter_number = ? AND scene_number = ?",
            (project, chapter_number, scene_number),
        )
        conn.commit()
        success = cur.rowcount > 0
        conn.close()
        return success

    def detect_inconsistencies(self, project: str) -> List[str]:
        """Warn (never fix) about timeline inconsistencies (plan §4).

        The user remains the authority: this only returns human-readable
        warnings, e.g. a later scene happening on an earlier day.
        """
        warnings = []
        entries = self.list_entries(project)
        prev = None
        for e in entries:
            if prev is not None:
                if e["day"] < prev["day"]:
                    warnings.append(
                        f"Chapter {e['chapter_number']} scene {e['scene_number']} "
                        f"(day {e['day']}) happens before the previous entry "
                        f"(day {prev['day']}). Intentional flashback?"
                    )
                elif e["day"] == prev["day"] and e["time_of_day"] < prev["time_of_day"]:
                    warnings.append(
                        f"Chapter {e['chapter_number']} scene {e['scene_number']} "
                        f"({e['time_of_day']}) is earlier in the day than the "
                        f"previous entry ({prev['time_of_day']})."
                    )
            prev = e
        return warnings


# Singleton instance
timeline_service = TimelineService()
