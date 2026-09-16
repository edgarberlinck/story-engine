"""Project settings service: generic per-project key/value settings.

Used for project-level configuration such as the narrator voice
(plan §2 "Narrator configuration"): the narrator can be a dedicated
designed voice (described by a prompt) or one of the existing characters.
"""

import sqlite3
import json
from typing import Optional, Dict, Any

# Narrator configuration keys/modes
NARRATOR_KEY = "narrator"
NARRATOR_MODE_DEDICATED = "dedicated"   # designed voice from a prompt
NARRATOR_MODE_CHARACTER = "character"   # reuse an existing character's voice

DEFAULT_NARRATOR = {
    "mode": NARRATOR_MODE_DEDICATED,
    "voice_prompt": "A neutral, warm storytelling voice, clear and engaging.",
    "character": None,
}


class ProjectSettingsService:
    def __init__(self, db_path: str = "story_engine.db"):
        self.db_path = db_path
        self._init_table()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_table(self):
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS project_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT NOT NULL,
                key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                UNIQUE(project, key)
            )
            """
        )
        conn.commit()
        conn.close()

    def set_setting(self, project: str, key: str, value: Any) -> None:
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO project_settings (project, key, value_json)
            VALUES (?, ?, ?)
            ON CONFLICT(project, key) DO UPDATE SET
                value_json = excluded.value_json
            """,
            (project, key, json.dumps(value)),
        )
        conn.commit()
        conn.close()

    def get_setting(self, project: str, key: str, default: Any = None) -> Any:
        conn = self._connect()
        row = conn.execute(
            "SELECT value_json FROM project_settings WHERE project = ? AND key = ?",
            (project, key),
        ).fetchone()
        conn.close()
        if row is None:
            return default
        return json.loads(row[0])

    # -- Narrator configuration -------------------------------------------

    def get_narrator(self, project: str) -> Dict[str, Any]:
        """Return the narrator configuration for a project (with defaults)."""
        stored = self.get_setting(project, NARRATOR_KEY, {})
        config = dict(DEFAULT_NARRATOR)
        if isinstance(stored, dict):
            config.update(stored)
        return config

    def set_narrator(
        self,
        project: str,
        mode: str,
        voice_prompt: Optional[str] = None,
        character: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist the narrator configuration.

        mode: "dedicated" (designed voice from voice_prompt) or
              "character" (an existing character narrates).
        """
        if mode not in (NARRATOR_MODE_DEDICATED, NARRATOR_MODE_CHARACTER):
            raise ValueError(f"Unknown narrator mode: {mode}")
        if mode == NARRATOR_MODE_CHARACTER and not character:
            raise ValueError("Narrator mode 'character' requires a character name")
        config = {
            "mode": mode,
            "voice_prompt": voice_prompt or DEFAULT_NARRATOR["voice_prompt"],
            "character": character,
        }
        self.set_setting(project, NARRATOR_KEY, config)
        return config


# Singleton instance
project_settings_service = ProjectSettingsService()
