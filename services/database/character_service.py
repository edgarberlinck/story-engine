"""Character service: persists character references (prompt, seed, model,
reference image path) so scenes can reference characters by name only.
"""

import sqlite3
import json
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
                voice_path TEXT,
                voice_prompt TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(project, name)
            )
            """
        )
        # Add voice_path / voice_prompt to pre-existing databases (columns
        # were added after the first release of the characters table).
        cols = {row[1] for row in conn.execute("PRAGMA table_info(characters)")}
        if "voice_path" not in cols:
            conn.execute("ALTER TABLE characters ADD COLUMN voice_path TEXT")
        if "voice_prompt" not in cols:
            conn.execute("ALTER TABLE characters ADD COLUMN voice_prompt TEXT")
        # The `character_attributes` table is required by save_character/
        # get_character (stored attributes for style-aware scene generation).
        # Created here so the CharacterService is self-contained (matches the
        # schema defined in services/database/migrations.py).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS character_attributes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT NOT NULL,
                character_name TEXT NOT NULL,
                attributes_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(project, character_name)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_char_attrs_proj_name "
            "ON character_attributes(project, character_name)"
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
        attributes: Optional[Dict[str, Any]] = None,
        voice_path: Optional[str] = None,
        voice_prompt: Optional[str] = None,
    ) -> int:
        """Insert or update a character reference. Returns the row id.
        
        If attributes is provided, it will be stored in character_attributes table
        for use in scene generation with style-aware prompt separation.
        """
        conn = self._connect()
        cursor = conn.execute(
            """
            INSERT INTO characters
                (project, name, prompt, seed, model, reference_image, voice_path, voice_prompt, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(project, name) DO UPDATE SET
                prompt = excluded.prompt,
                seed = excluded.seed,
                model = excluded.model,
                reference_image = excluded.reference_image,
                voice_path = COALESCE(excluded.voice_path, characters.voice_path),
                voice_prompt = COALESCE(excluded.voice_prompt, characters.voice_prompt)
            """,
            (project, name, prompt, seed, model, reference_image, voice_path, voice_prompt, datetime.now()),
        )
        
        # Store attributes if provided
        if attributes is not None:
            self._save_character_attributes(conn, project, name, attributes)
        
        conn.commit()
        row_id = cursor.lastrowid
        conn.close()
        return row_id
    
    def _save_character_attributes(self, conn: sqlite3.Connection, project: str, 
                                    character_name: str, attributes: Dict[str, Any]) -> None:
        """Save character attributes to character_attributes table."""
        json_str = json.dumps(attributes)
        conn.execute("""
            INSERT OR REPLACE INTO character_attributes
            (project, character_name, attributes_json, updated_at)
            VALUES (?, ?, ?, ?)
        """, (project, character_name, json_str, datetime.now()))

    def set_voice_path(self, name: str, project: str, voice_path: str,
                       voice_prompt: Optional[str] = None) -> bool:
        """Persist the character's generated voice WAV path (and the prompt
        that produced it, if any)."""
        conn = self._connect()
        if voice_prompt is not None:
            cur = conn.execute(
                "UPDATE characters SET voice_path = ?, voice_prompt = ? "
                "WHERE project = ? AND name = ?",
                (voice_path, voice_prompt, project, name),
            )
        else:
            cur = conn.execute(
                "UPDATE characters SET voice_path = ? WHERE project = ? AND name = ?",
                (voice_path, project, name),
            )
        conn.commit()
        success = cur.rowcount > 0
        conn.close()
        return success

    def get_character(
        self, name: str, project: str = "test_project"
    ) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM characters WHERE project = ? AND name = ?",
            (project, name),
        ).fetchone()
        
        if row:
            character = dict(row)
            # Load attributes if available
            attr_row = conn.execute("""
                SELECT attributes_json FROM character_attributes
                WHERE project = ? AND character_name = ?
            """, (project, name)).fetchone()
            
            if attr_row and attr_row["attributes_json"]:
                try:
                    character["attributes"] = json.loads(attr_row["attributes_json"])
                except json.JSONDecodeError:
                    pass
        
        conn.close()
        return character if row else None

    def list_characters(self, project: str = "test_project") -> List[Dict[str, Any]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM characters WHERE project = ? ORDER BY name", (project,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def delete_character(self, name: str, project: str = "test_project") -> bool:
        conn = self._connect()
        cur = conn.execute(
            "DELETE FROM characters WHERE project = ? AND name = ?",
            (project, name),
        )
        conn.commit()
        success = cur.rowcount > 0
        conn.close()
        return success

    def find_characters_in_text(
        self, text: str, project: str = "test_project"
    ) -> List[Dict[str, Any]]:
        """Return all known characters whose name appears in the given text."""
        import re
        found = []
        text_lower = text.lower()
        
        for c in self.list_characters(project):
            name_lower = c["name"].lower()
            # Use word boundaries to avoid false positives
            # e.g., "Al" shouldn't match "also", "Roger" shouldn't match "Rogers"
            pattern = r'\b' + re.escape(name_lower) + r'\b'
            if re.search(pattern, text_lower):
                found.append(c)
        
        return found


# Singleton instance
character_service = CharacterService()
