"""
Character attributes service for storing builder attributes.
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, Optional, Any


class CharacterAttributeService:
    def __init__(self, db_path: str = "story_engine.db"):
        self.db_path = db_path

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def save_attributes(self, project: str, character_name: str, attributes: Dict[str, Any]) -> bool:
        conn = self._connect()
        json_str = json.dumps(attributes)
        conn.execute("""
            INSERT OR REPLACE INTO character_attributes
            (project, character_name, attributes_json, updated_at)
            VALUES (?, ?, ?, ?)
        """, (project, character_name, json_str, datetime.now()))
        conn.commit()
        conn.close()
        return True

    def get_attributes(self, project: str, character_name: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        row = conn.execute("""
            SELECT attributes_json FROM character_attributes
            WHERE project = ? AND character_name = ?
        """, (project, character_name)).fetchone()
        conn.close()
        if row:
            return json.loads(row["attributes_json"])
        return None

    def generate_prompt_from_attributes(self, attributes: Dict[str, Any]) -> str:
        """Convert attributes to prompt using template."""
        parts = []
        if attributes.get("gender"):
            parts.append(f"{attributes['gender']} person")
        if attributes.get("age_range"):
            parts.append(f"age {attributes['age_range']}")
        if attributes.get("body_type"):
            parts.append(f"{attributes['body_type']} build")
        if attributes.get("hair_color"):
            parts.append(f"{attributes['hair_color']} hair")
        if attributes.get("skin_tone"):
            parts.append(f"{attributes['skin_tone']} skin")
        if attributes.get("eye_color"):
            parts.append(f"{attributes['eye_color']} eyes")

        base_prompt = ", ".join(parts)
        return f"Create a photorealistic portrait of a {base_prompt} character, high detail, studio lighting"


character_attribute_service = CharacterAttributeService()
