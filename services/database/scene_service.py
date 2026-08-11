"""
Scene metadata service for tracking generated scenes.
"""

import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Any


class SceneService:
    def __init__(self, db_path: str = "story_engine.db"):
        self.db_path = db_path

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def list_scenes(self, project: str) -> List[Dict[str, Any]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM scenes WHERE project = ? ORDER BY scene_number DESC
        """, (project,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def save_scene(self, project: str, scene_number: int, prompt: str,
                   image_path: str, seed: Optional[int] = None, model: Optional[str] = None) -> Dict[str, Any]:
        conn = self._connect()
        conn.execute("""
            INSERT OR REPLACE INTO scenes
            (project, scene_number, prompt, image_path, seed, model, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (project, scene_number, prompt, image_path, seed, model, datetime.now()))
        conn.commit()
        conn.close()
        return self.get_scene(project, scene_number)

    def get_scene(self, project: str, scene_number: int) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        row = conn.execute("""
            SELECT * FROM scenes WHERE project = ? AND scene_number = ?
        """, (project, scene_number)).fetchone()
        conn.close()
        return dict(row) if row else None


scene_service = SceneService()
