"""Manuscript service: the writing layer ported from the writing-tools app.

Data model mirrors writing-tools (React/Tauri) in SQLite:
- chapters: typed (regular/introduction/foreword/epigraph/appendix/
  about_author), ordered per project.
- chapter_contents: independent content per language (EN, PT-BR, ES, FR, DE),
  written in story markup (see core/story_markup.py) so any chapter can be
  compiled into audio scenes.
"""

import sqlite3
import json
from typing import Optional, Dict, Any, List

CHAPTER_TYPES = [
    "regular",
    "introduction",
    "foreword",
    "epigraph",
    "appendix",
    "about_author",
]

SUPPORTED_LOCALES = [
    ("en", "EN"),
    ("pt-BR", "PT-BR"),
    ("es", "ES"),
    ("fr", "FR"),
    ("de", "DE"),
]


class ManuscriptService:
    def __init__(self, db_path: str = "story_engine.db"):
        self.db_path = db_path
        self._init_tables()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_tables(self):
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT NOT NULL,
                title TEXT NOT NULL,
                chapter_type TEXT NOT NULL DEFAULT 'regular',
                position INTEGER NOT NULL,
                compiled_scenes_json TEXT NOT NULL DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chapter_contents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_id INTEGER NOT NULL,
                locale TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(chapter_id, locale)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chapters_proj ON chapters(project, position)"
        )
        conn.commit()
        conn.close()

    # -- Chapters ------------------------------------------------------------

    def create_chapter(
        self, project: str, title: str, chapter_type: str = "regular"
    ) -> int:
        if chapter_type not in CHAPTER_TYPES:
            raise ValueError(f"Unknown chapter type: {chapter_type}")
        conn = self._connect()
        row = conn.execute(
            "SELECT COALESCE(MAX(position), 0) + 1 FROM chapters WHERE project = ?",
            (project,),
        ).fetchone()
        position = row[0]
        cur = conn.execute(
            "INSERT INTO chapters (project, title, chapter_type, position) VALUES (?, ?, ?, ?)",
            (project, title, chapter_type, position),
        )
        conn.commit()
        chapter_id = cur.lastrowid
        conn.close()
        return chapter_id

    def list_chapters(self, project: str) -> List[Dict[str, Any]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM chapters WHERE project = ? ORDER BY position",
            (project,),
        ).fetchall()
        conn.close()
        result = []
        for r in rows:
            ch = dict(r)
            ch["compiled_scenes"] = json.loads(ch.get("compiled_scenes_json") or "{}")
            result.append(ch)
        return result

    def get_chapter(self, chapter_id: int) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM chapters WHERE id = ?", (chapter_id,)
        ).fetchone()
        conn.close()
        if not row:
            return None
        ch = dict(row)
        ch["compiled_scenes"] = json.loads(ch.get("compiled_scenes_json") or "{}")
        return ch

    def rename_chapter(self, chapter_id: int, title: str) -> bool:
        conn = self._connect()
        cur = conn.execute(
            "UPDATE chapters SET title = ? WHERE id = ?", (title, chapter_id)
        )
        conn.commit()
        ok = cur.rowcount > 0
        conn.close()
        return ok

    def set_chapter_type(self, chapter_id: int, chapter_type: str) -> bool:
        if chapter_type not in CHAPTER_TYPES:
            raise ValueError(f"Unknown chapter type: {chapter_type}")
        conn = self._connect()
        cur = conn.execute(
            "UPDATE chapters SET chapter_type = ? WHERE id = ?",
            (chapter_type, chapter_id),
        )
        conn.commit()
        ok = cur.rowcount > 0
        conn.close()
        return ok

    def delete_chapter(self, chapter_id: int) -> bool:
        conn = self._connect()
        conn.execute(
            "DELETE FROM chapter_contents WHERE chapter_id = ?", (chapter_id,)
        )
        cur = conn.execute("DELETE FROM chapters WHERE id = ?", (chapter_id,))
        conn.commit()
        ok = cur.rowcount > 0
        conn.close()
        return ok

    def move_chapter(self, project: str, chapter_id: int, direction: int) -> bool:
        """Swap a chapter with its neighbor (direction: -1 up, +1 down)."""
        if direction not in (-1, 1):
            return False
        chapters = self.list_chapters(project)
        idx = next((i for i, c in enumerate(chapters) if c["id"] == chapter_id), None)
        if idx is None:
            return False
        other = idx + direction
        if other < 0 or other >= len(chapters):
            return False
        a, b = chapters[idx], chapters[other]
        conn = self._connect()
        conn.execute("UPDATE chapters SET position = ? WHERE id = ?", (b["position"], a["id"]))
        conn.execute("UPDATE chapters SET position = ? WHERE id = ?", (a["position"], b["id"]))
        conn.commit()
        conn.close()
        return True

    def set_compiled_scenes(self, chapter_id: int, locale: str, scene_numbers: List[int]) -> None:
        """Remember which audio scenes a chapter+locale compiled into."""
        ch = self.get_chapter(chapter_id)
        if not ch:
            return
        compiled = ch["compiled_scenes"]
        compiled[locale] = scene_numbers
        conn = self._connect()
        conn.execute(
            "UPDATE chapters SET compiled_scenes_json = ? WHERE id = ?",
            (json.dumps(compiled), chapter_id),
        )
        conn.commit()
        conn.close()

    # -- Contents (per language) ----------------------------------------------

    def save_content(self, chapter_id: int, locale: str, content: str) -> None:
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO chapter_contents (chapter_id, locale, content, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(chapter_id, locale) DO UPDATE SET
                content = excluded.content,
                updated_at = CURRENT_TIMESTAMP
            """,
            (chapter_id, locale, content),
        )
        conn.commit()
        conn.close()

    def get_content(self, chapter_id: int, locale: str) -> str:
        conn = self._connect()
        row = conn.execute(
            "SELECT content FROM chapter_contents WHERE chapter_id = ? AND locale = ?",
            (chapter_id, locale),
        ).fetchone()
        conn.close()
        return row[0] if row else ""

    def get_contents(self, chapter_id: int) -> Dict[str, str]:
        """All language contents for a chapter: {locale: content}."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT locale, content FROM chapter_contents WHERE chapter_id = ?",
            (chapter_id,),
        ).fetchall()
        conn.close()
        return {locale: content for locale, content in rows}

    def delete_project_chapters(self, project: str) -> None:
        """Remove all chapters+contents for a project (cascade helper)."""
        conn = self._connect()
        ids = [r[0] for r in conn.execute(
            "SELECT id FROM chapters WHERE project = ?", (project,)
        ).fetchall()]
        if ids:
            placeholders = ",".join("?" for _ in ids)
            conn.execute(
                f"DELETE FROM chapter_contents WHERE chapter_id IN ({placeholders})", ids
            )
        conn.execute("DELETE FROM chapters WHERE project = ?", (project,))
        conn.commit()
        conn.close()


# Singleton instance
manuscript_service = ManuscriptService()
