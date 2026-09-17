"""
Database migrations for Story Engine — Audiobook & AI Story Engine.
Handles project deletion, character/object/location/versioning and scene metadata.
"""

import sqlite3
from datetime import datetime


def migrate_database(db_path: str = "story_engine.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Enable foreign keys
    cur.execute("PRAGMA foreign_keys = ON")

    # Projects table already exists; add soft-delete column for safety (optional)
    cur.execute("""
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
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS scenes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL,
            scene_number INTEGER NOT NULL,
            prompt TEXT NOT NULL,
            image_path TEXT NOT NULL,
            seed INTEGER,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(project, scene_number)
        )
    """)

    # Character attributes table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS character_attributes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL,
            character_name TEXT NOT NULL,
            attributes_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(project, character_name)
        )
    """)

    # NEW: Objects/Artifacts table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS objects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL,
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

    # NEW: Locations table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL,
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
    """)

    # Timeline table removed: narrative order lives in the manuscript
    # (chapter positions + compiled scene numbers) and each audio scene
    # representation carries its own {day, time}. Old DBs may still contain
    # a legacy 'timeline' table; it is dropped here.
    cur.execute("DROP TABLE IF EXISTS timeline")

    # NEW: Audio scene representations table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audio_scene_representations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL,
            scene_id INTEGER NOT NULL,
            scene_number INTEGER NOT NULL,
            representation_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(project, scene_number)
        )
    """)

    # NEW: Project settings table (narrator configuration, etc.)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS project_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL,
            key TEXT NOT NULL,
            value_json TEXT NOT NULL,
            UNIQUE(project, key)
        )
    """)

    # NEW: Manuscript tables (writing layer ported from writing-tools)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chapters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL,
            title TEXT NOT NULL,
            chapter_type TEXT NOT NULL DEFAULT 'regular',
            position INTEGER NOT NULL,
            compiled_scenes_json TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chapter_contents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chapter_id INTEGER NOT NULL,
            locale TEXT NOT NULL,
            content TEXT NOT NULL DEFAULT '',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(chapter_id, locale)
        )
    """)

    # Create index for faster lookups
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_char_versions_proj_name ON character_versions(project, character_name)"
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_scenes_proj ON scenes(project)")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_char_attrs_proj_name ON character_attributes(project, character_name)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_objects_proj_name ON objects(project, name)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_chapters_proj ON chapters(project, position)"
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_locations_proj ON locations(project)")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_audio_scene_proj ON audio_scene_representations(project)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_audio_scene_scene ON audio_scene_representations(scene_id)"
    )

    conn.commit()
    conn.close()


def seed_character_versions_from_existing(db_path: str = "story_engine.db"):
    """Migrate existing characters table rows into character_versions with version 1."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Check if migration already done
    cur.execute("SELECT COUNT(*) as c FROM character_versions")
    count = cur.fetchone()["c"]
    if count > 0:
        conn.close()
        return

    cur.execute(
        "SELECT project, name, prompt, seed, model, reference_image, created_at FROM characters"
    )
    rows = cur.fetchall()

    for r in rows:
        cur.execute(
            """
            INSERT OR IGNORE INTO character_versions
            (project, character_name, version, prompt, seed, model, image_path, created_at, is_default)
            VALUES (?, ?, 1, ?, ?, ?, ?, ?)
        """,
            (
                r["project"],
                r["name"],
                r["prompt"],
                r["seed"],
                r["model"],
                r["reference_image"],
                r["created_at"],
            ),
        )

    # Mark as default
    cur.execute("""
        UPDATE character_versions SET is_default = 1 WHERE version = 1
    """)
    conn.commit()
    conn.close()
