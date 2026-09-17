"""Audio scene representation model and service.

Manages the intermediate structured representation of every scene used for
audio generation. This representation describes segments (narration, dialogue,
speaker, emotion, tone, intensity, delivery, voice, timing, sound effects,
music) and preserves all metadata needed for audio assembly.

The user can inspect and modify this representation before audio generation,
and can request LLM assistance to modify the representation (e.g. "Make Nikita
sound more irritated", "Rewrite this dialogue so Roger sounds afraid but tries
to hide it").
"""

import json
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any, List


class AudioSceneSegment:
    """One segment of an audio scene representation."""

    def __init__(
        self,
        segment_type: str,
        speaker: str,
        text: str,
        emotion: Optional[str] = None,
        tone: Optional[str] = None,
        intensity: Optional[float] = None,
        delivery: Optional[str] = None,
        voice: Optional[str] = None,
        timing: Optional[Dict[str, float]] = None,
        sound_effects: Optional[List[str]] = None,
        music: Optional[bool] = None,
        speed: Optional[float] = None,
        accepted: bool = False,
        accepted_hash: str = "",
    ):
        self.segment_type = segment_type
        self.speaker = speaker
        self.text = text
        self.emotion = emotion
        self.tone = tone
        self.intensity = intensity
        self.delivery = delivery
        self.voice = voice
        self.timing = timing or {}
        self.sound_effects = sound_effects or []
        self.music = music
        self.speed = speed
        self.accepted = bool(accepted)
        self.accepted_hash = accepted_hash or ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.segment_type,
            "speaker": self.speaker,
            "text": self.text,
            "emotion": self.emotion,
            "tone": self.tone,
            "intensity": self.intensity,
            "delivery": self.delivery,
            "voice": self.voice,
            "timing": self.timing,
            "sound_effects": self.sound_effects,
            "music": self.music,
            "speed": self.speed,
            "accepted": self.accepted,
            "accepted_hash": self.accepted_hash,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AudioSceneSegment":
        seg = AudioSceneSegment(
            segment_type=data.get("type", "dialogue"),
            speaker=data.get("speaker", ""),
            text=data.get("text", ""),
            emotion=data.get("emotion"),
            tone=data.get("tone"),
            intensity=data.get("intensity"),
            delivery=data.get("delivery"),
            voice=data.get("voice"),
            timing=data.get("timing", {}),
            sound_effects=data.get("sound_effects", []),
            music=data.get("music"),
            speed=data.get("speed"),
            # Backward compatible with old blobs lacking these fields.
            accepted=bool(data.get("accepted", False)),
            accepted_hash=data.get("accepted_hash", "") or "",
        )
        return seg


class AudioSceneRepresentation:
    """Complete intermediate representation of a scene for audio generation."""

    def __init__(
        self,
        scene_id: str,
        title: str,
        timeline: Dict[str, Any],
        segments: Optional[List[AudioSceneSegment]] = None,
        characters_present: Optional[List[str]] = None,
        objects_present: Optional[List[str]] = None,
        location: Optional[str] = None,
    ):
        self.scene_id = scene_id
        self.title = title
        self.timeline = timeline
        self.segments = segments or []
        self.characters_present = characters_present or []
        self.objects_present = objects_present or []
        self.location = location

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "title": self.title,
            "timeline": self.timeline,
            "segments": [s.to_dict() for s in self.segments],
            "characters_present": self.characters_present,
            "objects_present": self.objects_present,
            "location": self.location,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AudioSceneRepresentation":
        segments = []
        for sdata in data.get("segments", []):
            segments.append(AudioSceneSegment.from_dict(sdata))

        return AudioSceneRepresentation(
            scene_id=data.get("scene_id", ""),
            title=data.get("title", ""),
            timeline=data.get("timeline", {}),
            segments=segments,
            characters_present=data.get("characters_present", []),
            objects_present=data.get("objects_present", []),
            location=data.get("location"),
        )

    def add_segment(
        self,
        segment_type: str,
        speaker: str,
        text: str,
        emotion: Optional[str] = None,
        tone: Optional[str] = None,
        intensity: Optional[float] = None,
        delivery: Optional[str] = None,
        voice: Optional[str] = None,
        timing: Optional[Dict[str, float]] = None,
        sound_effects: Optional[List[str]] = None,
        music: Optional[bool] = None,
        speed: Optional[float] = None,
    ):
        """Add a segment to the representation."""
        self.segments.append(
            AudioSceneSegment(
                segment_type=segment_type,
                speaker=speaker,
                text=text,
                emotion=emotion,
                tone=tone,
                intensity=intensity,
                delivery=delivery,
                voice=voice,
                timing=timing,
                sound_effects=sound_effects,
                music=music,
                speed=speed,
            )
        )

    def get_segment_by_speaker(self, speaker: str) -> Optional[AudioSceneSegment]:
        for seg in self.segments:
            if seg.speaker == speaker:
                return seg
        return None

    def get_segment_by_type(self, segment_type: str) -> List[AudioSceneSegment]:
        return [seg for seg in self.segments if seg.segment_type == segment_type]


class AudioSceneService:
    """Service for persisting and retrieving audio scene representations."""

    def __init__(self, db_path: str = "story_engine.db"):
        self.db_path = db_path
        self._init_table()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_table(self):
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audio_scene_representations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT NOT NULL,
                scene_id TEXT NOT NULL,
                scene_number INTEGER NOT NULL,
                representation_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(project, scene_number)
            )
            """)
        # Also ensure the scenes table has a place to reference the representation
        conn.commit()
        conn.close()

    def save_representation(
        self,
        project: str,
        scene_id: str,
        scene_number: int,
        representation: AudioSceneRepresentation,
    ) -> int:
        """Save an audio scene representation. Returns the row id."""
        conn = self._connect()
        repr_json = json.dumps(representation.to_dict())
        cursor = conn.execute(
            """
            INSERT INTO audio_scene_representations
                (project, scene_id, scene_number, representation_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(project, scene_number) DO UPDATE SET
                scene_id = excluded.scene_id,
                representation_json = excluded.representation_json,
                created_at = CURRENT_TIMESTAMP
            """,
            (project, scene_id, scene_number, repr_json, datetime.now()),
        )
        conn.commit()
        row_id = cursor.lastrowid
        conn.close()
        return row_id

    def get_representation(
        self, project: str, scene_number: int
    ) -> Optional[AudioSceneRepresentation]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM audio_scene_representations WHERE project = ? AND scene_number = ?",
            (project, scene_number),
        ).fetchone()
        if row:
            data = dict(row)
            data["representation_json"] = data[
                "representation_json"
            ]  # already a string
            repr_data = json.loads(data["representation_json"])
            representation = AudioSceneRepresentation.from_dict(repr_data)
        else:
            representation = None
        conn.close()
        return representation

    def list_representations(self, project: str) -> List[Dict[str, Any]]:
        """List all representation entries for a project."""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM audio_scene_representations"
            " WHERE project = ? ORDER BY scene_number DESC",
            (project,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]


# Singleton instance
audio_scene_service = AudioSceneService()
