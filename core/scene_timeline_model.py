"""Pure timeline layout model for a scene's audio (multitrack view).

Computes where every fragment of a scene sits on a shared time axis using the
SAME layout math as ``audiobook_exporter._render_scene`` (segment gaps,
speaker-change gaps, bed offsets and bed sizing), so the timeline view is a
faithful preview of the final mix.

The result is a plain dataclass tree (tracks -> clips) with no Qt types, so
the UI is a thin renderer and the layout is unit-testable headless.

A segment's ``start_offset`` (manual drag on the timeline) overrides its
auto-layout position. The auto cursor keeps advancing from the *auto*
positions, so moving one clip never re-flows the others.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from core.audio_preview import segment_audio_path, _effective_speed
from core.audiobook_exporter import (
    SEGMENT_GAP_S,
    SPEAKER_CHANGE_GAP_S,
    MUSIC_MIN_S,
    _cue_seconds,
)
from generators.sound_engine import cue_audio_path
from services.audio_scene_service import AudioSceneRepresentation

# Track kinds
TRACK_VOICE = "voice"
TRACK_SFX = "sfx"
TRACK_MUSIC = "music"

# Rough speech-rate estimate for ghost (not yet generated) clips.
_WORDS_PER_SECOND = 2.6
_MIN_CLIP_SECONDS = 0.6


@dataclass
class TimelineClip:
    segment_index: int
    label: str
    start: float  # seconds on the shared time axis
    duration: float  # seconds
    path: Optional[Path] = None  # WAV path (may not exist yet)
    missing: bool = False  # True when no generated audio exists
    kind: str = TRACK_VOICE
    accepted: bool = False  # validly accepted (reviewed) fragment

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class TimelineTrack:
    name: str
    kind: str = TRACK_VOICE
    clips: List[TimelineClip] = field(default_factory=list)


@dataclass
class TimelineModel:
    tracks: List[TimelineTrack] = field(default_factory=list)
    duration: float = 0.0

    def all_clips(self) -> List[TimelineClip]:
        return [c for t in self.tracks for c in t.clips]


def estimate_speech_seconds(text: str, speed: float = 1.0) -> float:
    """Duration estimate for a line whose audio was not generated yet."""
    words = len((text or "").split())
    seconds = words / _WORDS_PER_SECOND / max(speed, 0.1)
    return max(_MIN_CLIP_SECONDS, seconds)


def _wav_duration(path: Path) -> Optional[float]:
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:  # noqa: BLE001
        return None


def _voice_key(segment) -> str:
    return (
        segment.voice or segment.speaker or "narrator"
    ).strip().lower() or "narrator"


def _segment_accepted(project: str, segment) -> bool:
    """Valid acceptance flag for clip rendering (lazy import: no cycles)."""
    try:
        from core.scene_status import segment_accepted

        return segment_accepted(project, segment)
    except Exception:  # noqa: BLE001
        return False


def _track_name(segment) -> str:
    name = (segment.voice or segment.speaker or "").strip()
    if not name or name.lower() == "narrator":
        return "Narrator"
    return name


def compute_timeline(
    project: str,
    scene_number: int,
    representation: AudioSceneRepresentation,
) -> TimelineModel:
    """Lay the scene's segments out on tracks along a shared time axis."""
    voice_tracks: dict = {}  # voice_key -> TimelineTrack (insertion ordered)
    sfx_track = TimelineTrack(name="Sound FX", kind=TRACK_SFX)
    music_track = TimelineTrack(name="Music", kind=TRACK_MUSIC)

    # Pass 1: speech layout (mirrors audiobook_exporter._render_scene).
    cursor = 0.0
    prev_voice: Optional[str] = None
    first_speech = True
    pending_beds = []  # (segment_index, segment, auto_offset)

    for seg_idx, segment in enumerate(representation.segments):
        text = (segment.text or "").strip()
        if segment.segment_type in ("sound_effect", "music"):
            if text:
                pending_beds.append((seg_idx, segment, cursor))
            continue
        if not text:
            continue

        vkey = _voice_key(segment)
        if not first_speech:
            gap = SEGMENT_GAP_S if vkey == prev_voice else SPEAKER_CHANGE_GAP_S
            cursor += gap
        first_speech = False
        prev_voice = vkey

        path = segment_audio_path(project, scene_number, seg_idx, segment)
        exists = path.exists()
        duration = (_wav_duration(path) if exists else None) or (
            estimate_speech_seconds(text, _effective_speed(segment))
        )
        auto_start = cursor
        start = (
            float(segment.start_offset)
            if getattr(segment, "start_offset", None) is not None
            else auto_start
        )
        track = voice_tracks.get(vkey)
        if track is None:
            track = TimelineTrack(name=_track_name(segment), kind=TRACK_VOICE)
            voice_tracks[vkey] = track
        track.clips.append(
            TimelineClip(
                segment_index=seg_idx,
                label=text[:40],
                start=start,
                duration=duration,
                path=path,
                missing=not exists,
                kind=TRACK_VOICE,
                accepted=_segment_accepted(project, segment),
            )
        )
        cursor += duration

    scene_len = cursor

    # Pass 2: beds sized against the scene (same rules as the exporter).
    for seg_idx, segment, auto_offset in pending_beds:
        remaining = max(MUSIC_MIN_S, scene_len - auto_offset)
        seconds = _cue_seconds(segment, remaining_hint=remaining)
        path = cue_audio_path(project, segment.text, seconds)
        exists = path.exists()
        duration = (_wav_duration(path) if exists else None) or seconds
        start = (
            float(segment.start_offset)
            if getattr(segment, "start_offset", None) is not None
            else auto_offset
        )
        is_music = segment.segment_type == "music"
        track = music_track if is_music else sfx_track
        track.clips.append(
            TimelineClip(
                segment_index=seg_idx,
                label=(segment.text or "")[:40],
                start=start,
                duration=duration,
                path=path,
                missing=not exists,
                kind=TRACK_MUSIC if is_music else TRACK_SFX,
                accepted=_segment_accepted(project, segment),
            )
        )

    tracks = list(voice_tracks.values())
    if sfx_track.clips:
        tracks.append(sfx_track)
    if music_track.clips:
        tracks.append(music_track)

    model = TimelineModel(tracks=tracks)
    model.duration = max((c.end for c in model.all_clips()), default=0.0)
    return model
