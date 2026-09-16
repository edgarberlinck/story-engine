"""Story markup: the notation authors use to write audio-ready stories.

Two notations compile to the SAME structure (scenes of audio segments):

1. XML-like markup (full control, nestable):

    <scene>
      <sound preset="adventure" />
      <character name="narrator" tone="neutral">
        And your story begins...
      </character>
      <character name="claude fable" tone="scary" emotion="fearful">
        No, it's too difficult for me
      </character>
      <sound prompt="a confusing swirling noise">
        <character name="narrator" tone="mysterious">
          But then your hero shows up
        </character>
      </sound>
      <music prompt="soft strings" />
    </scene>

2. Bracket shorthand (fast to type, inline):

    [Narrator]
    Some text [Feeling=Angry] and my tokens will end [Feeling=calm]
    [Nikita] Don't worry about it... [Speed=1.2] I'll talk faster now.

Speech speed: ``speed`` (or ``pace``) is a multiplier (0.5-2.0, 1.0 = model
pace) usable as ``<character speed="1.15">`` or ``[Speed=1.15]``. Segments
without an explicit speed use the engine's natural default.

Both may be mixed: bracket tokens work inside <character> text too.

The parser is deliberately tolerant: unclosed tags are auto-closed at the
end of input, unknown tags/attributes are ignored rather than fatal, and
plain text outside any tag becomes narration.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from services.audio_scene_service import AudioSceneRepresentation

NARRATOR_NAMES = {"narrator", "narrador", "narador", "narator"}

# Bracket-token attribute keys -> segment fields.
_BRACKET_ATTR_KEYS = {
    "feeling": "emotion",
    "emotion": "emotion",
    "tone": "tone",
    "delivery": "delivery",
    "intensity": "intensity",
    "voice": "voice",
    "speed": "speed",
    "pace": "speed",
}

_TAG_RE = re.compile(
    r"<\s*(/?)\s*([a-zA-Z_][\w-]*)((?:\s+[\w-]+\s*=\s*(?:\"[^\"]*\"|'[^']*'))*)\s*(/?)\s*>"
)
_ATTR_RE = re.compile(r"([\w-]+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)')")
_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")


@dataclass
class MarkupSegment:
    """One parsed segment, notation-agnostic."""
    segment_type: str = "narration"     # narration | dialogue | sound_effect | music
    speaker: str = "narrator"
    text: str = ""
    emotion: Optional[str] = None
    tone: Optional[str] = None
    intensity: Optional[float] = None
    delivery: Optional[str] = None
    voice: Optional[str] = None
    speed: Optional[float] = None
    sound_effects: List[str] = field(default_factory=list)
    music: Optional[bool] = None


@dataclass
class MarkupScene:
    """One parsed scene: an ordered list of segments."""
    title: Optional[str] = None
    segments: List[MarkupSegment] = field(default_factory=list)


@dataclass
class ParseIssue:
    """A non-fatal problem found while parsing (shown to the author)."""
    message: str
    position: int = 0


class _State:
    """Current speaker/performance state while walking the text."""

    def __init__(self):
        self.speaker = "narrator"
        self.emotion: Optional[str] = None
        self.tone: Optional[str] = None
        self.intensity: Optional[float] = None
        self.delivery: Optional[str] = None
        self.voice: Optional[str] = None
        self.speed: Optional[float] = None

    def copy(self) -> "_State":
        s = _State()
        s.__dict__.update(self.__dict__)
        return s

    def make_segment(self, text: str) -> MarkupSegment:
        is_narration = self.speaker.strip().lower() in NARRATOR_NAMES
        return MarkupSegment(
            segment_type="narration" if is_narration else "dialogue",
            speaker="narrator" if is_narration else self.speaker,
            text=text,
            emotion=self.emotion,
            tone=self.tone,
            intensity=self.intensity,
            delivery=self.delivery,
            voice=self.voice,
            speed=self.speed,
        )


def _parse_attrs(raw: str) -> Dict[str, str]:
    return {
        k.lower(): (v1 if v1 is not None else v2)
        for k, (v1, v2) in ((m.group(1), (m.group(2), m.group(3)))
                            for m in _ATTR_RE.finditer(raw or ""))
    }


def _parse_intensity(value: str, issues: List[ParseIssue], pos: int) -> Optional[float]:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        issues.append(ParseIssue(f"Invalid intensity '{value}' (expected 0.0-1.0)", pos))
        return None


def _parse_speed(value: str, issues: List[ParseIssue], pos: int) -> Optional[float]:
    """Speech speed multiplier: 1.0 = model pace, clamped to 0.5-2.0."""
    try:
        return max(0.5, min(2.0, float(value)))
    except (TypeError, ValueError):
        issues.append(ParseIssue(f"Invalid speed '{value}' (expected 0.5-2.0)", pos))
        return None


def _emit_text(text: str, state: _State, segments: List[MarkupSegment],
               issues: List[ParseIssue], offset: int = 0) -> None:
    """Emit plain text, honoring bracket shorthand tokens inside it.

    Bracket tokens either switch speaker ([Nikita], [Narrator]) or change a
    performance attribute ([Feeling=Angry], [Tone=cold], [Intensity=0.8]).
    Each change starts a new segment so the change applies from that point.
    """
    cursor = 0
    for m in _BRACKET_RE.finditer(text):
        chunk = text[cursor:m.start()]
        if chunk.strip():
            segments.append(state.make_segment(chunk.strip()))
        cursor = m.end()

        token = m.group(1).strip()
        if "=" in token:
            key, _, value = token.partition("=")
            key = key.strip().lower()
            value = value.strip()
            attr = _BRACKET_ATTR_KEYS.get(key)
            if attr == "intensity":
                state.intensity = _parse_intensity(value, issues, offset + m.start())
            elif attr == "speed":
                state.speed = _parse_speed(value, issues, offset + m.start())
            elif attr:
                setattr(state, attr, value or None)
            elif key in ("sound", "sfx"):
                segments.append(MarkupSegment(
                    segment_type="sound_effect", speaker="", text=value,
                    sound_effects=[value],
                ))
            elif key == "music":
                segments.append(MarkupSegment(
                    segment_type="music", speaker="", text=value, music=True,
                ))
            else:
                issues.append(ParseIssue(
                    f"Unknown bracket attribute '[{token}]' ignored",
                    offset + m.start(),
                ))
        else:
            # Bare token: speaker switch. Reset performance state — a new
            # speaker starts with a clean delivery.
            state.speaker = token
            state.emotion = None
            state.tone = None
            state.intensity = None
            state.delivery = None
            state.voice = None
            state.speed = None

    tail = text[cursor:]
    if tail.strip():
        segments.append(state.make_segment(tail.strip()))


def parse_story_markup(text: str):
    """Parse story markup (XML-like, bracket shorthand, or mixed).

    Returns (scenes, issues):
    - scenes: List[MarkupScene] — at least one scene when any content exists.
    - issues: List[ParseIssue] — non-fatal warnings for the author.
    """
    scenes: List[MarkupScene] = []
    issues: List[ParseIssue] = []

    current_scene: Optional[MarkupScene] = None
    implicit_scene = False

    # Stack of open XML-like elements: (tag, attrs, saved_state)
    stack: List[tuple] = []
    state = _State()

    def scene() -> MarkupScene:
        nonlocal current_scene, implicit_scene
        if current_scene is None:
            current_scene = MarkupScene()
            implicit_scene = True
        return current_scene

    def close_scene():
        nonlocal current_scene, implicit_scene
        if current_scene is not None and current_scene.segments:
            scenes.append(current_scene)
        current_scene = None
        implicit_scene = False

    def in_character() -> bool:
        return any(t == "character" for t, _, _ in stack)

    cursor = 0
    for m in _TAG_RE.finditer(text):
        # Text before this tag.
        chunk = text[cursor:m.start()]
        if chunk.strip():
            _emit_text(chunk, state, scene().segments, issues, cursor)
        cursor = m.end()

        closing, tag, raw_attrs, self_closing = (
            m.group(1) == "/", m.group(2).lower(), m.group(3), m.group(4) == "/",
        )
        attrs = _parse_attrs(raw_attrs)

        if closing:
            # Pop to the matching open tag (tolerate mismatches).
            matched = None
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == tag:
                    matched = i
                    break
            if matched is None:
                if tag == "scene":
                    close_scene()
                else:
                    issues.append(ParseIssue(
                        f"Closing </{tag}> without matching opening tag", m.start()
                    ))
                continue
            while len(stack) > matched:
                popped_tag, _, saved_state = stack.pop()
                if popped_tag == "character":
                    state = saved_state
                if popped_tag == "scene" and not stack:
                    close_scene()
            continue

        if tag == "scene":
            if not stack:
                # Top-level scene boundary: flush any implicit/previous scene
                # and start a fresh one with a clean performance state.
                close_scene()
                current_scene = MarkupScene(title=attrs.get("title"))
                state = _State()
            # A nested <scene> simply layers into the parent (used for
            # immersion, e.g. a scene wrapped in a <sound>).
            if not self_closing:
                stack.append((tag, attrs, None))
            continue

        if tag == "character":
            saved = state.copy()
            state = _State()
            state.speaker = attrs.get("name", "narrator") or "narrator"
            state.emotion = attrs.get("emotion") or attrs.get("feeling")
            state.tone = attrs.get("tone")
            state.delivery = attrs.get("delivery")
            state.voice = attrs.get("voice")
            if "intensity" in attrs:
                state.intensity = _parse_intensity(attrs["intensity"], issues, m.start())
            if "speed" in attrs or "pace" in attrs:
                state.speed = _parse_speed(
                    attrs.get("speed") or attrs.get("pace"), issues, m.start()
                )
            if self_closing:
                issues.append(ParseIssue(
                    "<character /> is self-closing and has no text; ignored", m.start()
                ))
                state = saved
            else:
                stack.append((tag, attrs, saved))
            continue

        if tag in ("sound", "sfx"):
            label = attrs.get("prompt") or attrs.get("preset") or ""
            if not label:
                issues.append(ParseIssue(
                    "<sound> needs a 'preset' or 'prompt' attribute", m.start()
                ))
            scene().segments.append(MarkupSegment(
                segment_type="sound_effect", speaker="", text=label,
                sound_effects=[label] if label else [],
            ))
            if not self_closing:
                stack.append((tag, attrs, None))
            continue

        if tag == "music":
            label = attrs.get("prompt") or attrs.get("preset") or ""
            scene().segments.append(MarkupSegment(
                segment_type="music", speaker="", text=label, music=True,
            ))
            if not self_closing:
                stack.append((tag, attrs, None))
            continue

        issues.append(ParseIssue(f"Unknown tag <{tag}> ignored", m.start()))

    # Trailing text after the last tag.
    tail = text[cursor:]
    if tail.strip():
        _emit_text(tail, state, scene().segments, issues, cursor)

    # Auto-close anything left open (tolerant parsing).
    if stack:
        open_tags = ", ".join(t for t, _, _ in stack)
        issues.append(ParseIssue(f"Unclosed tag(s) auto-closed at end: {open_tags}"))
    close_scene()

    return scenes, issues


def scenes_to_representations(
    scenes: List[MarkupScene],
    scene_id_prefix: str = "scene",
    start_number: int = 1,
    timeline: Optional[Dict[str, Any]] = None,
) -> List[AudioSceneRepresentation]:
    """Convert parsed scenes into audio-scene representations (plan §5)."""
    reps = []
    for offset, sc in enumerate(scenes):
        number = start_number + offset
        rep = AudioSceneRepresentation(
            scene_id=f"{scene_id_prefix}_{number:03d}",
            title=sc.title or f"Scene {number}",
            timeline=dict(timeline or {"day": 1, "time": "08:00"}),
        )
        speakers = []
        for seg in sc.segments:
            rep.add_segment(
                segment_type=seg.segment_type,
                speaker=seg.speaker,
                text=seg.text,
                emotion=seg.emotion,
                tone=seg.tone,
                intensity=seg.intensity,
                delivery=seg.delivery,
                voice=seg.voice,
                sound_effects=seg.sound_effects or None,
                music=seg.music,
                speed=seg.speed,
            )
            if seg.segment_type == "dialogue" and seg.speaker not in speakers:
                speakers.append(seg.speaker)
        rep.characters_present = speakers
        reps.append(rep)
    return reps


def compile_story_markup(
    text: str,
    scene_id_prefix: str = "scene",
    start_number: int = 1,
):
    """One-call helper: markup text -> (representations, issues)."""
    scenes, issues = parse_story_markup(text)
    return scenes_to_representations(scenes, scene_id_prefix, start_number), issues
