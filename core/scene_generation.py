"""Bulk generation of a scene's missing fragment audio.

Powers the explicit "Generate Scene" affordance in the scene editor: after
writing a scene, the author clicks Generate Scene once and every ungenerated
fragment (ghost clip) is rendered — speech through the per-segment preview
pipeline, sound/music beds at their timeline-computed durations. The same
per-clip generation is reused by the offset renderer (core.timeline_mixer),
so both paths produce identical audio.

Headless and unit-testable: engines are imported lazily so tests can patch
``core.audio_preview.generate_segment_audio`` and
``generators.sound_engine.generate_cue``.
"""

import logging
from typing import Callable, List, Optional

from core.scene_timeline_model import TimelineClip, compute_timeline

logger = logging.getLogger(__name__)

# progress_cb(done, total, label): called before/after each clip generates.
ProgressCallback = Callable[[int, int, str], None]


def missing_clips(
    project: str, scene_number: int, representation
) -> List[TimelineClip]:
    """Timeline clips whose audio has not been generated yet (ghost clips)."""
    model = compute_timeline(project, scene_number, representation)
    return [c for c in model.all_clips() if c.missing]


def count_missing_fragments(project: str, scene_number: int, representation) -> int:
    """How many fragments still need audio generated."""
    return len(missing_clips(project, scene_number, representation))


def generate_clip_audio(
    project: str, scene_number: int, representation, clip: TimelineClip
) -> None:
    """Generate the audio for one timeline clip (speech line or bed cue)."""
    from core.audio_preview import generate_segment_audio
    from generators.sound_engine import generate_cue

    segment = representation.segments[clip.segment_index]
    if segment.segment_type in ("sound_effect", "music"):
        generate_cue(project, segment.text, clip.duration)
    else:
        generate_segment_audio(project, scene_number, clip.segment_index, segment)


def generate_missing_fragments(
    project: str,
    scene_number: int,
    representation,
    progress_cb: Optional[ProgressCallback] = None,
) -> int:
    """Generate every ungenerated fragment of a scene. Returns the count.

    ``progress_cb(done, total, label)`` is invoked before each fragment
    starts (done = fragments already finished) and once more at the end with
    ``done == total``, so a UI can show "Generating k/n…".
    """
    clips = missing_clips(project, scene_number, representation)
    total = len(clips)
    for done, clip in enumerate(clips):
        if progress_cb:
            progress_cb(done, total, clip.label)
        logger.info("Generating scene %s fragment: %s", scene_number, clip.label)
        generate_clip_audio(project, scene_number, representation, clip)
    if progress_cb and total:
        progress_cb(total, total, "")
    return total
