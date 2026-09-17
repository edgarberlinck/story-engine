"""Scene audio review status (pure-ish, headless-testable).

Computes how far a scene's audio review has progressed:
- how many fragments (segments with text) exist,
- how many already have generated audio (a freshly written scene has none
  and must be generated via "Generate Scene" before review),
- how many are validly accepted (accepted AND the accepted hash still
  matches the segment's current cache key — an edit after acceptance makes
  it stale),
- whether the scene is finalized (scene_final.wav exists AND every fragment
  is validly accepted, so editing any fragment after finalization
  invalidates the finalized state).
"""

from dataclasses import dataclass

from core.audio_preview import segment_cache_key
from core.scene_finalizer import final_scene_path
from core.scene_generation import count_missing_fragments


@dataclass
class SceneStatus:
    total: int = 0
    accepted: int = 0
    generated: int = 0  # fragments whose audio exists on disk
    finalized: bool = False

    @property
    def started(self) -> bool:
        return self.accepted > 0

    @property
    def label(self) -> str:
        if self.finalized:
            return "\U0001f7e2 Finalized"
        if self.total > 0 and self.generated == 0 and self.accepted == 0:
            return "\u26aa not generated"
        return f"\U0001f7e1 {self.accepted}/{self.total} accepted"


def segment_accepted(project: str, segment) -> bool:
    """True when a segment's acceptance is still valid (not edited since)."""
    if not getattr(segment, "accepted", False):
        return False
    return segment.accepted_hash == segment_cache_key(project, segment)


def scene_status(project: str, scene_number: int, representation) -> SceneStatus:
    """Compute the review status of one scene."""
    segments = [s for s in representation.segments if (s.text or "").strip()]
    total = len(segments)
    accepted = sum(1 for s in segments if segment_accepted(project, s))
    missing = count_missing_fragments(project, scene_number, representation)
    generated = max(0, total - missing)
    final_exists = final_scene_path(project, scene_number).exists()
    finalized = final_exists and total > 0 and accepted == total
    return SceneStatus(
        total=total, accepted=accepted, generated=generated, finalized=finalized
    )
