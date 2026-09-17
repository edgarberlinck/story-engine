"""Tests for scene review status (core/scene_status.py)."""

import unittest
from pathlib import Path
from unittest.mock import patch

from core import scene_status as ss
from services.audio_scene_service import AudioSceneRepresentation


def _rep(segs):
    """segs: list of (text, accepted, accepted_hash)."""
    rep = AudioSceneRepresentation(scene_id="s1", title="T", timeline={})
    for text, accepted, accepted_hash in segs:
        rep.add_segment("dialogue", "nikita", text)
        rep.segments[-1].accepted = accepted
        rep.segments[-1].accepted_hash = accepted_hash
    return rep


class _FakePath:
    def __init__(self, exists):
        self._exists = exists

    def exists(self):
        return self._exists


def _status(rep, final_exists=False, current_hash="HASH", missing=0):
    with patch.object(ss, "segment_cache_key", return_value=current_hash), patch.object(
        ss, "final_scene_path", return_value=_FakePath(final_exists)
    ), patch.object(ss, "count_missing_fragments", return_value=missing):
        return ss.scene_status("p", 1, rep)


class TestSceneStatus(unittest.TestCase):
    def test_not_generated(self):
        """A freshly written scene (no WAVs) reads as needing generation."""
        status = _status(_rep([("a", False, ""), ("b", False, "")]), missing=2)
        self.assertEqual((status.total, status.accepted, status.generated), (2, 0, 0))
        self.assertFalse(status.finalized)
        self.assertEqual(status.label, "\u26aa not generated")

    def test_generated_but_unreviewed_shows_zero_accepted(self):
        status = _status(_rep([("a", False, ""), ("b", False, "")]), missing=0)
        self.assertEqual(status.generated, 2)
        self.assertEqual(status.label, "\U0001f7e1 0/2 accepted")

    def test_partially_accepted(self):
        status = _status(_rep([("a", True, "HASH"), ("b", False, "")]))
        self.assertEqual((status.total, status.accepted), (2, 1))
        self.assertFalse(status.finalized)
        self.assertEqual(status.label, "\U0001f7e1 1/2 accepted")

    def test_finalized_when_all_accepted_and_final_exists(self):
        rep = _rep([("a", True, "HASH"), ("b", True, "HASH")])
        status = _status(rep, final_exists=True)
        self.assertTrue(status.finalized)
        self.assertEqual(status.label, "\U0001f7e2 Finalized")

    def test_all_accepted_but_no_final_file_is_not_finalized(self):
        rep = _rep([("a", True, "HASH")])
        status = _status(rep, final_exists=False)
        self.assertFalse(status.finalized)
        self.assertEqual(status.label, "\U0001f7e1 1/1 accepted")

    def test_edit_after_acceptance_invalidates_finalized(self):
        """Stale accepted_hash (segment edited) drops acceptance + finalized."""
        rep = _rep([("a", True, "OLD_HASH"), ("b", True, "HASH")])
        status = _status(rep, final_exists=True, current_hash="HASH")
        self.assertEqual(status.accepted, 1)
        self.assertFalse(status.finalized)

    def test_empty_scene_never_finalized(self):
        rep = AudioSceneRepresentation(scene_id="s1", title="T", timeline={})
        status = _status(rep, final_exists=True)
        self.assertEqual((status.total, status.accepted), (0, 0))
        self.assertFalse(status.finalized)

    def test_segments_without_text_ignored(self):
        rep = _rep([("a", True, "HASH"), ("   ", False, "")])
        status = _status(rep, final_exists=True)
        self.assertEqual(status.total, 1)
        self.assertTrue(status.finalized)

    def test_segment_accepted_helper(self):
        rep = _rep([("a", True, "HASH"), ("b", True, "OLD")])
        with patch.object(ss, "segment_cache_key", return_value="HASH"):
            self.assertTrue(ss.segment_accepted("p", rep.segments[0]))
            self.assertFalse(ss.segment_accepted("p", rep.segments[1]))

    def test_final_scene_path_is_real_path_helper(self):
        # Sanity: the real helper returns a Path under the scene folder.
        with patch("core.scene_finalizer.scene_dir", return_value=Path("/tmp/scene_x")):
            from core.scene_finalizer import final_scene_path

            self.assertEqual(
                final_scene_path("p", 1),
                Path("/tmp/scene_x/audio/scene_final.wav"),
            )


if __name__ == "__main__":
    unittest.main()
