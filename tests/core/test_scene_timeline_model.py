"""Tests for the pure scene timeline layout model."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from core import scene_timeline_model as stm
from core.audiobook_exporter import SEGMENT_GAP_S, SPEAKER_CHANGE_GAP_S
from services.audio_scene_service import AudioSceneRepresentation


def _tiny_wav(path: Path, seconds=0.4, sr=24000, freq=440):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(str(path), (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr)
    return path


class TestComputeTimeline(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.workdir.name)
        self._seg_paths = {}

        def fake_segment_path(project, scene_number, seg_idx, segment):
            return self._seg_paths.get(seg_idx, self.dir / f"missing_{seg_idx}.wav")

        def fake_cue_path(project, prompt, seconds):
            return self.dir / f"cue_{prompt.replace(' ', '_')}.wav"

        self.patches = [
            patch.object(stm, "segment_audio_path", fake_segment_path),
            patch.object(stm, "cue_audio_path", fake_cue_path),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.workdir.cleanup()

    def _rep(self):
        rep = AudioSceneRepresentation(scene_id="s1", title="T", timeline={})
        rep.add_segment("narration", "narrator", "Once upon a time.")
        rep.add_segment("dialogue", "nikita", "Hello there.")
        rep.add_segment("dialogue", "nikita", "Still me.")
        return rep

    def _with_audio(self, indices, seconds=0.4):
        for i in indices:
            self._seg_paths[i] = _tiny_wav(self.dir / f"seg_{i}.wav", seconds)

    def test_layout_matches_exporter_gap_logic(self):
        self._with_audio([0, 1, 2], seconds=0.4)
        model = stm.compute_timeline("p", 1, self._rep())
        clips = sorted(model.all_clips(), key=lambda c: c.segment_index)
        self.assertAlmostEqual(clips[0].start, 0.0, places=3)
        # Speaker change: longer gap.
        self.assertAlmostEqual(clips[1].start, 0.4 + SPEAKER_CHANGE_GAP_S, places=2)
        # Same speaker: short gap.
        self.assertAlmostEqual(
            clips[2].start, clips[1].start + 0.4 + SEGMENT_GAP_S, places=2
        )
        self.assertAlmostEqual(model.duration, clips[2].end, places=3)

    def test_tracks_per_voice_plus_bed_lanes(self):
        rep = self._rep()
        rep.add_segment("sound_effect", "", "wind howling")
        rep.add_segment("music", "", "sad piano")
        model = stm.compute_timeline("p", 1, rep)
        names = [t.name for t in model.tracks]
        self.assertIn("Narrator", names)
        self.assertIn("nikita", names)
        self.assertIn("Sound FX", names)
        self.assertIn("Music", names)
        kinds = {t.name: t.kind for t in model.tracks}
        self.assertEqual(kinds["Sound FX"], stm.TRACK_SFX)
        self.assertEqual(kinds["Music"], stm.TRACK_MUSIC)

    def test_bed_offset_is_speech_cursor_position(self):
        self._with_audio([0], seconds=0.4)
        rep = AudioSceneRepresentation(scene_id="s1", title="T", timeline={})
        rep.add_segment("narration", "narrator", "Once upon a time.")
        rep.add_segment("sound_effect", "", "wind howling")
        rep.add_segment("dialogue", "nikita", "Hello there.")
        model = stm.compute_timeline("p", 1, rep)
        bed = [c for c in model.all_clips() if c.kind == stm.TRACK_SFX][0]
        # Bed starts where the speech cursor was: after the first clip.
        self.assertAlmostEqual(bed.start, 0.4, places=2)

    def test_missing_audio_clips_flagged_with_estimated_duration(self):
        model = stm.compute_timeline("p", 1, self._rep())
        clips = model.all_clips()
        self.assertTrue(all(c.missing for c in clips))
        self.assertTrue(all(c.duration > 0 for c in clips))

    def test_manual_start_offset_overrides_auto_layout(self):
        self._with_audio([0, 1, 2], seconds=0.4)
        rep = self._rep()
        rep.segments[1].start_offset = 7.5
        model = stm.compute_timeline("p", 1, rep)
        clips = sorted(model.all_clips(), key=lambda c: c.segment_index)
        self.assertAlmostEqual(clips[1].start, 7.5, places=3)
        # Auto cursor unaffected: clip 2 keeps its auto position.
        self.assertAlmostEqual(
            clips[2].start,
            0.4 + SPEAKER_CHANGE_GAP_S + 0.4 + SEGMENT_GAP_S,
            places=2,
        )

    def test_accepted_segments_flag_their_clips(self):
        self._with_audio([0, 1, 2], seconds=0.4)
        rep = self._rep()
        rep.segments[1].accepted = True
        with patch(
            "core.scene_status.segment_accepted",
            side_effect=lambda project, seg: seg.accepted,
        ):
            model = stm.compute_timeline("p", 1, rep)
        clips = sorted(model.all_clips(), key=lambda c: c.segment_index)
        self.assertFalse(clips[0].accepted)
        self.assertTrue(clips[1].accepted)

    def test_estimate_speech_seconds(self):
        self.assertGreater(
            stm.estimate_speech_seconds("one two three four five six"),
            stm.estimate_speech_seconds("hi"),
        )
        self.assertGreaterEqual(stm.estimate_speech_seconds(""), 0.5)


if __name__ == "__main__":
    unittest.main()
