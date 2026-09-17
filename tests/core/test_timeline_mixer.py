"""Tests for core/timeline_mixer.py (real ffmpeg, tiny WAVs)."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from core import timeline_mixer as tm
from core.scene_timeline_model import (
    TimelineClip,
    TimelineModel,
    TimelineTrack,
)

FFMPEG = shutil.which("ffmpeg") is not None


def _tiny_wav(path: Path, seconds=0.4, sr=24000, freq=440, amp=0.3):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(str(path), (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr)
    return path


def _rms(path):
    data, _sr = sf.read(str(path))
    return float(np.sqrt(np.mean(np.square(data))))


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class TestMixTimeline(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.workdir.name)
        self.wav_a = _tiny_wav(self.dir / "a.wav", seconds=0.4, freq=440)
        self.wav_b = _tiny_wav(self.dir / "b.wav", seconds=0.4, freq=220)

    def tearDown(self):
        self.workdir.cleanup()

    def _model(self, a_start=0.0, b_start=0.0):
        clip_a = TimelineClip(0, "a", a_start, 0.4, path=self.wav_a)
        clip_b = TimelineClip(1, "b", b_start, 0.4, path=self.wav_b)
        model = TimelineModel(
            tracks=[
                TimelineTrack(name="A", clips=[clip_a]),
                TimelineTrack(name="B", clips=[clip_b]),
            ]
        )
        model.duration = max(clip_a.end, clip_b.end)
        return model

    def test_offsets_position_clips_on_the_axis(self):
        out = self.dir / "mix.wav"
        tm.mix_timeline(self._model(b_start=1.0), None, out)
        # Clip B at 1.0s + 0.4s duration -> ~1.4s mix.
        self.assertAlmostEqual(sf.info(str(out)).duration, 1.4, delta=0.1)

    def test_mute_skips_track(self):
        out = self.dir / "mix.wav"
        states = {"B": tm.TrackState(mute=True)}
        tm.mix_timeline(self._model(b_start=3.0), states, out)
        # Without B (at 3s) the mix is only clip A (~0.4s).
        self.assertLess(sf.info(str(out)).duration, 1.0)

    def test_solo_hides_other_tracks(self):
        out = self.dir / "mix.wav"
        states = {"A": tm.TrackState(solo=True)}
        tm.mix_timeline(self._model(b_start=3.0), states, out)
        self.assertLess(sf.info(str(out)).duration, 1.0)

    def test_volume_scales_track(self):
        loud = self.dir / "loud.wav"
        quiet = self.dir / "quiet.wav"
        model = TimelineModel(
            tracks=[
                TimelineTrack(
                    name="A", clips=[TimelineClip(0, "a", 0.0, 0.4, path=self.wav_a)]
                ),
            ]
        )
        model.duration = 0.4
        tm.mix_timeline(model, {"A": tm.TrackState(volume=1.0)}, loud)
        tm.mix_timeline(model, {"A": tm.TrackState(volume=0.5)}, quiet)
        self.assertAlmostEqual(_rms(quiet), _rms(loud) * 0.5, delta=0.02)

    def test_start_at_trims_head(self):
        out = self.dir / "mix.wav"
        tm.mix_timeline(self._model(b_start=1.0), None, out, start_at=1.0)
        self.assertAlmostEqual(sf.info(str(out)).duration, 0.4, delta=0.1)

    def test_all_muted_produces_silence(self):
        out = self.dir / "mix.wav"
        states = {"A": tm.TrackState(mute=True), "B": tm.TrackState(mute=True)}
        tm.mix_timeline(self._model(), states, out)
        self.assertTrue(out.exists())
        self.assertLess(_rms(out), 1e-4)

    def test_missing_clips_are_skipped(self):
        model = self._model()
        model.tracks[1].clips[0].missing = True
        model.tracks[1].clips[0].path = self.dir / "gone.wav"
        out = self.dir / "mix.wav"
        tm.mix_timeline(model, None, out)
        self.assertTrue(out.exists())


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class TestRenderSceneWithOffsets(unittest.TestCase):
    def test_generates_missing_then_mixes(self):
        workdir = tempfile.TemporaryDirectory()
        self.addCleanup(workdir.cleanup)
        d = Path(workdir.name)
        wav = _tiny_wav(d / "seg.wav")

        from services.audio_scene_service import AudioSceneRepresentation

        rep = AudioSceneRepresentation(scene_id="s1", title="T", timeline={})
        rep.add_segment("dialogue", "nikita", "Hello there.")
        rep.segments[0].start_offset = 0.5

        generated = []

        def fake_gen(project, scene_number, idx, segment, force=False):
            generated.append(idx)
            return wav

        clip = TimelineClip(0, "Hello there.", 0.5, 0.4, path=wav)
        ghost = TimelineClip(
            0, "Hello there.", 0.5, 0.4, path=d / "missing.wav", missing=True
        )
        ghost_model = TimelineModel(
            tracks=[TimelineTrack(name="nikita", clips=[ghost])], duration=0.9
        )
        real_model = TimelineModel(
            tracks=[TimelineTrack(name="nikita", clips=[clip])], duration=0.9
        )
        models = [ghost_model, real_model]

        out = d / "mix.wav"
        with patch.object(
            tm, "compute_timeline", side_effect=lambda *a, **k: models.pop(0)
        ), patch("core.audio_preview.generate_segment_audio", fake_gen):
            result = tm.render_scene_with_offsets("p", 1, rep, out)
        self.assertTrue(result.exists())
        self.assertEqual(generated, [0])
        # Offset honored: 0.5 + 0.4 ≈ 0.9s.
        self.assertAlmostEqual(sf.info(str(result)).duration, 0.9, delta=0.1)


if __name__ == "__main__":
    unittest.main()
