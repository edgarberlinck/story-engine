"""Tests for core/scene_finalizer.py.

Voice/cue generation is mocked with tiny WAVs (like the exporter tests);
ffmpeg runs for real so the mix pipeline is exercised end to end.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from core import audiobook_exporter as ax
from core import scene_finalizer
from services.audio_scene_service import AudioSceneRepresentation

FFMPEG = shutil.which("ffmpeg") is not None


def _tiny_wav(path: Path, seconds=0.4, sr=24000, freq=440):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(str(path), (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr)
    return path


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class TestSceneFinalizer(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.TemporaryDirectory()
        self.scene_dir = Path(self.workdir.name) / "scene_1"
        self.scene_patch = patch.object(
            scene_finalizer, "scene_dir", return_value=self.scene_dir
        )
        self.scene_patch.start()

    def tearDown(self):
        self.scene_patch.stop()
        self.workdir.cleanup()

    def _rep(self, with_cue=False):
        rep = AudioSceneRepresentation(scene_id="s1", title="T", timeline={})
        if with_cue:
            rep.add_segment("music", "", "sad piano")
        rep.add_segment("narration", "narrator", "Once upon a time.")
        rep.add_segment("dialogue", "nikita", "Hello there.")
        return rep

    def _fake_segment_audio(self, project, scene_number, seg_idx, segment, force=False):
        return _tiny_wav(Path(self.workdir.name) / f"seg_{seg_idx}.wav")

    def _fake_cue(self, project, prompt, seconds, force=False):
        return _tiny_wav(Path(self.workdir.name) / "cue.wav", seconds=1.0, freq=110)

    def test_finalize_produces_final_wav(self):
        with patch.object(
            ax, "generate_segment_audio", self._fake_segment_audio
        ), patch.object(ax, "generate_cue", self._fake_cue):
            out = scene_finalizer.finalize_scene("p", 1, self._rep(with_cue=True))
        self.assertTrue(out.exists())
        self.assertEqual(out.name, scene_finalizer.FINAL_FILENAME)
        self.assertEqual(out.parent, self.scene_dir / "audio")
        info = sf.info(str(out))
        self.assertEqual(info.samplerate, ax.SAMPLE_RATE)
        self.assertGreater(info.duration, 1.0)

    def test_finalize_loads_representation_when_not_given(self):
        rep = self._rep()
        with patch.object(
            scene_finalizer.audio_scene_service, "get_representation", return_value=rep
        ), patch.object(ax, "generate_segment_audio", self._fake_segment_audio):
            out = scene_finalizer.finalize_scene("p", 1)
        self.assertTrue(out.exists())

    def test_missing_representation_raises(self):
        with patch.object(
            scene_finalizer.audio_scene_service, "get_representation", return_value=None
        ):
            with self.assertRaises(ValueError):
                scene_finalizer.finalize_scene("p", 42)

    def test_empty_scene_raises(self):
        rep = AudioSceneRepresentation(scene_id="s1", title="T", timeline={})
        with self.assertRaises(RuntimeError):
            scene_finalizer.finalize_scene("p", 1, rep)

    def test_progress_callback_is_invoked(self):
        messages = []
        with patch.object(ax, "generate_segment_audio", self._fake_segment_audio):
            scene_finalizer.finalize_scene(
                "p", 1, self._rep(), progress=messages.append
            )
        self.assertTrue(any("Finalizing scene 1" in m for m in messages))
        self.assertTrue(any("Finalized:" in m for m in messages))


if __name__ == "__main__":
    unittest.main()
