"""Tests for bulk missing-fragment generation (core/scene_generation.py)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from core import scene_generation as sg
from core import scene_timeline_model as stm
from services.audio_scene_service import AudioSceneRepresentation


def _tiny_wav(path: Path, seconds=0.4, sr=24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(str(path), (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)
    return path


class TestSceneGeneration(unittest.TestCase):
    """Engines are mocked; timeline paths are redirected to a temp dir."""

    def setUp(self):
        self.workdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.workdir.cleanup)
        self.dir = Path(self.workdir.name)
        self._seg_paths = {}

        def fake_segment_path(project, scene_number, seg_idx, segment):
            return self._seg_paths.get(seg_idx, self.dir / f"missing_{seg_idx}.wav")

        def fake_cue_path(project, prompt, seconds):
            return self.dir / f"cue_{prompt.replace(' ', '_')}.wav"

        for target, repl in (
            ("segment_audio_path", fake_segment_path),
            ("cue_audio_path", fake_cue_path),
        ):
            p = patch.object(stm, target, repl)
            p.start()
            self.addCleanup(p.stop)

    def _rep(self, with_bed=False):
        rep = AudioSceneRepresentation(scene_id="s1", title="T", timeline={})
        rep.add_segment("narration", "narrator", "Once upon a time.")
        rep.add_segment("dialogue", "nikita", "Hello there.")
        if with_bed:
            rep.add_segment("music", "", "sad piano")
        return rep

    def _with_audio(self, indices):
        for i in indices:
            self._seg_paths[i] = _tiny_wav(self.dir / f"seg_{i}.wav")

    def test_count_missing_fragments(self):
        rep = self._rep(with_bed=True)
        self.assertEqual(sg.count_missing_fragments("p", 1, rep), 3)
        self._with_audio([0, 1])
        self.assertEqual(sg.count_missing_fragments("p", 1, rep), 1)

    def test_generate_missing_speech_and_beds(self):
        rep = self._rep(with_bed=True)
        speech_calls = []
        cue_calls = []

        def fake_gen(project, scene_number, idx, segment, force=False):
            speech_calls.append(idx)
            return self.dir / f"seg_{idx}.wav"

        def fake_cue(project, prompt, seconds, force=False):
            cue_calls.append((prompt, seconds))
            return self.dir / "cue.wav"

        with patch("core.audio_preview.generate_segment_audio", fake_gen), patch(
            "generators.sound_engine.generate_cue", fake_cue
        ):
            count = sg.generate_missing_fragments("p", 1, rep)
        self.assertEqual(count, 3)
        self.assertEqual(sorted(speech_calls), [0, 1])
        self.assertEqual(len(cue_calls), 1)
        self.assertEqual(cue_calls[0][0], "sad piano")
        self.assertGreater(cue_calls[0][1], 0)  # bed sized to the timeline

    def test_generate_skips_existing_audio(self):
        rep = self._rep()
        self._with_audio([0])
        calls = []

        def fake_gen(project, scene_number, idx, segment, force=False):
            calls.append(idx)
            return self.dir / f"seg_{idx}.wav"

        with patch("core.audio_preview.generate_segment_audio", fake_gen):
            count = sg.generate_missing_fragments("p", 1, rep)
        self.assertEqual(count, 1)
        self.assertEqual(calls, [1])

    def test_nothing_missing_is_a_noop(self):
        rep = self._rep()
        self._with_audio([0, 1])
        progress = []
        with patch("core.audio_preview.generate_segment_audio") as gen:
            count = sg.generate_missing_fragments(
                "p", 1, rep, progress_cb=lambda *a: progress.append(a)
            )
        self.assertEqual(count, 0)
        gen.assert_not_called()
        self.assertEqual(progress, [])

    def test_progress_callback_sequence(self):
        rep = self._rep()
        progress = []

        with patch(
            "core.audio_preview.generate_segment_audio",
            lambda *a, **k: self.dir / "x.wav",
        ):
            sg.generate_missing_fragments(
                "p", 1, rep, progress_cb=lambda d, t, label: progress.append((d, t))
            )
        self.assertEqual(progress, [(0, 2), (1, 2), (2, 2)])

    def test_generation_error_propagates(self):
        rep = self._rep()

        def boom(*a, **k):
            raise RuntimeError("engine down")

        with patch("core.audio_preview.generate_segment_audio", boom):
            with self.assertRaises(RuntimeError):
                sg.generate_missing_fragments("p", 1, rep)


if __name__ == "__main__":
    unittest.main()
