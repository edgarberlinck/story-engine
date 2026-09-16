"""
Tests for the audiobook exporter (core/audiobook_exporter.py).

Voice/cue generation is mocked with tiny WAVs; ffmpeg runs for real
(normalization, silence, bed mixing with ducking, concat, loudnorm) so the
whole assembly pipeline is actually exercised.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from core import audiobook_exporter as ax
from services.audio_scene_service import (
    AudioSceneService, AudioSceneRepresentation, AudioSceneSegment,
)
from services.database.manuscript_service import ManuscriptService

FFMPEG = shutil.which("ffmpeg") is not None


def _tiny_wav(path: Path, seconds=0.4, sr=24000, freq=440):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(str(path), (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr)
    return path


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class TestAudiobookExporter(unittest.TestCase):
    def setUp(self):
        self.tmpdb = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.tmpdb.name
        self.tmpdb.close()
        self.workdir = tempfile.TemporaryDirectory()
        self.manuscripts = ManuscriptService(self.db_path)
        self.audio = AudioSceneService(self.db_path)
        self.patches = [
            patch.object(ax, "manuscript_service", self.manuscripts),
            patch.object(ax, "audio_scene_service", self.audio),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.workdir.cleanup()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def _make_chapter(self, title, scene_number, with_cue=False, music=False):
        ch = self.manuscripts.create_chapter("p", title)
        rep = AudioSceneRepresentation(
            scene_id=f"s{scene_number}", title=title, timeline={},
        )
        if with_cue:
            rep.add_segment("music" if music else "sound_effect", "",
                            "wind howling", sound_effects=["wind howling"])
        rep.add_segment("narration", "narrator", "Once upon a time.")
        rep.add_segment("dialogue", "nikita", "Hello there.", emotion="warm")
        self.audio.save_representation("p", f"s{scene_number}", scene_number, rep)
        self.manuscripts.set_compiled_scenes(ch, "en", [scene_number])
        return ch

    def _fake_segment_audio(self, project, scene_number, seg_idx, segment, force=False):
        return _tiny_wav(
            Path(self.workdir.name) / f"seg_{scene_number}_{seg_idx}.wav"
        )

    def _fake_cue(self, project, prompt, seconds, force=False):
        return _tiny_wav(
            Path(self.workdir.name) / f"cue_{seconds:.0f}.wav",
            seconds=min(seconds, 2.0), freq=110,
        )

    def test_export_single_chapter(self):
        ch = self._make_chapter("C1", 1)
        out = Path(self.workdir.name) / "book.wav"
        with patch.object(ax, "generate_segment_audio", self._fake_segment_audio), \
             patch.object(ax, "generate_cue", self._fake_cue):
            result = ax.export_chapters("p", [ch], str(out))
        self.assertIsNone(result.error)
        self.assertTrue(out.exists())
        self.assertEqual(result.segments_rendered, 2)
        info = sf.info(str(out))
        self.assertEqual(info.samplerate, ax.SAMPLE_RATE)
        # two 0.4s clips + one 0.5s gap ≈ 1.3s
        self.assertGreater(info.duration, 1.0)

    def test_export_multiple_chapters_in_manuscript_order(self):
        ch1 = self._make_chapter("C1", 1)
        ch2 = self._make_chapter("C2", 2)
        out = Path(self.workdir.name) / "book.m4a"
        with patch.object(ax, "generate_segment_audio", self._fake_segment_audio), \
             patch.object(ax, "generate_cue", self._fake_cue):
            # Pass out of order; exporter must follow manuscript order.
            result = ax.export_chapters("p", [ch2, ch1], str(out))
        self.assertIsNone(result.error)
        self.assertTrue(out.exists())
        self.assertEqual(result.segments_rendered, 4)

    def test_cue_is_mixed_under_speech_not_appended(self):
        """The bed overlaps the voice: the scene is NOT lengthened by the cue."""
        ch_plain = self._make_chapter("Plain", 1, with_cue=False)
        ch_cue = self._make_chapter("Cue", 2, with_cue=True)
        out_plain = Path(self.workdir.name) / "plain.wav"
        out_cue = Path(self.workdir.name) / "cue.wav"
        with patch.object(ax, "generate_segment_audio", self._fake_segment_audio), \
             patch.object(ax, "generate_cue", self._fake_cue):
            ax.export_chapters("p", [ch_plain], str(out_plain))
            result = ax.export_chapters("p", [ch_cue], str(out_cue))
        self.assertEqual(result.cues_rendered, 1)
        d_plain = sf.info(str(out_plain)).duration
        d_cue = sf.info(str(out_cue)).duration
        # Mixed under: at most the bed tail may extend the scene slightly,
        # but the cue must not be prepended sequentially (which would add
        # its full duration + a gap).
        self.assertLess(d_cue, d_plain + 2.5)

    def test_music_bed_sizes_to_scene(self):
        ch = self._make_chapter("M", 1, with_cue=True, music=True)
        out = Path(self.workdir.name) / "music.wav"
        requested = {}

        def spy_cue(project, prompt, seconds, force=False):
            requested["seconds"] = seconds
            return self._fake_cue(project, prompt, seconds, force)

        with patch.object(ax, "generate_segment_audio", self._fake_segment_audio), \
             patch.object(ax, "generate_cue", spy_cue):
            ax.export_chapters("p", [ch], str(out))
        # Music bed sized against the (short) scene, clamped to the minimum.
        self.assertEqual(requested["seconds"], ax.MUSIC_MIN_S)

    def test_cues_can_be_disabled(self):
        ch = self._make_chapter("C1", 1, with_cue=True)
        out = Path(self.workdir.name) / "no_cues.wav"
        with patch.object(ax, "generate_segment_audio", self._fake_segment_audio), \
             patch.object(ax, "generate_cue", self._fake_cue):
            result = ax.export_chapters("p", [ch], str(out), include_cues=False)
        self.assertEqual(result.cues_rendered, 0)
        self.assertTrue(out.exists())

    def test_unavailable_cue_is_skipped_not_fatal(self):
        ch = self._make_chapter("C1", 1, with_cue=True)
        out = Path(self.workdir.name) / "book.wav"
        with patch.object(ax, "generate_segment_audio", self._fake_segment_audio), \
             patch.object(ax, "generate_cue", lambda *a, **k: None):
            result = ax.export_chapters("p", [ch], str(out))
        self.assertIsNone(result.error)
        self.assertTrue(any("cue skipped" in s for s in result.skipped))

    def test_uncompiled_chapter_is_reported(self):
        ch = self.manuscripts.create_chapter("p", "Empty")
        out = Path(self.workdir.name) / "book.wav"
        result = ax.export_chapters("p", [ch], str(out))
        self.assertIsNotNone(result.error)
        self.assertTrue(any("no compiled" in s for s in result.skipped))

    def test_unknown_chapter_id_errors(self):
        out = Path(self.workdir.name) / "book.wav"
        result = ax.export_chapters("p", [999], str(out))
        self.assertIn("Unknown chapter", result.error)

    def test_progress_callback_is_invoked(self):
        ch = self._make_chapter("C1", 1, with_cue=True)
        out = Path(self.workdir.name) / "book.wav"
        messages = []
        with patch.object(ax, "generate_segment_audio", self._fake_segment_audio), \
             patch.object(ax, "generate_cue", self._fake_cue):
            ax.export_chapters("p", [ch], str(out), progress=messages.append)
        self.assertTrue(any("Chapter: C1" in m for m in messages))
        self.assertTrue(any("bed [" in m for m in messages))
        self.assertTrue(any("Mastering" in m for m in messages))
        self.assertTrue(any("Exported" in m for m in messages))

    def test_cue_seconds_rules(self):
        seg = AudioSceneSegment("music", "", "x", timing={"start": 0, "end": 7})
        self.assertEqual(ax._cue_seconds(seg), 7.0)
        seg2 = AudioSceneSegment("music", "", "x")
        self.assertEqual(ax._cue_seconds(seg2, remaining_hint=12.0), 12.0)
        self.assertEqual(ax._cue_seconds(seg2, remaining_hint=200.0), ax.MUSIC_MAX_S)
        self.assertEqual(ax._cue_seconds(seg2, remaining_hint=1.0), ax.MUSIC_MIN_S)
        seg3 = AudioSceneSegment("sound_effect", "", "x")
        self.assertEqual(ax._cue_seconds(seg3), 5.0)

    def test_loudness_is_normalized(self):
        """Master loudness should land near the -16 LUFS audiobook target."""
        ch = self._make_chapter("C1", 1)
        out = Path(self.workdir.name) / "book.wav"
        with patch.object(ax, "generate_segment_audio", self._fake_segment_audio), \
             patch.object(ax, "generate_cue", self._fake_cue):
            ax.export_chapters("p", [ch], str(out))
        data, sr = sf.read(str(out))
        rms = np.sqrt(np.mean(data ** 2))
        # -16 LUFS integrated is roughly -16 dBFS RMS for a steady tone;
        # allow a generous window, we only assert it was leveled sanely.
        db = 20 * np.log10(max(rms, 1e-9))
        self.assertGreater(db, -30)
        self.assertLess(db, -6)


class TestSoundEngineCache(unittest.TestCase):
    def test_cache_key_stability(self):
        from generators import sound_engine as se
        a = se.cue_audio_path("p", "wind", 5.0)
        b = se.cue_audio_path("p", "wind", 5.0)
        c = se.cue_audio_path("p", "wind", 8.0)
        d = se.cue_audio_path("p", "rain", 5.0)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)

    def test_generate_cue_skips_empty_prompt_and_missing_model(self):
        from generators import sound_engine as se
        self.assertIsNone(se.generate_cue("p", "   "))
        with patch.object(se.SoundEngine, "available", return_value=False):
            self.assertIsNone(se.generate_cue("p", "thunder"))


if __name__ == "__main__":
    unittest.main()
