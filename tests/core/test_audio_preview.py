"""
Tests for per-segment audio preview (plan §7: regenerate only the affected
line, never the entire audiobook).
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core import audio_preview
from services.audio_scene_service import AudioSceneSegment


def _seg(**kwargs):
    base = dict(segment_type="dialogue", speaker="nikita", text="Hello there.")
    base.update(kwargs)
    return AudioSceneSegment(**base)


class TestSegmentCache(unittest.TestCase):
    def test_cache_key_stable_for_identical_segments(self):
        self.assertEqual(
            audio_preview._segment_cache_key(_seg()),
            audio_preview._segment_cache_key(_seg()),
        )

    def test_cache_key_changes_when_audio_fields_change(self):
        base = audio_preview._segment_cache_key(_seg())
        for change in (
            {"text": "Different."}, {"emotion": "angry"}, {"tone": "cold"},
            {"intensity": 0.9}, {"delivery": "fast"}, {"voice": "roger"},
            {"speaker": "roger"},
        ):
            self.assertNotEqual(base, audio_preview._segment_cache_key(_seg(**change)))

    def test_segment_audio_path_contains_index_and_key(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(audio_preview, "scene_dir", return_value=Path(tmp)):
            path = audio_preview.segment_audio_path("p", 1, 3, _seg())
            self.assertTrue(path.name.startswith("segment_003_"))
            self.assertTrue(path.name.endswith(".wav"))


class TestGenerateSegmentAudio(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.scene_patch = patch.object(
            audio_preview, "scene_dir", return_value=Path(self.tmp.name)
        )
        self.scene_patch.start()

    def tearDown(self):
        self.scene_patch.stop()
        self.tmp.cleanup()

    def _fake_wav(self):
        return np.zeros(16, dtype=np.float32), 24000

    def test_skips_empty_and_non_speech_segments(self):
        self.assertIsNone(
            audio_preview.generate_segment_audio("p", 1, 0, _seg(text="  "))
        )
        self.assertIsNone(
            audio_preview.generate_segment_audio(
                "p", 1, 0, _seg(segment_type="sound_effect", text="boom")
            )
        )

    def test_generates_dialogue_with_segment_performance(self):
        captured = {}

        def fake_pick(char_type, attributes, **kwargs):
            captured.update(kwargs)
            return "Speaker1", "instr"

        with patch.object(audio_preview, "pick_speaker", side_effect=fake_pick), \
             patch.object(audio_preview.voice_engine, "generate_voice_line",
                          return_value=self._fake_wav()) as gen, \
             patch.object(audio_preview, "_character_attributes", return_value={}):
            seg = _seg(emotion="angry", tone="confrontational",
                       intensity=0.8, delivery="fast and forceful")
            path = audio_preview.generate_segment_audio("p", 1, 0, seg)

        self.assertTrue(path.exists())
        gen.assert_called_once()
        self.assertEqual(captured["emotion"], "angry")
        self.assertEqual(captured["intensity"], 0.8)

    def test_cache_hit_skips_regeneration_and_force_regenerates(self):
        with patch.object(audio_preview, "pick_speaker",
                          return_value=("Speaker1", "i")), \
             patch.object(audio_preview.voice_engine, "generate_voice_line",
                          return_value=self._fake_wav()) as gen, \
             patch.object(audio_preview, "_character_attributes", return_value={}):
            seg = _seg()
            audio_preview.generate_segment_audio("p", 1, 0, seg)
            audio_preview.generate_segment_audio("p", 1, 0, seg)   # cache hit
            self.assertEqual(gen.call_count, 1)
            audio_preview.generate_segment_audio("p", 1, 0, seg, force=True)
            self.assertEqual(gen.call_count, 2)

    def test_changed_line_regenerates_only_itself(self):
        with patch.object(audio_preview, "pick_speaker",
                          return_value=("Speaker1", "i")), \
             patch.object(audio_preview.voice_engine, "generate_voice_line",
                          return_value=self._fake_wav()) as gen, \
             patch.object(audio_preview, "_character_attributes", return_value={}):
            seg_a, seg_b = _seg(), _seg(speaker="roger", text="Morning.")
            audio_preview.generate_segment_audio("p", 1, 0, seg_a)
            audio_preview.generate_segment_audio("p", 1, 1, seg_b)
            self.assertEqual(gen.call_count, 2)
            # Only line B changes; line A stays cached.
            seg_b.text = "Good morning!"
            audio_preview.generate_segment_audio("p", 1, 0, seg_a)
            audio_preview.generate_segment_audio("p", 1, 1, seg_b)
            self.assertEqual(gen.call_count, 3)

    def test_narrator_uses_character_mode_config(self):
        narrator = {"mode": "character", "character": "Nikita", "voice_prompt": ""}
        with patch.object(audio_preview.project_settings_service, "get_narrator",
                          return_value=narrator), \
             patch.object(audio_preview, "_character_attributes",
                          return_value={"gender": "Female"}) as attrs, \
             patch.object(audio_preview, "pick_speaker",
                          return_value=("Speaker1", "i")), \
             patch.object(audio_preview.voice_engine, "generate_voice_line",
                          return_value=self._fake_wav()):
            seg = _seg(segment_type="narration", speaker="narrator", voice=None)
            path = audio_preview.generate_segment_audio("p", 1, 0, seg)
        self.assertTrue(path.exists())
        attrs.assert_called_once_with("p", "Nikita")

    def test_narrator_dedicated_uses_designed_voice_when_available(self):
        narrator = {"mode": "dedicated", "character": None,
                    "voice_prompt": "A deep cinematic voice"}
        with patch.object(audio_preview.project_settings_service, "get_narrator",
                          return_value=narrator), \
             patch.object(audio_preview.voice_engine, "design_model_available",
                          return_value=True), \
             patch.object(audio_preview.voice_engine, "generate_designed_voice",
                          return_value=self._fake_wav()) as design:
            seg = _seg(segment_type="narration", speaker="narrator",
                       voice=None, emotion="calm")
            path = audio_preview.generate_segment_audio("p", 1, 0, seg)
        self.assertTrue(path.exists())
        design.assert_called_once()
        self.assertIn("A deep cinematic voice", design.call_args.args[1])
        self.assertIn("calm", design.call_args.args[1])

    def test_narrator_dedicated_falls_back_to_preset_speaker(self):
        narrator = {"mode": "dedicated", "character": None, "voice_prompt": "X"}
        with patch.object(audio_preview.project_settings_service, "get_narrator",
                          return_value=narrator), \
             patch.object(audio_preview.voice_engine, "design_model_available",
                          return_value=False), \
             patch.object(audio_preview.voice_engine, "generate_voice_line",
                          return_value=self._fake_wav()) as gen:
            seg = _seg(segment_type="narration", speaker="narrator", voice=None)
            path = audio_preview.generate_segment_audio("p", 1, 0, seg)
        self.assertTrue(path.exists())
        gen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
