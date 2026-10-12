"""Sound-effects inference and cue routing without downloading model weights."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
import torch

from generators import sound_effects_engine as sfx
from generators import sound_engine as cues
from models import MODEL_METADATA, MODEL_PATHS, SOUND_EFFECTS_MODELS


class TestSoundEffectsEngine(unittest.TestCase):
    def test_registry_and_local_path_agree(self):
        key = "stable_audio_open"
        self.assertEqual(SOUND_EFFECTS_MODELS[key], "stabilityai/stable-audio-open-1.0")
        self.assertEqual(MODEL_METADATA[key]["repo_id"], SOUND_EFFECTS_MODELS[key])
        self.assertEqual(MODEL_METADATA[key]["type"], "sound_effects")
        self.assertEqual(sfx.STABLE_AUDIO_DIR, Path(MODEL_PATHS["sound_effects"]) / key)

    def test_available_checks_diffusers_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = sfx.SoundEffectsEngine(Path(tmp))
            self.assertFalse(engine.available())
            (Path(tmp) / "config.json").write_text("{}")
            self.assertFalse(engine.available())
            (Path(tmp) / "model_index.json").write_text("{}")
            self.assertTrue(engine.available())

    def test_load_is_lazy_local_and_uses_configured_device(self):
        pipeline = Mock()
        pipeline.to.return_value = pipeline
        pipeline_class = Mock()
        pipeline_class.from_pretrained.return_value = pipeline
        engine = sfx.SoundEffectsEngine(Path("/tmp/sfx-model"))
        with patch.dict(
            "sys.modules",
            {"diffusers": SimpleNamespace(StableAudioPipeline=pipeline_class)},
        ), patch.object(sfx, "get_model_config", return_value=("cpu", torch.float32)):
            self.assertIs(engine._load(), pipeline)
            self.assertIs(engine._load(), pipeline)
        pipeline_class.from_pretrained.assert_called_once_with(
            "/tmp/sfx-model", torch_dtype=torch.float32, local_files_only=True
        )
        pipeline.to.assert_called_once_with("cpu")

    def test_generate_preserves_stereo_and_requested_duration(self):
        audio = np.arange(1000, dtype=np.float32).reshape(2, 500)
        pipeline = Mock(return_value=SimpleNamespace(audios=np.array([audio])))
        pipeline.vae.config.sampling_rate = 44100
        engine = sfx.SoundEffectsEngine()
        engine._pipeline = pipeline
        wav, sr = engine.generate("  a door slamming  ", 0.01)
        self.assertEqual(sr, 44100)
        self.assertEqual(wav.shape, (441, 2))
        np.testing.assert_array_equal(wav, audio.T[:441])
        pipeline.assert_called_once_with(
            prompt="a door slamming",
            negative_prompt="Low quality.",
            audio_start_in_s=0.0,
            audio_end_in_s=0.01,
            num_inference_steps=100,
            guidance_scale=7.0,
            output_type="np",
        )

    def test_invalid_input_does_not_load_model(self):
        engine = sfx.SoundEffectsEngine()
        with patch.object(engine, "_load") as load:
            for seconds in (0, -1, 48, float("nan"), float("inf")):
                with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                    engine.generate("door", seconds)
            with self.assertRaises(ValueError):
                engine.generate("  ")
        load.assert_not_called()

    def test_unload_releases_pipeline_and_device_cache(self):
        engine = sfx.SoundEffectsEngine()
        engine._pipeline = Mock()
        with patch("torch.backends.mps.is_available", return_value=False), patch(
            "torch.cuda.is_available", return_value=True
        ), patch("torch.cuda.empty_cache") as empty_cache:
            engine.unload()
            engine.unload()
        self.assertIsNone(engine._pipeline)
        empty_cache.assert_called_once()


class TestCueRouting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.effects = Mock()
        self.effects.available.return_value = True
        self.effects.generate.return_value = (np.zeros((441, 2)), 44100)
        self.music = Mock()
        self.music.available.return_value = True
        self.music.generate.return_value = (np.zeros(320), 32000)
        for target, value in (
            ("project_dir", self.root),
            ("SoundEffectsEngine.get_instance", self.effects),
            ("SoundEngine.get_instance", self.music),
        ):
            p = patch(f"generators.sound_engine.{target}", return_value=value)
            p.start()
            self.addCleanup(p.stop)

    def test_effects_and_music_route_to_separate_models_and_caches(self):
        effects_path = cues.generate_cue("p", "door", 0.01)
        music_path = cues.generate_cue("p", "door", 0.01, cue_type="music")
        self.assertNotEqual(effects_path, music_path)
        self.effects.generate.assert_called_once_with("door", 0.01)
        self.music.generate.assert_called_once_with("door", 0.01)
        self.assertEqual(sf.info(str(effects_path)).channels, 2)
        self.assertEqual(sf.info(str(effects_path)).samplerate, 44100)
        self.assertEqual(sf.info(str(music_path)).samplerate, 32000)

    def test_cache_reuse_force_and_offline_playback(self):
        path = cues.generate_cue("p", "door", 0.01)
        self.assertEqual(cues.generate_cue("p", "  door  ", 0.01), path)
        self.assertEqual(self.effects.generate.call_count, 1)
        cues.generate_cue("p", "door", 0.01, force=True)
        self.assertEqual(self.effects.generate.call_count, 2)
        self.effects.available.return_value = False
        self.assertEqual(cues.generate_cue("p", "door", 0.01), path)
        self.assertIsNone(cues.generate_cue("p", "door", 0.01, force=True))

    def test_missing_effects_does_not_disable_music_or_fall_back(self):
        self.effects.available.return_value = False
        self.assertIsNone(cues.generate_cue("p", "door", 0.01))
        self.music.generate.assert_not_called()
        self.assertIsNotNone(cues.generate_cue("p", "piano", 0.01, cue_type="music"))

    def test_cache_identity_includes_model_type_and_precise_duration(self):
        path = cues.cue_audio_path("p", "door", 5.0)
        self.assertEqual(path, cues.cue_audio_path("p", "door", 5))
        self.assertNotEqual(path, cues.cue_audio_path("p", "door", 5.01))
        self.assertNotEqual(path, cues.cue_audio_path("p", "door", 5.0, "music"))
        with patch.dict(cues.SOUND_EFFECTS_MODELS, stable_audio_open="new/model"):
            self.assertNotEqual(path, cues.cue_audio_path("p", "door", 5.0))

    def test_empty_prompt_and_unknown_type(self):
        self.assertIsNone(cues.generate_cue("p", "  "))
        self.effects.generate.assert_not_called()
        self.music.generate.assert_not_called()
        with self.assertRaises(ValueError):
            cues.generate_cue("p", "door", cue_type="dialogue")


class TestSoundEffectsInstallation(unittest.TestCase):
    def test_installer_downloads_effects_in_registered_directory(self):
        from scripts import install

        with patch.object(install, "setup_directories"), patch.object(
            install, "get_network_speed", return_value=1
        ), patch.object(install, "download_model", return_value=True) as download:
            install.install_all_models()
        download.assert_any_call(
            "stable_audio_open",
            "sound_effects",
            "models/sound_effects/stable_audio_open",
        )


class TestAudioEngineCleanup(unittest.TestCase):
    def test_export_cleanup_unloads_both_audio_backends(self):
        from core.audiobook_exporter import _unload_generation_models

        with patch("core.voice_engine.voice_engine.unload") as voice, patch(
            "generators.sound_engine.sound_engine.unload"
        ) as music, patch(
            "generators.sound_effects_engine.sound_effects_engine.unload"
        ) as effects:
            _unload_generation_models()
        voice.assert_called_once()
        music.assert_called_once()
        effects.assert_called_once()
