"""Sound-effects inference and cue routing tests without downloaded weights."""

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf
import torch

from generators import sound_effects_generator as sfx
from generators import sound_engine as cues
from models import MODEL_METADATA, MODEL_PATHS, SOUND_EFFECTS_MODELS


class TestSoundEffectsGenerator(unittest.TestCase):
    def test_registered_checkpoint(self):
        self.assertEqual(SOUND_EFFECTS_MODELS[sfx.MODEL_NAME], sfx.MODEL_ID)
        self.assertEqual(MODEL_METADATA[sfx.MODEL_NAME]["type"], "sound_effects")
        self.assertEqual(MODEL_METADATA[sfx.MODEL_NAME]["repo_id"], sfx.MODEL_ID)
        self.assertEqual(
            sfx.MODEL_DIR,
            Path(__file__).resolve().parent.parent
            / MODEL_PATHS["sound_effects"]
            / sfx.MODEL_NAME,
        )

    def test_available_checks_diffusers_index_not_musicgen_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = sfx.SoundEffectsGenerator(Path(tmp))
            self.assertFalse(generator.available())
            (Path(tmp) / "config.json").write_text("{}")
            self.assertFalse(generator.available())
            (Path(tmp) / "model_index.json").write_text("{}")
            self.assertTrue(generator.available())

    def test_load_is_local_safe_lazy_and_uses_device_precision(self):
        for device, expected_dtype in (
            ("cpu", torch.float32),
            ("mps", torch.float32),
            ("cuda", torch.float16),
        ):
            with self.subTest(device=device):
                pipeline_class = MagicMock()
                pipeline = pipeline_class.from_pretrained.return_value
                pipeline.to.return_value = pipeline
                generator = sfx.SoundEffectsGenerator(Path("/tmp/sfx-test-model"))
                with patch.dict(
                    sys.modules,
                    {"diffusers": SimpleNamespace(StableAudioPipeline=pipeline_class)},
                ), patch.object(
                    sfx, "get_model_config", return_value=(device, torch.float16)
                ):
                    self.assertIs(generator._load(), pipeline)
                    self.assertIs(generator._load(), pipeline)
                pipeline_class.from_pretrained.assert_called_once_with(
                    str(generator.model_dir),
                    torch_dtype=expected_dtype,
                    local_files_only=True,
                    use_safetensors=True,
                )
                pipeline.to.assert_called_once_with(device)

    def test_generate_downmixes_stereo_and_trims_to_duration(self):
        generator = sfx.SoundEffectsGenerator()
        pipeline = MagicMock()
        pipeline.vae.sampling_rate = 44100
        stereo = torch.stack((torch.ones(44110), torch.zeros(44110)))
        pipeline.return_value.audios = stereo.unsqueeze(0)
        generator._pipeline = pipeline
        wav, sr = generator.generate("rain on a roof", 1.0)
        self.assertEqual(sr, 44100)
        self.assertEqual(wav.shape, (44100,))
        np.testing.assert_allclose(wav, 0.5)
        pipeline.assert_called_once_with(
            prompt="rain on a roof",
            negative_prompt=sfx.NEGATIVE_PROMPT,
            audio_end_in_s=1.0,
            num_inference_steps=sfx.INFERENCE_STEPS,
            num_waveforms_per_prompt=1,
            output_type="pt",
        )

    def test_invalid_durations_do_not_load_model(self):
        generator = sfx.SoundEffectsGenerator()
        with patch.object(generator, "_load") as load:
            for seconds in (0, -1, 48, float("nan"), float("inf")):
                with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                    generator.generate("thunder", seconds)
            load.assert_not_called()

    def test_unload_releases_pipeline(self):
        generator = sfx.SoundEffectsGenerator()
        generator._pipeline = MagicMock()
        with patch("torch.backends.mps.is_available", return_value=False), patch(
            "torch.cuda.is_available", return_value=False
        ):
            generator.unload()
            generator.unload()
        self.assertIsNone(generator._pipeline)


class TestSoundEffectsInstallation(unittest.TestCase):
    def test_model_installer_downloads_registered_sound_effects(self):
        from scripts import install

        with patch.object(install, "setup_directories"), patch.object(
            install, "get_network_speed", return_value=1
        ), patch.object(
            install, "download_model", return_value=True
        ) as download, redirect_stdout(
            io.StringIO()
        ):
            install.install_all_models()
        download.assert_any_call(
            sfx.MODEL_NAME,
            "sound_effects",
            f"{MODEL_PATHS['sound_effects']}/{sfx.MODEL_NAME}",
        )

    def test_model_installer_creates_sound_effects_directory(self):
        from scripts import install

        with patch.object(install.os, "makedirs") as mkdir, redirect_stdout(
            io.StringIO()
        ):
            install.setup_directories()
        mkdir.assert_any_call(MODEL_PATHS["sound_effects"], exist_ok=True)


class TestCueRouting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project_dir = patch.object(
            cues, "project_dir", return_value=Path(self.tmp.name)
        )
        self.project_dir.start()
        self.addCleanup(self.project_dir.stop)
        self.sfx_engine = MagicMock()
        self.music_engine = MagicMock()
        for engine in (self.sfx_engine, self.music_engine):
            engine.available.return_value = True
            engine.generate.return_value = (np.zeros(100, dtype=np.float32), 44100)
        for cls, engine in (
            (cues.SoundEffectsGenerator, self.sfx_engine),
            (cues.SoundEngine, self.music_engine),
        ):
            mock_instance = patch.object(cls, "get_instance", return_value=engine)
            mock_instance.start()
            self.addCleanup(mock_instance.stop)

    def test_sfx_and_music_use_different_engines_and_cache_paths(self):
        effect = cues.generate_cue("p", "rain", 5)
        music = cues.generate_cue("p", "rain", 5, cue_type="music")
        self.assertNotEqual(effect, music)
        self.assertEqual(effect, cues.cue_audio_path("p", "rain", 5))
        self.assertEqual(music, cues.cue_audio_path("p", "rain", 5, cue_type="music"))
        self.sfx_engine.generate.assert_called_once_with("rain", 5)
        self.music_engine.generate.assert_called_once_with("rain", 5)
        self.assertEqual(sf.info(str(effect)).samplerate, 44100)

    def test_cached_audio_reused_without_model_and_force_regenerates(self):
        first = cues.generate_cue("p", "thunder", 5)
        self.sfx_engine.available.return_value = False
        self.assertEqual(cues.generate_cue("p", "thunder", 5), first)
        self.sfx_engine.generate.assert_called_once()
        self.sfx_engine.available.return_value = True
        self.assertEqual(cues.generate_cue("p", "thunder", 5, force=True), first)
        self.assertEqual(self.sfx_engine.generate.call_count, 2)

    def test_unavailable_sfx_does_not_fall_back_to_music(self):
        self.sfx_engine.available.return_value = False
        self.assertIsNone(cues.generate_cue("p", "thunder", 5))
        self.music_engine.generate.assert_not_called()
        self.assertIsNotNone(cues.generate_cue("p", "piano", 5, cue_type="music"))

    def test_unavailable_music_does_not_disable_sfx(self):
        self.music_engine.available.return_value = False
        self.assertIsNone(cues.generate_cue("p", "piano", 5, cue_type="music"))
        self.assertIsNotNone(cues.generate_cue("p", "thunder", 5))

    def test_cache_identity_includes_model_settings_type_and_precise_duration(self):
        base = cues.cue_audio_path("p", "wind", 5.01)
        self.assertEqual(base, cues.cue_audio_path("p", " wind ", 5.01))
        self.assertNotEqual(base, cues.cue_audio_path("p", "wind", 5.02))
        self.assertNotEqual(
            base, cues.cue_audio_path("p", "wind", 5.01, cue_type="music")
        )
        with patch.object(cues, "SFX_MODEL_ID", "another/model"):
            self.assertNotEqual(base, cues.cue_audio_path("p", "wind", 5.01))
        with patch.object(cues, "INFERENCE_STEPS", 200):
            self.assertNotEqual(base, cues.cue_audio_path("p", "wind", 5.01))
        legacy = cues.hashlib.sha1(b"wind|5.0").hexdigest()[:16]
        self.assertNotEqual(base.name, f"cue_{legacy}.wav")

    def test_invalid_cue_type_is_rejected(self):
        with self.assertRaises(ValueError):
            cues.generate_cue("p", "wind", 5, cue_type="speech")
        self.sfx_engine.generate.assert_not_called()
        self.music_engine.generate.assert_not_called()

    def test_blank_prompt_does_not_load_either_engine(self):
        self.assertIsNone(cues.generate_cue("p", "   "))
        self.sfx_engine.generate.assert_not_called()
        self.music_engine.generate.assert_not_called()

    def test_export_unloads_both_audio_models(self):
        from core.audiobook_exporter import _unload_generation_models

        with patch("core.voice_engine.voice_engine.unload"), patch(
            "generators.sound_engine.sound_engine.unload"
        ) as unload_music, patch(
            "generators.sound_effects_generator.sound_effects_generator.unload"
        ) as unload_sfx:
            _unload_generation_models()
        unload_music.assert_called_once()
        unload_sfx.assert_called_once()


if __name__ == "__main__":
    unittest.main()
