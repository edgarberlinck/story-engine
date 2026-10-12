"""Sound-effects generation and cue routing without downloading model weights."""

import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from generators import sound_engine as cues
from generators.sound_effects_generator import SoundEffectsGenerator
from models import MODEL_METADATA, MODEL_PATHS, SOUND_EFFECT_MODELS


class TestSoundEffectsGenerator(unittest.TestCase):
    def test_registry(self):
        self.assertEqual(
            SOUND_EFFECT_MODELS["stable_audio_open"],
            MODEL_METADATA["stable_audio_open"]["repo_id"],
        )
        self.assertEqual(MODEL_METADATA["stable_audio_open"]["type"], "sound_effects")
        self.assertEqual(MODEL_PATHS["sound_effects"], "models/sound_effects")

    def test_installer_includes_sound_effects_checkpoint(self):
        from scripts import install

        with patch.object(install, "setup_directories"), patch.object(
            install, "get_network_speed", return_value=1
        ), patch.object(
            install, "download_model", return_value=True
        ) as download, patch.object(
            install, "tqdm", side_effect=lambda iterable, **kwargs: iterable
        ), redirect_stdout(io.StringIO()):
            install.install_all_models()
        download.assert_any_call(
            "stable_audio_open",
            "sound_effects",
            "models/sound_effects/stable_audio_open",
        )

    def test_available_checks_pipeline_index_not_musicgen_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            engine = SoundEffectsGenerator(directory)
            self.assertFalse(engine.available())
            (directory / "config.json").touch()
            self.assertFalse(engine.available())
            (directory / "model_index.json").touch()
            self.assertTrue(engine.available())

    def test_load_is_lazy_local_and_uses_safe_device_dtype(self):
        for device, expected in (
            ("cpu", torch.float32),
            ("mps", torch.float32),
            ("cuda", torch.float16),
        ):
            with self.subTest(device=device), patch(
                "diffusers.StableAudioPipeline.from_pretrained"
            ) as load, patch(
                "generators.sound_effects_generator.get_model_config",
                return_value=(device, torch.float32),
            ):
                engine = SoundEffectsGenerator(Path("/local/model"))
                pipeline = load.return_value.to.return_value
                self.assertIs(engine._load(), pipeline)
                self.assertIs(engine._load(), pipeline)
                load.assert_called_once_with(
                    "/local/model", torch_dtype=expected, local_files_only=True
                )
                load.return_value.to.assert_called_once_with(device)

    def test_generate_uses_duration_and_downmixes_to_mono(self):
        pipeline = MagicMock()
        pipeline.vae.sampling_rate = 44100
        pipeline.return_value = SimpleNamespace(
            audios=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        )
        engine = SoundEffectsGenerator()
        engine._pipeline = pipeline
        wav, sr = engine.generate("Rain on a metal roof", 8.0)
        np.testing.assert_allclose(wav, [0.5, 0.5])
        self.assertEqual(sr, 44100)
        self.assertEqual(pipeline.call_args.kwargs["audio_end_in_s"], 8.0)
        self.assertEqual(pipeline.call_args.kwargs["num_waveforms_per_prompt"], 1)

    def test_invalid_duration_and_empty_prompt_do_not_load_model(self):
        engine = SoundEffectsGenerator()
        with patch.object(engine, "_load") as load:
            for seconds in (0, -1, 48, float("nan"), float("inf")):
                with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                    engine.generate("Rain", seconds)
            with self.assertRaises(ValueError):
                engine.generate(" ")
            load.assert_not_called()

    def test_unload_releases_pipeline(self):
        engine = SoundEffectsGenerator()
        engine._pipeline = MagicMock()
        with patch("torch.backends.mps.is_available", return_value=False), patch(
            "torch.cuda.is_available", return_value=False
        ):
            engine.unload()
            engine.unload()
        self.assertIsNone(engine._pipeline)


class TestCueRouting(unittest.TestCase):
    def test_sound_effects_and_music_have_separate_cache_paths(self):
        sfx = cues.cue_audio_path("p", "wind", 5.0)
        music = cues.cue_audio_path("p", "wind", 5.0, cue_type="music")
        old_key = hashlib.sha1(b"wind|5.0").hexdigest()[:16]
        self.assertNotEqual(sfx, music)
        self.assertEqual(music.name, f"cue_{old_key}.wav")
        self.assertNotEqual(sfx, cues.cue_audio_path("p", "wind", 5.01))

    def test_invalid_cue_type_is_rejected(self):
        with self.assertRaises(ValueError):
            cues.cue_audio_path("p", "rain", 5.0, cue_type="speech")

    def test_routing_cache_and_force(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            cues, "project_dir", return_value=Path(tmp)
        ), patch.object(cues.SoundEngine, "get_instance") as music, patch.object(
            cues.SoundEffectsGenerator, "get_instance"
        ) as sfx:
            for factory in (music, sfx):
                factory.return_value.available.return_value = True
                factory.return_value.generate.return_value = (
                    np.zeros(100, dtype=np.float32), 44100
                )
            effect_path = cues.generate_cue("p", "rain", 5.0)
            self.assertTrue(effect_path.exists())
            sfx.return_value.generate.assert_called_once_with("rain", 5.0)
            music.assert_not_called()
            self.assertEqual(cues.generate_cue("p", "rain", 5.0), effect_path)
            self.assertEqual(sfx.return_value.generate.call_count, 1)
            cues.generate_cue("p", "rain", 5.0, force=True)
            self.assertEqual(sfx.return_value.generate.call_count, 2)
            music_path = cues.generate_cue("p", "rain", 5.0, cue_type="music")
            self.assertTrue(music_path.exists())
            self.assertNotEqual(effect_path, music_path)
            music.return_value.generate.assert_called_once_with("rain", 5.0)

    def test_missing_sound_model_does_not_fall_back_to_music(self):
        with patch.object(cues.SoundEffectsGenerator, "get_instance") as sfx, patch.object(
            cues.SoundEngine, "get_instance"
        ) as music, patch.object(cues, "cue_audio_path") as path:
            path.return_value.exists.return_value = False
            sfx.return_value.available.return_value = False
            self.assertIsNone(cues.generate_cue("p", "rain"))
            sfx.return_value.generate.assert_not_called()
            music.assert_not_called()


if __name__ == "__main__":
    unittest.main()
