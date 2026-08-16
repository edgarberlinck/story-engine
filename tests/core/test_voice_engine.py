"""
Tests for character voice line building and speaker selection.

The model itself is NOT loaded in these tests (it is heavy); only the pure
functions that map character attributes to a line + (speaker, instruct) are
covered, plus the DB persistence of the voice path.
"""

import os
import tempfile
import unittest

from core.voice_engine import (
    build_voice_line,
    pick_speaker,
    VoiceEngine,
    infer_gender_from_prompt,
    FEMALE_SPEAKERS,
    MALE_SPEAKERS,
)
from services.database.character_service import CharacterService


class TestBuildVoiceLine(unittest.TestCase):
    def test_basic_line(self):
        line = build_voice_line(
            "Leila", "woman", {"age": "Elderly", "ethnicity": "Mediterranean"}
        )
        self.assertEqual(
            line,
            "Hi, my name is Leila, I'm an elderly mediterranean woman.",
        )

    def test_article_a_vs_an(self):
        self.assertTrue(
            build_voice_line("X", "man", {"age": "Adult"}).startswith(
                "Hi, my name is X, I'm an adult"
            )
        )
        self.assertTrue(
            build_voice_line("X", "man", {"age": "Child"}).startswith(
                "Hi, my name is X, I'm a child"
            )
        )

    def test_empty_attributes(self):
        line = build_voice_line("X", "person", {})
        self.assertEqual(line, "Hi, my name is X, I'm a person.")

    def test_builder_vocabulary_age_range(self):
        """The builder screen stores age_range ('50+', 'Teen', ...)."""
        line = build_voice_line(
            "Leila", "woman", {"gender": "Female", "age_range": "50+"}
        )
        self.assertIn("an elderly woman", line)

    def test_personality_appended(self):
        line = build_voice_line(
            "Aiden", "man", {"personality": "Curious", "age": "Child"}
        )
        self.assertIn("with a curious personality", line)

    def test_animal_species(self):
        line = build_voice_line("Rex", "animal", {"species": "Dog", "age": "Adult"})
        self.assertIn("an adult dog", line)


class TestPickSpeaker(unittest.TestCase):
    def test_female_default(self):
        speaker, _ = pick_speaker("woman", {"gender": "Female"})
        self.assertIn(speaker, FEMALE_SPEAKERS)

    def test_male_default(self):
        speaker, _ = pick_speaker("man", {"gender": "Male"})
        self.assertIn(speaker, MALE_SPEAKERS)

    def test_elderly_uses_older_preset(self):
        speaker, _ = pick_speaker("woman", {"gender": "Female", "age": "Elderly"})
        self.assertEqual(speaker, "vivian")

    def test_instruct_from_personality(self):
        _, instruct = pick_speaker("man", {"personality": "Friendly"})
        self.assertIn("warm, friendly", instruct)

    def test_instruct_from_mood(self):
        _, instruct = pick_speaker("woman", {"gender": "Female", "mood": "Happy"})
        self.assertIn("happy", instruct)

    def test_default_instruct(self):
        _, instruct = pick_speaker("person", {})
        self.assertTrue(instruct)


class TestVoicePathPersistence(unittest.TestCase):
    def setUp(self):
        self.tmpdb = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.tmpdb.name
        self.tmpdb.close()
        self.service = CharacterService(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_voice_path_roundtrip(self):
        self.service.save_character(
            name="Leila",
            prompt="a woman",
            seed=1,
            model="flux_dev",
            reference_image="/img.png",
            project="proj",
        )
        self.assertTrue(
            self.service.set_voice_path("Leila", "proj", "/voice/leila.wav")
        )
        char = self.service.get_character("Leila", "proj")
        self.assertEqual(char["voice_path"], "/voice/leila.wav")

    def test_voice_path_preserved_on_update(self):
        """Re-saving a character must not wipe the voice path."""
        self.service.save_character(
            name="Leila",
            prompt="a woman",
            seed=1,
            model="flux_dev",
            reference_image="/img.png",
            project="proj",
        )
        self.service.set_voice_path("Leila", "proj", "/voice/leila.wav")
        # Update prompt/seed only (voice_path omitted)
        self.service.save_character(
            name="Leila",
            prompt="a new woman",
            seed=2,
            model="flux_dev",
            reference_image="/img2.png",
            project="proj",
        )
        char = self.service.get_character("Leila", "proj")
        self.assertEqual(char["voice_path"], "/voice/leila.wav")

    def test_voice_prompt_roundtrip(self):
        self.service.save_character(
            name="Leila",
            prompt="a woman",
            seed=1,
            model="flux_dev",
            reference_image="/img.png",
            project="proj",
        )
        self.assertTrue(
            self.service.set_voice_path(
                "Leila", "proj", "/voice/leila.wav", voice_prompt="a calm warm voice"
            )
        )
        char = self.service.get_character("Leila", "proj")
        self.assertEqual(char["voice_prompt"], "a calm warm voice")


class TestInferGenderFromPrompt(unittest.TestCase):
    def test_female_hints(self):
        self.assertEqual(
            infer_gender_from_prompt(
                "A beautiful, elegant female voice... she sounds mature"
            ),
            "female",
        )
        self.assertEqual(
            infer_gender_from_prompt("a warm woman's voice, her delivery calm"),
            "female",
        )

    def test_male_hints(self):
        self.assertEqual(
            infer_gender_from_prompt("a deep male voice, he speaks slowly"),
            "male",
        )

    def test_no_hints(self):
        self.assertIsNone(infer_gender_from_prompt("a calm, clear voice"))
        self.assertIsNone(infer_gender_from_prompt(""))


class _FakeTTSModel:
    """Minimal stand-in for qwen_tts.Qwen3TTSModel (no torch/MPS needed)."""

    def __init__(self):
        self.calls = []

    def generate_custom_voice(
        self, text, speaker, language, instruct, non_streaming_mode=True
    ):
        import numpy as np

        self.calls.append({"text": text, "speaker": speaker, "instruct": instruct})
        wav = np.zeros(2400, dtype=np.float32)  # 0.1 s @ 24 kHz
        return [wav], 24000


class _FakeDesignModel:
    """Stand-in for the VoiceDesign checkpoint."""

    def __init__(self):
        self.calls = []

    def generate_voice_design(
        self, text, instruct, language=None, non_streaming_mode=True
    ):
        import numpy as np

        self.calls.append({"text": text, "instruct": instruct, "language": language})
        wav = np.zeros(2400, dtype=np.float32)
        return [wav], 24000


class TestVoiceEngineGeneration(unittest.TestCase):
    """Tests the generate/cache/force/prompt logic without loading the model."""

    def setUp(self):
        import tempfile
        from pathlib import Path

        self.tmpdir = tempfile.TemporaryDirectory()
        # Fake design dir that "exists" with a weights file so
        # design_model_available() is True.
        fake_design_dir = Path(self.tmpdir.name) / "design"
        fake_design_dir.mkdir()
        (fake_design_dir / "model.safetensors").touch()
        self.engine = VoiceEngine(
            model_dir="not/a/real/dir",
            design_model_dir=fake_design_dir,
        )
        self.engine._model = _FakeTTSModel()  # bypass real CustomVoice
        self.engine._design_model = _FakeDesignModel()  # bypass real VoiceDesign
        self.out = self.tmpdir.name

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_generates_and_caches(self):
        path1, _ = self.engine.generate_character_voice(
            "Leila",
            "proj",
            attributes={"gender": "Female"},
            output_dir=self.out,
        )
        path2, _ = self.engine.generate_character_voice(
            "Leila",
            "proj",
            attributes={"gender": "Female"},
            output_dir=self.out,
        )
        self.assertEqual(path1, path2)
        self.assertTrue(path1.exists())
        # Cached: only one model call.
        self.assertEqual(len(self.engine._model.calls), 1)
        self.assertIn("voice", str(path1))  # voice/ subfolder layout

    def test_force_regenerates(self):
        path1, _ = self.engine.generate_character_voice(
            "Leila",
            "proj",
            output_dir=self.out,
        )
        path2, _ = self.engine.generate_character_voice(
            "Leila",
            "proj",
            output_dir=self.out,
            force=True,
        )
        self.assertEqual(path1, path2)
        self.assertEqual(len(self.engine._model.calls), 2)

    def test_instruct_routes_to_voicedesign(self):
        """A regeneration prompt must be handled by the VoiceDesign model
        (it creates the timbre from the prompt)."""
        _, instruct_used = self.engine.generate_character_voice(
            "Nikita",
            "proj",
            output_dir=self.out,
            instruct="A warm, elegant and feminine female voice. She sounds intelligent.",
        )
        self.assertEqual(
            instruct_used,
            "A warm, elegant and feminine female voice. She sounds intelligent.",
        )
        self.assertEqual(len(self.engine._design_model.calls), 1)
        self.assertEqual(len(self.engine._model.calls), 0)  # CustomVoice untouched

    def test_no_prompt_uses_customvoice_attribute_gender(self):
        self.engine.generate_character_voice(
            "Nikita",
            "proj",
            attributes={"gender": "Male"},
            output_dir=self.out,
        )
        speaker = self.engine._model.calls[-1]["speaker"]
        self.assertIn(speaker, MALE_SPEAKERS)
        self.assertEqual(len(self.engine._design_model.calls), 0)

    def test_prompt_falls_back_when_design_model_missing(self):
        """Without the VoiceDesign checkpoint, a prompt degrades to
        CustomVoice with the prompt as instruct."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            engine = VoiceEngine(
                model_dir="not/a/real/dir",
                design_model_dir=Path(d),  # no model.safetensors inside
            )
            engine._model = _FakeTTSModel()
            out_dir = Path(d) / "out"
            _, instruct_used = engine.generate_character_voice(
                "Nikita",
                "proj",
                output_dir=out_dir,
                instruct="a deep booming voice",
            )
            self.assertEqual(instruct_used, "a deep booming voice")
            self.assertEqual(len(engine._model.calls), 1)


if __name__ == "__main__":
    unittest.main()
