#!/usr/bin/env python3
"""Unit tests for video generation: paths, character service, video engine."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from utils import project_paths
from services.database.character_service import CharacterService
from generators.video_generator import (
    AVAILABLE_VIDEO_MODELS,
    DEFAULT_VIDEO_MODEL,
    MODEL_GENERATION_PARAMS,
    generate_video,
)
from generators import video_engine


class TestProjectPaths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._old_root = project_paths.OUTPUTS_ROOT
        project_paths.OUTPUTS_ROOT = project_paths.Path(self.tmp)

    def tearDown(self):
        project_paths.OUTPUTS_ROOT = self._old_root
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_structure(self):
        char = project_paths.character_dir("Richard Morton")
        self.assertTrue(char.is_dir())
        self.assertTrue(str(char).endswith("test_project/characters/Richard_Morton"))

        self.assertEqual(project_paths.next_scene_number(), 1)
        s1 = project_paths.scene_dir()
        self.assertTrue(str(s1).endswith("scenes/scene_1"))
        self.assertEqual(project_paths.next_scene_number(), 2)

        out = project_paths.scene_out_dir(1)
        self.assertTrue(str(out).endswith("scenes/scene_1/out"))
        self.assertTrue(out.is_dir())

    def test_slugify(self):
        self.assertEqual(project_paths.slugify("Richard Morton!"), "Richard_Morton")


class TestCharacterService(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.svc = CharacterService(db_path=self.db_path)

    def tearDown(self):
        os.remove(self.db_path)

    def test_save_and_get(self):
        self.svc.save_character(
            "Richard Morton", "a tall archeologist", 42, "flux_dev", "/tmp/ref.png"
        )
        c = self.svc.get_character("Richard Morton")
        self.assertEqual(c["prompt"], "a tall archeologist")
        self.assertEqual(c["seed"], 42)
        self.assertEqual(c["model"], "flux_dev")
        self.assertEqual(c["reference_image"], "/tmp/ref.png")

    def test_upsert(self):
        self.svc.save_character("R", "p1", 1, "flux_dev", "a.png")
        self.svc.save_character("R", "p2", 2, "sdxl", "b.png")
        c = self.svc.get_character("R")
        self.assertEqual(c["prompt"], "p2")
        self.assertEqual(len(self.svc.list_characters()), 1)

    def test_find_in_text(self):
        self.svc.save_character("Richard Morton", "p", 1, "flux_dev", "a.png")
        found = self.svc.find_characters_in_text("richard morton waves")
        self.assertEqual(len(found), 1)
        self.assertEqual(self.svc.find_characters_in_text("nobody here"), [])


class TestVideoGenerator(unittest.TestCase):
    def test_default_model_is_wan(self):
        self.assertEqual(DEFAULT_VIDEO_MODEL, "wan22_i2v")

    def test_all_models_have_params(self):
        self.assertEqual(set(AVAILABLE_VIDEO_MODELS), set(MODEL_GENERATION_PARAMS))
        for params in MODEL_GENERATION_PARAMS.values():
            for key in ("width", "height", "num_frames", "fps",
                        "guidance_scale", "num_inference_steps"):
                self.assertIn(key, params)

    def test_invalid_model_rejected(self):
        with self.assertRaises(ValueError):
            generate_video("img.png", "prompt", model_name="not_a_model")

    def test_missing_image_rejected(self):
        with self.assertRaises(FileNotFoundError):
            generate_video("/nope/missing.png", "prompt")


class TestVideoEngine(unittest.TestCase):
    def test_benchmark_runs_all_models(self):
        scene = {"scene_number": 1, "image_path": "x.png", "prompt": "p"}
        with patch.object(video_engine, "animate_scene", return_value={"ok": True}) as m:
            results = video_engine.benchmark_scene_video(scene)
        self.assertEqual(set(results), set(AVAILABLE_VIDEO_MODELS))
        self.assertEqual(m.call_count, len(AVAILABLE_VIDEO_MODELS))

    def test_benchmark_tolerates_failures(self):
        scene = {"scene_number": 1, "image_path": "x.png", "prompt": "p"}
        with patch.object(video_engine, "animate_scene", side_effect=RuntimeError("boom")):
            results = video_engine.benchmark_scene_video(scene)
        self.assertTrue(all(v is None for v in results.values()))

    def test_scene_retries_until_verified(self):
        character = {"name": "R", "reference_image": "ref.png"}
        scenes = [
            {"scene_number": 1, "image_path": "a.png", "prompt": "p"},
            {"scene_number": 1, "image_path": "b.png", "prompt": "p"},
        ]
        with patch.object(video_engine, "get_character", return_value=character), \
             patch.object(video_engine, "generate_scene", side_effect=scenes) as gen, \
             patch.object(video_engine, "verify_character_in_scene",
                          side_effect=[False, True]):
            scene = video_engine.create_validated_scene("p", character_name="R")
        self.assertTrue(scene["character_verified"])
        self.assertEqual(gen.call_count, 2)
        # Retry must reuse the same scene folder and bump the seed
        self.assertEqual(gen.call_args_list[1].kwargs["scene_number"], 1)
        self.assertEqual(gen.call_args_list[1].kwargs["seed"], 43)

    def test_unknown_character_raises(self):
        with patch.object(video_engine, "get_character", return_value=None):
            with self.assertRaises(ValueError):
                video_engine.create_validated_scene("p", character_name="Ghost")

    def test_inconclusive_verification_accepts_scene(self):
        character = {"name": "R", "reference_image": "ref.png"}
        scene_result = {"scene_number": 2, "image_path": "a.png", "prompt": "p"}
        with patch.object(video_engine, "get_character", return_value=character), \
             patch.object(video_engine, "generate_scene", return_value=scene_result), \
             patch.object(video_engine, "verify_character_in_scene", return_value=None):
            scene = video_engine.create_validated_scene("p", character_name="R")
        self.assertIsNone(scene["character_verified"])


if __name__ == "__main__":
    unittest.main()


class TestMultiCharacterVerification(unittest.TestCase):
    def _chars(self):
        return {
            "Yamu": {"name": "Yamu", "reference_image": "y.png"},
            "Cristal": {"name": "Cristal", "reference_image": "c.png"},
        }

    def test_all_characters_must_be_verified(self):
        chars = self._chars()
        scene_result = {"scene_number": 1, "image_path": "a.png", "prompt": "p"}
        # First attempt: Cristal missing; second attempt: both found
        with patch.object(video_engine, "get_character",
                          side_effect=lambda n, p=None: chars[n]), \
             patch.object(video_engine, "generate_scene",
                          return_value=scene_result) as gen, \
             patch.object(video_engine, "verify_character_in_scene",
                          side_effect=[True, False, True, True]):
            scene = video_engine.create_validated_scene(
                "p", character_names=["Yamu", "Cristal"])
        self.assertTrue(scene["character_verified"])
        self.assertEqual(gen.call_count, 2)

    def test_require_verification_raises_when_inconclusive(self):
        chars = self._chars()
        scene_result = {"scene_number": 1, "image_path": "a.png", "prompt": "p"}
        with patch.object(video_engine, "get_character",
                          side_effect=lambda n, p=None: chars[n]), \
             patch.object(video_engine, "generate_scene",
                          return_value=scene_result), \
             patch.object(video_engine, "verify_character_in_scene",
                          return_value=None):
            with self.assertRaises(RuntimeError):
                video_engine.create_validated_scene(
                    "p", character_names=["Yamu"], require_verification=True)

    def test_require_verification_raises_after_max_attempts(self):
        chars = self._chars()
        scene_result = {"scene_number": 1, "image_path": "a.png", "prompt": "p"}
        with patch.object(video_engine, "get_character",
                          side_effect=lambda n, p=None: chars[n]), \
             patch.object(video_engine, "generate_scene",
                          return_value=scene_result), \
             patch.object(video_engine, "verify_character_in_scene",
                          return_value=False):
            with self.assertRaises(RuntimeError):
                video_engine.create_validated_scene(
                    "p", character_names=["Yamu"], require_verification=True,
                    max_attempts=2)


class TestFaceRecognitionAvailable(unittest.TestCase):
    def test_face_recognition_installed(self):
        from utils.face_check import is_face_check_available
        self.assertTrue(is_face_check_available(),
                        "face_recognition must be installed for benchmarks")


class TestFaceBenchmark(unittest.TestCase):
    def test_random_characters_unique_and_seeded(self):
        import random
        from generators.benchmark_face_recognition import build_random_characters
        chars_a = build_random_characters(random.Random(42), 8)
        chars_b = build_random_characters(random.Random(42), 8)
        self.assertEqual(chars_a, chars_b)  # reproducible
        names = [c["name"] for c in chars_a]
        self.assertEqual(len(names), len(set(names)))  # unique
        for c in chars_a:
            self.assertIn(c["name"], c["prompt"])

    def test_summary_and_report(self):
        from generators import benchmark_face_recognition as bfr
        results = {
            "characters": [
                {"name": "A", "prompt": "p", "reference_image": "a.png",
                 "scene_prompt": "s", "scene_image": "sa.png",
                 "self_check": True, "scene_check": True},
                {"name": "B", "prompt": "p", "reference_image": "b.png",
                 "scene_prompt": "s", "scene_image": "sb.png",
                 "self_check": True, "scene_check": False},
                {"name": "C", "prompt": "p", "reference_image": "c.png",
                 "scene_prompt": "s", "scene_image": "sc.png",
                 "self_check": None, "scene_check": None},
            ],
            "cross_checks": [
                {"character": "A", "scene_of": "B", "scene_image": "sb.png",
                 "match": True},
                {"character": "B", "scene_of": "A", "scene_image": "sa.png",
                 "match": False},
            ],
        }
        summary = bfr.summarize(results)
        self.assertEqual(summary["self_check_pass"], 2)
        self.assertEqual(summary["self_check_total"], 2)
        self.assertEqual(summary["scene_check_pass"], 1)
        self.assertEqual(summary["scene_check_total"], 2)
        self.assertEqual(summary["false_positives"], 1)
        self.assertEqual(summary["cross_check_total"], 2)

        import tempfile, shutil
        from utils import project_paths
        tmp = tempfile.mkdtemp()
        old_root = project_paths.OUTPUTS_ROOT
        project_paths.OUTPUTS_ROOT = project_paths.Path(tmp)
        try:
            md_path = bfr.write_report(results, summary)
            content = md_path.read_text()
            self.assertIn("Face Recognition Benchmark Report", content)
            self.assertIn("INCONCLUSIVE", content)
            self.assertIn("False positives", content)
            self.assertTrue((md_path.parent / "report.json").exists())
        finally:
            project_paths.OUTPUTS_ROOT = old_root
            shutil.rmtree(tmp, ignore_errors=True)
