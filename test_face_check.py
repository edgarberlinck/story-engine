import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from utils import face_check


class FaceCheckTest(unittest.TestCase):
    def test_available_false_when_import_fails(self):
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "face_recognition":
                raise ImportError("missing")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            self.assertFalse(face_check.is_face_check_available())
            self.assertIsNone(face_check.character_appears_in_image("ref", "scene"))

    def test_missing_files_return_none_when_available(self):
        fake_fr = SimpleNamespace()
        with patch.dict(sys.modules, {"face_recognition": fake_fr}):
            self.assertTrue(face_check.is_face_check_available())
            self.assertIsNone(
                face_check.character_appears_in_image("/missing/ref", "/missing/scene")
            )

    def test_reference_and_scene_no_face_are_inconclusive(self):
        fake_fr = SimpleNamespace(
            load_image_file=lambda path: path,
            face_encodings=lambda image: [],
        )
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            sys.modules, {"face_recognition": fake_fr}
        ):
            ref = os.path.join(tmp, "ref.jpg")
            scene = os.path.join(tmp, "scene.jpg")
            open(ref, "w").close()
            open(scene, "w").close()
            self.assertIsNone(face_check.character_appears_in_image(ref, scene))

        calls = []

        def enc(image):
            calls.append(image)
            return ["refenc"] if image.endswith("ref.jpg") else []

        fake_fr = SimpleNamespace(load_image_file=lambda path: path, face_encodings=enc)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            sys.modules, {"face_recognition": fake_fr}
        ):
            ref = os.path.join(tmp, "ref.jpg")
            scene = os.path.join(tmp, "scene.jpg")
            open(ref, "w").close()
            open(scene, "w").close()
            self.assertIsNone(face_check.character_appears_in_image(ref, scene))

    def test_character_found_and_not_found(self):
        def run(matches):
            fake_fr = SimpleNamespace(
                load_image_file=lambda path: path,
                face_encodings=lambda image: (
                    ["ref"] if image.endswith("ref.jpg") else ["s1", "s2"]
                ),
                compare_faces=lambda scene_encodings, ref, tolerance=0.6: matches,
            )
            with tempfile.TemporaryDirectory() as tmp, patch.dict(
                sys.modules, {"face_recognition": fake_fr}
            ):
                ref = os.path.join(tmp, "ref.jpg")
                scene = os.path.join(tmp, "scene.jpg")
                open(ref, "w").close()
                open(scene, "w").close()
                return face_check.character_appears_in_image(ref, scene, tolerance=0.3)

        self.assertTrue(run([False, True]))
        self.assertFalse(run([False, False]))


if __name__ == "__main__":
    unittest.main()
