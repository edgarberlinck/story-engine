"""Face verification utilities.

Checks whether a character (via a reference image) appears in a generated
scene. Uses the optional `face_recognition` library (dlib-based). If it is
not installed, the check is skipped with a warning instead of failing.
"""

from typing import Optional


def is_face_check_available() -> bool:
    try:
        import face_recognition  # noqa: F401
        return True
    except ImportError:
        return False


def character_appears_in_image(
    reference_image_path: str,
    scene_image_path: str,
    tolerance: float = 0.6,
) -> Optional[bool]:
    """Compare the face in a character reference against faces in a scene.

    Args:
        reference_image_path: Path to the character's reference image.
        scene_image_path: Path to the generated scene image.
        tolerance: Face distance threshold (lower = stricter).

    Returns:
        True if a matching face is found, False if faces were detected in the
        scene but none matched the reference. Returns None (inconclusive) if the
        check cannot determine anything: face recognition is unavailable, no face
        is found in the reference, or no face is found in the scene (no detection
        does not prove the character is absent).
    """
    if not is_face_check_available():
        print(
            "Warning: `face_recognition` is not installed; skipping character "
            "verification. Install it with `pip install face_recognition`."
        )
        return None

    import os

    for label, path in (("reference", reference_image_path),
                        ("scene", scene_image_path)):
        if not os.path.isfile(path):
            print(f"Warning: {label} image not found: {path}; skipping check.")
            return None

    import face_recognition

    reference = face_recognition.load_image_file(reference_image_path)
    reference_encodings = face_recognition.face_encodings(reference)
    if not reference_encodings:
        print(f"Warning: no face found in reference image {reference_image_path}")
        return None

    scene = face_recognition.load_image_file(scene_image_path)
    scene_encodings = face_recognition.face_encodings(scene)
    if not scene_encodings:
        # No faces detected: this does not prove the character is absent (the
        # detector may simply fail to find a face), so the check is inconclusive.
        print(f"No faces detected in scene {scene_image_path}; check inconclusive.")
        return None

    matches = face_recognition.compare_faces(
        scene_encodings, reference_encodings[0], tolerance=tolerance
    )
    return any(matches)
