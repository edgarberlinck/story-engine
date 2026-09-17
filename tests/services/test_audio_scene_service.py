"""Tests for AudioSceneSegment persistence (accepted fields + old blobs)."""

import unittest

from services.audio_scene_service import AudioSceneSegment


class TestAcceptedFieldsRoundTrip(unittest.TestCase):
    def test_defaults(self):
        seg = AudioSceneSegment("dialogue", "nikita", "Hello.")
        self.assertFalse(seg.accepted)
        self.assertEqual(seg.accepted_hash, "")

    def test_round_trip(self):
        seg = AudioSceneSegment(
            "dialogue",
            "nikita",
            "Hello.",
            accepted=True,
            accepted_hash="abc123",
        )
        data = seg.to_dict()
        self.assertTrue(data["accepted"])
        self.assertEqual(data["accepted_hash"], "abc123")
        restored = AudioSceneSegment.from_dict(data)
        self.assertTrue(restored.accepted)
        self.assertEqual(restored.accepted_hash, "abc123")

    def test_old_blob_without_accept_fields_is_compatible(self):
        old = {
            "type": "dialogue",
            "speaker": "nikita",
            "text": "Hello.",
            "emotion": "warm",
            "tone": None,
            "intensity": None,
            "delivery": None,
            "voice": None,
            "timing": {},
            "sound_effects": [],
            "music": None,
            "speed": None,
        }
        seg = AudioSceneSegment.from_dict(old)
        self.assertFalse(seg.accepted)
        self.assertEqual(seg.accepted_hash, "")

    def test_none_accepted_hash_normalized(self):
        seg = AudioSceneSegment.from_dict(
            {
                "type": "dialogue",
                "speaker": "x",
                "text": "y",
                "accepted": None,
                "accepted_hash": None,
            }
        )
        self.assertFalse(seg.accepted)
        self.assertEqual(seg.accepted_hash, "")


if __name__ == "__main__":
    unittest.main()
