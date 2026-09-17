"""Tests for stable per-character voice resolution (core/voice_resolver.py)."""

import unittest
from unittest.mock import patch

from core import voice_resolver
from core.voice_resolver import ResolvedVoice, resolve_character_voice


class TestVoiceResolver(unittest.TestCase):
    def setUp(self):
        voice_resolver.clear_voice_cache()

    def tearDown(self):
        voice_resolver.clear_voice_cache()

    def test_designed_voice_used_when_voice_prompt_exists(self):
        record = {"name": "Nikita", "voice_prompt": "a warm husky alto"}
        with patch(
            "services.database.character_service.character_service" ".get_character",
            return_value=record,
        ):
            resolved = resolve_character_voice("p", "Nikita")
        self.assertEqual(resolved.kind, "designed")
        self.assertEqual(resolved.voice_prompt, "a warm husky alto")
        self.assertIn("a warm husky alto", resolved.identity)

    def test_preset_pinned_from_attributes(self):
        attrs = {"type": "woman", "gender": "Female"}
        with patch(
            "services.database.character_service.character_service" ".get_character",
            return_value=None,
        ), patch(
            "services.database.character_attribute_service"
            ".character_attribute_service.get_attributes",
            return_value=attrs,
        ):
            resolved = resolve_character_voice("p", "Nikita")
        self.assertEqual(resolved.kind, "preset")
        self.assertEqual(resolved.speaker, "serena")  # female default preset

    def test_resolution_is_stable_across_fragments(self):
        """The second call must NOT hit the services again (pinned voice)."""
        with patch(
            "services.database.character_service.character_service" ".get_character",
            return_value=None,
        ) as get_char, patch(
            "services.database.character_attribute_service"
            ".character_attribute_service.get_attributes",
            return_value={"gender": "Female"},
        ) as get_attrs:
            first = resolve_character_voice("p", "Nikita")
            second = resolve_character_voice("p", "nikita")  # case-insensitive
        self.assertEqual(first, second)
        self.assertEqual(get_char.call_count, 1)
        self.assertEqual(get_attrs.call_count, 1)

    def test_attribute_failure_falls_back_deterministically_and_pins(self):
        """A transient failure must not flip the voice for one fragment."""
        with patch(
            "services.database.character_service.character_service" ".get_character",
            return_value=None,
        ), patch(
            "services.database.character_attribute_service"
            ".character_attribute_service.get_attributes",
            side_effect=RuntimeError("db down"),
        ):
            with self.assertLogs("core.voice_resolver", level="WARNING"):
                failed = resolve_character_voice("p", "Roger")
        self.assertEqual(failed.kind, "preset")
        self.assertEqual(failed.speaker, "ryan")  # deterministic default

        # Even when the DB recovers, the pinned resolution is reused.
        with patch(
            "services.database.character_attribute_service"
            ".character_attribute_service.get_attributes",
            return_value={"gender": "Female"},
        ):
            again = resolve_character_voice("p", "Roger")
        self.assertEqual(again, failed)

    def test_identity_differs_between_kinds(self):
        a = ResolvedVoice(kind="preset", speaker="ryan")
        b = ResolvedVoice(kind="designed", voice_prompt="ryan")
        self.assertNotEqual(a.identity, b.identity)


if __name__ == "__main__":
    unittest.main()
