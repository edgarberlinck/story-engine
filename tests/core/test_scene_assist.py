"""
Tests for LLM scene assistance (plan §5: LLM edits the structured
representation, and the user reviews before anything is applied).
"""

import json
import unittest
from unittest.mock import patch

from core import scene_assist
from services.audio_scene_service import AudioSceneRepresentation


def _make_representation():
    rep = AudioSceneRepresentation(
        scene_id="scene_001",
        title="The Morning",
        timeline={"day": 1, "time": "08:30"},
        characters_present=["nikita", "roger"],
        location="central_square",
    )
    rep.add_segment(
        "narration",
        "narrator",
        "Nikita entered the kitchen.",
        emotion="neutral",
        tone="descriptive",
    )
    rep.add_segment(
        "dialogue",
        "nikita",
        "Good morning, Roger.",
        emotion="warm",
        tone="friendly",
        intensity=0.3,
        delivery="calm",
    )
    return rep


class TestSceneAssist(unittest.TestCase):
    def test_returns_none_when_llm_unavailable(self):
        rep = _make_representation()
        with patch.object(scene_assist, "LLM_AVAILABLE", False):
            self.assertIsNone(scene_assist.assist_scene(rep, "more tension"))

    def test_applies_llm_modification(self):
        rep = _make_representation()
        modified = rep.to_dict()
        modified["segments"][1]["emotion"] = "angry"
        modified["segments"][1]["intensity"] = 0.8
        with patch.object(scene_assist, "LLM_AVAILABLE", True), patch.object(
            scene_assist,
            "generate_text_with_llm",
            return_value=json.dumps(modified),
            create=True,
        ):
            proposal = scene_assist.assist_scene(rep, "Make Nikita sound angry")
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.segments[1].emotion, "angry")
        self.assertEqual(proposal.segments[1].intensity, 0.8)
        # Untouched fields survive.
        self.assertEqual(proposal.segments[0].text, "Nikita entered the kitchen.")

    def test_preserves_identity_fields_against_llm_drift(self):
        rep = _make_representation()
        modified = rep.to_dict()
        modified["scene_id"] = "totally_wrong"  # LLM must not change this
        del modified["characters_present"]  # LLM must not lose this
        with patch.object(scene_assist, "LLM_AVAILABLE", True), patch.object(
            scene_assist,
            "generate_text_with_llm",
            return_value=json.dumps(modified),
            create=True,
        ):
            proposal = scene_assist.assist_scene(rep, "anything")
        self.assertEqual(proposal.scene_id, "scene_001")
        self.assertEqual(proposal.characters_present, ["nikita", "roger"])

    def test_strips_unknown_segment_keys(self):
        rep = _make_representation()
        modified = rep.to_dict()
        modified["segments"][0]["hallucinated_field"] = "x"
        with patch.object(scene_assist, "LLM_AVAILABLE", True), patch.object(
            scene_assist,
            "generate_text_with_llm",
            return_value=json.dumps(modified),
            create=True,
        ):
            proposal = scene_assist.assist_scene(rep, "anything")
        self.assertNotIn("hallucinated_field", proposal.segments[0].to_dict())

    def test_rejects_garbage_output(self):
        rep = _make_representation()
        for bad in [
            None,
            "",
            "not json",
            json.dumps([1, 2]),
            json.dumps({"no": "segments"}),
        ]:
            with patch.object(scene_assist, "LLM_AVAILABLE", True), patch.object(
                scene_assist, "generate_text_with_llm", return_value=bad, create=True
            ):
                self.assertIsNone(scene_assist.assist_scene(rep, "x"))

    def test_handles_fenced_json(self):
        rep = _make_representation()
        fenced = "Here you go:\n```json\n" + json.dumps(rep.to_dict()) + "\n```"
        with patch.object(scene_assist, "LLM_AVAILABLE", True), patch.object(
            scene_assist, "generate_text_with_llm", return_value=fenced, create=True
        ):
            proposal = scene_assist.assist_scene(rep, "x")
        self.assertIsNotNone(proposal)
        self.assertEqual(len(proposal.segments), 2)


if __name__ == "__main__":
    unittest.main()
