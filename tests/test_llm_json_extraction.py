"""
Focused unit tests for the LLM structured-output plumbing that makes the
multi-character scene generator reliable:

- `_extract_json_blocks` in core/scene_planner.py strips ```json fences and
  prose lead-ins/tails so `json.loads` can parse the model's raw completion.
- `generate_text_with_llm` in generators/text_generator.py applies the chat
  template (so instruct models obey "return JSON only") and passes generation
  knobs as call arguments (so structured output is not silently truncated).

These tests mock the transformers pipeline so they run without model weights.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.scene_planner import _extract_json_blocks
import generators.text_generator as tg


class TestExtractJsonBlocks(unittest.TestCase):
    """The fence/lead-in stripper must turn raw completions into JSON."""

    def test_strips_json_fences(self):
        raw = '```json\n{"strategy": "asset_composition"}\n```'
        self.assertEqual(_extract_json_blocks(raw), '{"strategy": "asset_composition"}')

    def test_plain_object_passthrough(self):
        raw = '{"strategy": "progressive"}'
        self.assertEqual(_extract_json_blocks(raw), raw)

    def test_trims_prose_lead_in(self):
        raw = 'Here is the answer:\n{"strategy": "single_pass"}'
        self.assertEqual(_extract_json_blocks(raw), '{"strategy": "single_pass"}')

    def test_trims_prose_tail_after_object(self):
        raw = '{"strategy": "single_pass"}\nHope that helps!'
        self.assertEqual(_extract_json_blocks(raw), '{"strategy": "single_pass"}')

    def test_array_inside_fences(self):
        raw = '```\n[{"name": "A"}, {"name": "B"}]\n```'
        self.assertEqual(
            _extract_json_blocks(raw), '[{"name": "A"}, {"name": "B"}]'
        )

    def test_empty_input(self):
        self.assertEqual(_extract_json_blocks(""), "")
        self.assertIsNone(_extract_json_blocks(None))

    def test_round_trips_realistic_stage_a(self):
        # A realistic phi3-mini Stage A completion (complete, both characters).
        raw = (
            "```json\n"
            "[\n"
            '  {"name": "Nikita", "identity": ["young woman"], '
            '"presentation_decision": "REPLACE", "scene_pose": "sitting on a chair", '
            '"scene_action": "playing a black Gibson Explorer guitar", '
            '"scene_position_hint": "left side of the stage", "dropped": []},\n'
            '  {"name": "Roger", "identity": ["muscular man"], '
            '"presentation_decision": "ADAPT", "scene_pose": "behind drum kit", '
            '"scene_action": "playing drums", "dropped": []}\n'
            "]\n"
            "```"
        )
        import json
        parsed = json.loads(_extract_json_blocks(raw))
        self.assertEqual([c["name"] for c in parsed], ["Nikita", "Roger"])


class TestGenerateTextWithLlm(unittest.TestCase):
    """The wrapper must apply the chat template and honor max_new_tokens."""

    def _pipeline_mock(self, tokenizer, generated_text):
        gen = MagicMock()
        gen.tokenizer = tokenizer
        gen.model.generation_config = MagicMock()
        gen.return_value = [{"generated_text": generated_text}]
        return gen

    def test_returns_model_completion_and_passes_call_args(self):
        tokenizer = MagicMock()
        tokenizer.chat_template = "phi3"
        tokenizer.apply_chat_template.return_value = "<user>hi</user><assistant>"
        gen = self._pipeline_mock(tokenizer, '{"strategy": "asset_composition"}')
        # Avoid the finally-block cleanup touching real imports.
        with patch("generators.text_generator.resolve_model_path", return_value="m"), \
             patch("generators.text_generator.get_model_config", return_value=("cpu", "float32")), \
             patch("generators.text_generator.hf_pipeline", return_value=gen), \
             patch("generators.image_generator.cleanup_pipeline"):
            out = tg.generate_text_with_llm(
                "Return JSON only", model_name="phi3_mini", max_new_tokens=512
            )
        self.assertEqual(out, '{"strategy": "asset_composition"}')
        # The generation knobs must be passed to the call, not ignored.
        _, kwargs = gen.call_args
        self.assertEqual(kwargs["max_new_tokens"], 512)
        self.assertIs(kwargs["do_sample"], False)
        # Chat template applied so the instruct model obeys instructions.
        tokenizer.apply_chat_template.assert_called_once()
        gen.assert_called_once()

    def test_falls_back_to_raw_prompt_without_chat_template(self):
        tokenizer = MagicMock()
        tokenizer.chat_template = None  # e.g. a base (non-instruct) model
        gen = self._pipeline_mock(tokenizer, "some output")
        with patch("generators.text_generator.resolve_model_path", return_value="m"), \
             patch("generators.text_generator.get_model_config", return_value=("cpu", "float32")), \
             patch("generators.text_generator.hf_pipeline", return_value=gen), \
             patch("generators.image_generator.cleanup_pipeline"):
            out = tg.generate_text_with_llm("raw prompt", model_name="phi3_mini")
        self.assertEqual(out, "some output")
        # Raw prompt forwarded unchanged.
        self.assertIn("raw prompt", gen.call_args.args[0])

    def test_returns_none_on_failure(self):
        def boom(*a, **k):
            raise RuntimeError("load failed")

        with patch("generators.text_generator.resolve_model_path", side_effect=boom), \
             patch("generators.image_generator.cleanup_pipeline"):
            out = tg.generate_text_with_llm("anything", model_name="phi3_mini")
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
