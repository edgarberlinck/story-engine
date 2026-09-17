import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from generators import text_generator as tg


class _FakeGenerator:
    def __init__(self, text="expanded prompt"):
        self.model = SimpleNamespace(generation_config=SimpleNamespace())
        self.tokenizer = None
        self.calls = []
        self.text = text

    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return [{"generated_text": self.text}]


class TextGeneratorLLMTest(unittest.TestCase):
    def test_generate_prompt_with_llm_returns_first_line_and_cleans_up(self):
        fake = _FakeGenerator("rich cinematic prompt\nextra")
        with patch.object(
            tg, "hf_logging", SimpleNamespace(set_verbosity_error=MagicMock())
        ), patch.object(
            tg, "resolve_model_path", return_value="local/model"
        ), patch.object(
            tg, "get_model_config", return_value=("cpu", "float16")
        ), patch.object(
            tg, "hf_pipeline", return_value=fake
        ) as pipeline, patch(
            "generators.image_generator.cleanup_pipeline"
        ) as cleanup:
            out = tg.generate_prompt_with_llm("small cat")

        self.assertEqual(out, "rich cinematic prompt")
        pipeline.assert_called_once()
        cleanup.assert_called_once_with(fake)
        self.assertIn("Description: small cat", fake.calls[0][0])
        self.assertFalse(fake.model.generation_config.do_sample)

    def test_generate_prompt_with_llm_falls_back_on_failure(self):
        with patch.object(
            tg, "hf_logging", SimpleNamespace(set_verbosity_error=MagicMock())
        ), patch.object(
            tg, "resolve_model_path", side_effect=RuntimeError("boom")
        ), patch(
            "generators.image_generator.cleanup_pipeline"
        ) as cleanup:
            self.assertEqual(tg.generate_prompt_with_llm("raw"), "raw")
        cleanup.assert_called_once_with(None)

    def test_generate_text_with_llm_uses_chat_template_and_temperature(self):
        fake = _FakeGenerator(' {"ok": true} ')
        fake.tokenizer = SimpleNamespace(
            chat_template="template",
            apply_chat_template=MagicMock(return_value="CHAT:prompt"),
        )
        with patch.object(
            tg, "hf_logging", SimpleNamespace(set_verbosity_error=MagicMock())
        ), patch.object(
            tg, "resolve_model_path", return_value="local/model"
        ), patch.object(
            tg, "get_model_config", return_value=("cpu", "float16")
        ), patch.object(
            tg, "hf_pipeline", return_value=fake
        ), patch(
            "generators.image_generator.cleanup_pipeline"
        ):
            out = tg.generate_text_with_llm(
                "prompt", max_new_tokens=77, temperature=0.4
            )

        self.assertEqual(out, '{"ok": true}')
        fake.tokenizer.apply_chat_template.assert_called_once()
        sent_prompt, kwargs = fake.calls[0]
        self.assertEqual(sent_prompt, "CHAT:prompt")
        self.assertEqual(kwargs["max_new_tokens"], 77)
        self.assertTrue(kwargs["do_sample"])
        self.assertEqual(kwargs["temperature"], 0.4)

    def test_generate_text_with_llm_returns_none_for_empty_or_exception(self):
        fake = _FakeGenerator("   ")
        with patch.object(
            tg, "hf_logging", SimpleNamespace(set_verbosity_error=MagicMock())
        ), patch.object(
            tg, "resolve_model_path", return_value="local/model"
        ), patch.object(
            tg, "get_model_config", return_value=("cpu", "float16")
        ), patch.object(
            tg, "hf_pipeline", return_value=fake
        ), patch(
            "generators.image_generator.cleanup_pipeline"
        ):
            self.assertIsNone(tg.generate_text_with_llm("prompt"))

        with patch.object(
            tg, "hf_logging", SimpleNamespace(set_verbosity_error=MagicMock())
        ), patch.object(tg, "resolve_model_path", side_effect=Exception("nope")), patch(
            "generators.image_generator.cleanup_pipeline"
        ):
            self.assertIsNone(tg.generate_text_with_llm("prompt"))

    def test_resolve_model_path_prefers_nonempty_local_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(tg, "project_root", tmp), patch(
                "models.MODEL_PATHS", {"text_generation": "models/text"}
            ):
                local = os.path.join(tmp, "models/text/phi")
                os.makedirs(local)
                self.assertEqual(
                    tg.resolve_model_path("text_generation", "phi", "hub"), "hub"
                )
                with open(os.path.join(local, "config.json"), "w") as f:
                    f.write("{}")
                self.assertEqual(
                    tg.resolve_model_path("text_generation", "phi", "hub"), local
                )


if __name__ == "__main__":
    unittest.main()
