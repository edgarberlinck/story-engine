"""
Unit tests for core.prompt_builder (SemanticPromptBuilder + module helpers).

Run with stdlib unittest.  Any LLM extraction is mocked so no real model /
network is touched — only the prompt-building logic under test is exercised.
"""

import unittest
from unittest.mock import patch, MagicMock

from core.scene_semantics import (
    SceneComposition,
    ScenePromptSpec,
    CharacterIdentity,
    CharacterSceneData,
)
from core.prompt_builder import (
    SemanticPromptBuilder,
    create_optimized_scene_prompt,
    example_usage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nick():
    return CharacterIdentity(
           gender="young woman",
           age_range="25-35",
           hair_color="red",
           hairstyle="curly long",
           skin_tone="fair",
        )


def _roger():
    return CharacterIdentity(
           gender="muscular man",
           skin_tone="dark",
           hairstyle="short bald",
        )


def _make_spec(environment="", scene=None, chars=None, style="photorealistic"):
    if scene is None:
        scene = SceneComposition(
                environment=environment,
             camera_position="back of the bar",
             shot_type="wide shot",
             spatial_composition="audience in foreground",
             lighting="dim",
             atmosphere="moody",
          )
    if chars is None:
        chars = [
             CharacterSceneData(
                  "Nikita", _nick(), "left side of the stage",
                  "black suit", "playing a black Gibson Explorer"),
             CharacterSceneData(
                  "Roger", _roger(), "right side of the stage",
                  "dark formal outfit", "sitting behind a drum kit"),
          ]
    return ScenePromptSpec(scene=scene, characters=chars, style=style)


def _fake_extractor_class(returned_spec):
    """Return a replacement for LLMSceneSemanticExtractor producing a fixed spec."""
    class _FakeExtractor:
        def __init__(self, *a, **k):
            pass

        def extract_semantic_spec(self, scene_description, character_records,
                                   project_style="photorealistic"):
            return returned_spec

    return _FakeExtractor


class TestBuildCompositionPrompt(unittest.TestCase):

    def test_with_full_scene(self):
        spec = _make_spec(environment="an old bar")
        prompt = SemanticPromptBuilder.build_composition_prompt(spec)
        self.assertIn("an old bar", prompt)
        self.assertIn("wide shot", prompt)
        self.assertIn("from back of the bar", prompt)
        self.assertIn("audience in foreground", prompt)
        self.assertIn("lighting: dim", prompt)
        self.assertIn("atmosphere: moody", prompt)
        self.assertTrue(prompt.endswith("."))

    def test_minimal_scene_no_lighting_atmosphere(self):
        scene = SceneComposition(
                environment="a field",
             camera_position="front",
             shot_type="close up",
             spatial_composition="",
             lighting=None,
             atmosphere=None,
           )
        spec = ScenePromptSpec(scene=scene, characters=[])
        prompt = SemanticPromptBuilder.build_composition_prompt(spec)
        self.assertIn("a field", prompt)
        self.assertIn("close up shot", prompt)
        self.assertNotIn("lighting:", prompt)
        self.assertNotIn("atmosphere:", prompt)

    def test_no_camera_position(self):
        scene = SceneComposition(
                environment="here",
             camera_position="",
             shot_type="wide shot",
             spatial_composition="",
           )
        spec = ScenePromptSpec(scene=scene, characters=[])
        prompt = SemanticPromptBuilder.build_composition_prompt(spec)
           # No "from <empty>" fragment should appear.
        self.assertNotIn("from   ", prompt)
        self.assertNotIn("from None", prompt)

    def test_none_scene_returns_dot(self):
        spec = _make_spec()
        spec.scene = None
        prompt = SemanticPromptBuilder.build_composition_prompt(spec)
        self.assertEqual(prompt, ".")

    def test_scene_present_without_environment(self):
        scene = SceneComposition(
                environment="",
             camera_position="low angle",
             shot_type="medium shot",
             spatial_composition="centered",
             lighting="bright",
             atmosphere="clear",
           )
        spec = ScenePromptSpec(scene=scene, characters=[])
        prompt = SemanticPromptBuilder.build_composition_prompt(spec)
        self.assertIn("medium shot", prompt)
        self.assertIn("from low angle", prompt)
        self.assertIn("lighting: bright", prompt)


class TestBuildCharacterIdentityPrompt(unittest.TestCase):

    def test_basic(self):
        char = CharacterSceneData(
                "Nikita", _nick(), "left", "suit", "playing guitar")
        prompt = SemanticPromptBuilder.build_character_identity_prompt(char)
        self.assertTrue(prompt.startswith("Nikita,"))
        self.assertIn("fair skin", prompt)

    def test_empty_identity(self):
        char = CharacterSceneData(
                "X", CharacterIdentity(), "left", "suit", "acting")
        prompt = SemanticPromptBuilder.build_character_identity_prompt(char)
        # "X, " - identity description is empty, but name prefix retained.
        self.assertTrue(prompt.startswith("X,"))


class TestBuildSceneAppearanceActionPrompt(unittest.TestCase):

    def test_all_fields(self):
        char = CharacterSceneData(
                "Nikita", _nick(), "left side", "black suit", "playing guitar")
        prompt = SemanticPromptBuilder.build_scene_appearance_action_prompt(char)
        self.assertIn("positioned left side", prompt)
        self.assertIn("wearing black suit", prompt)
        self.assertIn("playing guitar", prompt)

    def test_none_scene_appearance_action(self):
        char = CharacterSceneData(
                "X", CharacterIdentity(), "", "", "")
        prompt = SemanticPromptBuilder.build_scene_appearance_action_prompt(char)
        self.assertEqual(prompt, "")

    def test_only_action(self):
        char = CharacterSceneData(
                "X", CharacterIdentity(), "", "", "dancing")
        prompt = SemanticPromptBuilder.build_scene_appearance_action_prompt(char)
        self.assertEqual(prompt, "dancing")

    def test_only_clothing(self):
        char = CharacterSceneData(
                "X", CharacterIdentity(), "", "gown", "")
        prompt = SemanticPromptBuilder.build_scene_appearance_action_prompt(char)
        self.assertEqual(prompt, "wearing gown")

    def test_only_position(self):
        char = CharacterSceneData(
                "X", CharacterIdentity(), "center", "", "")
        prompt = SemanticPromptBuilder.build_scene_appearance_action_prompt(char)
        self.assertEqual(prompt, "positioned center")


class TestBuildExplicitIdentityPrompt(unittest.TestCase):

    def test_composition_plus_characters_plus_style(self):
        spec = _make_spec(environment="an old bar")
        prompt = SemanticPromptBuilder.build_explicit_identity_prompt(spec)
        self.assertIn("an old bar", prompt)
        self.assertIn("Nikita", prompt)
        self.assertIn("Roger", prompt)
        self.assertIn("style: photorealistic", prompt)
          # Explicit association: "On <position>".
        self.assertIn("On left side of the stage", prompt)

    def test_clothing_and_action_verb_form(self):
        spec = _make_spec()
        prompt = SemanticPromptBuilder.build_explicit_identity_prompt(spec)
         # "is wearing <clothing> and <action>"
        self.assertIn("is wearing black suit and playing a black Gibson Explorer", prompt)

    def test_only_action_no_clothing(self):
        char = CharacterSceneData(
                "Nikita", _nick(), "left", "", "playing guitar")
        spec = _make_spec(chars=[char])
        prompt = SemanticPromptBuilder.build_explicit_identity_prompt(spec)
        self.assertIn("is playing guitar", prompt)
        self.assertNotIn("is wearing", prompt)

    def test_neither_action_nor_clothing(self):
        char = CharacterSceneData(
                "Nikita", _nick(), "left", "", "")
        spec = _make_spec(chars=[char])
        prompt = SemanticPromptBuilder.build_explicit_identity_prompt(spec)
         # Falls back to "Nikita, <identity>".
        self.assertIn("Nikita, ", prompt)
        self.assertNotIn("is wearing", prompt)
        self.assertNotIn("is ", prompt)

    def test_missing_position_uses_in_the_scene(self):
        char = CharacterSceneData(
                "Nikita", _nick(), "", "suit", "playing guitar")
        spec = _make_spec(chars=[char])
        prompt = SemanticPromptBuilder.build_explicit_identity_prompt(spec)
        self.assertIn("In the scene", prompt)

    def test_no_style_omits_style_phrase(self):
        spec = _make_spec(style="")
        prompt = SemanticPromptBuilder.build_explicit_identity_prompt(spec)
        self.assertNotIn("style:", prompt)

    def test_none_scene_handled(self):
        spec = _make_spec()
        spec.scene = None
        prompt = SemanticPromptBuilder.build_explicit_identity_prompt(spec)
        self.assertIn("Nikita", prompt)


class TestBuildLayeredPrompts(unittest.TestCase):

    def test_returns_all_layers(self):
        spec = _make_spec(environment="an old bar")
        layers = SemanticPromptBuilder.build_layered_prompts(spec)
        self.assertIn("composition", layers)
        self.assertIn("identities", layers)
        self.assertIn("appearances_actions", layers)
        self.assertIn("full_explicit", layers)
        self.assertIn("Nikita", layers["full_explicit"])
         # identities are "Name, <identity_desc>" joined by " | ".
        self.assertIn("Nikita,", layers["identities"])
         # appearances_actions are "Name: <appearance>".
        self.assertIn("Nikita: ", layers["appearances_actions"])
        self.assertNotEqual(layers["composition"], layers["identities"])

    def test_identities_joined_by_pipe(self):
        spec = _make_spec()
        layers = SemanticPromptBuilder.build_layered_prompts(spec)
        self.assertIn(" | ", layers["identities"])
        self.assertIn("Nikita", layers["identities"])
        self.assertIn("Roger", layers["identities"])


class TestOptimizeForTokenBudget(unittest.TestCase):

    def test_within_budget_returns_full(self):
        spec = _make_spec(environment="a bar")
        builder = SemanticPromptBuilder()
        result = builder.optimize_for_token_budget(spec, max_tokens=100000)
         # Within budget -> the full explicit prompt.
        full = builder.build_explicit_identity_prompt(spec)
        self.assertEqual(result, full)

    def test_over_budget_returns_minimal(self):
        spec = _make_spec(environment="a very long old bar with many details")
        builder = SemanticPromptBuilder()
        result = builder.optimize_for_token_budget(spec, max_tokens=3)
          # Over budget -> minimal version listing chars + environment.
        self.assertIn("Nikita", result)
        self.assertIn("Roger", result)
        self.assertIn("a very long old bar with many details", result)
          # The minimal build does NOT contain the full explicit prompt.
        self.assertNotEqual(result, builder.build_explicit_identity_prompt(spec))

    def test_over_budget_no_environment(self):
        spec = _make_spec(environment="")
        spec.scene.environment = ""
        builder = SemanticPromptBuilder()
        result = builder.optimize_for_token_budget(spec, max_tokens=3)
        self.assertIn("Nikita", result)
        self.assertNotIn("|", result)


class TestCreateOptimizedScenePrompt(unittest.TestCase):

    def test_fits_budget_returns_full(self):
         # Patch the LLM extractor (constructed inside the function) to return
        # a small spec whose full prompt fits the budget.
        small_spec = _make_spec(environment="small bar")
        with patch("core.scene_semantics.LLMSceneSemanticExtractor",
                   _fake_extractor_class(small_spec)):
            result = create_optimized_scene_prompt(
                  "scene desc",
                  [{"name": "Nikita", "prompt": "..."}],
                  project_style="photorealistic",
                  max_tokens=100000)
            full = SemanticPromptBuilder().build_explicit_identity_prompt(small_spec)
            self.assertEqual(result, full)

    def test_exceeds_budget_optimizes(self):
        big_spec = _make_spec(environment="a very long bar with many things")
        with patch("core.scene_semantics.LLMSceneSemanticExtractor",
                   _fake_extractor_class(big_spec)):
            result = create_optimized_scene_prompt(
                  "scene desc",
                  [{"name": "Nikita", "prompt": "..."}],
                  project_style="photorealistic",
                  max_tokens=3)
            self.assertIn("Nikita", result)
            self.assertIn("Roger", result)


class TestExampleUsage(unittest.TestCase):

    def test_example_usage_returns_string(self):
         # Mock the LLM extractor so example_usage does not hit a real model.
        with patch("core.scene_semantics.LLMSceneSemanticExtractor",
                   _fake_extractor_class(_make_spec(environment="an old bar"))):
            prompt = example_usage()
        self.assertIsInstance(prompt, str)
        self.assertIn("Nikita", prompt)


if __name__ == "__main__":
    unittest.main()
