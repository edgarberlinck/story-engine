"""Unit tests for LLM scene planning and semantic scene extraction."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.scene_planner import (
    LLMScenePlanner,
    ResolvedCharacter,
    SceneLayerPlan,
    ScenePlan,
    StrategyDecision,
    STRATEGY_ASSET_COMPOSITION,
    STRATEGY_PROGRESSIVE,
    STRATEGY_SINGLE_PASS,
    _extract_character_snippets,
    _pre_llm_strategy_gate,
    create_llm_scene_plan,
    stage_a_context_resolution,
    stage_b_decompose_scene,
    stage_c_token_budget_fit,
    stage_d_select_strategy,
)
from core.scene_semantics import (
    CharacterIdentity,
    CharacterSceneData,
    LLMSceneSemanticExtractor,
    SceneComposition,
    SemanticSceneExtractor,
)


def _stage_a_json():
    return "```json\n" + json.dumps([
        {
            "name": "Nikita",
            "identity": ["young woman", "long black hair", "fair skin"],
            "default_presentation": ["ornate elven armor"],
            "presentation_decision": "REPLACE",
            "scene_presentation": ["black suit"],
            "scene_pose": "sitting on a chair",
            "scene_action": "playing a black Gibson Explorer guitar",
            "scene_position_hint": "left side of the stage",
            "dropped": ["ornate elven armor"],
            "dropped_reason": "bar scene overrides fantasy clothing",
        },
        {
            "name": "Roger",
            "identity": ["adult man", "muscular build"],
            "default_presentation": [],
            "presentation_decision": "KEEP",
            "scene_presentation": ["dark formal outfit"],
            "scene_pose": "behind drum kit",
            "scene_action": "playing drums",
            "scene_position_hint": "right side of the stage",
            "dropped": [],
            "dropped_reason": "",
        },
    ]) + "\n```"


def _stage_b_json():
    return "Here is the plan:\n```json\n" + json.dumps({
        "camera": "wide shot from the back of the bar",
        "layers": [
            {
                "name": "base_environment",
                "prompt": "small empty stage with red curtains and cracked wood, cinematic lighting",
                "must_include": ["red curtains", "cracked wood"],
                "region_hint": "full frame",
            },
            {
                "name": "character_Nikita",
                "prompt": "Nikita, young woman, long black hair, black suit, playing guitar, cinematic lighting",
                "must_include": ["long black hair"],
                "region_hint": "left third",
                "depends_on": "base_environment",
            },
        ],
        "single_pass_feasible": False,
        "rationale": "multiple subjects require layers",
        "canvas_layout": {"width": 1024, "height": 1024, "placements": [{"name": "Nikita", "anchor": [0.3, 0.85], "scale": 0.5, "z": 1}]},
    }) + "\n```"


class TestLLMScenePlannerStages(unittest.TestCase):
    def setUp(self):
        self.scene = (
            "There's a very small stage with red curtains. Nikita is sitting on a chair "
            "playing a black Gibson Explorer guitar. Roger is on the right playing drums."
        )
        self.characters = [
            {"name": "Nikita", "prompt": "young woman, long black hair, ornate elven armor"},
            {"name": "Roger", "prompt": "adult man, muscular build"},
        ]

    def test_stage_a_parses_fenced_json_with_optional_asset_fields(self):
        with patch("core.scene_planner.LLM_AVAILABLE", True), \
             patch("core.scene_planner.generate_text_with_llm", return_value=_stage_a_json()) as mock_llm:
            resolved = stage_a_context_resolution(self.scene, self.characters)

        self.assertEqual([c.name for c in resolved], ["Nikita", "Roger"])
        self.assertEqual(resolved[0].scene_pose, "sitting on a chair")
        self.assertEqual(resolved[0].scene_position_hint, "left side of the stage")
        self.assertIn("ornate elven armor", resolved[0].dropped)
        mock_llm.assert_called_once()

    def test_stage_a_fallback_uses_scene_snippets_not_stored_prompt(self):
        with patch("core.scene_planner.LLM_AVAILABLE", True), \
             patch("core.scene_planner.generate_text_with_llm", return_value=None):
            resolved = stage_a_context_resolution(
                "Nikita, a woman with long black hair. She wears a black suit. Roger plays drums.",
                [{"name": "Nikita", "prompt": "fantasy armor"}, {"name": "Roger", "prompt": "man"}],
            )

        self.assertEqual(resolved[0].identity, ["a woman with long black hair"])
        self.assertEqual(resolved[0].scene_presentation, ["She wears a black suit"])
        self.assertEqual(resolved[0].presentation_decision, "ADAPT")
        self.assertNotIn("fantasy armor", " ".join(resolved[0].identity + resolved[0].scene_presentation))

    def test_character_snippet_stops_when_another_character_appears(self):
        snippets = _extract_character_snippets(
            "Nikita, a guitarist. She smiles. Roger, a drummer. He waves.",
            "Nikita",
            ["Nikita", "Roger"],
        )
        self.assertEqual(snippets, ["a guitarist", "She smiles"])

    def test_stage_b_parses_layers_canvas_layout_and_dependency(self):
        resolved = [ResolvedCharacter("Nikita", ["young woman"], [], "KEEP", [], [], "")]
        with patch("core.scene_planner.generate_text_with_llm", return_value=_stage_b_json()):
            plan = stage_b_decompose_scene(self.scene, resolved, project_style="noir")
        self.assertIsInstance(plan, ScenePlan)
        self.assertFalse(plan.single_pass_feasible)
        self.assertEqual(plan.layers[0].name, "base_environment")
        self.assertEqual(plan.layers[1].depends_on, "base_environment")
        self.assertEqual(plan.canvas_layout["placements"][0]["name"], "Nikita")

    def test_stage_b_returns_simple_fallback_on_bad_llm_json(self):
        with patch("core.scene_planner.generate_text_with_llm", return_value="not json"):
            plan = stage_b_decompose_scene("A quiet empty room", [])
        self.assertTrue(plan.single_pass_feasible)
        self.assertEqual(plan.camera, "wide shot")
        self.assertEqual(plan.layers[0].prompt, "A quiet empty room")

    def test_stage_c_returns_original_when_within_budget(self):
        layer = SceneLayerPlan("short", "short prompt", ["short"])
        self.assertIs(stage_c_token_budget_fit(layer, token_limit=77), layer)
        self.assertIsNone(layer.original_prompt)

    def test_stage_c_uses_llm_compression_when_under_limit(self):
        layer = SceneLayerPlan("long", "one two three four five six seven eight nine ten")
        with patch("core.scene_planner.count_tokens", side_effect=[10, 3]), \
             patch("core.scene_planner.generate_text_with_llm", return_value="one two three"):
            fitted = stage_c_token_budget_fit(layer, token_limit=4)
        self.assertEqual(fitted.prompt, "one two three")
        self.assertIn("one two three four", fitted.original_prompt)

    def test_stage_c_falls_back_to_token_truncation(self):
        layer = SceneLayerPlan("long", "one two three four five six seven eight nine ten")
        with patch("core.scene_planner.generate_text_with_llm", side_effect=RuntimeError("boom")):
            fitted = stage_c_token_budget_fit(layer, token_limit=4)
        self.assertLessEqual(len(fitted.prompt.split()), 4)
        self.assertIsNotNone(fitted.original_prompt)

    def test_stage_d_pre_llm_gate_and_llm_decision(self):
        self.assertEqual(_pre_llm_strategy_gate(0), STRATEGY_SINGLE_PASS)
        self.assertEqual(_pre_llm_strategy_gate(1, over_clip_budget=True), STRATEGY_PROGRESSIVE)
        self.assertIsNone(_pre_llm_strategy_gate(2))
        chars = [ResolvedCharacter("A", [], [], "KEEP", [], [], ""), ResolvedCharacter("B", [], [], "KEEP", [], [], "")]
        with patch("core.scene_planner.LLM_AVAILABLE", True), \
             patch("core.scene_planner.generate_text_with_llm", return_value='```json\n{"strategy":"progressive","reason":"loose"}\n```'):
            decision = stage_d_select_strategy("A and B play together", chars)
        self.assertEqual(decision.strategy, STRATEGY_PROGRESSIVE)
        self.assertEqual(decision.reason, "loose")

    def test_stage_d_invalid_llm_defaults_to_asset_for_spatial_scene(self):
        chars = [
            ResolvedCharacter("A", [], [], "KEEP", [], [], "", scene_position_hint="left"),
            ResolvedCharacter("B", [], [], "KEEP", [], [], "", scene_position_hint="right"),
        ]
        with patch("core.scene_planner.LLM_AVAILABLE", True), \
             patch("core.scene_planner.generate_text_with_llm", return_value='{"strategy":"nonsense"}'):
            decision = stage_d_select_strategy("A left, B right", chars)
        self.assertEqual(decision.strategy, STRATEGY_ASSET_COMPOSITION)
        self.assertTrue(decision.requires_spatial_precision)
        self.assertTrue(decision.heuristic)

    def test_full_planner_orchestrates_all_stages_and_saves_plan(self):
        responses = [_stage_a_json(), _stage_b_json(), '{"strategy":"asset_composition","reason":"positions"}']
        with patch("core.scene_planner.LLM_AVAILABLE", True), \
             patch("core.scene_planner.generate_text_with_llm", side_effect=responses):
            plan = LLMScenePlanner().plan_scene(self.scene, self.characters, "cinematic")
        self.assertEqual(plan.strategy, STRATEGY_ASSET_COMPOSITION)
        self.assertEqual(len(plan.resolved_characters), 2)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "plan.json"
            LLMScenePlanner().save_plan(plan, out)
            saved = json.loads(out.read_text())
        self.assertEqual(saved["strategy"], STRATEGY_ASSET_COMPOSITION)
        self.assertEqual(saved["resolved_characters"][0]["scene_action"], "playing a black Gibson Explorer guitar")

    def test_planner_disabled_returns_empty_plan(self):
        plan = LLMScenePlanner(use_llm=False).plan_scene("anything", [])
        self.assertEqual(plan.rationale, "LLM disabled")
        self.assertEqual(plan.layers, [])

    def test_create_llm_scene_plan_gets_style_and_swallows_save_errors(self):
        fake_plan = ScenePlan("cam", [], True, "ok")
        with patch("services.database.project_service.get_project", return_value={"style": "watercolor"}, create=True), \
             patch.object(LLMScenePlanner, "plan_scene", return_value=fake_plan) as mock_plan, \
             patch.object(LLMScenePlanner, "save_plan", side_effect=OSError("no write")):
            out = create_llm_scene_plan("scene", [], project_name="proj")
        self.assertIs(out, fake_plan)
        self.assertEqual(mock_plan.call_args.args[2], "watercolor")


class TestSceneSemantics(unittest.TestCase):
    def test_character_identity_descriptions_and_lists(self):
        ident = CharacterIdentity(
            name="Nikita", gender="woman", age_range="young", hair_color="black",
            hairstyle="long", skin_tone="fair", body_type="slender",
            facial_traits=["green eyes"], permanent_traits=["small scar"],
        )
        self.assertEqual(
            ident.to_concise_description(),
            "young woman, with long black hair, fair skin, slender build, green eyes",
        )
        self.assertEqual(
            ident.to_list(),
            ["woman", "young", "long black hair", "fair", "slender build", "green eyes", "small scar"],
        )

    def test_character_identity_handles_partial_hair_and_age_only(self):
        self.assertEqual(CharacterIdentity(age_range="adult", hair_color="red").to_concise_description(), "adult, with red hair")
        self.assertEqual(CharacterIdentity(gender="man", hairstyle="curly long").to_list(), ["man"])

    def test_extract_identity_heuristics_for_known_traits(self):
        identity = SemanticSceneExtractor._extract_identity(
            {"name": "Roger", "prompt": "adult muscular male with dark-skinned complexion and curly red hair"},
            "Roger plays drums",
        )
        self.assertEqual(identity.name, "Roger")
        self.assertEqual(identity.gender, "man")
        self.assertEqual(identity.age_range, "adult")
        self.assertEqual(identity.hair_color, "red")
        self.assertEqual(identity.skin_tone, "dark")
        self.assertEqual(identity.body_type, "muscular")

    def test_extract_scene_attributes_positions_clothing_and_actions(self):
        scene = "Nikita is on the left wearing a black suit and playing guitar. Roger is right in a ceremonial gown and sitting."
        self.assertEqual(
            SemanticSceneExtractor._extract_scene_attributes("Nikita", scene),
            ("left side of stage", "black suit", "playing guitar"),
        )
        self.assertEqual(
            SemanticSceneExtractor._extract_scene_attributes("Roger", scene),
            ("right side of stage", "ceremonial gown", "sitting"),
        )
        self.assertEqual(
            SemanticSceneExtractor._extract_scene_attributes("Missing", scene),
            ("in scene", "scene-appropriate clothing", "performing action"),
        )

    def test_extract_scene_layers_and_to_dict(self):
        spec = SemanticSceneExtractor.extract_scene_layers(
            "A bar room with a stage. Camera close wide shot. Nikita is left in a suit playing piano.",
            [{"name": "Nikita", "prompt": "young woman, fair skin, long black hair, slender"}],
            project_style="noir",
        )
        data = spec.to_dict()
        self.assertIn("bar", data["scene"]["environment"].lower())
        self.assertEqual(data["style"], "noir")
        self.assertEqual(data["characters"][0]["name"], "Nikita")
        self.assertIn("long black hair", data["characters"][0]["identity"])

    def test_scene_composition_and_character_scene_data_to_dict(self):
        scene = SceneComposition("club", "low camera", "wide", "A left B right", "neon", "smoky", {"A": "left"})
        ident = CharacterIdentity(name="A", gender="woman")
        data = CharacterSceneData("A", ident, "left", "jacket", "singing").to_dict()
        self.assertEqual(scene.to_dict()["character_positions"], {"A": "left"})
        self.assertEqual(data["identity"], ["woman"])

    def test_llm_semantic_extractor_parses_output(self):
        payload = json.dumps({
            "scene": {"environment": "bar stage", "camera_position": "back of bar", "shot_type": "wide shot", "spatial_composition": "Nikita left", "lighting": "warm", "atmosphere": "busy"},
            "characters": [{"name": "Nikita", "identity": ["young woman", "black hair"], "position": "left", "clothing": "black suit", "action": "playing guitar"}],
        })
        with patch("generators.text_generator.generate_prompt_with_llm", return_value=payload):
            spec = LLMSceneSemanticExtractor().extract_semantic_spec("scene", [{"name": "Nikita", "prompt": "woman"}], "cinematic")
        self.assertEqual(spec.scene.environment, "bar stage")
        self.assertEqual(spec.characters[0].identity.gender, "woman")
        self.assertEqual(spec.style, "cinematic")

    def test_llm_semantic_extractor_falls_back_on_bad_json(self):
        with patch("generators.text_generator.generate_prompt_with_llm", return_value="not json"):
            spec = LLMSceneSemanticExtractor().extract_semantic_spec(
                "A room. Nikita is sitting.", [{"name": "Nikita", "prompt": "female young"}], "fallback-style"
            )
        self.assertEqual(spec.style, "fallback-style")
        self.assertEqual(spec.characters[0].action, "sitting")

    def test_parse_llm_output_defaults_missing_fields_and_man_identity(self):
        spec = LLMSceneSemanticExtractor.__new__(LLMSceneSemanticExtractor)._parse_llm_output(
            {"scene": {}, "characters": [{"name": "Roger", "identity": ["adult performer"]}]},
            "style",
        )
        self.assertEqual(spec.scene.camera_position, "wide shot")
        self.assertEqual(spec.characters[0].identity.gender, "man")
        self.assertEqual(spec.characters[0].position, "in scene")


if __name__ == "__main__":
    unittest.main()
