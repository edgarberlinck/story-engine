"""
Focused unit tests for the design-doc gaps implemented to make the scene
generator match `docs/story-engine-multi-character-scene-design.md`:

- Stage D strategy selection (§2.1) with deterministic guardrails.
- ResolvedCharacter scene_pose / scene_action / scene_position_hint (§2.2).
- ScenePlan.canvas_layout + strategy (§2.3).
- The character_service character_attributes table fix.
- The prompt_builder optimize_for_token_budget `specs` bug fix.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.scene_planner import (
    ResolvedCharacter,
    ScenePlan,
    SceneLayerPlan,
    stage_d_select_strategy,
    _pre_llm_strategy_gate,
    STRATEGY_SINGLE_PASS,
    STRATEGY_PROGRESSIVE,
    STRATEGY_ASSET_COMPOSITION,
)


def _rc(name, **kwargs):
    defaults = dict(
        name=name,
        identity=["trait"],
        default_presentation=[],
        presentation_decision="KEEP",
        scene_presentation=[],
        dropped=[],
        dropped_reason="",
    )
    defaults.update(kwargs)
    return ResolvedCharacter(**defaults)


class TestStrategyGuardrailGate(unittest.TestCase):
    def test_zero_characters_single_pass(self):
        self.assertEqual(_pre_llm_strategy_gate(0), STRATEGY_SINGLE_PASS)

    def test_one_character_single_pass(self):
        self.assertEqual(_pre_llm_strategy_gate(1), STRATEGY_SINGLE_PASS)

    def test_one_character_over_budget_progressive(self):
        self.assertEqual(_pre_llm_strategy_gate(1, over_clip_budget=True), STRATEGY_PROGRESSIVE)

    def test_two_characters_deferred_to_llm(self):
        # The gate returns None (non-trivial) so Stage D / LLM decides.
        self.assertIsNone(_pre_llm_strategy_gate(2))


class TestStageDSelectStrategy(unittest.TestCase):
    def test_gate_short_circuits_single_pass(self):
        d = stage_d_select_strategy("empty", [])
        self.assertEqual(d.strategy, STRATEGY_SINGLE_PASS)
        self.assertTrue(d.heuristic)

    def test_one_character_single_pass(self):
        d = stage_d_select_strategy("one person", [_rc("A")])
        self.assertEqual(d.strategy, STRATEGY_SINGLE_PASS)

    def test_two_characters_no_hints_defaults_progressive(self):
        # Simulate LLM unavailability (returns None) so the heuristic fallback
        # is exercised deterministically regardless of whether the real local
        # model is available/working. Without explicit spatial hints the
        # heuristic defaults to progressive (cheaper than asset_composition).
        with patch("core.scene_planner.generate_text_with_llm", return_value=None):
            d = stage_d_select_strategy("two people on stage", [_rc("A"), _rc("B")])
        self.assertEqual(d.strategy, STRATEGY_PROGRESSIVE)

    def test_two_characters_with_position_hints_asset_composition(self):
        with patch("core.scene_planner.generate_text_with_llm", return_value=None):
            d = stage_d_select_strategy(
                "Nikita on the left, Roger on the right of the stage",
                [_rc("Nikita", scene_position_hint="left of the stage"),
                 _rc("Roger", scene_position_hint="right of the stage")],
            )
        self.assertEqual(d.strategy, STRATEGY_ASSET_COMPOSITION)
        self.assertTrue(d.requires_spatial_precision)

    def test_reason_and_metadata_populated(self):
        with patch("core.scene_planner.generate_text_with_llm", return_value=None):
            d = stage_d_select_strategy(
                "Nikita left, Roger right",
                [_rc("Nikita"), _rc("Roger")],
            )
        self.assertIsInstance(d.reason, str)
        self.assertEqual(d.character_count, 2)


class TestResolvedCharacterSceneFields(unittest.TestCase):
    def test_new_fields_default_to_none(self):
        rc = ResolvedCharacter(
            name="Nikita",
            identity=["woman"],
            default_presentation=[],
            presentation_decision="KEEP",
            scene_presentation=[],
            dropped=[],
            dropped_reason="",
        )
        self.assertIsNone(rc.scene_pose)
        self.assertIsNone(rc.scene_action)
        self.assertIsNone(rc.scene_position_hint)

    def test_new_fields_assignable(self):
        rc = ResolvedCharacter(
            name="Nikita",
            identity=["woman"],
            default_presentation=[],
            presentation_decision="KEEP",
            scene_presentation=["black suit"],
            dropped=[],
            dropped_reason="",
            scene_pose="sitting on a chair",
            scene_action="playing a black Gibson Explorer guitar",
            scene_position_hint="left side of the stage",
        )
        self.assertEqual(rc.scene_pose, "sitting on a chair")
        self.assertEqual(rc.scene_action, "playing a black Gibson Explorer guitar")
        self.assertEqual(rc.scene_position_hint, "left side of the stage")


class TestScenePlanCompositionFields(unittest.TestCase):
    def test_canvas_layout_and_strategy_defaults(self):
        plan = ScenePlan(
            camera="wide shot",
            layers=[SceneLayerPlan(name="base", prompt="env")],
            single_pass_feasible=True,
            rationale="test",
        )
        self.assertIsNone(plan.canvas_layout)
        self.assertEqual(plan.strategy, STRATEGY_SINGLE_PASS)

    def test_canvas_layout_assignable(self):
        layout = {
            "width": 1024,
            "height": 1024,
            "placements": [
                {"name": "Nikita", "anchor": [0.28, 0.62], "scale": 0.42, "z": 1},
            ],
        }
        plan = ScenePlan(
            camera="wide shot",
            layers=[],
            single_pass_feasible=False,
            rationale="test",
            strategy=STRATEGY_ASSET_COMPOSITION,
            canvas_layout=layout,
        )
        self.assertEqual(plan.strategy, STRATEGY_ASSET_COMPOSITION)
        self.assertEqual(plan.canvas_layout["placements"][0]["name"], "Nikita")


class TestCharacterServiceAttributesTable(unittest.TestCase):
    def test_save_and_get_character_with_attributes(self):
        from services.database.character_service import CharacterService

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            svc = CharacterService(db_path=db_path)
            svc.save_character(
                name="Nikita",
                prompt="woman, red hair",
                seed=42,
                model="flux_dev",
                reference_image="ref.png",
                project="test_project",
                attributes={"gender": "woman", "hair_color": "red"},
            )
            # get_character must not crash on the character_attributes table.
            char = svc.get_character("Nikita", project="test_project")
            self.assertIsNotNone(char)
            self.assertEqual(char["attributes"]["hair_color"], "red")


class TestPromptBuilderBugFix(unittest.TestCase):
    def test_optimize_for_token_budget_compresses(self):
        from core.scene_semantics import (
            ScenePromptSpec,
            SceneComposition,
            CharacterSceneData,
            CharacterIdentity,
        )
        from core.prompt_builder import SemanticPromptBuilder

        spec = ScenePromptSpec(
            scene=SceneComposition(
                environment="an old bar with a small stage",
                camera_position="back of the bar",
                shot_type="wide shot",
                spatial_composition="stage visible",
            ),
            characters=[
                CharacterSceneData(
                    name="Nikita",
                    identity=CharacterIdentity(name="Nikita", gender="woman", hair_color="red"),
                    position="left",
                    clothing="black suit",
                    action="playing guitar",
                ),
                CharacterSceneData(
                    name="Roger",
                    identity=CharacterIdentity(name="Roger", gender="man", hair_color="black"),
                    position="right",
                    clothing="formal",
                    action="playing drums",
                ),
            ],
            style="photorealistic",
        )
        builder = SemanticPromptBuilder()
        # Regression: previously raised NameError (`specs.scene.environment`).
        result = builder.optimize_for_token_budget(spec, max_tokens=20)
        self.assertIsInstance(result, str)
        self.assertIn("Nikita", result)
        self.assertIn("Roger", result)


if __name__ == "__main__":
    unittest.main()
