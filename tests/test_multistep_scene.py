"""Unit tests for multi-step scene composition and pipeline fallbacks."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.scene_composer import MultiStepSceneGenerator, SceneComposer, SceneCompositionPlan, SceneLayer
from core.scene_planner import ResolvedCharacter, SceneLayerPlan, ScenePlan
from core import scene_pipeline as sp
from core.character_asset_generator import CharacterAsset


DESCRIPTION = """There's a very small stage with red curtains. The stage is made of old, cracked wood.

In front of the stage, there are tables with people sitting and watching the performance.

On the stage, Nikita is sitting on a chair and playing a black Gibson Explorer guitar.

Roger is sitting behind a drum kit and playing drums.

The camera is positioned at the back of the bar, showing the crowd in the foreground and the entire stage.

Wide shot, both characters clearly visible."""


def _characters():
    return [
        {"name": "Nikita", "prompt": "woman, long black hair, fantasy armor"},
        {"name": "Roger", "prompt": "man, bald head"},
    ]


def _image(path: Path, color=(120, 80, 40)) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)
    return str(path)


def _plan(strategy="asset_composition", canvas_layout=None):
    resolved = [
        ResolvedCharacter("Nikita", ["woman"], [], "KEEP", ["black suit"], [], "", scene_position_hint="left"),
        ResolvedCharacter("Roger", ["man"], [], "KEEP", ["shirt"], [], "", scene_position_hint="right"),
    ]
    return ScenePlan(
        camera="wide",
        layers=[SceneLayerPlan("base_environment", "empty bar stage", ["stage"], "full")],
        single_pass_feasible=False,
        rationale="two people",
        resolved_characters=resolved,
        strategy=strategy,
        canvas_layout=canvas_layout,
    )


class TestSceneComposer(unittest.TestCase):
    def test_scene_decomposition_creates_base_and_character_layers(self):
        plan = SceneComposer.decompose_scene(DESCRIPTION, _characters())

        self.assertIsInstance(plan, SceneCompositionPlan)
        self.assertIn("stage", plan.base_environment.lower())
        self.assertGreaterEqual(len(plan.layers), 3)
        self.assertEqual(plan.layers[0].name, "base_environment")
        self.assertEqual([layer.priority for layer in plan.layers], [1, 2, 3])
        self.assertTrue(any(layer.name == "character_Nikita" for layer in plan.layers))
        self.assertTrue(any("playing a black Gibson" in layer.prompt for layer in plan.layers))
        self.assertFalse(any("fantasy armor" in layer.prompt for layer in plan.layers))

    def test_decomposition_uses_default_environment_when_no_env_lines(self):
        plan = SceneComposer.decompose_scene("Nikita smiles.", [{"name": "Nikita", "prompt": "woman"}])
        self.assertEqual(plan.layers[0].prompt, "A bar with stage and audience")
        self.assertEqual(plan.base_environment, "")

    def test_build_incremental_prompts_include_camera_and_continue_prefix(self):
        plan = SceneCompositionPlan(
            base_environment="red curtain stage",
            characters=_characters(),
            camera_setup="wide shot",
            layers=[
                SceneLayer("base_environment", "red curtain stage"),
                SceneLayer("character_Nikita", "Nikita playing guitar", "left area", 2),
                SceneLayer("character_Roger", "Roger playing drums", "right area", 3),
            ],
        )
        steps = SceneComposer.build_incremental_prompts(plan)
        self.assertEqual(steps[0], ("base", "red curtain stage. wide shot"))
        self.assertEqual(steps[1][0], "character_Nikita")
        self.assertTrue(steps[1][1].startswith("Continue the scene."))
        self.assertIn("Roger playing drums", steps[2][1])

    def test_incremental_prompts_handle_blank_camera_setup(self):
        plan = SceneCompositionPlan("empty room", [], "", [SceneLayer("base_environment", "empty room")])
        self.assertEqual(SceneComposer.build_incremental_prompts(plan), [("base", "empty room")])


class TestMultiStepSceneGenerator(unittest.TestCase):
    def test_generate_scene_incrementally_mocks_image_generation_and_copies_steps(self):
        with tempfile.TemporaryDirectory() as td:
            temp_root = Path(td)
            generated = [
                _image(temp_root / "gen_base.png", (10, 10, 10)),
                _image(temp_root / "gen_nikita.png", (20, 20, 20)),
                _image(temp_root / "gen_roger.png", (30, 30, 30)),
            ]

            calls = []
            def fake_generate_images(**kwargs):
                calls.append(kwargs)
                return [generated[len(calls) - 1]]

            with patch("generators.image_engine.generate_images", side_effect=fake_generate_images), \
                 patch("utils.project_paths.scene_dir", return_value=temp_root / "scene_001"):
                result = MultiStepSceneGenerator().generate_scene_incrementally(
                    DESCRIPTION, _characters(), "Project", 1, model="sdxl", seed=100
                )

            self.assertEqual(result["steps_generated"], 3)
            self.assertTrue(Path(result["image_path"]).exists())
            self.assertEqual([c["seed"] for c in calls], [100, 101, 102])
            self.assertIn("Maintaining the existing composition", calls[1]["prompt"])
            self.assertEqual(calls[0]["task_name"], "scene_1_step0")


class TestScenePipelineHelpers(unittest.TestCase):
    def test_select_strategy_and_background_prompt_variants(self):
        self.assertEqual(sp._select_strategy(0), "single_pass")
        self.assertEqual(sp._select_strategy(1), "single_pass")
        self.assertEqual(sp._select_strategy(2), "asset_composition")

        plan = _plan()
        self.assertEqual(sp._build_background_prompt("Nikita plays", plan, ["Nikita"]), "Nikita plays. The scene is empty, no characters visible yet.")
        plan.layers.append(SceneLayerPlan("character_Nikita", "Nikita", []))
        self.assertEqual(sp._build_background_prompt("raw", plan, ["Nikita"]), "empty bar stage")
        stripped = sp._build_background_prompt("A smoky bar. Nikita plays. She smiles. The crowd cheers.", ScenePlan("", [], True, ""), ["Nikita"])
        self.assertIn("A smoky bar", stripped)
        self.assertNotIn("Nikita", stripped)

    def test_reference_prompt_and_refinement_prompt_and_canvas_layout(self):
        plan = _plan(canvas_layout={"width": 512, "height": 512, "placements": [{"name": "Nikita", "anchor": [0.2, 0.8], "scale": 0.4, "z": 2}]})
        prompt = sp._build_reference_conditioned_prompt("A stage.", plan, [{"name": "Nikita", "path": "n.png"}])
        self.assertIn("Reference image 1 is Nikita", prompt)
        self.assertIn("position: left", prompt)
        self.assertIs(sp._canvas_layout_from_plan(plan, ["Nikita"]), plan.canvas_layout)
        fallback = sp._canvas_layout_from_plan(ScenePlan("", [], True, ""), ["A", "B"])
        self.assertEqual(fallback["width"], 1024)
        refinement = sp._build_refinement_prompt("photorealistic wide cinematic shot with natural stage lighting")
        self.assertIn("coherent lighting", refinement)
        self.assertEqual(refinement.count("stage lighting"), 1)

    def test_qa_scene_success_missing_reference_and_exception(self):
        records = {
            "Nikita": {"reference_image": "ref.png"},
            "Roger": {"reference_image": None},
            "Bad": {"reference_image": "bad.png"},
        }
        def fake_face(ref, img):
            if ref == "bad.png":
                raise RuntimeError("optional dependency failed")
            return True
        with patch("generators.image_engine.get_character", side_effect=lambda name, project: records.get(name)), \
             patch("utils.face_check.character_appears_in_image", side_effect=fake_face):
            qa = sp._qa_scene("final.png", ["Nikita", "Roger", "Missing", "Bad"], "Project")
        self.assertEqual(qa["per_character"]["Nikita"], True)
        self.assertIsNone(qa["per_character"]["Roger"])
        self.assertIsNone(qa["per_character"]["Bad"])
        self.assertTrue(qa["passed"])

    def test_qa_scene_marks_false_as_failed(self):
        with patch("generators.image_engine.get_character", return_value={"reference_image": "ref.png"}), \
             patch("utils.face_check.character_appears_in_image", return_value=False):
            qa = sp._qa_scene("final.png", ["Nikita"], "Project")
        self.assertFalse(qa["passed"])


class TestScenePipelineAssetComposition(unittest.TestCase):
    def test_asset_composition_refinement_disabled_uses_composite_and_region_hints(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_bg = _image(root / "source_bg.png")
            source_asset = _image(root / "nikita_asset.png")
            cutout = _image(root / "nikita_cutout.png")
            plan = ScenePlan(
                "wide",
                [
                    SceneLayerPlan("base_environment", "empty stage prompt", ["stage"], "full frame"),
                    SceneLayerPlan("character_Nikita", "Nikita playing guitar", ["Nikita"], "left third"),
                ],
                False,
                "r",
                resolved_characters=[ResolvedCharacter("Nikita", ["woman"], [], "KEEP", [], [], "")],
                strategy="asset_composition",
                canvas_layout={"width": 640, "height": 480, "placements": [{"name": "Nikita", "anchor": [0.25, 0.75], "scale": 0.35, "z": 4}]},
            )

            def fake_compose(background_path, placements, canvas_width, canvas_height, output_path):
                self.assertEqual(canvas_width, 640)
                self.assertEqual(canvas_height, 480)
                self.assertEqual(placements[0]["anchor"], (0.25, 0.75))
                _image(Path(output_path), (1, 2, 3))

            with patch("core.scene_pipeline.scene_dir", return_value=root / "scene"), \
                 patch("generators.image_engine.generate_images", return_value=[source_bg]) as mock_bg, \
                 patch("core.scene_pipeline.generate_character_assets", return_value=[CharacterAsset("Nikita", source_asset, "prompt", "ref")]), \
                 patch("core.scene_compositor.segment_character", return_value=("mask.png", cutout, (0, 0, 10, 10), "mock")), \
                 patch("core.scene_compositor.validate_character_asset", return_value={"valid": True, "issues": [], "metrics": {}, "name": "Nikita"}), \
                 patch("core.scene_pipeline.compose_scene", side_effect=fake_compose), \
                patch("core.scene_pipeline._qa_scene", return_value={"per_character": {"Nikita": True}, "passed": True}):
                result = sp._run_asset_composition("Nikita plays on a stage", "Project", 7, plan, "sdxl", 9, enable_refinement=False)
                result_path_exists = Path(result["image_path"]).exists()

        self.assertEqual(result["strategy"], "asset_composition")
        self.assertFalse(result["refinement_applied"])
        self.assertIsNone(result["refined_image_path"])
        self.assertEqual(result["background_prompt"], "empty stage prompt")
        self.assertTrue(result_path_exists)
        self.assertEqual(mock_bg.call_args.kwargs["task_name"], "scene_7_background")

    def test_asset_composition_invalid_segmentation_retries_and_warns_then_refinement_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_bg = _image(root / "source_bg.png")
            source_asset = _image(root / "raw_asset.png")

            checks = [
                {"valid": False, "issues": ["opaque background"], "metrics": {}, "name": "Roger"},
                {"valid": False, "issues": ["still opaque"], "metrics": {}, "name": "Roger"},
            ]
            def fake_compose(output_path, **kwargs):
                _image(Path(output_path), (8, 8, 8))

            with patch("core.scene_pipeline.scene_dir", return_value=root / "scene"), \
                 patch("generators.image_engine.generate_images", return_value=[source_bg]), \
                 patch("core.scene_pipeline.generate_character_assets", return_value=[CharacterAsset("Roger", source_asset, "prompt", None)]), \
                 patch("core.scene_compositor.segment_character", side_effect=[("m", "bad_cutout.png", (1, 2, 3, 4), "simple"), ("m2", None, None, "aggressive")]) as mock_segment, \
                 patch("core.scene_compositor.validate_character_asset", side_effect=checks), \
                 patch("core.scene_pipeline.compose_scene", side_effect=fake_compose), \
                 patch("generators.img2img_engine.refine_composite", side_effect=RuntimeError("refine no")), \
                 patch("core.scene_pipeline._qa_scene", return_value={"per_character": {"Roger": False}, "passed": False}):
                result = sp._run_asset_composition("A bar", "Project", 8, _plan(), "sdxl", 1, enable_refinement=True)

        self.assertEqual(mock_segment.call_count, 2)
        self.assertFalse(result["assets"][0]["segmentation_ok"])
        self.assertGreaterEqual(len(result["warnings"]), 3)
        self.assertIn("Refinement pass errored", " ".join(result["warnings"]))
        self.assertFalse(result["qa"]["passed"])


class TestGenerateScenePipelineFallbacks(unittest.TestCase):
    def test_single_pass_path_without_characters_uses_generate_scene(self):
        with patch("core.scene_pipeline.generate_scene", return_value={"image_path": "scene.png"}) as mock_generate:
            result = sp.generate_scene_pipeline("empty landscape", project="P", scene_number=1, characters=[], model="sdxl", seed=5)
        self.assertEqual(result["strategy"], "single_pass")
        mock_generate.assert_called_once_with(
            prompt="empty landscape", project="P", scene_number=1, model="sdxl", seed=5, use_asset_pipeline=False
        )

    def test_reference_generation_failure_falls_back_to_progressive(self):
        plan = _plan(strategy="progressive")
        prog_path = _image(Path(tempfile.mkdtemp()) / "prog.png")
        with patch("core.scene_pipeline.LLMScenePlanner", **{
                "return_value.plan_scene.return_value": plan,
                "return_value.save_plan.return_value": None,
             }), \
             patch("core.scene_pipeline._collect_character_references", return_value=[{"name": "Nikita", "path": "ref.png"}]), \
             patch("core.scene_pipeline._run_reference_conditioned_scene", side_effect=RuntimeError("reference backend died")), \
             patch("core.scene_pipeline._run_progressive", return_value={"image_path": prog_path, "scene_number": 2, "strategy": "progressive"}) as mock_prog:
            result = sp.generate_scene_pipeline("band", project="P", scene_number=2, characters=_characters(), model="flux_klein", seed=3)
        self.assertEqual(result["fallback_from"], "reference_conditioned_single_pass")
        self.assertEqual(result["fallback_reason"], "reference backend died")
        self.assertEqual(result["strategy"], "progressive")
        mock_prog.assert_called_once()

    def test_asset_composition_failure_falls_back_to_progressive(self):
        plan = _plan(strategy="asset_composition")
        prog_path = _image(Path(tempfile.mkdtemp()) / "prog.png")
        with patch("core.scene_pipeline.LLMScenePlanner", **{
                "return_value.plan_scene.return_value": plan,
                "return_value.save_plan.return_value": None,
             }), \
             patch("core.scene_pipeline.supports_reference_conditioning", return_value=False), \
             patch("core.scene_pipeline._run_asset_composition", side_effect=RuntimeError("asset failed")), \
             patch("core.scene_pipeline._run_progressive", return_value={"image_path": prog_path, "scene_number": 3}) as mock_prog:
            result = sp.generate_scene_pipeline("band", project="P", scene_number=3, characters=_characters(), model="sdxl", seed=4)
        self.assertEqual(result["fallback_from"], "asset_composition")
        self.assertEqual(result["fallback_reason"], "asset failed")
        self.assertEqual(result["strategy"], "asset_composition")
        mock_prog.assert_called_once()


if __name__ == "__main__":
    unittest.main()
