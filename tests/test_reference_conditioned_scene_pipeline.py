"""Unit tests for the reference-conditioned routing in core.scene_pipeline.

These mock the planner, reference collection, and generation backends so no
real diffusion or MPS inference runs. They verify:
- reference-capable models with usable references route to the holistic backend,
- sdxl/flux_dev never route to the reference backend,
- missing references fall back to the existing strategy,
- backend failure records fallback metadata,
- the holistic path skips img2img refinement and still runs QA.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import scene_pipeline as sp
from core.scene_planner import ScenePlan, ResolvedCharacter


def _make_plan(strategy="asset_composition", names=("Nikita", "Roger")) -> ScenePlan:
    resolved = [
        ResolvedCharacter(
            name=name, identity=["identity"], default_presentation=[],
            presentation_decision="KEEP", scene_presentation=[], dropped=[],
            dropped_reason="",
        )
        for name in names
    ]
    return ScenePlan(
        camera="wide shot", layers=[], single_pass_feasible=True,
        rationale="x", resolved_characters=resolved, strategy=strategy,
    )


def _make_ref_file() -> str:
    d = tempfile.mkdtemp()
    p = os.path.join(d, "ref.png")
    Image.new("RGB", (64, 64)).save(p)
    return p


class ReferenceConditioningSupportTest(unittest.TestCase):
    def test_supports_reference_conditioning(self):
        self.assertTrue(sp.supports_reference_conditioning("flux_klein"))
        self.assertFalse(sp.supports_reference_conditioning("sdxl"))
        self.assertFalse(sp.supports_reference_conditioning("flux_dev"))


class CollectCharacterReferencesTest(unittest.TestCase):
    def test_returns_existing_records_in_character_order(self):
        ref_a = _make_ref_file()
        ref_b = _make_ref_file()
        records = {
            "Nikita": {"name": "Nikita", "reference_image": ref_a},
            "Roger": {"name": "Roger", "reference_image": ref_b},
        }
        def fake_get(name, project):
            return records.get(name)
        with patch("generators.image_engine.get_character", side_effect=fake_get):
            out = sp._collect_character_references(
                [{"name": "Nikita"}, {"name": "Roger"}], "p"
            )
        self.assertEqual([r["name"] for r in out], ["Nikita", "Roger"])
        self.assertEqual([r["path"] for r in out], [ref_a, ref_b])

    def test_skips_missing_or_unreadable_records(self):
        ref_ok = _make_ref_file()
        records = {
            "A": {"name": "A", "reference_image": ref_ok},
            "B": {"name": "B", "reference_image": None},
            "C": {"name": "C", "reference_image": "/nonexistent.png"},
        }
        def fake_get(name, project):
            return records.get(name)
        with patch("generators.image_engine.get_character", side_effect=fake_get):
            out = sp._collect_character_references(
                [{"name": "A"}, {"name": "B"}, {"name": "C"}], "p"
            )
        self.assertEqual([r["name"] for r in out], ["A"])

    def test_skips_character_when_lookup_raises(self):
        def fake_get(name, project):
            if name == "Bad":
                raise RuntimeError("boom")
            return {"name": name, "reference_image": _make_ref_file()}
        with patch("generators.image_engine.get_character", side_effect=fake_get):
            out = sp._collect_character_references([{"name": "Bad"}, {"name": "Good"}], "p")
        self.assertEqual([r["name"] for r in out], ["Good"])


def _run_pipeline(**overrides):
    """Run generate_scene_pipeline with the planner/reference backend mocked."""
    plan = _make_plan()
    ref_file = _make_ref_file()
    ref_result_path = os.path.join(tempfile.mkdtemp(), "scene_reference_conditioned.png")
    Image.new("RGB", (64, 64)).save(ref_result_path)
    ref_result = {
        "scene_number": 1,
        "image_path": ref_result_path,
        "prompt": "p",
        "reference_conditioned_prompt": "p",
        "seed": 42,
        "model": "flux_klein",
        "strategy": "reference_conditioned_single_pass",
        "reference_images": [{"name": "Nikita", "path": ref_file}],
        "refinement_applied": False,
        "qa": {"per_character": {"Nikita": True}, "passed": True},
        "warnings": [],
    }
    default_patches = {
        "core.scene_pipeline.LLMScenePlanner": patch(
            "core.scene_pipeline.LLMScenePlanner",
            **{"return_value.plan_scene": lambda self, *a, **k: plan,
               "return_value.save_plan": lambda *a, **k: None},
        ),
        "core.scene_pipeline._collect_character_references": patch(
            "core.scene_pipeline._collect_character_references",
            return_value=[{"name": "Nikita", "path": ref_file}],
        ),
        "core.scene_pipeline._run_reference_conditioned_scene": patch(
            "core.scene_pipeline._run_reference_conditioned_scene",
            return_value=ref_result,
        ),
    }
    defaults = {
        "prompt": "p", "project": "Test", "scene_number": 1,
        "characters": [{"name": "Nikita"}, {"name": "Roger"}],
        "model": "flux_klein", "seed": 42,
    }
    defaults.update(overrides.get("kwargs", {}))
    patches = dict(default_patches)
    patches.update(overrides.get("patches", {}))
    for p in patches.values():
        p.start()
    try:
        return sp.generate_scene_pipeline(**defaults)
    finally:
        for p in patches.values():
            p.stop()

class ReferenceRoutingTest(unittest.TestCase):
    def test_flux_klein_with_references_routes_to_holistic_backend(self):
        with patch("core.scene_pipeline._run_asset_composition") as mock_asset, \
             patch("core.scene_pipeline._run_progressive") as mock_prog:
            result = _run_pipeline()
        self.assertEqual(result["strategy"], "reference_conditioned_single_pass")
        mock_asset.assert_not_called()
        mock_prog.assert_not_called()
        self.assertIn("reference_images", result)
        # Canonical scene.png should have been produced.
        self.assertTrue(result["image_path"].endswith("scene.png"))

    def test_sdxl_does_not_route_to_reference_backend(self):
        with patch("core.scene_pipeline._run_reference_conditioned_scene") as mock_ref, \
             patch("core.scene_pipeline._run_asset_composition") as mock_asset:
            mock_asset.return_value = {
                "scene_number": 1, "image_path": _make_ref_file(),
                "qa": {"per_character": {}, "passed": True}, "warnings": [],
            }
            _run_pipeline(kwargs={"model": "sdxl"})
        mock_ref.assert_not_called()
        mock_asset.assert_called()

    def test_flux_dev_does_not_route_to_reference_backend(self):
        with patch("core.scene_pipeline._run_reference_conditioned_scene") as mock_ref, \
             patch("core.scene_pipeline._run_asset_composition") as mock_asset:
            mock_asset.return_value = {
                "scene_number": 1, "image_path": _make_ref_file(),
                "qa": {"per_character": {}, "passed": True}, "warnings": [],
            }
            _run_pipeline(kwargs={"model": "flux_dev"})
        mock_ref.assert_not_called()
        mock_asset.assert_called()

    def test_flux_klein_without_references_uses_existing_strategy(self):
        asset_result = {
            "scene_number": 1, "image_path": _make_ref_file(),
            "qa": {"per_character": {}, "passed": True}, "warnings": [],
        }
        with patch("core.scene_pipeline._run_asset_composition",
                   return_value=asset_result) as mock_asset:
            result = _run_pipeline(patches={
                "core.scene_pipeline._collect_character_references": patch(
                    "core.scene_pipeline._collect_character_references",
                    return_value=[],
                ),
                "core.scene_pipeline._run_reference_conditioned_scene": patch(
                    "core.scene_pipeline._run_reference_conditioned_scene",
                ),
            })
        self.assertEqual(result["strategy"], "asset_composition")
        mock_asset.assert_called()

    def test_reference_backend_failure_records_fallback(self):
        asset_result = {
            "scene_number": 1, "image_path": _make_ref_file(),
            "qa": {"per_character": {}, "passed": True}, "warnings": [],
        }
        with patch("core.scene_pipeline._run_asset_composition",
                   return_value=asset_result) as mock_asset:
            result = _run_pipeline(patches={
                "core.scene_pipeline._run_reference_conditioned_scene": patch(
                    "core.scene_pipeline._run_reference_conditioned_scene",
                    side_effect=RuntimeError("model load failed"),
                ),
            })
        self.assertEqual(result["fallback_from"], "reference_conditioned_single_pass")
        self.assertEqual(result["fallback_reason"], "model load failed")
        mock_asset.assert_called()


class ReferenceHolisticBehaviorTest(unittest.TestCase):
    def test_holistic_path_skips_img2img_refinement(self):
        ref = _make_ref_file()
        result_path = os.path.join(tempfile.mkdtemp(), "scene_reference_conditioned.png")
        Image.new("RGB", (64, 64)).save(result_path)
        with patch("generators.reference_scene_generator.generate_reference_conditioned_scene",
                   return_value=[result_path]) as mock_gen, \
             patch("core.scene_pipeline._qa_scene",
                   return_value={"per_character": {}, "passed": True}) as mock_qa, \
             patch("generators.img2img_engine.refine_composite") as mock_refine:
            result = sp._run_reference_conditioned_scene(
                "p", "Test", 1, _make_plan(), [{"name": "Nikita", "path": ref}],
                "flux_klein", 42,
            )
        mock_refine.assert_not_called()
        mock_qa.assert_called()
        self.assertIs(result["refinement_applied"], False)
        self.assertEqual(result["strategy"], "reference_conditioned_single_pass")


if __name__ == "__main__":
    unittest.main()
