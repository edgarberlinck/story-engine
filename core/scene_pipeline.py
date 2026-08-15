"""
Top-level scene generation orchestrator.

Sequences existing infrastructure (LLM scene planner, image engine) with the
two new modules (`character_asset_generator.py`, `scene_compositor.py`) to
implement the "character asset generation + deterministic composition"
strategy described in
`docs/story-engine-multi-character-scene-design.md`.

Strategy selection (cheap deterministic gate, per design doc §2.1):
- 0 characters               -> single_pass (existing `generate_scene`)
- 1 character                -> single_pass (existing `generate_scene`)
- >=2 characters             -> asset_composition (this module), falling
                                 back to progressive prompt-only generation
                                 (`MultiStepSceneGenerator`) if asset
                                 composition fails at any stage.

This module never crashes: every failure point degrades to a cheaper
strategy and records a warning in the result metadata, per the design doc's
"always degrade, never crash, always flag" principle (§6).
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.scene_planner import (
    LLMScenePlanner,
    ScenePlan,
    stage_d_select_strategy,
    STRATEGY_SINGLE_PASS,
    STRATEGY_ASSET_COMPOSITION,
)
from core.character_asset_generator import generate_character_assets, CharacterAsset
from core.scene_compositor import segment_character, compose_scene, default_canvas_layout
from generators.image_engine import generate_scene
from utils.project_paths import scene_dir, DEFAULT_PROJECT
from utils.scene_logger import scene_logging


def _select_strategy(num_characters: int) -> str:
    """Deterministic pre-LLM guardrail gate (design doc §2.1) kept as a thin
    helper for the trivial cases. For >=2 characters this returns None so the
    caller delegates to Stage D (LLM) for the real decision."""
    from core.scene_planner import _pre_llm_strategy_gate
    gated = _pre_llm_strategy_gate(num_characters)
    return gated if gated is not None else STRATEGY_ASSET_COMPOSITION


def _build_background_prompt(prompt: str, plan: ScenePlan, character_names: Optional[List[str]] = None) -> str:
    """Prefer the LLM-planned base_environment layer (no characters) when
    available; otherwise deterministically strip character sentences from the
    raw scene prompt so the background contains only environment/camera/
    composition information."""
    for layer in plan.layers:
        if layer.name == "base_environment" and len(plan.layers) > 1:
            return layer.prompt

    import re
    names = [n.lower() for n in (character_names or [])]
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|\n+", prompt)
        if s.strip()
    ]
    env_sentences = []
    character_words = re.compile(
        r"\b(musicians?|performers?|characters?|both)\b", re.IGNORECASE
    )
    for s in sentences:
        lower = s.lower()
        if any(n in lower for n in names):
            continue
        # Drop pronoun continuations of character sentences.
        if re.match(r"^(she|he|they)\b", lower):
            continue
        # Drop sentences that talk about the characters generically.
        if character_words.search(s):
            continue
        env_sentences.append(s.rstrip("."))
    if not env_sentences:
        return f"{prompt}. The scene is empty, no characters visible yet."
    return ". ".join(env_sentences) + ". The stage is empty, no performers on stage yet."


def _qa_scene(final_image_path: str, character_names: List[str], project: str) -> Dict[str, Any]:
    """Best-effort automated QA using `utils/face_check.py`. Skips
    gracefully (inconclusive) if `face_recognition` isn't installed or a
    character has no reference image, matching the project's existing
    optional-dependency philosophy."""
    from generators.image_engine import get_character
    from utils.face_check import character_appears_in_image

    results = {}
    for name in character_names:
        character = get_character(name, project)
        if not character or not character.get("reference_image"):
            results[name] = None
            continue
        try:
            results[name] = character_appears_in_image(
                character["reference_image"], final_image_path
            )
        except Exception as e:
            print(f"QA check failed for '{name}': {e}")
            results[name] = None

    passed = all(v is not False for v in results.values())
    return {"per_character": results, "passed": passed}


# Models that expose FLUX.2-style image-reference conditioning (character
# reference images passed as reference latents). Currently only flux_klein.
# An explicit set is used rather than `"flux" in model.lower()` because
# flux_dev is not wired for the reference-latent interface.
REFERENCE_CONDITIONED_SCENE_MODELS = {"flux_klein"}


def supports_reference_conditioning(model: str) -> bool:
    """Whether the given model can condition a scene on character references."""
    return model in REFERENCE_CONDITIONED_SCENE_MODELS


def _collect_character_references(
    characters: List[Dict[str, Any]], project: str
) -> List[Dict[str, str]]:
    """Resolve character reference image paths in character order.

    Uses the stored character record's ``reference_image``. Characters without
    a usable reference are skipped (never fatal); the caller decides whether to
    invoke the reference backend based on what comes back.
    """
    from generators.image_engine import get_character

    references: List[Dict[str, str]] = []
    for character in characters or []:
        name = character.get("name") if isinstance(character, dict) else getattr(character, "name", None)
        if not name:
            continue
        try:
            record = get_character(name, project)
        except Exception as e:
            print(f"[scene_pipeline] Could not resolve character '{name}' reference: {e}")
            continue
        if not record or not record.get("reference_image"):
            continue
        path = record["reference_image"]
        if path and Path(path).exists():
            references.append({"name": name, "path": path})
        else:
            print(f"[scene_pipeline] Character '{name}' has no usable reference image at {path}")
    return references


def _build_reference_conditioned_prompt(
    prompt: str, plan: ScenePlan, references: List[Dict[str, str]]
) -> str:
    """Append a compact ordered identity binding to the holistic scene prompt.

    Matches reference ordinal/name to the ``image=[...]`` order passed to the
    backend. Uses Stage A ``ResolvedCharacter`` scene-specific attributes when
    available.
    """
    resolved_by_name = {rc.name: rc for rc in plan.resolved_characters}
    parts = [prompt.rstrip().rstrip("."), "", "Character-reference binding:"]
    for idx, ref in enumerate(references, start=1):
        name = ref["name"]
        rc = resolved_by_name.get(name)
        bits = []
        if rc is not None:
            if rc.identity:
                bits.append(", ".join(rc.identity))
            if rc.scene_pose:
                bits.append(rc.scene_pose)
            if rc.scene_action:
                bits.append(rc.scene_action)
            if rc.scene_position_hint:
                bits.append(f"position: {rc.scene_position_hint}")
        detail = "; ".join(bits).strip("; ")
        suffix = f" ({detail})" if detail else ""
        parts.append(f"Reference image {idx} is {name}{suffix}.")
    parts.append("Render all referenced individuals as themselves in one coherent wide cinematic shot.")
    return "\n".join(parts)


def _run_reference_conditioned_scene(
    prompt: str,
    project: str,
    scene_number: int,
    plan: ScenePlan,
    references: List[Dict[str, str]],
    model: str,
    seed: int,
) -> Dict[str, Any]:
    """Generate a complete scene in one reference-conditioned diffusion call."""
    from generators.reference_scene_generator import generate_reference_conditioned_scene

    target_dir = scene_dir(scene_number, project)
    target_dir.mkdir(parents=True, exist_ok=True)

    reference_conditioned_prompt = _build_reference_conditioned_prompt(prompt, plan, references)
    print(f"[scene_pipeline] Reference-conditioned scene prompt:\n{reference_conditioned_prompt}")
    print(f"[scene_pipeline] Reference images: {[r['name'] for r in references]}")

    files = generate_reference_conditioned_scene(
        prompt=reference_conditioned_prompt,
        reference_image_paths=[r["path"] for r in references],
        model_name=model,
        seed=seed,
        task_name=f"scene_{scene_number}_reference",
    )

    final_image_path = str(target_dir / "scene_reference_conditioned.png")
    import shutil
    shutil.copy(files[0], final_image_path)

    character_names = [rc.name for rc in plan.resolved_characters]
    qa = _qa_scene(final_image_path, character_names, project)
    warnings: List[str] = []
    if not qa["passed"]:
        warnings.append("QA check failed for one or more characters (see qa.per_character).")

    return {
        "scene_number": scene_number,
        "image_path": final_image_path,
        "prompt": prompt,
        "reference_conditioned_prompt": reference_conditioned_prompt,
        "seed": seed,
        "model": model,
        "strategy": "reference_conditioned_single_pass",
        "reference_images": [{"name": r["name"], "path": r["path"]} for r in references],
        "refinement_applied": False,
        "qa": qa,
        "warnings": warnings,
    }


def _build_refinement_prompt(prompt: str) -> str:
    """Build a short, neutral scene-level integration prompt from the original
    scene prompt — environment/lighting/style only, NO character names or
    identity attributes, so the img2img pass doesn't drift identity."""
    import re

    style_keywords = [
        "photorealistic", "cinematic", "wide shot", "wide cinematic shot",
        "natural stage lighting", "stage lighting", "natural lighting",
        "detailed environment", "dramatic lighting", "soft lighting",
        "film grain", "depth of field",
    ]
    lower = prompt.lower()
    found = [kw for kw in style_keywords if kw in lower]
    # Drop keywords subsumed by longer matches (e.g. "stage lighting" when
    # "natural stage lighting" already matched).
    found = [kw for kw in found if not any(kw != o and kw in o for o in found)]

    parts = ["Photorealistic scene"] if "photorealistic" not in found else []
    parts += found
    parts += [
        "integrated environment",
        "coherent lighting across all subjects",
        "natural shadows",
        "photographic quality",
    ]
    # De-dup while preserving order, keep it short (< ~60 tokens).
    seen = set()
    unique = []
    for p in parts:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return ", ".join(unique[:10])


def _canvas_layout_from_plan(plan: ScenePlan, character_names: List[str]) -> Dict[str, Any]:
    """Use the LLM-provided `canvas_layout` (design doc §2.3) when present,
    otherwise fall back to an evenly-spaced deterministic layout. This keeps
    placement explicit and inspectable rather than guessed from prompt text
    (§4.3).
    """
    layout = getattr(plan, "canvas_layout", None)
    if isinstance(layout, dict) and layout.get("placements"):
        return layout
    return default_canvas_layout(character_names)


def _run_asset_composition(
    prompt: str,
    project: str,
    scene_number: int,
    plan: ScenePlan,
    model: str,
    seed: int,
    enable_refinement: bool = True,
    refinement_strength: float = 0.25,
    refinement_model: str = "sdxl",
) -> Dict[str, Any]:
    target_dir = scene_dir(scene_number, project)
    assets_dir = target_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []

    # Step 1: background/environment (text-to-image, no characters).
    resolved_characters = plan.resolved_characters
    character_names = [rc.name for rc in resolved_characters]

    bg_prompt = _build_background_prompt(prompt, plan, character_names)
    print(f"[scene_pipeline] Generating background: {bg_prompt}")
    from generators.image_engine import generate_images
    bg_files = generate_images(
        prompt=bg_prompt,
        model_name=model,
        seed=seed,
        task_name=f"scene_{scene_number}_background",
    )
    background_path = str(assets_dir / "background.png")
    import shutil
    shutil.copy(bg_files[0], background_path)

    # Step 2: per-character asset generation (plain background).
    assets: List[CharacterAsset] = generate_character_assets(
        resolved_characters,
        project=project,
        model="sdxl",
        base_seed=seed,
        output_dir=assets_dir,
    )

    # Step 3: segmentation, with graceful per-character fallback.
    placements_meta = []
    canvas_layout = _canvas_layout_from_plan(plan, character_names)
    canvas_placements = {p["name"]: p for p in canvas_layout["placements"]}

    from core.scene_compositor import (
        segment_character,
        validate_character_asset,
    )

    for asset in assets:
        mask_path, cutout_path, bbox, method = segment_character(
            asset.image_path, output_dir=assets_dir, name_hint=asset.name
        )
        # `check` describes the cutout that will actually be used. It starts
        # as "no usable cutout" and is replaced once a valid cutout exists.
        check: Dict[str, Any] = {
            "valid": False,
            "issues": ["segmentation produced no cutout"],
            "metrics": {},
            "name": asset.name,
        }

        # Phase 2: asset validation. Segmentation can "succeed" yet still ship
        # a broken cutout (e.g. background remaining / opaque rectangle). If
        # validation fails, retry segmentation once with an aggressive
        # alternative strategy; if still invalid, fall back to the raw asset
        # (opaque rectangle) and flag it loudly so we never silently paste a
        # broken cutout.
        if cutout_path is not None:
            check = validate_character_asset(cutout_path, name_hint=asset.name)
            if not check["valid"]:
                print(f"[scene_pipeline] Asset validation failed for '{asset.name}': "
                      f"{check['issues']}; retrying segmentation (aggressive).")
                mask_path, cutout_path, bbox, method = segment_character(
                    asset.image_path, output_dir=assets_dir,
                    name_hint=asset.name, retry=True,
                )
                if cutout_path is not None:
                    check = validate_character_asset(cutout_path, name_hint=asset.name)
                else:
                    check = {"valid": False, "issues": ["retry segmentation returned nothing"],
                             "metrics": {}, "name": asset.name}

        if not check.get("valid", True):
            warnings.append(
                f"Asset '{asset.name}' failed validation: "
                f"{check.get('issues', 'segmentation failed')}. Using raw asset "
                "image as an opaque rectangle placement (visibly worse but non-fatal)."
            )
            cutout_path = asset.image_path
            asset.segmentation_ok = False
        else:
            asset.segmentation_ok = True
        asset.validation = check
        asset.mask_path = mask_path
        asset.cutout_path = cutout_path
        asset.bbox = bbox
        asset.segmentation_method = method

        layout = canvas_placements.get(asset.name, {"anchor": [0.5, 0.85], "scale": 0.5, "z": 1})
        placements_meta.append({
            "name": asset.name,
            "cutout_path": cutout_path,
            "anchor": tuple(layout["anchor"]),
            "scale": layout["scale"],
            "z": layout["z"],
        })

    # Step 4: deterministic composition. Saved under a distinct filename so
    # it is never silently clobbered if a later fallback (progressive
    # generation) also writes to `scene.png` in the same scene directory.
    final_image_path = str(target_dir / "scene_asset_composition.png")
    compose_scene(
        background_path=background_path,
        placements=placements_meta,
        canvas_width=canvas_layout["width"],
        canvas_height=canvas_layout["height"],
        output_path=final_image_path,
    )

    # Step 5: optional low-strength img2img refinement pass to visually
    # integrate the pasted cutouts (harmonize lighting/shadows) while
    # preserving composition. The deterministic composite is NEVER
    # overwritten — it remains the correctness ground truth.
    result_image_path = final_image_path
    refinement_applied = False
    refined_image_path = None
    refinement_prompt = None
    if enable_refinement:
        try:
            from generators.img2img_engine import refine_composite
            refinement_prompt = _build_refinement_prompt(prompt)
            print(f"[scene_pipeline] Refinement pass enabled "
                  f"(model={refinement_model}, strength={refinement_strength})")
            print(f"[scene_pipeline] Refinement prompt: {refinement_prompt}")
            refined_image_path = refine_composite(
                composite_image_path=final_image_path,
                prompt=refinement_prompt,
                model_name=refinement_model,
                strength=refinement_strength,
                seed=seed,
                output_path=str(target_dir / "scene_refined.png"),
            )
            if refined_image_path != final_image_path and Path(refined_image_path).exists():
                refinement_applied = True
                result_image_path = refined_image_path
            else:
                warnings.append("Refinement pass failed; keeping deterministic composite.")
        except Exception as e:
            warnings.append(f"Refinement pass errored ({e}); keeping deterministic composite.")
            print(f"[scene_pipeline] Refinement pass errored: {e}")
    else:
        print("[scene_pipeline] Refinement pass disabled (enable_refinement=False).")

    # Step 6: automated QA (on the image that will ship).
    qa = _qa_scene(result_image_path, character_names, project)
    if not qa["passed"]:
        warnings.append("QA check failed for one or more characters (see qa.per_character).")

    return {
        "scene_number": scene_number,
        "image_path": result_image_path,
        "composite_image_path": final_image_path,
        "refined_image_path": refined_image_path if refinement_applied else None,
        "refinement_applied": refinement_applied,
        "refinement_strength": refinement_strength if refinement_applied else None,
        "refinement_model": refinement_model if refinement_applied else None,
        "refinement_prompt": refinement_prompt,
        "prompt": prompt,
        "seed": seed,
        "model": model,
        "strategy": "asset_composition",
        "background_prompt": bg_prompt,
        "assets": [asdict(a) for a in assets],
        "canvas_layout": canvas_layout,
        "qa": qa,
        "warnings": warnings,
    }


def _run_progressive(
    prompt: str,
    project: str,
    scene_number: int,
    characters: List[Dict[str, Any]],
    model: str,
    seed: int,
) -> Dict[str, Any]:
    """Prompt-only progressive fallback using the existing
    `MultiStepSceneGenerator` (kept as a cheaper fallback, per design doc)."""
    from core.scene_composer import MultiStepSceneGenerator

    gen = MultiStepSceneGenerator()
    result = gen.generate_scene_incrementally(
        description=prompt,
        characters=characters,
        project=project,
        scene_number=scene_number,
        model=model,
        seed=seed,
    )
    result["strategy"] = "progressive"
    return result


@scene_logging(scene_name_arg="scene_number", prompt_arg="prompt")
def generate_scene_pipeline(
    prompt: str,
    project: str = DEFAULT_PROJECT,
    scene_number: Optional[int] = None,
    characters: Optional[List[Dict[str, Any]]] = None,
    model: str = "flux_dev",
    seed: int = 42,
    enable_refinement: bool = True,
    refinement_strength: float = 0.25,
    refinement_model: str = "sdxl",
) -> Dict[str, Any]:
    """Main orchestrator entry point.

    Selects a generation strategy based on character count, runs the LLM
    planner (Stage A/B/C) for context, and executes the selected strategy
    with layered fallbacks so the pipeline never crashes.
    """
    from services.database.character_service import character_service
    from utils.project_paths import next_scene_number

    if characters is None:
        characters = character_service.find_characters_in_text(prompt, project)

    if scene_number is None:
        scene_number = next_scene_number(project)

    # Trivial cases (0-1 characters) are decided by the deterministic gate.
    # Non-trivial (>=2) cases are decided by Stage D during LLM planning below.
    strategy = _select_strategy(len(characters))
    print(f"[scene_pipeline] Selected strategy: {strategy} ({len(characters)} characters)")

    if strategy == "single_pass":
        result = generate_scene(
            prompt=prompt,
            project=project,
            scene_number=scene_number,
            model=model,
            seed=seed,
            use_asset_pipeline=False,
        )
        result.setdefault("strategy", "single_pass")
        return result

    # asset_composition / progressive path, with LLM Stage D deciding between
    # them for >=2 characters, and progressive fallback on any failure.
    planner = LLMScenePlanner(use_llm=True)
    plan = planner.plan_scene(prompt, characters)

    # Stage D may have selected a strategy (single_pass / progressive /
    # asset_composition). Respect it.
    strategy = plan.strategy or strategy
    print(f"[scene_pipeline] Stage D strategy: {strategy}")

    # Tracks a failed reference-conditioned attempt so the fallback result can
    # record where it came from (design §11). None when no reference attempt ran
    # or it succeeded.
    reference_fallback = None

    # Reference-conditioned holistic override (design:
    # docs/story-engine-reference-conditioned-scene-design.md §10/§11). When a
    # reference-capable model is selected AND at least one character reference
    # image is usable, generate the whole scene in one reference-conditioned
    # call instead of asset-composition. Falls back to the Stage D strategy on
    # any failure so the scene never silently breaks.
    if supports_reference_conditioning(model):
        references = _collect_character_references(characters, project)
        if references:
            print(f"[scene_pipeline] Model '{model}' supports reference conditioning; "
                  f"using holistic reference-conditioned generation "
                  f"({len(references)} reference image(s)).")
            try:
                result = _run_reference_conditioned_scene(
                    prompt, project, scene_number, plan, references, model, seed
                )
                if not result["qa"]["passed"]:
                    print("[scene_pipeline] WARNING: QA check flagged one or more characters; "
                          "keeping reference-conditioned result (see result['qa']).")

                target_dir = scene_dir(scene_number, project)
                target_dir.mkdir(parents=True, exist_ok=True)
                plan_path = target_dir / "llm_plan.json"
                try:
                    planner.save_plan(plan, plan_path)
                except Exception as e:
                    print(f"Could not save plan: {e}")
                result["llm_plan_path"] = str(plan_path)

                canonical_path = target_dir / "scene.png"
                raw_path = Path(result["image_path"])
                if raw_path.resolve() != canonical_path.resolve():
                    import shutil as _shutil
                    _shutil.copy(raw_path, canonical_path)
                    result["raw_image_path"] = str(raw_path)
                    result["image_path"] = str(canonical_path)

                try:
                    metadata_path = target_dir / "pipeline_result.json"
                    metadata_path.write_text(json.dumps(result, indent=2, default=str))
                except Exception as e:
                    print(f"Could not save pipeline metadata: {e}")
                return result
            except Exception as e:
                print(f"[scene_pipeline] Reference-conditioned generation failed ({e}); "
                      f"falling back to {strategy} strategy.")
                result = None
                reference_fallback = {
                    "fallback_from": "reference_conditioned_single_pass",
                    "fallback_reason": str(e),
                }
            else:
                reference_fallback = None

    if strategy == "single_pass":
        result = generate_scene(
            prompt=prompt,
            project=project,
            scene_number=scene_number,
            model=model,
            seed=seed,
            use_asset_pipeline=False,
        )
        result.setdefault("strategy", "single_pass")
        result["llm_plan_path"] = None
        if reference_fallback:
            result.update(reference_fallback)
        return result

    if strategy == "progressive":
        result = _run_progressive(prompt, project, scene_number, characters, model, seed)
        result["llm_plan_path"] = None
        result.setdefault("strategy", "progressive")
        if reference_fallback:
            result.update(reference_fallback)
        return result

    target_dir = scene_dir(scene_number, project)
    target_dir.mkdir(parents=True, exist_ok=True)
    plan_path = target_dir / "llm_plan.json"
    try:
        planner.save_plan(plan, plan_path)
    except Exception as e:
        print(f"Could not save plan: {e}")

    try:
        result = _run_asset_composition(
            prompt, project, scene_number, plan, model, seed,
            enable_refinement=enable_refinement,
            refinement_strength=refinement_strength,
            refinement_model=refinement_model,
        )

        if not result["qa"]["passed"]:
            # Per design doc §6: degrade gracefully and FLAG, never throw the
            # composed result away. QA is best-effort (face_recognition may
            # mismatch cross-model assets), so a failed check is a warning,
            # not grounds for regenerating the whole scene.
            print("[scene_pipeline] WARNING: QA check flagged one or more characters; "
                  "keeping asset-composition result (see result['qa']).")

    except Exception as e:
        print(f"[scene_pipeline] Asset composition failed ({e}); falling back to progressive generation.")
        result = _run_progressive(prompt, project, scene_number, characters, model, seed)
        result["fallback_from"] = "asset_composition"
        result["fallback_reason"] = str(e)

    if reference_fallback:
        result.update(reference_fallback)

    result["llm_plan_path"] = str(plan_path)
    result.setdefault("strategy", strategy)

    # Normalize to a stable "scene.png" filename regardless of which
    # strategy ultimately produced the result, so downstream consumers can
    # rely on a consistent path convention (matching single_pass/progressive
    # behavior). The original path (e.g. scene_asset_composition.png) is kept
    # for audit purposes.
    canonical_path = target_dir / "scene.png"
    raw_path = Path(result["image_path"])
    if raw_path.resolve() != canonical_path.resolve():
        import shutil as _shutil
        _shutil.copy(raw_path, canonical_path)
        result["raw_image_path"] = str(raw_path)
        result["image_path"] = str(canonical_path)

    # Persist final pipeline metadata for audit (extends plan.json pattern).
    try:
        metadata_path = target_dir / "pipeline_result.json"
        metadata_path.write_text(json.dumps(result, indent=2, default=str))
    except Exception as e:
        print(f"Could not save pipeline metadata: {e}")

    return result
