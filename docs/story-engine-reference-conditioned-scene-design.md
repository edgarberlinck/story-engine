# Story Engine — Holistic Reference-Conditioned Multi-Character Scene Design

Status: **Proposed** (pending review)
Owner: Story Engine agent

## 1. Goal

Pivot multi-character scene generation away from the current pipeline:

1. background text-to-image generation,
2. one text-to-image asset per character,
3. segmentation + validation (DETR panoptic / chroma-key / border flood-fill),
4. deterministic compositing (PIL paste),
5. SDXL img2img harmonization (`refine_composite`),

toward **one holistic diffusion invocation** that generates the complete scene while
conditioning on each stored character reference image as a FLUX.2 **reference latent**.

The primary benefit: in-context lighting, occlusion, perspective, interaction, and scene
coherence — with no pasted character cutouts, no background leakage, and no "sticker"
artifacts. This mirrors how the reference project works: one
cinematic scene prompt describing all subjects in context, character reference images fed
as reference-latent conditioning, and **no segmentation/compositing at all**.

## 2. Non-goals

- Do NOT remove `single_pass`, `progressive`, or `asset_composition`.
- Do NOT remove segmentation, composition, or `refine_composite`; they remain fallbacks for
  non-reference-capable models and future workflows.
- Do NOT alter character-reference persistence (character generation still saves
  `reference.png` + DB records as today).
- Do NOT require CUDA; the target environment is MPS-first.
- Do NOT run real diffusion in unit tests.
- Do NOT weaken/delete existing tests; the 135-test baseline must remain green.
- Do NOT make reference conditioning mandatory when no usable character reference exists.
- Do NOT treat face QA as a hard failure; it remains best-effort diagnostics.

## 3. Verified current state

- `generators/image_engine.py::generate_scene()` delegates scenes with >= 2 resolved
  characters to `core/scene_pipeline.generate_scene_pipeline()` (line 326).
- `generate_scene_pipeline()` runs `LLMScenePlanner.plan_scene()` (Stages A/B/C/D), then
  executes a selected strategy: `single_pass` / `progressive` / `asset_composition`. The
  multi-character path is `_run_asset_composition()`.
- Character records already contain the reference path:
  `generators.image_engine.get_character(name, project)["reference_image"]`.
- `models.py::DIFFUSION_MODELS = {"sdxl": ..., "flux_dev": ...}`. `flux_klein` is NOT
  registered, but a local model dir exists at `models/diffusion/flux_klein/`.
- `diffusers==0.39.0` exposes `Flux2Pipeline`, `Flux2KleinPipeline`, and
  `Flux2KleinKVPipeline`. The KV pipeline's `__call__` accepts `image=...` (PIL or list of
  PIL) as reference-latent conditioning.

## 4. FLUX.2 / Diffusers findings

The local model at `models/diffusion/flux_klein/` is structurally complete:

- `model_index.json` with `_class_name: "Flux2KleinPipeline"` (distilled), `is_distilled: true`
- `flux-2-klein-4b.safetensors`
- `transformer/`, `vae/`, `text_encoder/` (Qwen3), `tokenizer/`, `scheduler/`

Both installed Klein pipeline classes accept `image=`:

- `Flux2KleinPipeline` — 50 steps default, supports `guidance_scale`, accepts reference images.
- `Flux2KleinKVPipeline` — purpose-built for reference conditioning (KV-cached), 4 steps
  default, does **not** accept `guidance_scale`.

**Recommendation:** use `Flux2KleinKVPipeline` for the reference scene backend. Explicit
`Flux2KleinKVPipeline.from_pretrained(local_path)` is safe even though `model_index.json`
declares `Flux2KleinPipeline`, because both classes share compatible required components
(scheduler, VAE, Qwen3 text encoder, tokenizer, transformer). Do NOT use generic
`DiffusionPipeline.from_pretrained()` for this path (that would resolve to the non-KV
class).

## 5. Model registry + loader fix

### 5.1 Registry entry

Add to `models.py::DIFFUSION_MODELS`:

```python
"flux_klein": "black-forest-labs/FLUX.2-klein-4B",
```

`resolve_model_path("diffusion", "flux_klein", ...)` (defined in
`generators/image_generator.py:77`) already resolves local-first to
`models/diffusion/flux_klein` and falls back to the hub id. No change needed there.

### 5.2 Fix the general image-generator loader bug

`generate_images()` currently branches `sdxl` / `flux_dev` and defaults everything else to
`StableDiffusionXLPipeline`. Registering `flux_klein` without a branch would try to load
Klein as SDXL and fail.

Add an explicit `model_name == "flux_klein"` loader branch using `Flux2KleinPipeline` for
ordinary text-only generation, and dispatch with `guidance_scale` + `max_sequence_length=512`.
Do NOT route `flux_klein` through the generic `"flux" in model_name.lower()` branch (that
passes `guidance_scale`, which the KV reference pipeline rejects).

## 6. New backend API

New module: `generators/reference_scene_generator.py`

```python
def generate_reference_conditioned_scene(
    prompt: str,
    reference_image_paths: list[str],
    *,
    model_name: str = "flux_klein",
    seed: int = 42,
    steps: int = 4,
    width: int = 1024,
    height: int = 1024,
    num_images: int = 1,
    task_name: str | None = None,
) -> list[str]:
```

Contract:

- `reference_image_paths` ordered + nonempty.
- `steps=4` matches the distilled KV default.
- Returns `list[str]` of PNG paths (like `generate_images`).
- Raises on real backend failure; orchestration owns fallback decisions.
- Avoid the `generate_images` bug of reusing one filename for multiple images; suffix
  numerics.

Module-level test seam:

```python
def load_reference_scene_pipeline(model_path: str, torch_dtype):
    return Flux2KleinKVPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
```

Tests patch this function (consistent with how `hf_pipeline` / `generate_images` are
patched elsewhere).

### Reference image preprocessing

1. Validate each path; open with `Image.open(path).convert("RGB")`.
2. Resize long side to max 512 preserving aspect ratio; no upscaling.
3. Round dimensions down to multiple of 16 (min 64 per side).
4. Keep in memory; don't overwrite stored references.
5. Skip unreadable references with a warning.

### Device / dtype / lifecycle

- `device, torch_dtype = get_model_config("diffusion")` (MPS-first; from `models.py:116`).
- `pipe = load_reference_scene_pipeline(model_path, torch_dtype).to(device)`.
- `generator = torch.Generator(device=device).manual_seed(seed)`.
- `cleanup_pipeline(pipe)` in `finally` (defined `generators/image_generator.py:115`).
- No `guidance_scale` for KV pipeline.
- MPS: start 1 image @1024, 1–2 references; consider 768 square as configurable fallback.

## 7. Reference capability policy

In `core/scene_pipeline.py`:

```python
REFERENCE_CONDITIONED_SCENE_MODELS = {"flux_klein"}

def supports_reference_conditioning(model: str) -> bool:
    return model in REFERENCE_CONDITIONED_SCENE_MODELS
```

Use an explicit set, not `"flux" in model.lower()` (flux_dev is not wired for reference).

## 8. Character reference collection

```python
def _collect_character_references(characters, project) -> list[dict]:
    # [{name, path}, ...] in character order
```

- For each resolved character name, call `get_character(name, project)` and read
  `record.get("reference_image")`; confirm the file exists.
- Per-character try/except; skip missing/unreadable; don't fail the whole scene.
- If no valid references: don't invoke the holistic backend; continue with the existing
  planner-selected strategy.
- If some characters have references: invoke holistic with available references, keep all
  characters in the text prompt, record missing names in metadata/warnings.

## 9. Prompt strategy

One scene-level prompt. Use the caller's scene prompt + a compact ordered identity binding
built from Stage A `ResolvedCharacter` data, matching reference ordinal/name order to
`image=[...]`. Example suffix:

```
Character-reference binding:
Reference image 1 is Nikita: young woman, long curly red hair, fair skin;
she is on the left side of the stage, sitting and playing a black guitar.
Reference image 2 is Roger: muscular dark-skinned man, bald;
he is on the right side behind a drum kit, playing drums.
Render both as the referenced individuals in one coherent wide cinematic shot.
```

- Preserve caller's setting/camera/action/spatial instructions.
- Use `ResolvedCharacter.identity / scene_presentation / scene_pose / scene_action /
  scene_position_hint`.
- Do NOT reinsert full stored character generation prompts (clothing/style artifacts).
- No 77-token CLIP cap; Klein supports `max_sequence_length=512`.

Helper:

```python
def _build_reference_conditioned_prompt(prompt, plan, references) -> str:
```

Map by name, not by index order assumption.

## 10. Wiring / dispatch

Keep `image_engine.generate_scene()` signature unchanged. Its existing delegation
(`if use_asset_pipeline and len(characters) >= 2: generate_scene_pipeline(...)`) stays.

Inside `generate_scene_pipeline()`, after the planner builds `plan`, before executing
Stage D strategy:

1. `_collect_character_references(characters, project)`
2. `supports_reference_conditioning(model)`
3. if capable AND >= 1 valid reference:
   - build holistic prompt
   - `_run_reference_conditioned_scene(...)`
   - QA + normalize to `scene.png`
   - persist metadata + return

This is an override of Stage D for reference-capable models. Run the planner first (smallest
safe change; Stage A supplies useful identity/action/position text). Later optimization: a
Stage-A-only planner method for the reference path.

New orchestration helper:

```python
def _run_reference_conditioned_scene(prompt, project, scene_number, plan, references, model, seed) -> dict:
```

- build prompt; call `generate_reference_conditioned_scene`
- copy first output to `outputs/<project>/scenes/<n>/scene_reference_conditioned.png`
- `_qa_scene(...)`
- return metadata:

```python
{
    "scene_number": scene_number,
    "image_path": ...,
    "prompt": prompt,
    "reference_conditioned_prompt": ...,
    "seed": seed,
    "model": model,
    "strategy": "reference_conditioned_single_pass",
    "reference_images": [{"name": ..., "path": ...}, ...],
    "missing_reference_characters": [],
    "refinement_applied": False,
    "qa": ...,
    "warnings": [],
}
```

## 11. Fallback / failure semantics

The new backend must never silently break a scene. Wrap the reference route in try/except:

- log exception; add warning:
  `Reference-conditioned generation failed (<reason>); falling back to <existing strategy>.`
- resume the pre-existing selected path (`asset_composition` / `progressive` / `single_pass`).
- metadata: `fallback_from`, `fallback_reason`.
- Do NOT fall back to text-only `generate_images(model_name="flux_klein")` (a load failure
  would affect that too); fall back to the established strategy + its selected model.

## 12. Refinement and QA

### Refinement

Skip `refine_composite()` by default for `reference_conditioned_single_pass`:

- no composite to harmonize;
- the scene was generated in one diffusion process;
- a subsequent img2img pass could change faces and undermine reference conditioning;
- saves MPS time/memory.

Keep `enable_refinement` + existing refinement for `asset_composition`. If refinement is
later offered for reference scenes, use a distinct opt-in flag
(`enable_reference_scene_refinement=False`) with low strength.

### QA

Continue `_qa_scene(final_image_path, character_names, project)`. The stored reference
images are now both generation references and QA references. Missing face-recognition
support / unavailable references -> `None`; only a definite `False` fails aggregate QA; QA
failure is a warning, never a discarded scene.

## 13. Testing plan

All new tests mock loaders/filesystem; none instantiate Klein or run MPS inference.

### `tests/test_reference_scene_generator.py`

- `test_load_reference_scene_pipeline_uses_kv_pipeline` — patch `Flux2KleinKVPipeline.from_pretrained`; assert KV loader gets exact path/dtype; ordinary Klein loader not called.
- `test_generate_reference_conditioned_scene_passes_pil_reference_list` — patch loader + device + cleanup; assert `image` is a list of PIL RGB, `num_inference_steps=4`, `max_sequence_length=512`, no `guidance_scale`, `num_images_per_prompt=1`.
- `test_reference_preprocessing_scales_long_side_to_512_preserving_ratio` — non-square large image -> long side 512; small image not upscaled.
- `test_generate_reference_conditioned_scene_returns_output_path_list` — two-image fake -> two distinct existing PNGs.
- `test_generate_reference_conditioned_scene_rejects_empty_references` — `ValueError` before pipeline load.

### Image-generator regression

- `test_flux_klein_is_registered` — key + repo id in `DIFFUSION_MODELS`.
- `test_generate_images_flux_klein_loads_klein_pipeline_not_sdxl` — patch resolution/device/`Flux2KleinPipeline.from_pretrained`; assert SDXL loader not called.
- `test_generate_images_flux_klein_uses_supported_text_only_kwargs` — normal Klein text-only receives `guidance_scale`.

### `tests/test_reference_conditioned_scene_pipeline.py`

- `test_collect_character_references_returns_existing_records_in_character_order`
- `test_collect_character_references_skips_missing_or_unreadable_records`
- `test_flux_klein_with_one_or_more_references_routes_to_holistic_backend` — asset/progressive not called; strategy `reference_conditioned_single_pass`.
- `test_sdxl_multi_character_scene_does_not_route_to_reference_backend`
- `test_flux_dev_multi_character_scene_does_not_route_to_reference_backend`
- `test_flux_klein_without_usable_references_uses_existing_strategy`
- `test_reference_backend_failure_falls_back_to_asset_composition`
- `test_reference_backend_failure_falls_back_to_progressive_when_plan_selected_progressive`
- `test_reference_holistic_path_skips_img2img_refinement`
- `test_reference_holistic_path_runs_existing_qa`

### Existing test updates

Tests asserting "2 characters always => asset_composition" need a model-capability guard.
Keep existing smoke tests on `model="sdxl"` so they test the fallback path, not accidentally
become Klein tests.

## 14. Risks / open questions

- **MPS memory**: 4B Klein + Qwen + VAE + multiple ref latents. Mitigate: 4 steps, refs <=512,
  1 image, release pipeline, configurable resolution.
- **model_index mismatch**: local declares `Flux2KleinPipeline`; backend loads
  `Flux2KleinKVPipeline`. Compatible in 0.39.0; verify with one manual MPS smoke test.
  Fallback: use ordinary `Flux2KleinPipeline` (still accepts `image=`) if KV load fails.
  Do NOT mutate `model_index.json` initially.
- **Dependency pin**: `diffusers==0.39.0` vs model metadata `0.37.0.dev0`. Don't loosen the
  pin without a verified reason.
- **Reference-name association**: relies on ordered prompt wording; verify 2-char scene for
  identity swaps, then refine wording.
- **>2 references**: cost grows with each; target 1–2 on MPS.

## 15. Rollout order

1. Register `flux_klein` + fix its normal text-only loader branch.
2. Implement + unit-test isolated reference-scene generator.
3. Manually verify `Flux2KleinKVPipeline.from_pretrained(local_path)` on MPS with 1 reference.
4. Add scene-pipeline routing + fallbacks.
5. Run full suite: `.venv/bin/python -m unittest discover -s . -p "test_*.py"`.
6. Run one 2-character reference-conditioned smoke generation; inspect load, ordering,
   coherence, `scene.png`, QA metadata, fallback on mocked load failure.
