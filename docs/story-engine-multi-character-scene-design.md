# Multi-Character Scene Generation — Design Document

Status: proposed
Companion doc: `docs/story-engine-multi-character-scene-investigation.md`

## 0. Investigation summary (what the codebase actually supports today)

Confirmed by reading `generators/image_generator.py`, `generators/image_engine.py`,
`models.py`, `core/scene_planner.py`, `core/scene_composer.py`, `utils/token_budget.py`,
`utils/face_check.py`:

- **Diffusion models are text-to-image only.** `generate_images()` only ever
  instantiates `StableDiffusionXLPipeline` or `FluxPipeline`. There is no
  `Img2Img`, `Inpaint`, `ControlNet`, or `IPAdapter` pipeline anywhere in the
  codebase, and `DIFFUSION_MODELS` in `models.py` lists only `sdxl` and
  `flux_dev`. **No reference-image conditioning, inpainting, or masking
  capability exists today.** Any design that assumes image-to-image editing
  is not implementable without adding new model downloads and pipeline code
  (large scope, explicitly out of scope per the task).
- `core/scene_composer.py::MultiStepSceneGenerator` already stages generation
  into "base_environment" then one layer per character, but each step is a
  **fresh independent text-to-image call** (comment at line 169: "For
  subsequent steps, we would ideally use inpainting... For now, generate with
  modified prompt that references previous"). The previous image is only
  used as `shutil.copy`'d evidence, never fed back into the pipeline. This is
  the concrete gap the investigation doc identifies.
- `core/scene_planner.py` already implements the 3-stage LLM planning
  pipeline (context resolution, decomposition, token-budget fitting) using
  `generate_prompt_with_llm(..., model_name="phi3_mini")`. This is solid
  infrastructure and should be reused/extended, not replaced.
- `SEGMENTATION_MODELS` includes `facebook/detr-resnet-50-panoptic`
  (panoptic segmentation), already wired into `MODEL_PATHS["segmentation"]`
  and `DEFAULT_MODEL_CONFIGS`. This is locally available and usable for
  **background/character segmentation**, which is exactly what's needed for
  asset isolation and deterministic composition — no new model family
  required, just a new invocation path.
- `utils/face_check.py` provides `character_appears_in_image()` using the
  optional `face_recognition` library — usable today as an **automated QA
  signal** for identity preservation, not just a manual check.
- `utils/token_budget.py` has `CHARACTER_GEN_INSTRUCTIONS`/`SCENE_SAFE_STRIP`
  regex lists and a `TokenBudgetManager.truncate_to_tokens` fallback — good
  last-resort mechanism, but truncation is not semantic; LLM compression
  (already in `stage_c_token_budget_fit`) should be preferred.
- `core/scene_workflow.py::generate_scene_with_llm_orchestration` is the
  single existing high-level entry point tying planner + composer together.
  It currently branches on `plan.single_pass_feasible` and otherwise calls
  `MultiStepSceneGenerator`, i.e. today's "progressive generation" degrades
  to N independent unconditioned generations, which is exactly why identity
  is lost (as the investigation observed).

**Conclusion: this is not purely a prompting problem, and it is not solvable
by image-editing either, because no image-editing capability exists in this
project's model stack right now.** The sustainable path with *currently
available* models (SDXL/FLUX text-to-image + DETR panoptic segmentation) is
the **"character asset generation + deterministic composition"** strategy
described in the investigation doc, with progressive prompt-only generation
kept as a cheaper fallback for simple scenes, and true image-to-image /
ControlNet / IP-Adapter noted as a future upgrade path (not built now).

---

## 1. Architecture overview

```
Scene Request (description + character refs)
        │
        v
┌───────────────────────────┐
│ SceneStrategySelector (LLM)│  <- new, built on scene_planner infra
└───────────────────────────┘
        │
        ├─ COMPLEXITY = simple (0-1 characters, short prompt)
        │       └──> Single-pass text-to-image (existing generate_scene)
        │
        ├─ COMPLEXITY = moderate (2 characters, fits with compression)
        │       └──> Progressive prompt-only generation (existing
        │            MultiStepSceneGenerator, prompt continuity only —
        │            kept as a fallback, not primary path)
        │
        └─ COMPLEXITY = complex (2+ characters, strict spatial/identity
                        requirements, or moderate path fails QA)
                └──> Character-Asset + Deterministic Composition pipeline (NEW)
                             │
                             ├─ 1. Generate background/environment (text2img)
                             ├─ 2. Generate each character asset
                             │      (text2img, scene-specific pose/clothing,
                             │       identity-preserving prompt, simple/plain
                             │       background for easy segmentation)
                             ├─ 3. Segment each character asset
                             │      (DETR panoptic -> largest "person" mask,
                             │       or rembg/simple heuristic fallback)
                             ├─ 4. Deterministic composition
                             │      (PIL/np: place, scale, layer-order,
                             │       soft-edge blend, approximate shadow)
                             ├─ 5. Optional AI refinement
                             │      (NOT true img2img; see §5 — a bounded,
                             │       explicitly-flagged experimental step)
                             └─ 6. Automated QA (face_check + heuristics)
                                     -> pass/fail -> pick best candidate
```

Everything downstream of the LLM strategy decision is orchestrated by a new
`core/scene_pipeline.py` module that composes the existing planner
(`core/scene_planner.py`), a new character-asset generator, a new
segmentation/composition module, and the existing image engine — it does not
replace any existing module, it sequences them.

---

## 2. Leveraging existing LLM planning infrastructure

`core/scene_planner.py` already gives us Stage A (context resolution), Stage
B (decomposition), Stage C (token fitting). Extend it, don't replace it:

### 2.1 New Stage D: Strategy Selection

Add a function `stage_d_select_strategy(scene_description, resolved_characters, plan) -> StrategyDecision`
in `scene_planner.py`, called by a new orchestrator (`scene_pipeline.py`)
right after Stage B/C run. It should ask the LLM (same `phi3_mini` call
pattern) to decide among the three strategies given:

```json
{
  "strategy": "asset_composition",
  "reason": "2 characters with distinct spatial positions and required identity preservation; single-pass previously failed to bind identity",
  "character_count": 2,
  "requires_spatial_precision": true
}
```

Deterministic guardrails (cheap pre-filters before even calling the LLM, to
save latency/cost — this is a heuristic gate, not a heuristic *policy*):

- 0 characters → always `single_pass`.
- 1 character → `single_pass` unless the token budget after Stage C
  compression still exceeds the CLIP limit (77 tokens) → then
  `progressive` (background first, then character in isolation, still
  prompt-only since there's no binding problem with only one subject).
- ≥2 characters → ask the LLM to choose between `progressive` and
  `asset_composition`; default to `asset_composition` when the scene
  specifies explicit distinct positions/spatial layout for each character
  (this is precisely the failure mode identified in the investigation).

This keeps hard-coded heuristics only for the trivial/cheap cases, per the
investigation's explicit request to avoid unnecessary heuristics, while
avoiding LLM round-trips for scenes that obviously don't need them.

### 2.2 Extending `ResolvedCharacter` for asset generation

`ResolvedCharacter` (Stage A output) already separates `identity` vs
`scene_presentation` vs `dropped`. This is exactly the payload the character
asset generator needs — no new extraction step required. Add two optional
fields used only by the asset-composition path:

```python
@dataclass
class ResolvedCharacter:
    ...
    scene_pose: Optional[str] = None      # e.g. "sitting on a chair"
    scene_action: Optional[str] = None    # e.g. "playing a black Gibson Explorer guitar"
    scene_position_hint: Optional[str] = None  # e.g. "left third of frame"
```

Extend the Stage A LLM prompt to also emit `scene_pose`, `scene_action`,
`scene_position_hint` per character (the LLM is already reading the full
scene description, so this is additive, not a new pass).

### 2.3 `ScenePlan` gains composition metadata

Add to `ScenePlan`:

```python
strategy: str  # "single_pass" | "progressive" | "asset_composition"
canvas_layout: Optional[Dict[str, Any]] = None  # positions/scale/z-order hints
```

`canvas_layout` is produced by Stage B (which already knows region hints per
layer) — e.g.:

```json
{
  "width": 1024, "height": 1024,
  "placements": [
    {"name": "Nikita", "anchor": [0.28, 0.62], "scale": 0.42, "z": 1},
    {"name": "Roger",  "anchor": [0.72, 0.62], "scale": 0.42, "z": 1}
  ]
}
```

This gives the deterministic compositor (section 4) explicit, LLM-reasoned
placement instead of guessing from `region_hint` strings.

---

## 3. Character asset generation workflow

New module: `core/character_asset_generator.py`.

### 3.1 Input

A `ResolvedCharacter` (with `identity`, `scene_presentation`,
`scene_pose`, `scene_action`) plus project style.

### 3.2 Prompt construction (reuses existing pieces)

```
{identity traits joined}, {scene_presentation}, {scene_pose}, {scene_action},
{project_style}, full body shot, entire body visible from head to toe,
simple plain neutral background, studio lighting, centered subject,
photorealistic detail on face and hands
```

- Reuses `core/prompt_decomposer.py::build_appearance_prompt`-style
  assembly and `utils/token_budget.py` stripping/priority logic — the same
  budget-fitting Stage C function in `scene_planner.py` is called per
  character asset prompt (each asset prompt has its own 77-token budget,
  since it's generated independently, which is actually *easier* than the
  combined-scene budget problem).
- Deliberately **forces a plain/neutral background** ("simple plain neutral
  background", "studio lighting", "centered subject") — this is the
  low-tech trick that makes segmentation reliable without training/using
  a matting model: DETR panoptic segmentation (or even a simple
  saliency/GrabCut fallback) works far better on high-contrast plain
  backgrounds than on the final busy bar/stage background.
- Uses the *same* seed-per-character strategy already used for
  reproducibility (`seed=42 + character_index`), and reuses
  `generate_images()` from `generators/image_engine.py` unchanged — no
  new pipeline classes needed for this step, it's still SDXL/FLUX
  text-to-image.

### 3.3 Output

```python
@dataclass
class CharacterAsset:
    name: str
    image_path: str          # raw generated asset (plain background)
    mask_path: Optional[str] # segmentation result (section 4)
    cutout_path: Optional[str]  # RGBA isolated character
    bbox: Optional[Tuple[int,int,int,int]]
    prompt_used: str
```

### 3.4 Identity consistency across multiple character images

SDXL/FLUX have no native "reference image" conditioning, so identity
consistency for a given character *within one scene* relies on:
1. Reusing the exact same `identity` trait string verbatim (Stage A already
   guarantees this — "NEVER drop identity traits; copy them verbatim").
2. Reusing a **fixed per-character seed** (derived deterministically from the
   character name/id, not scene number) so the same character always starts
   from a similar noise basin across scenes — an approximation, not a
   guarantee, but free and already partially present in the codebase's
   seed-handling pattern.
3. Recording `prompt_used` and seed in scene metadata (extends the existing
   `plan.json`/`llm_plan.json` audit trail pattern already in
   `scene_planner.py`/`scene_workflow.py`) so mismatches are debuggable.

This is explicitly flagged as a **known limitation**: true identity
preservation across independent generations would require IP-Adapter/
InstantID/LoRA-per-character, none of which exist in this repo today (see
§5, future work).

---

## 4. Composition strategy without true image editing

New module: `core/scene_compositor.py`. Pure deterministic image ops (PIL +
numpy), no diffusion calls.

### 4.1 Segmentation

```python
def segment_character(image_path) -> (mask, cutout_rgba):
```

- Primary: use the already-available `facebook/detr-resnet-50-panoptic`
  model (`SEGMENTATION_MODELS["detr_resnet_50_panoptic"]`,
  `MODEL_PATHS["segmentation"]`) via the `transformers` panoptic
  segmentation pipeline (already a dependency — `transformers` is in
  `.venv`), select the "person" class mask with the largest area.
- Fallback (if segmentation model unavailable/not installed): since the
  asset was generated against a plain/neutral background (§3.2), a simple
  background-color flood-fill / chroma-distance threshold (GrabCut seeded
  from the border, via `opencv-python` if available, else a naive
  histogram-based background removal) produces an adequate cutout. This
  fallback must degrade gracefully (log a warning, keep working) rather than
  fail the whole pipeline — same philosophy as `utils/face_check.py`'s
  optional-dependency handling.
- Output: alpha-matted RGBA cutout + bounding box.

### 4.2 Deterministic placement

```python
def compose_scene(background_path, assets: List[CharacterAsset],
                   canvas_layout: Dict) -> Image:
```

Using `canvas_layout` from Stage B (§2.3):
1. Load background image at target canvas size.
2. For each character (sorted by `z`), scale the cutout so its bounding-box
   height matches `scale * canvas_height`, anchor its bottom-center at
   `anchor` (normalized coords) to approximate "feet/base on the stage
   floor" placement, and alpha-composite onto the canvas.
3. Cheap shadow approximation: render a soft dark ellipse under the
   character's anchor point (blurred, low-opacity) before compositing the
   cutout — purely deterministic, not AI.
4. Optional edge feathering (Gaussian blur on the mask edge before
   compositing) to reduce hard cutout edges.

This directly implements the investigation's "deterministic composition
layer" — position, scale, z-order, and blending are explicit and
inspectable, not left to the diffusion model to infer from text.

### 4.3 Why this is better than progressive img2img here

Given the model stack constraint (no image editing at all), deterministic
compositing is strictly more controllable than prompt-chained regeneration
(the current `MultiStepSceneGenerator` behavior), because placement is
exact rather than hoped-for through prompt wording like "on the left side of
the stage."

---

## 5. Optional AI refinement (bounded, explicitly experimental)

Because there is **no img2img/inpainting pipeline in this codebase**, a
literal "refine the composed image while preserving structure" step (as
suggested in the investigation doc) is **not implementable today** without
adding a new pipeline class (e.g., `StableDiffusionXLImg2ImgPipeline` /
`FluxImg2ImgPipeline` from the already-installed `diffusers` package — the
classes exist in the dependency but are never imported/used in this repo).

Recommendation:
- **Phase 1 (this design): ship without AI refinement.** The deterministic
  composite from §4 is the final output. This keeps the whole feature
  buildable with zero new model downloads or pipeline code.
- **Phase 2 (flagged future work, separate design/approval needed):** add an
  optional low-strength img2img pass (`strength` ~0.15–0.3) using
  `StableDiffusionXLImg2ImgPipeline` seeded from the composite, purely to
  blend lighting/shadow edges. This is a genuinely new capability
  (image-to-image) and must be scoped, tested, and gated behind a config
  flag (`enable_ai_refinement=False` by default) since it changes pixels
  the deterministic compositor already placed correctly and risks
  destroying identity/position guarantees. Not part of this implementation.

---

## 6. Fallback strategies

Layered fallbacks, mirroring the defensive patterns already used throughout
the codebase (`LLM_AVAILABLE` fallback in `scene_planner.py`, optional
`face_recognition` in `face_check.py`, truncation fallback in
`token_budget.py`):

| Failure point | Fallback |
|---|---|
| LLM unavailable (Stage A–D) | `scene_planner.py` already falls back to KEEP-all-fields / single fallback plan; Stage D defaults to the pre-LLM heuristic gate (§2.1) instead of failing |
| Segmentation model unavailable | Chroma/GrabCut-based background removal (§4.1); if that also fails, place the character asset as an opaque rectangle crop (visibly worse but non-fatal) and log a warning, so the pipeline never crashes |
| Character asset generation produces a bad cutout (e.g., segmentation captures <30% expected area) | Regenerate the asset once with a fixed alternate seed; if still bad, fall back to the *progressive prompt-only* path for that character (i.e., degrade from asset-composition to progressive strategy for the whole scene) |
| Asset-composition scene fails QA (see §7) | Retry once with strategy=`progressive`; if that also fails QA, return the best-scoring candidate of the two with a warning flag in the result metadata (never silently return a broken scene as if it succeeded) |
| Token budget still exceeded after LLM compression (Stage C) | Existing `TokenBudgetManager.truncate_to_tokens` hard truncation (already implemented) |
| No characters detected | Existing direct `generate_scene()` single-pass path (already implemented in `scene_workflow.py`) |

This "always degrade, never crash, always flag" approach matches the
project's existing defensive coding style.

---

## 7. Automated QA (new, small addition)

Add `core/scene_qa.py` using the already-present `utils/face_check.py`:

```python
def qa_scene(final_image_path, character_assets: List[CharacterAsset]) -> QAResult:
    # For each character, check that character_appears_in_image(asset.cutout_path, final_image_path)
    # returns True (or None=inconclusive, treated as "pass with warning").
```

Used by the orchestrator to:
1. Decide whether to accept an asset-composition result as final.
2. Decide whether to fall back to progressive generation (§6).
3. Attach a `qa` block to the scene's saved metadata (extends existing
   `plan.json`/`llm_plan.json` audit pattern) so QA results are inspectable
   later, not just used transiently.

If `face_recognition` isn't installed, QA is skipped (returns
inconclusive/pass), matching the existing optional-dependency philosophy —
this feature must not introduce a hard new dependency.

---

## 8. Summary of new/changed files

| File | Change |
|---|---|
| `core/scene_planner.py` | Add Stage D (`stage_d_select_strategy`), extend `ResolvedCharacter` and `ScenePlan` |
| `core/character_asset_generator.py` | **New.** Generates per-character, scene-specific, plain-background assets |
| `core/scene_compositor.py` | **New.** Segmentation + deterministic placement/compositing |
| `core/scene_qa.py` | **New.** Wraps `utils/face_check.py` for automated identity QA |
| `core/scene_pipeline.py` | **New.** Top-level orchestrator: strategy selection → asset generation → composition → QA → fallback, replacing ad-hoc branching currently in `core/scene_workflow.py` |
| `core/scene_workflow.py` | Refactor `generate_scene_with_llm_orchestration` to delegate to `scene_pipeline.py` (keep as thin backward-compatible wrapper) |
| `core/scene_composer.py` | Kept as-is; used unchanged for the `progressive` strategy fallback path |
| `utils/token_budget.py` | No changes required (reused as-is) |

No changes to `generators/image_generator.py` or `models.py` — no new
diffusion model families are required for the default (Phase 1) build.

---

## 9. Non-goals / explicitly deferred

- True image-to-image, inpainting, ControlNet, IP-Adapter, InstantID: not
  present in the codebase, not added in this design (Phase 2, separate
  proposal, per §5).
- Perfect pixel-identical identity preservation across independently
  generated character assets: approximated via verbatim identity strings +
  fixed per-character seeds; not guaranteed without adapter/LoRA-based
  conditioning.
- Automatic perspective-correct 3D placement: the compositor uses simple
  anchor+scale placement, not full perspective/depth-aware warping.
</content>
