# Image-to-Video (i2v) — Definitive Specification

> **Status:** archived. Image-to-video generation has been archived due to
> hardware constraints (see §8). This document is preserved for reference; i2v
> model re-introduction requires a machine with >= 80 GB unified memory or
> quantized/MLX-ified models.
>
> Operating principles are informed by the working reference implementation
> (production i2v in that reference: Wan 2.2 ComfyUI/xfuser, LTX-2,
> cloud backends), adapted to this codebase's diffusers-based runtime.
> Story Engine's chosen i2v model is **LTX-Video 0.9.5** (the LTX lineage).

---

## 1. Domain workflow

Video generation is **image-to-video (i2v)**: we never generate video from
text alone. A still image is produced first, validated, and only then
animated.

```
character reference ──► scene image ──► face validation ──► i2v animation ──► audio + lip sync
        (DB)                (image engine)   (gate)          (video engine)      (post)
```

The pipeline is a hard sequence — each stage depends on the previous one:

1. **Character** — `generate_character(name, prompt, model="flux_dev")`
   persists a reference image + prompt + seed + model in the database
   (`characters` table) under `outputs/<project>/characters/<name>/`.
2. **Scene** — `create_validated_scene(prompt, character_name=...)` in
   `generators/video_engine.py` resolves characters by name from the DB,
   generates the scene image, and runs face recognition against the
   character reference.
3. **Validation gate** — a scene is only eligible for video once every
   listed character is verified in it. On failure the scene is regenerated
   with a new seed (bounded retries, default 3). If verification is
   required but impossible (`require_verification=True`), the pipeline
   aborts instead of silently producing unverified video.
4. **Animation** — the validated scene becomes video via
   `animate_scene(scene)` (single model) or `benchmark_scene_video(scene)`
   (all models).
5. **Audio + lip sync** — most scenes contain a speaking character and
   background music. The spoken line is synthesized (TTS), the background
   track is generated (music), and the speaking character's mouth is
   animated to match the audio (lip sync). See §10.

> **Benchmark policy (i2v model reduction):** i2v generation has been
> archived. The benchmark suite (`make benchmark-video`) is disabled until a
> suitable model and machine are available. Future work: re-introduce when a
> machine with >= 80 GB unified memory is available for LTX-Video or when models
> are quantized/MLX-ified for local execution.

## 2. Architecture

| Component | Responsibility |
|---|---|
| `models.py` | `IMAGE_TO_VIDEO_MODELS` registry, `MODEL_PATHS["image_to_video"]`, `get_model_config("image_to_video")` → `(device, torch_dtype)`; plus audio registries: `TEXT_TO_SPEECH_MODELS`, `VOICE_GENERATION_MODELS`, `LIP_SYNC_MODELS`, `MUSIC_GENERATION_MODELS` |
| `generators/video_generator.py` | Low-level i2v: per-model pipelines, generation params, video + metrics output |
| `generators/video_engine.py` | High-level orchestration: validated scene → animation, benchmarking |
| `generators/benchmark_video_generator.py` | Benchmark suite (café conversation, all models) |
| `utils/project_paths.py` | Canonical project-scoped output layout |
| `utils/model_metrics.py` | RSS memory sampling for peak-memory metrics |
| `Makefile` | `make benchmark-video` runs the full i2v benchmark suite |

### Model resolution

`video_generator.resolve_video_model_path()` prefers the locally installed
copy at `models/image_to_video/<model_name>/` (downloaded by
`scripts/install.py`, `make install`). If absent it warns and falls back to
downloading from the Hugging Face hub id at runtime.

## 3. Models

| Key | Model | Repo | Default |
|---|---|---|---|
| — | — | — | — |

- **Status (2026-08):** ALL i2v models have been archived due to hardware
  constraints. The `IMAGE_TO_VIDEO_MODELS` registry in `models.py` is empty
  (`ltx_video_095_i2v` and prior models such as Wan 2.2 I2V A14B and
  HunyuanVideo-I2V have been removed). Re-introduction requires a machine with
  >= 80 GB unified memory or quantized/MLX-ified model weights.
- The registered repo and diffusers-format checkpoint information is preserved
  in `MODEL_METADATA` for reference, but no models are loaded or invoked.
- Device/dtype configuration via `get_model_config("image_to_video")` is
  preserved but will default to CPU with float32 when no model is selected.

### Audio & lip-sync model registries

Registered in `models.py` alongside the i2v models (all verified
identifiers, sizes from Hugging Face). One winner per category; small
alternatives kept where useful. This project is **CC BY-NC 4.0
(non-commercial, personal)** — see `LICENSE.md` and `CONTRIBUTING.md` — so
model license permissiveness is not the deciding factor; size and quality
are:

| Registry | Models | Size | Why this one |
|---|---|---|---|
| `TEXT_TO_SPEECH_MODELS` | `qwen3_tts` (`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`), `qwen3_tts_base` (`Qwen/Qwen3-TTS-12Hz-0.6B-Base`) | 4.2 GB / 2.3 GB | Apache-2.0 SOTA; CustomVoice = voice clone, Base = light fallback |
| `LIP_SYNC_MODELS` | `latentsync_1_6` (`ByteDance/LatentSync-1.6`) | 9.0 GB | Best fidelity; 1.5 dropped (bigger, 9.2 GB), MuseTalk dropped (256×256, lower fidelity), Wav2Lip dropped (research-grade, GitHub-only) |
| `MUSIC_GENERATION_MODELS` | `musicgen_medium` (`facebook/musicgen-medium`) | 11.1 GB | Smallest viable; large (19 GB) dropped for +8 GB marginal gain, stable-audio (14.6 GB) dropped for size |

- Voice cloning is a capability of the TTS models (Qwen3-TTS CustomVoice
  *is* the clone model), so no separate voice registry is kept.
- Cloud/API TTS (ElevenLabs, MiniMax) and music (MiniMax music-01) were
  dropped: paid keys, no disk-cost benefit.
- `scripts/install.py` downloads these Hugging Face-sourced entries
  (`source: huggingface` in `MODEL_METADATA`) into their `MODEL_PATHS`
  directories; each category has a `DEFAULT_MODEL_CONFIGS` entry
  (float16, cpu by default) so `get_model_config("lip_sync")` etc.
  resolve correctly.

## 4. Inputs

### Conditioning image

- Must be an **already-validated scene image** (see §1). i2v is never fed an
  unverified scene.
- Loaded, converted to RGB, and resized with Lanczos to the model's target
  `width × height`.
- **Aspect preservation:** the conditioning frame is *fit, never stretched*. When
 the source already matches the model's target aspect ratio it is a clean Lanczos
resize; when it does not (e.g. a square 1024×1024 scene) the frame is *contain*-fit
 into the target box and the leftover edges are filled by *edge-replication*
 (border padding) — so the picture is never distorted. This used to squash a
 square scene and make the "aspect ratio look off / movements distort the video".
- Benchmark/ideal practice: generate the scene at the video target resolution
 (1024×576 for LTX-Video 0.9.5, now the native output) so the conditioning frame
 matches the output (reference: the reference implementation renders the first
 frame at the exact video resolution). The scene pipeline threads `width`/`height`
 through so the *scene* and the *video* share one 16:9 frame.

### Prompt

- The **enriched scene prompt** (`scene["enriched_prompt"]`, falling back to
  `scene["prompt"]`) is used as the video prompt. It must describe both the
  **scene content** and the **motion** ("cinematic motion", gestures,
  camera behaviour) — the model needs explicit motion language.
- Negative prompts: per-model defaults are defined in
   `MODEL_GENERATION_PARAMS`. The LTX-Video 0.9.5 negative block forbids
  static/still-frame results, blur, text/subtitles, deformities, extra limbs
  and a messy background.
- Reference operating rule: a good i2v prompt is a short cinematic beat
  ("Yamu killing a tiger with a long bow arrow, dramatic action, the arrow
  flies and strikes the tiger, cinematic motion") — scene + action +
  motion style, and **never** text/symbols/numbers in frame.

### Seed

- Fixed seed (default `42`) for reproducibility. Benchmark runs use the
  same seed across models so differences are attributable to the model.

## 5. Per-model generation parameters

Base parameters (`MODEL_GENERATION_PARAMS`):

| Model | Resolution | Frames | FPS | Guidance | Steps | Negatives |
|---|---|---|---|---|---|---|
| `ltx_video_095_i2v` | 576×1024 | 81 | 25 | 3.0 | 64 | yes |

**Frame-count rule (8k+1):** LTX-Video's VAE has a temporal stride of 8, so
`num_frames ≡ 1 (mod 8)`; 81 (= 1 + 8*10) satisfies this. 81 frames @ 25 fps
is a **~3.24 s** clip — appropriate for the new vertical benchmark target of
~3 seconds.

**Where the model parameters live (single source of truth, 2026-08):** every i2v
model's resolution / frame count / fps / guidance is owned by
`video_generator.MODEL_GENERATION_PARAMS`, **not** by the benchmark. The café
benchmark no longer carries its own `BENCHMARK_VIDEO_PARAMS` override — when it
generates videos it must not pass model-specific parameters, so it supplies only
the conditioning image, motion prompt, model name, output location and a fixed
seed and lets `video_generator` apply each model's native parameters. The new
vertical benchmark target uses ~3.24 s (81 frames @ 25 fps).

| Model | Resolution | Frames | FPS | Duration |
|---|---|---|---|---|
| `ltx_video_095_i2v` | 576×1024 (9:16) | 81 | 25 | ~3.24 s |

> **Vertical 9:16 benchmark configuration:** the native output moved to
> **576×1024 (9:16)** so the conditioning image and the video share one
> aspect ratio (no more squished source — see §4). The original landscape
> 1024×576 (16:9) default has been replaced by this vertical configuration
> for the 2026 benchmark. The benchmark target is ~3.24 s (81 frames @ 25 fps)
> focused on subtle motion and character consistency. The post-generation
> **enhancement pass** (§11) — and its opt-in *super-resolution seam* (§11) —
> is the path to longer durations and different aspect ratios, without
> re-loading a heavy i2v model and **without auto-downloading** anything.

Every generation is wrapped in timing (`duration_ms`) and RSS sampling
(`peak_memory_mb`) and the pipeline is torn down (`cleanup_pipeline`) in a
`finally` block so consecutive model runs do not accumulate VRAM/RAM.

## 6. Outputs & project layout

```
outputs/<project>/
├── characters/<name>/reference.png     # character reference
└── scenes/
    ├── scene_<n>/scene.png             # validated scene image
    └── scene_<n>/out/
         ├── scene_<n>_<model>.mp4        # raw single-model animation (H.264 MP4)
         ├── scene_<n>_<model>_benchmark_metrics.json
         ├── benchmark_<model>.mp4        # raw benchmark run output
         ├── benchmark_<model>_benchmark_metrics.json
         ├── *_*_enhanced_enhanced.mkv    # high-quality enhanced clip (see §11)
         └── *_*_enhanced_metrics.json    # enhancement metrics next to it
```

- Videos: H.264 MP4 (`diffusers.utils.export_to_video`, model fps).
- Metrics JSON next to every video:

```json
{
   "model": "ltx_video_095_i2v",
    "prompt": "<video prompt>",
   "image": "<scene image path>",
   "seed": 42,
   "fps": 25,
   "duration_ms": 123456,
   "peak_memory_mb": 8123,
   "output": "<video path>",
    "width": 1024, "height": 576,
    "num_frames": 161, "guidance_scale": 3.0, "num_inference_steps": 64
}
```

The default project is the mock `test_project` until real project wiring
lands.

## 7. Benchmark suite

Run with `make benchmark-video` (or
`python generators/benchmark_video_generator.py`).

**Goal:** produce the same validated scene, animated by **every** i2v model,
so outputs are directly comparable. The suite follows the rules:

- Face verification is **soft** (`require_verification=False`): both
  characters are checked for presence in the scene, but a non-match or an
  inconclusive check does **not** abort the run — the benchmark's purpose is to
  see whether a video *can* be generated, not to gate it on perfect face
  detection.
- Videos are ≥ 4 s (see §5 overrides), seeded identically, named
  `benchmark_<model>.mp4`. The ≥ 720p bar was dropped with Wan (see §5).
- A summary prints path, resolution, duration, latency and peak memory for
  each model.

### The benchmark scenario — "Café conversation (vertical 9:16)"

The definitive test scene for the vertical 9:16 LTX Video benchmark:
- **Aspect ratio:** 576×1024 (9:16), optimized for mobile / Instagram Story /
  Reels-style presentation on a phone.
- **Duration:** ~3.24 seconds (81 frames @ 25 fps), focused on a single simple
  action.
- **Purpose:** test whether LTX Video can maintain character identity, facial
  consistency, body anatomy, clothing consistency, scene consistency, temporal
  stability, and natural subtle movement over ~3 seconds.
- **Characters:** Nikita and Roger — both visually consistent throughout the
  shot.
- **Scene:** Nikita and Roger are sitting together at a small table in a quiet,
  realistic café during the morning. Nikita looks at Roger and makes a subtle
  natural movement, gently turning her head toward him and smiling. Roger remains
  mostly still and looks at Nikita. Both characters remain visually consistent
  throughout the shot. Natural breathing and subtle facial movement. Stable
  camera, realistic lighting, realistic anatomy, cinematic composition.
  Minimal movement, calm and natural acting.
- **Motion:** deliberately uses very simple and controlled motion — subtle head
  movement, natural blinking, breathing, slight body movement, looking toward
  another character, small hand movements, subtle facial expressions. Slow
  camera movement, if any. Static camera preferred.
- **Camera:** stable cinematic camera. Prefer static camera, subtle push-in,
  subtle pull-out, very slow camera movement. Avoid handheld camera shake,
  rapid camera movement, dramatic camera rotations, fast tracking shots.
- **Motion forbids:** complex body movements, running, jumping, dancing,
  fighting, fast camera movement, multiple simultaneous actions, complicated
  interactions between characters, exaggerated gestures, rapid changes in pose,
  large movements across the frame.
- **Negative prompt:** concise, focused on observed failure modes:
  distorted face, warped face, deformed anatomy, distorted body, extra limbs,
  deformed hands, temporal flickering, frame artifacts, warped background,
  unnatural motion, unstable identity, blurry details, duplicated body parts.

### Comparing results

- **Quantitative:** the per-model `*_benchmark_metrics.json` files
  (latency, peak memory, resolution, frames, fps).
- **Qualitative:** inspect `scene_*/out/benchmark_<model>.mp4` for motion
  quality, character consistency and prompt adherence.
- Benchmark decision (2026-08): `ltx_video_095_i2v` is the reference/surviving
   model. Wan 2.2 I2V A14B was dropped because it OOMs on the 64 GB host, and
   HunyuanVideo-I2V was dropped for architecture/size — see §3.

## 8. Operating constraints (this machine) — ARCHIVED

> **Status: i2v generation ARCHIVED.** Video generation has been disabled
> due to insufficient GPU/unified memory on this machine.
>
> Machine: Apple M5 Pro MacBook Pro (Mac17,9), 18-core CPU / 20-core GPU,
> Metal 4, 64 GB unified memory, torch 2.13 (MPS), diffusers 0.39.0.
>
> **Verdict: memory is the binding constraint.** LTX-Video 0.9.5 (~3.6 GB
> transformer, ~24 GB total) nearly fits but ultimately requires more VRAM than
> available for reliable operation without offload. ALL i2v models have been
> archived.
>
> **Required for re-introduction:**
> - A machine with >= 80 GB unified memory (e.g., Mac Pro with M2 Ultra, or
>   a multi-GPU workstation), **or**
> - Model quantization/MLX-ification to reduce VRAM footprint below 32 GB,
>   **or**
> - GPU offload to a discrete NVIDIA accelerator with >= 16 GB VRAM.
>
> Practical rules (archived):
> 1. **Memory decides the model.** Insufficient VRAM prevents any i2v model
>    from loading, regardless of resolution/frame count adjustments.
> 2. Future re-introduction requires addressing the memory gap before any
>    configuration changes are meaningful.

## 9. Verification & tests

- `test_video_engine.py` covers: project path layout, character service
  persistence, per-model params completeness, invalid model/image rejection,
  benchmark fan-out across all models (incl. failure tolerance), scene
  retry-until-verified (same folder, seed bump), unknown-character
  rejection, and face-verification policy (required vs. inconclusive).
- Enhancement coverage: `test_video_enhancer.py` (temporal smoothing, denoise,
  read/write round-trip, end-to-end `enhance_video`, and the opt-in
  super-resolution no-download seam), `test_video_generator_aspect.py`
  (aspect-preserving `_prepare_image`: clean-resize vs. contain+pad), and
  `test_video_engine_enhance.py` (`animate_scene(enhance=…)` wiring +
  graceful degradation).
- Run `make test` before committing; the suite must pass.
- `make benchmark-video` is the end-to-end smoke check for the i2v stack.

## 10. Talking scenes: dialogue, lip sync & background music

Most scenes contain a **speaking character** and **background music**. The
audio stage turns a silent animated clip into a complete scene:

```
validated scene ──► silent i2v video
                        │
                        ├──► TTS speech (spoken line) ──┐
                        ├──► music generation ──────────┤──► mix ──► final scene
                        └──► lip sync (mouth ↔ audio) ──┘
```

### 10.1 Spoken dialogue (TTS / voice)

- The scene carries a **spoken line** (per-character dialogue, generated
  upstream like the scene prompt). The line is synthesized with a TTS
  engine; the character's voice is either a **voice clone** (from the
  character reference audio, if available) or a **designed voice**
  (natural-language description).
- Local defaults: `qwen3_tts` (voice cloning via the CustomVoice model;
  `qwen3_tts_base` 0.6B as a light fallback). Cloud alternatives were
  dropped for now (paid keys; no disk-cost benefit).
- Timing rule (from the reference implementation): speech length is measured with
  ffprobe (`get_audio_length`) and speed-fitted to the scene's video
  duration so dialogue and picture stay aligned.

### 10.2 Lip sync (mouth matches speech)

Realism requirement: **the speaking character's mouth must move in sync
with what is said**. Story Engine does this with a dedicated lip-sync model
applied to the animated clip + the TTS audio (the reference implementation relies on
native joint audio+video models like LTX-2; Story Engine's diffusers i2v
models are silent, so a post-hoc lip-sync stage is required).

- **Default: `latentsync_1_6`** (`ByteDance/LatentSync-1.6`) — diffusion
  lip sync, highest visual fidelity (~9 GB weights, needs ~18 GB VRAM).
- **Accepted risk:** LatentSync is CUDA-oriented; on the M5 Pro it may run
  flakily on MPS. That is acceptable — intermittent failures are tolerated
  as long as a usable final video is produced (retry on failure).
- The lip-sync pass must preserve the original frame rate and resolution of
  the i2v output; the output replaces the silent clip in `scene_*/out/`.

### 10.3 Background music

- Per-scene background track from `MUSIC_GENERATION_MODELS`:
  `musicgen_medium` (local).
- The track is mixed under the dialogue (ducked/at lower volume) and cut to
  the scene length.

### 10.4 Assembly

- Final scene = lip-synced video + dialogue + music, encoded H.264 MP4
  (AAC audio), same fps/resolution as the i2v output, saved to
  `scene_*/out/` alongside the silent clips and metrics.
- Benchmark scope note: the i2v benchmark measures the **silent** video
  models only; TTS / lip-sync / music models are evaluated separately in a
  later audio benchmark.

## 11. Post-generation enhancement

LTX-Video 0.9.5 is compact and fast, so its raw output can look a little "off":
the motion can flicker/warp between frames and diffusers' `export_to_video` writes
a lossy H.264 MP4. `generators/video_enhancer.py` adds an **opt-in enhancement
pass** over the generated clip that:

1. **reads** every frame of the clip,
2. **temporally smooths** jitter between frames (`temporal_smooth` — a moving
   window blend) to kill flicker/warping,
3. applies a **light spatial denoise** (`spatial_denoise` — per-frame Gaussian
   smoothing) to calm per-frame grain, and
4. **re-encodes** to a high-quality, low-CRF container (default **MKV**, CRF 12).

All of the base path is **dependency-free** (numpy + scipy + imageio's *bundled*
ffmpeg); **no new model is downloaded**, so the resident i2v footprint stays
unchanged.

**Entry point:**

```
enhance_video(video_path, output_dir=None, output_basename=None,
              fps=25.0, enhancement=None, sr_model=None, sr_dir=None)
   -> {"video_path", "metrics_path", "metrics"}
```

`DEFAULT_ENHANCEMENT = {temporal: 0.18, temporal_window: 1, denoise: 0.5,
container: "mkv", crf: 12, super_resolve: False}`; the `enhancement` dict
overrides any subset. Output is `<stem>_enhanced.<container>` plus a
`<stem>_enhanced_metrics.json` next to it (a trailing `_enhanced` in the source
name is stripped so re-runs don't stack suffixes). The metrics JSON records the
source/output paths, frame count/shape, fps, container, crf, the temporal/denoise
settings applied, `super_resolve`, `mean_abs_frame_change` (the mean per-pixel
change — 0 means no change), and `duration_ms`.

**Wiring:** `video_engine.animate_scene(..., enhance=True)` runs enhancer over the
raw clip and attaches `enhanced_video_path` / `enhanced_metrics_path` /
`enhanced_metrics` to the result; failures degrade gracefully to the raw clip. The
café benchmark (`benchmark_video_generator`) likewise writes an enhanced MKV per
model.

**Opt-in super-resolution (no auto-download):** `super_resolve_video(frames,
sr_model=None, sr_dir=None, scale=2)` is the "specialized video-enhancement
model" seam. It only runs when a local SR model is already installed
(`_resolve_local_sr_model` searches `models/super_resolution` and the i2v model
tree for an up-sampler); otherwise it returns the frames **unchanged** and prints a
warning. Nothing is ever fetched automatically — this honours the ROADMAP "do NOT
auto-download" constraint. Enable it explicitly with
`enhancement={"super_resolve": True, ...}` or by placing a local SR model on disk.

**Aspect ratio:** the enhancer preserves whatever resolution the input already has;
it never stretches. The 1024×576 (16:9) output (§5) and the aspect-preserving
`_prepare_image` (§4) are what keep the picture from looking squished in the first
place.

**Tests:** `test_video_enhancer.py`, `test_video_generator_aspect.py`, and
`test_video_engine_enhance.py` (see §9).