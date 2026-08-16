# Image-to-Video (i2v) — Definitive Specification

> **Status:** authoritative. This document is the single source of truth for
> how image-to-video generation operates inside Story Engine. It replaces the
> former `docs/humans/video-generation.md`, `docs/llm/video-generation.md`,
> and `docs/video-generation-caveats.md`.
>
> Operating principles are informed by the working reference implementation
> (production i2v: Wan 2.2 ComfyUI/xfuser, LTX-2,
> cloud backends), adapted to this codebase's diffusers-based runtime.

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

> **Benchmark policy (i2v model reduction):** the benchmark suite exists to
> pick the winning image-to-video model. Once the benchmark comparison is
> final, **exactly one i2v model remains** in `models.py`; the losing model
> is removed from the project (registry, install script, params, tests).

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
| `wan22_i2v` | Wan 2.2 I2V A14B | `Wan-AI/Wan2.2-I2V-A14B` | **yes** |
| `hunyuan_video_i2v` | HunyuanVideo I2V | `tencent/HunyuanVideo-I2V` | no |

- `AVAILABLE_VIDEO_MODELS` = the full registry; `DEFAULT_VIDEO_MODEL` is
  `wan22_i2v`.
- Each model is invoked through its own diffusers pipeline class and
  model-specific parameters (`MODEL_GENERATION_PARAMS` in
  `video_generator.py`) — never a shared generic call.
- Device/dtype come from `get_model_config("image_to_video")`
  (`bfloat16`; device resolves mps > cuda > cpu).

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
- Benchmark/ideal practice: generate the scene at the video target
  resolution (e.g. 1280×720) so the conditioning frame matches the output
  (reference: the reference implementation renders the first frame at the exact
  video resolution).

### Prompt

- The **enriched scene prompt** (`scene["enriched_prompt"]`, falling back to
  `scene["prompt"]`) is used as the video prompt. It must describe both the
  **scene content** and the **motion** ("cinematic motion", gestures,
  camera behaviour) — the model needs explicit motion language.
- Negative prompts: per-model defaults are defined in
  `MODEL_GENERATION_PARAMS`. The Wan negative block forbids
  static/still-frame results, blur, text/subtitles, deformities and extra
  limbs. HunyuanVideo I2V uses no CFG negatives.
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
| `wan22_i2v` | 832×480 | 81 | 16 | 3.5 | 40 | yes |
| `hunyuan_video_i2v` | 720×480 | 61 | 15 | 6.5 | 30 | no |

**Frame-count rule (4k+1):** Wan and HunyuanVideo require
`num_frames ≡ 1 (mod 4)`. 81 and 61 both satisfy this.

Benchmark overrides (`BENCHMARK_VIDEO_PARAMS`) raise the bar to
**≥ 720p and ≥ 4 s**:

| Model | Resolution | Frames | FPS | Duration |
|---|---|---|---|---|
| `wan22_i2v` | 1280×720 | 81 | 16 | 5.06 s |
| `hunyuan_video_i2v` | 1280×720 | 61 | 15 | 4.07 s |

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
        ├── scene_<n>_<model>.mp4       # single-model animation
        ├── scene_<n>_<model>_benchmark_metrics.json
        ├── benchmark_<model>.mp4       # benchmark run output
        └── benchmark_<model>_benchmark_metrics.json
```

- Videos: H.264 MP4 (`diffusers.utils.export_to_video`, model fps).
- Metrics JSON next to every video:

```json
{
  "model": "wan22_i2v",
  "prompt": "<video prompt>",
  "image": "<scene image path>",
  "seed": 42,
  "fps": 16,
  "duration_ms": 123456,
  "peak_memory_mb": 8123,
  "output": "<video path>",
  "width": 1280, "height": 720,
  "num_frames": 81, "guidance_scale": 3.5, "num_inference_steps": 40
}
```

The default project is the mock `test_project` until real project wiring
lands.

## 7. Benchmark suite

Run with `make benchmark-video` (or
`python generators/benchmark_video_generator.py`).

**Goal:** produce the same validated scene, animated by **every** i2v model,
so outputs are directly comparable. The suite follows the rules:

- Face verification is **required** (`require_verification=True`); scenes
  regenerate until every character is verified, and the benchmark aborts if
  verification is impossible.
- Videos are ≥ 4 s and ≥ 720p (see §5 overrides), seeded identically,
  named `benchmark_<model>.mp4`.
- A summary prints path, resolution, duration, latency and peak memory for
  each model.

### The benchmark scenario — "Café conversation"

The definitive test scene is a **two-character conversation** (chosen to
exercise the talking-scenes requirement of §10):

- Characters: **Nikita** (left, long curly red hair, natural friendly
  expression) and **Roger** (right, bald dark-skinned man, muscular build,
  calm friendly expression) — both face-validated in one frame.
- Scene: both sitting at a small table in a quiet, stylish café in the
  morning, waist-up, facing each other, medium cinematic shot, warm morning
  light, photorealistic.
- Video: natural conversation, subtle gestures, gentle head movements,
  cinematic motion.
- Dialogue (for the audio stage): Nikita *"Good morning, Roger."* / Roger
  *"Good morning to you too, Nikita."*; background: quiet, relaxed café
  ambience.

### Comparing results

- **Quantitative:** the per-model `*_benchmark_metrics.json` files
  (latency, peak memory, resolution, frames, fps).
- **Qualitative:** inspect `scene_*/out/benchmark_<model>.mp4` side by side
  for motion quality, character consistency and prompt adherence.
- Default benchmark expectations: `wan22_i2v` = reference quality;
  `hunyuan_video_i2v` = faster but lower fidelity.

## 8. Operating constraints (this machine)

Machine: Apple M5 Pro MacBook Pro (Mac17,9), 18-core CPU / 20-core GPU,
Metal 4, 64 GB unified memory, torch 2.13 (MPS), diffusers 0.39.0.

**Verdict: the constraint is throughput, not memory.**

| Wan 2.2 A14B variant | Weight size | Verdict on 64 GB |
|---|---|---|
| FP16 via diffusers/MPS | ~28 GB + text encoder + Wan-VAE + activations | Fits, but slow — MPS falls back to CPU for many Wan ops |
| MLX / GGUF Q4–Q5 | ~9–12 GB | Sweet spot — comfortable headroom, ~2–5 min per 81-frame @ 832×480 clip |

Practical rules:

1. **Skip torch-MPS for Wan 2.2** when speed matters. Diffusers' Wan2.2
   pipeline on Metal is partially unaccelerated; the 20-core GPU is left
   mostly idle. `mlx-wan` (or ComfyUI + GGUF, as in the reference
   implementation) is dramatically faster.
2. Thermal note: sustained 14B inference throttles the M5 Pro over long
   runs, but a 40-step @ Q4 generation finishes before that matters.
3. Memory is not the issue — quantization gives comfortable headroom. The
   practical ceiling is speed, not capacity.
4. HunyuanVideo I2V is lighter and acceptable on MPS for quick iterations.

## 9. Verification & tests

- `test_video_engine.py` covers: project path layout, character service
  persistence, per-model params completeness, invalid model/image rejection,
  benchmark fan-out across all models (incl. failure tolerance), scene
  retry-until-verified (same folder, seed bump), unknown-character
  rejection, and face-verification policy (required vs. inconclusive).
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