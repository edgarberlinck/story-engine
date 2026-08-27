# Image-to-Video (i2v) — Definitive Specification

> **Status:** authoritative. This document is the single source of truth for
> how image-to-video generation operates inside Story Engine. It replaces the
> former `docs/humans/video-generation.md`, `docs/llm/video-generation.md`,
> and `docs/video-generation-caveats.md`.
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
| `ltx_video_095_i2v` | LTX-Video 0.9.5 I2V | `Lightricks/LTX-Video-0.9.5` | **yes** |

- **Benchmark decision (2026-08):** LTX-Video 0.9.5 is the surviving i2v
  model. Wan 2.2 I2V A14B and HunyuanVideo-I2V were **dropped** per the
  registry policy in `models.py`. Wan 2.2 A14B is a dual 14B-expert MoE
  stored **F32** on disk (~119 GB across ~53 GB `transformer` + ~53 GB
  `transformer_2` + a ~11 GB text encoder) and **OOMs during load on the 64 GB
  Apple-Silicon host** (SIGKILL `Killed: 9` at "Loading pipeline components",
  before any sampling step) — so it cannot run here regardless of resolution or
  frame count. LTX-Video 0.9.5 (~3.6 GB transformer, ~24 GB total) fits
  comfortably and runs on MPS.
- The registered repo is the **diffusers-format** checkpoint (loadable via
   `from_pretrained`, here `LTXImageToVideoPipeline`).
- `AVAILABLE_VIDEO_MODELS` = the full registry; `DEFAULT_VIDEO_MODEL` is
   `ltx_video_095_i2v`.
- Each model is invoked through its own diffusers pipeline class and
  model-specific parameters (`MODEL_GENERATION_PARAMS` in
   `video_generator.py`) — never a shared generic call.
- Device/dtype come from `get_model_config("image_to_video")`
   (`bfloat16`; device resolves mps > cuda > cpu). To keep the footprint
   within the machine's free memory, `_load_pipeline` calls
   `enable_model_cpu_offload()` on accelerator (cuda/mps) backends so the ~17 GB
   text encoder is streamed off-device one component at a time.

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
   resolution (e.g. 704×512 for LTX-Video 0.9.5) so the conditioning frame
   matches the output (reference: the reference implementation renders the first
   frame at the exact video resolution).

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
| `ltx_video_095_i2v` | 704×512 | 161 | 25 | 3.0 | 50 | yes |

**Frame-count rule (8k+1):** LTX-Video's VAE has a temporal stride of 8, so
`num_frames ≡ 1 (mod 8)`; 161 (= 1 + 8×20) satisfies this. 161 frames @ 25 fps
is a **6.44 s** clip — already past the benchmark's ≥ 4 s bar without needing an
upsample.

Benchmark overrides (`BENCHMARK_VIDEO_PARAMS`) pin the model to its LTX-native
resolution; the ≥ 4 s bar is met by the 161-frame count:

| Model | Resolution | Frames | FPS | Duration |
|---|---|---|---|---|
| `ltx_video_095_i2v` | 704×512 | 161 | 25 | 6.44 s |

> **Resolution caveat (honest):** the original benchmark bar was also
 > "≥ 720p". LTX-Video 0.9.5's native output is 704×512, so the ≥ 720p bar is
 > **not met** by the default params. Hitting 1280×720 would require an
 > upsampled generation that the 64 GB run avoids (risking OOM / artifacts);
 > the ≥ 4 s duration bar is the one that's retained. The video-quality
 > enhancement option (§3 note + ROADMAP) is the path to higher resolution.

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
   "model": "ltx_video_095_i2v",
    "prompt": "<video prompt>",
   "image": "<scene image path>",
   "seed": 42,
   "fps": 25,
   "duration_ms": 123456,
   "peak_memory_mb": 8123,
   "output": "<video path>",
   "width": 704, "height": 512,
   "num_frames": 161, "guidance_scale": 3.0, "num_inference_steps": 50
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
- **Qualitative:** inspect `scene_*/out/benchmark_<model>.mp4` for motion
  quality, character consistency and prompt adherence.
- Benchmark decision (2026-08): `ltx_video_095_i2v` is the reference/surviving
   model. Wan 2.2 I2V A14B was dropped because it OOMs on the 64 GB host, and
   HunyuanVideo-I2V was dropped for architecture/size — see §3.

## 8. Operating constraints (this machine)

Machine: Apple M5 Pro MacBook Pro (Mac17,9), 18-core CPU / 20-core GPU,
Metal 4, 64 GB unified memory, torch 2.13 (MPS), diffusers 0.39.0.

**Verdict: memory is the binding constraint — it is what forced the model
choice.** The benchmark's original candidate, Wan 2.2 I2V A14B, is a dual
14B-expert MoE stored F32 on disk (~119 GB: ~53 GB `transformer` + ~53 GB
`transformer_2` + a ~11 GB text encoder) and is **killed (`Killed: 9` /
SIGKILL) inside `from_pretrained`, before any sampling step**, on this 64 GB
host even at the smallest config (480×320 / 33 frames / 20 steps — a trivial
load that would be comfortable on a desktop GPU). Lowering resolution, frame
count, or steps does not help — the OOM is at *load time*, so those knobs are
irrelevant to it. The surviving model is **LTX-Video 0.9.5**
(`Lightricks/LTX-Video-0.9.5`), whose ~3.6 GB transformer (≈24 GB total) loads
comfortably.

| i2v candidate | Footprint | Verdict on 64 GB |
|---|---|---|
| Wan 2.2 I2V A14B | ~119 GB on disk (F32), ~54 GB resident in bf16 | **OOM** — `Killed: 9` at load; abandoned |
| LTX-Video 0.9.5 | ~24 GB total, 3.6 GB transformer | **Survives** — fits with headroom; chosen |

Practical rules:

1. **Memory, not throughput, decides the model.** The Wan→LTX swap was driven
   by load-time OOM, not speed. A future move back to Wan (or to a larger
   model) is only viable with more unified memory, GPU offload to a real
   accelerator, or MLX/quantized weights small enough for the 64 GB pool.
2. `_load_pipeline` calls `enable_model_cpu_offload()` on the accelerator
   (cuda/mps) backend so the ≈17 GB text encoder is streamed off-device one
   component at a time, keeping the live footprint within the machine's free
   memory.
3. **LTX-Video 0.9.5 is not optimized for MPS.** Diffusers' LTX pipeline on
   Metal may fall back to CPU for some ops; a future move to `mlx` / GGUF
   weights (as the reference implementation uses for LTX-2) would sharpen speed.
4. **Thermal note:** sustained video inference throttles the M5 Pro over long
   runs; a single 50-step 6.44 s generation finishes before that matters, but
   the benchmark is run once, not in a tight loop.

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