# Character Voice Generation with Local Models

> Reference doc — how the reference project generates voices for characters
> and guarantees the same voice across clips. Investigated Aug 2026.

## TL;DR — important correction to the common assumption

**The reference project does NOT "generate a voice for a character" from a
description.** There is no character entity with a voice, and no
"after character → generate voice" step. What it actually does:

1. **Pick a voice** — either a preset speaker of a TTS model (e.g. one of XTTS v2's
   57 built-in voices) or a custom voice.
2. **Clone a voice from a reference WAV** — that is the mechanism that
   guarantees the character sounds the same on every clip.
3. **Synthesize every line with the same clone** (`tts_with_vc_to_file`).

Voice "design" from a text description (`VOICE_DESIGN` capability, MiniMax
`voice_design` API) exists as scaffolding but is **not wired into any engine,
pipeline, or UI** — it is future work.

> **Update:** prompt-based voice design **is** possible today with a fully local
> model — Qwen3-TTS `VoiceDesign` (Jan 2026). The reference project does not
> use it (it only loads the `CustomVoice` + `Base` variants). See
> [Prompt-based voice design (local)](#prompt-based-voice-design-local).

---

## Architecture overview

The project has two TTS layers that share the same engine abstraction
(`tts/tts_engine.py`):

| Layer | Entry point | Purpose |
|---|---|---|
| Standalone library | `tts/core.py::dub_video()` + `tts/dub_video_from_srt.py` CLI | Dubbing a video from SRT/TXT with one voice |
| Queue platform | `POST /api/generate_tts` → `song_generator/queue/pipelines/generate_tts.py` | Multi-track, per-clip TTS on the GPU worker |

Both end up calling the same engine methods:
- `tts_to_file(...)` — plain synthesis (no cloning)
- `tts_with_vc_to_file(..., speaker_wav=...)` — voice-cloned synthesis

### Engines (registered in `tts/tts_engine.py::_register_builtin_engines`)

| Engine | Type | Local? | Clone voice? | Notes |
|---|---|---|---|---|
| **XTTS v2** (`xtts_engine.py`) | Coqui `tts_models/multilingual/multi-dataset/xtts_v2` | ✅ LOCAL | ✅ `speaker_wav=` | 57 preset speakers; **the** local cloning engine |
| **Coqui** (`coqui_engine.py`) | per-language models (vits/tacotron2/glow-tts, see `config.py::DEFAULT_LANGUAGE_MODELS`) | ✅ LOCAL | ✅ `tts_with_vc_to_file` | single-speaker, one model per language |
| **Qwen3-TTS** (`qwen3_engine.py`) | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` + `-Base` | ✅ LOCAL | ✅ `generate_voice_clone` | **DISABLED** — not registered; docstring: "not giving good results and is not stable" |
| **MiniMax** (`MiniMax/tts_engine.py`) | `speech-2.x` | ❌ CLOUD | ✅ `VoiceCloneTask` | only engine with a persisted voice_id |
| **ElevenLabs** (`ElevenLabs/tts_engine.py`) | `eleven_multilingual_v2` | ❌ CLOUD | ✅ `clone_voice_from_wav` | |
| **OpenAI** (`openai_engine.py`) | `gpt-4o-mini-tts` | ❌ CLOUD | ❌ | voice selection only |

Local engines are registered by default; cloud engines are registered by the
TTS worker (`gpu_worker_tts.py`) and server (`server/tts_engines.py`).

---

## The full flow (queue pipeline, `generate_tts.py`)

```
POST /api/generate_tts
  └─ generate_tts(dispatcher, params)
      ├─ GenerateVideoDirectoryTask
      ├─ GetVideoTranscription            (if no original-audio SRT yet)
      ├─ if params.clone_voice:
      │    └─ ExtractSpeakerWavTask       → extracts speaker.wav (16 kHz mono PCM)
      │         └─ if engine == minimax:
      │              └─ VoiceCloneTask    → MiniMax API clone → voice_id
      │                                      (persisted in ai_assets/<id>/asset.json
      │                                       under minimax.voice_id, reused on reruns)
      ├─ per clip (SRT cue):
      │    └─ GenerateTTSClipTask          → runner → engine:
      │         ├─ clone_voice=True        → tts_with_vc_to_file(speaker_wav)
      │         ├─ voice_id cached (minimax) → tts_to_file(voice=voice_id)
      │         └─ else                    → tts_to_file(speaker=<preset>)
      ├─ GenerateTTSTask                   → assembles clips, fits speeds, muxes audio
      └─ (optional) GenerateSubtitlesTask
```

### 1. Get the reference voice (speaker WAV)

Two sources today:

- **Extracted from the input video** (`ExtractSpeakerWavTask` →
  `tts/utils/video_utils.py::extract_speaker_wav`): ffmpeg dumps the
  whole audio track as `speaker.wav` (`pcm_s16le`, 16 kHz, mono). This is the
  "the character in this video already has a voice, clone it" path — it is used
  for dubbing.
- **Uploaded asset** — a WAV under `ai_assets/<id>/` (used the same way via
  `speaker_path`).

There is **no local voice-synthesis-from-description step** today. To give a
brand-new character a voice you must provide a reference WAV (or pick a preset).

### 2. Clone / fix the voice (the "sounds the same" guarantee)

Once a `speaker.wav` exists, every clip for that character is rendered with
**zero-shot voice cloning**:

- **XTTS v2 (local)**: `tts_to_file(..., speaker_wav=speaker.wav, speaker=<name>)`
  — Coqui clones the timbre from the reference WAV on every call.
- **Coqui (local)**: `tts_with_vc_to_file(..., speaker_wav=speaker.wav)`.
- **Qwen3 (local, disabled)**: `model_base.create_voice_clone_prompt(ref_audio)`
  (x-vector-only) cached per WAV, then `model_base.generate_voice_clone(...)`.
- **MiniMax (cloud)**: `VoiceCloneTask` clones once via
  `/v1/voice_clone`, stores the resulting `voice_id`, and every clip reuses it —
  no per-clip cloning cost.

Because cloning is deterministic for the same reference WAV, every line a
character speaks comes out with the same voice — this is the project's answer
to "guarantee they will sound the same".

### 3. Synthesize the clip

`GenerateTTSClipTask.run()` builds a payload and calls
`runner.tts_with_vc_to_file(payload)` (clone) or `runner.tts_to_file(payload)`.
The runner (`song_generator/queue/runners/tts_clip_mixin.py`) resolves the
engine via `create_tts_engine`, applies options (voice/emotion/speed), and
dispatches to the TTS GPU worker (`gpu_worker/tts_app.py`) or in-process.

---

## Standalone CLI flow (same idea, simpler)

```
python tts/dub_video_from_srt.py video.mp4 subs.srt out.mp4 \
  --language en --clone-voice [--model tts_models/multilingual/multi-dataset/xtts_v2]
```

`dub_video()` in `tts/core.py`:
1. If `clone_voice=True` → `extract_speaker_wav(input_video, speaker.wav)`.
2. `resolve_tts_model_name()` → model for the language/engine
   (XTTS default `xtts_v2`, or per-language Coqui models).
3. `create_tts(...)` + optional `set_voice/set_emotion`.
4. Per subtitle: `generate_tts_clip(..., speaker_path=speaker.wav, clone_voice=True)`.
5. Assemble, normalize, mux.

---

## What "generate a voice for a character" would take (gap analysis)

If story-engine wants **local-only, character-first voice generation**, the
building blocks are already proven in the reference project:

| Capability | Status in the reference project | How to do it locally |
|---|---|---|
| Pick a stable voice without cloning | ✅ XTTS v2 57 preset speakers | `XTTSEngine.speakers` → `tts_to_file(speaker=<id>)` |
| Clone an existing voice for consistency | ✅ XTTS / Coqui / Qwen3 | reference WAV → `tts_with_vc_to_file` |
| Generate a voice from a character description | ⚠️ scaffolding only (`VOICE_DESIGN` enum; MiniMax `voice_design` client method, unused) | **Qwen3-TTS-12Hz-1.7B-VoiceDesign** (local, Jan 2026) — `generate_voice_design(instruct=...)`; not loaded by the reference project. See [below](#prompt-based-voice-design-local) |
| Persist one voice per character, reuse across generations | ⚠️ only for MiniMax (voice_id in `asset.json`) | store reference WAV per character and reuse as `speaker_wav` |

Practical local recipe to guarantee a consistent character voice:

1. Generate or source a 10–30 s clean reference WAV for the character
   (design one with Qwen3 `VoiceDesign`, pick a preset XTTS sample, extract
   audio from a video, or use a chosen recording).
2. Keep it in the character's asset folder (like `ai_assets/<id>/`).
3. Always synthesize that character's lines with
   `tts_with_vc_to_file(speaker_wav=<ref>)` on XTTS v2 (or Qwen3 once stable).

---

## Prompt-based voice design (local)

**Yes — you can generate a voice from a prompt and edit it with a prompt, fully
locally.** The model built for this is Qwen3-TTS (open-sourced Jan 2026, 10
languages, runs on one GPU). It exposes three modes:

| Model | Method | What it does | Local? |
|---|---|---|---|
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `generate_voice_design(text, language, instruct)` | Designs a **brand-new voice** from a natural-language description | ✅ |
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | `generate_custom_voice(text, language, speaker, instruct)` | Style control over 9 preset timbres via instructions (this is the only variant the reference project loads) | ✅ |
| `Qwen3-TTS-12Hz-1.7B-Base` (+ CustomVoice) | `create_voice_clone_prompt(ref_audio)` → `generate_voice_clone(...)` | Zero-shot **voice cloning** from a reference WAV (this is what the reference project's `qwen3_engine.py` uses) | ✅ |

Your example works verbatim as an `instruct`:

```python
wavs, sr = design_model.generate_voice_design(
    text="Hello, I am the fairy of the forest.",
    language="English",
    instruct="A calm, sweet female voice, soft and gentle, like a kind grandmother telling a bedtime story.",
)
sf.write("character_voice.wav", wavs[0], sr)
```

### The recommended workflow: design → lock → consistent

This is the officially recommended pattern and it matches the
"generate a voice, then guarantee it always sounds the same" goal exactly:

1. **Design** the voice once from the character description:
   `design_model.generate_voice_design(text=<sample line>, language=..., instruct=<character voice description>)`
   → produces a short reference WAV (`character_voice.wav`).
2. **Lock** it into a reusable identity:
   `prompt_items = model_base.create_voice_clone_prompt(ref_audio="character_voice.wav", x_vector_only_mode=True)`
   → a reusable voice prompt (cached per WAV, like `qwen3_engine.py` does).
3. **Synthesize every line** with the same prompt:
   `model_base.generate_voice_clone(text=<line>, language=..., voice_clone_prompt=prompt_items)`
   → identical voice across all lines.

This mirrors the reference project's XTTS clone pattern (reference WAV →
`tts_with_vc_to_file`), but the reference comes from a **generated** voice
instead of a recording.

### "Editing" a voice via prompt

- **Keep the timbre, change the delivery** → `CustomVoice.generate_custom_voice(..., instruct="...")`
  per line (emotion/pace/style on a fixed preset timbre).
- **Change the voice itself** → re-run `generate_voice_design` with a new
  description and re-clone. True identity-preserving edits (e.g. "same voice but
  calmer") are still easier on cloud APIs — MiniMax's `voice_modify` (pitch /
  intensity / timbre / sound effects) and `emotion` preset (`calm`, …) exist in
  the reference project's `MiniMax/types.py` but are not wired into any engine.

### Why the reference project's Qwen3 engine is disabled

Its `qwen3_engine.py` only loads `CustomVoice` + `Base` (never the
`VoiceDesign` model) and uses `instruct` only for speed control
(`instruct=f"Speak in {speed}x speed"`) — which the model docs warn is not
reliable (`# not reliable` in the code). The "random, unstable" results the
comment describes are consistent with that integration, not with the model
family as a whole.

### Caveats

- `instruct` is for **style/emotion/timbre**, not speed; drive pacing via
  `length_scale`/generation kwargs instead.
- Requires `flash-attn` or `sdpa` (`attn_implementation`) and `torch` in the
  TTS worker venv — same setup as the existing disabled Qwen3 engine.
- Batch inference is supported (pass lists for `text`/`language`/`instruct`),
  which is useful for generating the initial reference WAV cheaply.

---

## Key files

| File | Role |
|---|---|
| `tts/tts_engine.py` | Engine abstraction, registry, capabilities, tags (LOCAL/CLOUD) |
| `tts/xtts_engine.py` | Local XTTS v2, 57 presets, `speaker_wav` cloning |
| `tts/coqui_engine.py` | Local Coqui per-language models, `tts_with_vc_to_file` |
| `tts/qwen3_engine.py` | Local Qwen3-TTS clone via `CustomVoice`+`Base` (disabled; never loads the `VoiceDesign` model) |
| `tts/core.py` | Standalone `dub_video()` orchestration |
| `tts/dub_video_from_srt.py` | CLI entry point |
| `song_generator/queue/pipelines/generate_tts.py` | Queue pipeline (tasks, speaker WAV, MiniMax clone) |
| `song_generator/queue/tasks/extract_speaker_wav.py` | Extract speaker WAV from video |
| `song_generator/queue/tasks/voice_clone.py` | MiniMax voice clone + persisted voice_id |
| `song_generator/queue/tasks/generate_tts_clip.py` | Per-clip synthesis dispatch |
| `song_generator/queue/runners/tts_clip_mixin.py` | Runner glue: engine + payload → TTS call |
| `gpu_worker/tts_app.py` | TTS GPU worker (local engines run here) |
| `MiniMax/tts_engine.py`, `ElevenLabs/tts_engine.py` | Cloud engines with clone support |