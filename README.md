# Story Engine

![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue.svg)

## Overview

**Story Engine** is an ML-driven pipeline for producing **immersive audiobooks** from a
descriptive markup language. Authors write stories using XML-like tags or bracket shorthand,
and the pipeline generates per-character isolated audio, background beds, and a mixed final
scene — all orchestrated from a desktop UI or CLI. Video generation has been removed; the
focus is purely on high-quality, consistent audio production.

Story data is persisted in a lightweight SQLite database. Every scene, character, and
voice assignment is tracked, making it easy to preview, edit, accept, and finalize audio
productions with full reproducibility.

## Features

- **Text-to-audio markup** — write stories with `<scene>`, `<character>`, `<sound>`, and
  `<music>` tags (or bracket shorthand). The parser is tolerant and mixes notation styles.
- **Per-character voice assignment** — each character has a persistent voice profile
  (`voice_path`, `voice_prompt`) for consistent timbre across all fragments.
- **Background beds** — generate `<sound>` (sound effects) and `<music>` beds via MusicGen;
  dialogue is ducked under the bed automatically.
- **Per-fragment preview** — listen to any fragment in isolation (character or background);
  accept fragments to pre-generate and cache their WAV files.
- **Accept & pre-generate** — mark fragments as accepted; the system generates and caches
  their audio. Accepted fragments are visually indicated and persist across sessions.
- **Finalize scene** — stitch all pre-generated fragments into one mixed piece with proper
  gaps, speaker-change silences, and bed ducking. The final mix is saved as
  `scene_final.wav` and can be played from the UI.
- **Voice drift prevention** — a per-character voice resolver pins one stable voice per
  character (designed voice from `voice_prompt` if set, otherwise a preset pinned once),
  so timbre never drifts between fragments.
- **Multitrack timeline view** — Audacity-style track lanes with waveforms, mute/solo,
  volume sliders, drag-to-retime, and zoom. Honors per-segment start offsets.
- **Project management** — create, list, search, and update projects and scenes via UI or CLI.
- **CLI** — Typer-based interface for all project/character/scene operations.

## Getting Started

### 1. Install dependencies and download models

```bash
make install
```

Download models selectively:

```bash
make models
```

Or use the interactive Rust TUI model manager:

```bash
make models-ui
```

### 2. Launch the app

```bash
make ui     # Desktop UI (PySide6)
make cli    # CLI help
```

## Quick start: your first audiobook

Write a story using the markup language. Here's a minimal example:

```xml
<scene>
  <character name="narrator" tone="neutral">
    Once upon a time...
  </character>
  <character name="maestro" tone="confident">
    The orchestra tuned up.
  </character>
  <sound preset="orchestra warm"/>
  <music prompt="soft strings"/>
</scene>
```

Or using bracket shorthand:

```text
[narrator] Once upon a time...
[maestro] The orchestra tuned up.
[sound preset=orchestra warm]
[music prompt=soft strings]
```

Open the UI, create a new project, paste the markup, and click **Generate Scene** to
render the audio. Then use the scene editor to preview fragments, accept them, and
**Finalize Scene** to produce the final mixed audio.

## Project structure

```
.
├── Makefile              # Automation (test, install, UI/CLI)
├── models.py             # Model registries and config helpers
├── requirements.txt
├── core/                 # Project/scene/character managers (UI-facing logic)
├── services/database/    # SQLite-backed services (projects, scenes, characters)
├── generators/           # Audio generation engines (MusicGen, Qwen3-TTS)
├── utils/                # Shared helpers (model metrics, paths)
├── workers/              # Background task infrastructure
├── ui/                   # PySide6 desktop application
├── cli/                  # Typer CLI
├── scripts/install.py    # Model installation & verification
├── tests/                # Unit tests (cli, core, services, workers)
├── docs/                 # Human + LLM documentation
│   └── llm/              # Audiobook specification and API docs
├── outputs/              # Generated content (scenes, audio, waveforms)
├── models/               # Downloaded model weights
├── story_engine.db       # SQLite database (created at runtime)
└── README.md             # This file
```

## Model registries

`models.py` centralizes every model used by the project:

| Category | Models |
|---|---|
| **Text-to-audio** | Qwen3-TTS (`qwen3_tts`) |
| **Background audio** | MusicGen medium (`musicgen_medium`) |
| **Text generation** | Phi-3 Mini (`phi3_mini`), Gemma 2B (`gemma_2b`) |

All audio generation runs locally (MLX on Apple Silicon) or via CPU fallback. No video
models are included.

## Automation commands

| Command | Description |
|---|---|
| `make install` | Install dependencies and download models |
| `make models` | Download models only |
| `make models-ui` | Interactive Rust TUI for model downloads |
| `make ui` | Launch the desktop UI |
| `make cli` | Show CLI help |
| `make test` | Run all unit tests |
| `make watch` | Re-run tests on file changes |
| `make format` | Format code with Black |
| `make lint` | Lint code with Flake8 |
| `make check` | Run lint + test |
| `make clean` | Remove build artifacts and `story_engine.db` |

## Documentation

- `docs/humans/` — user guides (e.g. `project-service.md`)
- `docs/llm/` — technical specifications for API integration, including the full
  audiobook markup specification (`audiobook-specification.md`)
- `ROADMAP.md` — phase-by-phase development status

## Troubleshooting

- **Hugging Face auth errors**: run `huggingface-cli login` and confirm your account has
  access to the gated model repositories.
- **Large downloads timing out**: large models may require a stable connection;
  `scripts/install.py` retries and adapts concurrency to network speed.
- **No audio output**: ensure at least one `<sound>` or `<music>` tag (or bracketted
  equivalent) is present in your scene, or the scene will render as pure narration.
- **Voice sounds different between fragments**: this was a known issue (per-fragment voice
  derivation) and is now fixed by the per-character voice resolver — each character’s
  timbre is pinned once and reused across all segments.