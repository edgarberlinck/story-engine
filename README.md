# Story Engine

![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue.svg)

## Overview

Story Engine is an ML-driven pipeline for producing character-based video
content (e.g. vertical dramas). It generates images from text prompts,
animates them into video clips with image-to-video models, and manages the
supporting data (projects, scenes, characters) in a persistent SQLite
database — all from a desktop UI or CLI.

## Features

- **Image generation** — text-to-image via SDXL and FLUX.1-dev
- **Image-to-video generation** — Wan 2.2 I2V A14B, LTX-Video, HunyuanVideo I2V
- **Face recognition** — benchmarking for consistent character identity
- **Project management** — create, list, search, and update projects (SQLite-backed)
- **Scene & character management** — structured story data with versioned
  character attributes
- **Desktop UI** — PySide6-based application with stack navigation
- **CLI** — Typer-based interface for project/character/scene operations
- **Background workers** — thread-pool infrastructure for heavy generation tasks
- **Model management** — centralized registries, metadata, and an automated
  install script with concurrent downloads and progress tracking

## Prerequisites

- Python 3.11+ (developed on 3.14, Apple Silicon / macOS)
- A [Hugging Face](https://huggingface.co/) account authenticated for the
  models you want to use:

```bash
pip install huggingface_hub
huggingface-cli login
```

> Some models are gated. FLUX.1-dev requires accepting its license on
> Hugging Face before the download will succeed.

## Getting Started

### 1. Install dependencies and download models

```bash
make install
```

Downloads can be large (FLUX ~34GB, Wan 2.2 ~60GB). To download models
without reinstalling dependencies:

```bash
make models
```

Or use the interactive Rust TUI model manager:

```bash
make models-ui
```

### 2. Launch the app

```bash
make ui     # Desktop UI
make cli    # CLI help
```

## Project Structure

```
.
├── Makefile              # Automation (test, install, UI/CLI, benchmarks)
├── models.py             # Model registries, metadata, and config helpers
├── requirements.txt
├── core/                 # Project/scene/character managers (UI-facing logic)
├── services/database/    # SQLite-backed services (projects, scenes, characters)
├── infrastructure/database/  # Database connection infrastructure
├── generators/           # Image, video, and face-recognition generation engines
│   ├── image_generator.py
│   ├── video_generator.py
│   ├── image_engine.py
│   └── benchmark_*.py
├── utils/                # Shared helpers (model metrics, paths)
├── workers/              # Background task infrastructure
├── ui/                   # PySide6 desktop application
├── cli/                  # Typer CLI
├── scripts/install.py    # Model installation & verification
├── tests/                # Unit tests (cli, core, services, workers)
├── docs/                 # Human + LLM documentation
├── outputs/              # Generated content
├── models/               # Downloaded model weights
└── story_engine.db       # SQLite database (created at runtime)
```

## Model Registries

`models.py` centralizes every model used by the project:

| Category | Models |
|---|---|
| **Diffusion** (text-to-image) | SDXL (`sdxl`), FLUX.1-dev (`flux_dev`) |
| **Image-to-video** | Wan 2.2 I2V A14B (`wan22_i2v`), LTX-Video (`ltx_video`), HunyuanVideo I2V (`hunyuan_video_i2v`) |
| **Text generation** | Phi-3 Mini (`phi3_mini`), Gemma 2B (`gemma_2b`) |
| **Segmentation** | DETR ResNet-50 Panoptic (`detr_resnet_50_panoptic`) |

See [docs/video-generation-caveats.md](docs/video-generation-caveats.md) for
runtime guidance on the image-to-video models on this machine.

## Automation Commands

| Command | Description |
|---|---|
| `make install` | Install dependencies and download models |
| `make models` | Download models only |
| `make models-ui` | Interactive Rust TUI for model downloads |
| `make ui` | Launch the desktop UI |
| `make cli` | Show CLI help |
| `make test` | Run all unit tests |
| `make watch` | Re-run tests on file changes |
| `make benchmark-image` | Image generation benchmark suite |
| `make benchmark-video` | Image-to-video benchmark suite |
| `make benchmark-face` | Face recognition benchmark (writes report) |
| `make benchmark-face-sdxl` | Face recognition benchmark with SDXL |
| `make benchmark_fullbody_recognition` | Full-body face recognition benchmark |
| `make format` | Format code with Black |
| `make lint` | Lint code with Flake8 |
| `make check` | Run lint + test |
| `make clean` | Remove build artifacts and `story_engine.db` |

## Documentation

- `docs/humans/` — user guides (e.g. `project-service.md`)
- `docs/llm/` — technical specifications for API integration
- `ROADMAP.md` — phase-by-phase development status

## Troubleshooting

- **Hugging Face auth errors**: run `huggingface-cli login` and confirm your
  account has access to the gated model repositories.
- **Large downloads timing out**: large models (4GB+) may require a stable
  connection; `scripts/install.py` retries and adapts concurrency to network
  speed.
- **macOS semaphore warnings during generation**: `TOKENIZERS_PARALLELISM`
  is set to `false` in the generators to avoid fork-based tokenizer issues.
