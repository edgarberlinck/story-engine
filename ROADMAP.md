# Story Engine Roadmap

## Phase 1: Text-to-Image Generation ✅ Complete
- [x] Basic image generation using text prompts
- [x] Support for multiple diffusion models (SDXL, FLUX.1 Dev)
- [x] Organized model management system (`models.py` registries, paths, metadata)
- [x] Automated installation and verification (`scripts/install.py`, `make install`, `make models`)
- [x] Model benchmarking support (`benchmark_image_generator.py`, `make benchmark-image`)
- [x] LLM-assisted prompt/filename generation (`text_generator.py`, Phi-3)

## Phase 2: Generic Task-based Generation 🟡 In Progress
- [x] Support for switching between different models dynamically (`generate_image(model_name=...)`, per-type model policy)
- [x] Task framework with typed prompts (scenes, characters, environments)
  - [x] Image engine with typed generation (`GenerationType`: character/environment)
  - [x] Scene-driven generation pipeline (`image_engine.generate_scene`)
- [x] Video generation (Wan 2.2 I2V) — see Phase 4
- [ ] Extend task framework to text/audio tasks
- [ ] Flexible configuration options
  - [x] Per-model-type dtype/device configuration (`get_model_config`)
  - [ ] Config files / user-defined profiles (only directory setup in `configuration_manager.py` so far)
- [ ] API design for extensibility
  - [x] `CharacterReferenceStore` abstraction for identity references
  - [ ] Pluggable task/generator registry

## Phase 3: Image-to-Image Generation (Character Variation) 🟡 In Progress
- [x] Character generation for stories
  - [x] Generate and persist character reference images (`generate_character`, stored to `outputs/<project>/characters/<name>/reference.png` + DB)
  - [x] Characters reusable across scenes (prompt enrichment via `_enrich_scene_prompt`)
  - [ ] Actual identity preservation (IP-Adapter or similar) — currently a TODO
- [ ] Generate variations of existing images based on attributes
- [ ] Accept arbitrary image path as input (currently in-project references only)
- [ ] Support for concept variation (smiling, serious, etc.)
- [ ] Integration with segmentation models for analysis
  - [x] DETR panoptic model downloadable via install script
  - [ ] Wire segmentation into any code path (currently unused)

## Phase 4: Image-to-Video Generation 🟡 Mostly Complete
- [x] Register I2V models (Wan 2.2 I2V A14B; HunyuanVideo-I2V dropped 2026-08)
- [x] Install script downloads I2V models to `models/image_to_video/`
- [x] Video generator module (`generators/video_generator.py`, `make benchmark-video`)
- [x] Scene-driven video pipeline (`video_engine.py`: validated scene → character reference → animated clip)
- [x] Output management for video files (`outputs/<project>/scenes/scene_<n>/out/`, videos + metrics JSON)
- [x] Consolidate benchmarks: remove redundant `scripts/video_benchmark_nikita_roger.py`; use `generators/benchmark_video_generator.py` as the canonical video benchmark
- [ ] Per-model quantization/MLX runtime support (see `docs/image-to-video.md`)
- [ ] Audio & lip-sync pipeline — talking scenes (TTS → lip sync → music → mix, see `docs/image-to-video.md` §10)
  - [x] Audio model registries in `models.py` (TTS, lip sync, music) + install.py wiring — one winner per category, small fallbacks only
  - [ ] TTS/voice engine implementation (Qwen3-TTS local, 1.7B + 0.6B)
  - [ ] Lip-sync implementation (LatentSync 1.6; CUDA-oriented, MPS flakiness accepted — retry on failure)
  - [ ] Music generation + dialogue/music mix assembly (MusicGen medium)
- [ ] Single-i2v-model decision: after the benchmark comparison is final, keep exactly one i2v model and remove the other from the project
- [x] Fix FLUX `sentencepiece` crash: FLUX.1's T5 `text_encoder_2` tokenizer (`tokenizer_2/spiece.model`) required the `sentencepiece` package, which was absent from `requirements.txt` and the venv — caused `make benchmark-video` to abort with "Cannot instantiate this tokenizer from a slow version". Added `sentencepiece>=0.2` to `requirements.txt` (installed 0.2.2, cp314 arm64 wheel). FLUX now loads; benchmark runs the full pipeline.
- [x] **Mitigated "No faces detected" hard-abort in `make benchmark-video`** (after the sentencepiece fix, the benchmark ran the full pipeline but aborted at the validated-scene stage — *not* a crash):
   - Symptom that triggered the investigation: "No faces detected in scene ...; Characters NOT found in scene: Nikita, Roger; regenerating..." → "Could not verify all characters in scene after 3 attempts."
   - **Real root cause**: the benchmark verified against the default project (`test_project`), but Nikita & Roger live in the **`Test_ui`** project — it was checking the wrong project's data. (Initial misdiagnosis — a *broken* `dlib`/`face_recognition` embedder under Python 3.14 — was ruled out: face detection works fine on proper `Test_ui` scenes, e.g. "Nikita in scene_4/5/6: True".)
   - "For now" mitigation applied: `utils/face_check.character_appears_in_image` now returns **`None` (inconclusive)** instead of `False` when no faces are detected in the scene — no detection does not prove the character is absent. Note: with `require_verification=True` (as the benchmark uses) a `None` still raises "verification required but inconclusive"; deciding whether an inconclusive check should accept-or-abort is deferred.
   - Deeper fix deferred → see cross-project character import below.
   - Note: `Makefile` benchmark targets call bare `python` (not on PATH outside an activated venv); run via `source .venv/bin/activate && make benchmark-video` or fix the targets to use `.venv/bin/python`.

## Phase 5: Project & Data Management 🟡 In Progress
- [x] SQLite persistence layer (`services/database/`)
- [x] Project service: create, read, update, list, search
- [x] Delete operation for projects (`project_service.delete_project`, CLI command)
- [x] Character entities in the database (`character_service.py`, attributes + versions)
- [x] Scene entities in the database (`scene_service.py`)
- [x] Core managers bridging DB and UI (`core/project_manager.py`, `character_manager.py`, `scene_manager.py`)
- [ ] Associate generated assets with projects at the DB level (path-based storage exists)
- [ ] Migrations for evolving schemas (`migrations.py` exists — needs schema history review)
- [ ] **Cross-project character import / borrowing**: by design, characters must only be accessed via their owning project (today the scene pipeline resolves characters from the single project being processed — the `Test_ui` vs `test_project` mix-up exposed this). Add a deliberate "import a character from another project" feature so a character (e.g. an *older version* of a character in a different project) can be explicitly brought into the current project. Design considerations: DB record + reference-image copy/link across projects, version tracking, and provenance of the source project.

## Phase 6: Advanced Story Generation Features 🟡 In Progress
- [x] Multiple character management (DB-backed, CLI + UI character builder/viewer)
- [x] Scene progression (scene manager, scene cards, scene dialog)
- [ ] Story structure generation
- [ ] Narrative coherence
- [ ] Multi-modal content generation (image + video done; text/audio tasks pending via Phase 2)

## User Interfaces ✅ In Progress
- [x] Desktop UI (PySide6: project list/screens, character builder/viewer, project view)
- [x] CLI (Typer: project/character/scene operations)
- [x] Background workers (`workers/base_worker.py`)
- [x] Interactive model download TUI (`make models-ui`, Rust)
- [ ] Web interface (future)

## Technical Considerations
- [x] Keep model directories organized and scalable
- [x] Document all APIs and interfaces clearly (`docs/humans/`, `docs/llm/`)
- [x] Runtime/performance guidance documented (`docs/image-to-video.md`)
- [ ] Maintain backward compatibility with existing implementations (ongoing)
- [ ] Ensure smooth integration between different generation approaches
- [ ] Decide: wire up segmentation model or remove it from install list (~1.2GB unused)
- [ ] Flake8 cleanup: repo has pre-existing violations in legacy files (`cli/main.py` F401/E231, `core/advanced_prompting.py` W293/F841, `core/character_manager.py` F401/E226, `models.py` W291/E302, `services/database/character_service.py` W293/E127). Also add a `.flake8`/`setup.cfg` config (exclude `.venv`, outputs, node_modules) so `flake8 .` stops crashing on `.venv`'s sympy (RecursionError) — then enforce via `make lint` in CI

## Future Enhancements
- [x] Integration with text generation models (prompt enhancement via Phi-3)
- [x] Face recognition benchmarking for character consistency
- [ ] Web interface for easy access
- [ ] Export/import of projects
- [ ] Advanced search filters