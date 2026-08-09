# Story Engine Roadmap

## Phase 1: Text-to-Image Generation ✅ Complete
- [x] Basic image generation using text prompts
- [x] Support for multiple diffusion models (SDXL, FLUX.1 Dev)
- [x] Organized model management system (`models.py` registries, paths, metadata)
- [x] Automated installation and verification (`scripts/install.py`, `make install`, `make models`)
- [x] Model benchmarking support (`benchmark_models`)
- [x] LLM-assisted prompt/filename generation (Phi-3)

## Phase 2: Generic Task-based Generation 🟡 In Progress
- [x] Support for switching between different models dynamically (`generate_image(model_name=...)`, per-type model policy)
- [ ] Create generic task framework that accepts prompts
  - [x] Initial image engine with typed generation (`GenerationType`: character/environment)
  - [ ] Extend beyond image tasks (video, text, audio)
- [ ] Flexible configuration options
  - [x] Per-model-type dtype/device configuration (`get_model_config`)
  - [ ] Config files / user-defined profiles
- [ ] API design for extensibility
  - [x] `CharacterReferenceStore` abstraction for identity references
  - [ ] Pluggable task/generator registry

## Phase 3: Image-to-Image Generation (Character Variation) 🔴 Not Started
- [ ] Generate variations of existing images based on attributes
- [ ] Accept image path as input
- [ ] Support for concept variation (smiling, serious, etc.)
- [ ] Character generation for stories
  - [x] Architecture prepared (`character_id` + reference store in `image_engine.py`)
  - [ ] Actual identity preservation (IP-Adapter or similar) — currently a TODO
- [ ] Integration with segmentation models for analysis
  - [x] DETR panoptic model downloadable via install script
  - [ ] Wire segmentation into any code path (currently unused)

## Phase 4: Image-to-Video Generation 🟡 Models Ready
- [x] Register I2V models (Wan 2.2 I2V A14B, LTX-Video, HunyuanVideo-I2V)
- [x] Install script downloads I2V models to `models/image_to_video/`
- [ ] Video generator module (load pipeline, animate image, save video)
- [ ] Integrate video generation into task framework
- [ ] Output management for video files

## Phase 5: Project & Data Management 🟡 In Progress
- [x] SQLite persistence layer (`services/database/database_service.py`)
- [x] Project service: create, read, update, list, search
- [ ] Delete operation for projects
- [ ] Associate generated assets (images/videos) with projects
- [ ] Character/scene entities in the database

## Phase 6: Advanced Story Generation Features 🔴 Not Started
- [ ] Story structure generation
- [ ] Multiple character management
- [ ] Scene progression
- [ ] Narrative coherence
- [ ] Multi-modal content generation

## Technical Considerations
- [x] Keep model directories organized and scalable
- [x] Document all APIs and interfaces clearly (`docs/humans/`, `docs/llm/`)
- [ ] Maintain backward compatibility with existing implementations (ongoing)
- [ ] Ensure smooth integration between different generation approaches
- [ ] Decide: wire up segmentation model or remove it from install list (~1.2GB unused)

## Future Enhancements
- [x] Integration with text generation models (prompt enhancement via Phi-3)
- [ ] Web interface for easy access
- [ ] Export/import of projects
- [ ] Advanced search filters
