# Video Generation — Requirements Specification

Source: product owner instructions (2026-08-10). This is the authoritative
spec for the `video_generator` / `video_engine` work.

## Components to build

1. **`generators/video_engine.py`** — high-level orchestration API.
2. **`generators/video_generator.py`** — low-level image-to-video generation
   using the models in `IMAGE_TO_VIDEO_MODELS` (`models.py`).

## Models

- Use **all** image-to-video models: `wan22_i2v`, `ltx_video`,
  `hunyuan_video_i2v`.
- **Default model: `wan22_i2v` (Wan)**.
- Each model must be invoked with its correct, model-specific parameters
  (pipeline class, dtype, resolution, num_frames, fps, guidance, etc.).

## Project-scoped output structure (mandatory)

Everything lives inside a project. Until real project wiring is done, use a
mock project named `test_project`:

```
outputs/<project>/characters/<character_name>/...files
outputs/<project>/scenes/scene_1/<image>       # scene_2, scene_3, ...
outputs/<project>/scenes/scene_*/out/<videos>  # generated videos go in out/
```

## Character generation (prerequisite for scenes)

- API: `generate_character(name, prompt, model="flux_dev")` in the image
  engine (`generators/image_engine.py`) — create it if missing.
- Uses the default image engine model (**flux_dev**).
- The character reference MUST be persisted in the database with:
  - name, prompt used, seed, model.
- Reference image saved under `outputs/<project>/characters/<name>/`.

## Scene generation

- Generate a scene by prompt that references a character by name, e.g.:
  `"Richard Morton is entering a cave, he is staring at the wall and has a
  glimpse of various ancient drawings"`.
- The engine must resolve the character by name from the DB — the caller
  does NOT resend the character's original prompt.
- **Validation:** after generating the scene image, run face recognition
  against the stored character reference to verify the character actually
  appears in the scene (`check_character_on_scene(character_reference, scene)`).
- If validation fails: regenerate the scene (new seed) and retry until a
  good result (bounded retries).

## Video generation & benchmarking

- Only convert a scene to video after character validation passes:
  `image_to_video(scene)`.
- Because we are benchmarking, generate the **same scene with all three
  i2v models** and record benchmark results (timings/metrics JSON), same
  approach as `generators/benchmark_image_generator.py` +
  `utils/model_metrics.py`.

## Reference flow (from product owner)

```python
generate_character("Richard Morton", prompt)
scene = generate_image("Richard Morton waving to the camera")

character_reference = get_character("Richard Morton")
if check_character_on_scene(character_reference, scene):
    image_to_video(scene)  # runs across all models for benchmark
else:
    # regenerate the scene and try again
    ...
```

## Implemented API (final names)

- `utils/project_paths.py` — `character_dir`, `scene_dir`, `scene_out_dir`,
  `next_scene_number`; default project `test_project`.
- `services/database/character_service.py` — `CharacterService`
  (`save_character`, `get_character`, `list_characters`,
  `find_characters_in_text`); `characters` table in `story_engine.db`
  storing project, name, prompt, seed, model, reference_image.
- `utils/face_check.py` — `character_appears_in_image(ref, scene)`;
  requires optional `face_recognition`, returns None when unavailable.
- `generators/image_engine.py` — `generate_character(name, prompt,
  model="flux_dev")`, `get_character(name)`, `generate_scene(prompt)`
  (auto-injects stored character descriptions),
  `verify_character_in_scene(character, scene_image_path)`.
- `generators/video_generator.py` — `generate_video(image_path, prompt,
  model_name="wan22_i2v", ...)` with per-model params in
  `MODEL_GENERATION_PARAMS`; writes video + `*_benchmark_metrics.json`.
- `generators/video_engine.py` — `create_validated_scene(prompt,
  character_name=...)` (retry with seed bump on failed face check),
  `animate_scene(scene, model_name="wan22_i2v")`,
  `benchmark_scene_video(scene)` (all models), `main()` demo.
- Tests: `test_video_engine.py`.
