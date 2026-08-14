# Video Generation Guide

This guide describes how Story Engine generates videos from images, and the
workflow around characters and scenes.

## Overview

Video generation is image-to-video (i2v): we first create a still image of a
scene, validate it, then animate it. Three models are supported, and the
default is **Wan 2.2** (`wan22_i2v`):

| Model | Repo |
|---|---|
| Wan 2.2 I2V A14B (default) | `Wan-AI/Wan2.2-I2V-A14B` |
| HunyuanVideo I2V | `tencent/HunyuanVideo-I2V` |

## Project folder structure

All outputs are organized per project (currently the mock project
`test_project`):

```
outputs/test_project/
├── characters/
│   └── Richard Morton/        # character reference images
└── scenes/
    ├── scene_1/
    │   ├── <scene image>
    │   └── out/               # generated videos
    ├── scene_2/
    └── ...
```

## Workflow

### 1. Create a character

```python
from generators.image_engine import generate_character

generate_character("Richard Morton", prompt, model="flux_dev")
```

The character's prompt, seed, and model are stored in the database so scenes
can reference the character by name only. The reference image is saved to
`outputs/<project>/characters/<name>/reference.png`.

### 2. Generate and validate a scene

```python
from generators.video_engine import create_validated_scene

scene = create_validated_scene(
    "Richard Morton is entering a cave, staring at ancient drawings",
    character_name="Richard Morton",
)
```

The engine looks up "Richard Morton" and injects his stored description —
you never repeat the character prompt. After generation, face recognition
compares the scene against the character's reference image; if the character
isn't detected, the scene is regenerated with a new seed (up to 3 attempts).

Lower-level pieces are also available in `generators.image_engine`:
`generate_scene(prompt)` and `verify_character_in_scene(character, image)`.

> Face verification requires the optional `face_recognition` package
> (`pip install face_recognition`). Without it, the check is skipped with a
> warning.

### 3. Generate videos

```python
from generators.video_engine import animate_scene, benchmark_scene_video

animate_scene(scene)            # single model (default: wan22_i2v)
benchmark_scene_video(scene)    # all three models, with metrics
```

While benchmarking, the same scene is rendered with **all three models**, and
metrics (generation time, memory, parameters) are saved alongside the videos
in `scene_*/out/`, just like the image benchmark suite.

You can also run the full demo benchmark end-to-end:

```bash
python generators/video_engine.py
```
