#!/usr/bin/env python3
"""
Nikita & Roger 3-scene cinematic benchmark for the Test_ui project.

Benchmark spec (from the user):
  1. Scene 1 - Nikita Arrives:
     Nikita walking confidently toward the camera, elegant black suit, dark
     sunglasses, full body, modern cinematic location, natural daylight.
     No dialogue. (~5 s)
  2. Scene 2 - Nikita Greets Roger:
     Continuation. Nikita stops, removes her sunglasses and says
     "Good morning, Roger." Ideally, use the final frame of Scene 1 as the
     visual reference for Scene 2 (shot continuity).
  3. Scene 3 - Roger Responds:
     Roger near a large window with a cup of coffee, turns his head toward
     Nikita and says "Good morning, Nikita." Same cinematic location.

Each scene is animated with the winning i2v model wan22_i2v at benchmark
resolution (1280x720, >=4 s), and the 3 clips are joined into a single
sequence video with ffmpeg.

Usage:
  python scripts/video_benchmark_nikita_roger.py --scenes      # generate 3 scene images
  python scripts/video_benchmark_nikita_roger.py --videos      # animate scenes w/ both models
  python scripts/video_benchmark_nikita_roger.py --join        # ffmpeg concat per model
  python scripts/video_benchmark_nikita_roger.py --all         # everything
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generators.reference_scene_generator import generate_reference_conditioned_scene  # noqa: E402
from generators.video_generator import AVAILABLE_VIDEO_MODELS, generate_video  # noqa: E402
from services.database.scene_service import scene_service  # noqa: E402
from utils.project_paths import scene_dir, scene_out_dir  # noqa: E402

PROJECT = "Test_ui"
SEED = 42
SCENE_W, SCENE_H = 1280, 720

NIKITA_REF = "outputs/Test_ui/characters/Nikita/reference.png"
ROGER_REF = "outputs/Test_ui/characters/Roger/reference.png"

# Per-model benchmark overrides (>= 4 s, >= 720p) — see
# generators/benchmark_video_generator.py BENCHMARK_VIDEO_PARAMS.
BENCHMARK_VIDEO_PARAMS = {
    "wan22_i2v": {"width": 1280, "height": 720, "num_frames": 81, "fps": 16},
}

SCENES = [
    {
        "name": "Scene 1 - Nikita Arrives",
        "prompt": (
            "Nikita is walking confidently toward the camera along a modern "
            "cinematic location, a bright contemporary plaza with clean "
            "architecture and glass facades, natural daylight. She is a tall "
            "beautiful woman with long curly red hair and fair skin, wearing "
            "an elegant black suit with a fitted blazer and trousers, dark "
            "sunglasses covering her eyes, high heels, full body visible, "
            "walking with a confident stride, long legs, graceful posture. "
            "Wide cinematic shot, natural sunlight, photorealistic, high "
            "detail, cinematic composition, realistic body proportions, "
            "coherent lighting, shadows, perspective."
        ),
        "video_prompt": (
            "Nikita walking confidently toward the camera with a natural "
            "stride, gentle body movement, hair swaying slightly, cinematic "
            "motion, steady camera."
        ),
        "references": [NIKITA_REF],
        "dialogue": None,
    },
    {
        "name": "Scene 2 - Nikita Greets Roger",
        "prompt": (
            "Continuation of the same modern cinematic location. Nikita has "
            "stopped walking and is now standing, she removes her dark "
            "sunglasses with one hand and looks directly toward the camera "
            "with a friendly greeting expression, lips slightly parted as if "
            "speaking. She is a tall beautiful woman with long curly red "
            "hair and fair skin, wearing the same elegant black suit with a "
            "fitted blazer and trousers, sunglasses held in one hand, "
            "visible from the waist up. Same plaza with clean architecture, "
            "natural daylight. Medium cinematic shot, photorealistic, high "
            "detail, cinematic composition, realistic facial features, "
            "coherent lighting, shadows, perspective."
        ),
        "video_prompt": (
            "Nikita removes her dark sunglasses with one hand, lowers them, "
            "and speaks while looking at the camera, subtle head movement, "
            "natural facial expression, gentle arm motion, cinematic motion."
        ),
        # Scene 1's frame is used as an additional visual reference for
        # shot continuity (same character, outfit and location). The
        # previous scene image is injected at runtime by _resolve_references.
        "references": [NIKITA_REF],
        "dialogue": {"character": "Nikita", "line": "Good morning, Roger."},
    },
    {
        "name": "Scene 3 - Roger Responds",
        "prompt": (
            "Roger standing near a large window inside a modern elegant "
            "interior, holding a cup of coffee, he turns his head toward the "
            "camera with a calm friendly expression, lips slightly parted as "
            "if speaking. He is a bald dark-skinned man with a muscular "
            "build, wearing formal clothing, dark blazer over a white dress "
            "shirt. Visible from the waist up. Soft natural daylight from "
            "the large window, the same cinematic atmosphere as the plaza "
            "scene. Medium cinematic shot, photorealistic, high detail, "
            "cinematic composition, realistic facial features, coherent "
            "lighting, shadows, perspective."
        ),
        "video_prompt": (
            "Roger turns his head toward the camera, raises the coffee cup "
            "slightly and speaks, subtle facial expression, gentle arm "
            "movement, cinematic motion, soft daylight."
        ),
        "references": [ROGER_REF],
        "dialogue": {"character": "Roger", "line": "Good morning, Nikita."},
    },
]


_MANIFEST = Path("outputs") / PROJECT / "scenes" / "benchmark_sequence.json"


def _load_manifest() -> list:
    """Scene numbers for the 3 benchmark scenes (index -> scene number)."""
    if _MANIFEST.is_file():
        return json.loads(_MANIFEST.read_text())
    return []


def _save_manifest(scene_numbers: list):
    _MANIFEST.write_text(json.dumps(scene_numbers, indent=2))


def _resolve_references(scene_spec: dict, prev_scene_image: Path | None) -> list:
    """Resolve reference images, embedding the previous scene frame when
    available (Scene 2 references Scene 1 for shot continuity)."""
    if scene_spec["name"].startswith("Scene 2") and prev_scene_image is not None:
        references = [NIKITA_REF, str(prev_scene_image)]
    else:
        references = scene_spec["references"]
    valid = [r for r in references if Path(r).is_file()]
    if not valid:
        raise FileNotFoundError(f"No valid references for {scene_spec['name']}")
    return valid


# Standard directory for benchmark output videos
VIDEOS_DIR = Path("outputs/videos")


def generate_scene_images():
    """Generate the 3 scene images via reference-conditioned FLUX.2 Klein."""
    results = []
    manifest = _load_manifest()
    prev_scene_image = None

    for index, scene_spec in enumerate(SCENES):
        if index < len(manifest):
            scene_number = manifest[index]
        else:
            scene_number = next_scene_number()
            manifest.append(scene_number)
            _save_manifest(manifest)

        print(f"\n=== Generating {scene_spec['name']} (scene_{scene_number}) ===")
        target_dir = scene_dir(scene_number, PROJECT)
        image_path = str(target_dir / "scene.png")

        references = _resolve_references(scene_spec, prev_scene_image)
        print(f"References: {references}")
        files = generate_reference_conditioned_scene(
            prompt=scene_spec["prompt"],
            reference_image_paths=references,
            model_name="flux_klein",
            seed=SEED + index,
            width=SCENE_W,
            height=SCENE_H,
            task_name=f"scene_{scene_number}",
        )
        shutil.move(files[0], image_path)
        print(f"Scene image saved to: {image_path}")

        scene_service.save_scene(
            project=PROJECT,
            scene_number=scene_number,
            prompt=scene_spec["prompt"],
            image_path=image_path,
            seed=SEED + index,
            model="flux_klein",
        )
        results.append({"scene_number": scene_number, "image_path": image_path})
        prev_scene_image = Path(image_path)
        print(f"Scene {scene_number} registered in database.")

    return results


def next_scene_number() -> int:
    """Next scene number for the Test_ui project (from project_paths)."""
    from utils.project_paths import next_scene_number as _next

    return _next(PROJECT)


def _sequence_scenes() -> list:
    """The 3 benchmark scenes, in order, from the manifest + database."""
    manifest = _load_manifest()
    scenes = []
    for scene_number in manifest:
        scene = scene_service.get_scene(PROJECT, scene_number)
        if scene is not None:
            scenes.append(scene)
    return scenes


def animate_scenes(model_name: str):
    """Animate every scene of the sequence with a given i2v model."""
    sequence_scenes = _sequence_scenes()
    if not sequence_scenes:
        print("No benchmark scenes found. Run --scenes first.")
        return []
    print(f"\n=== Animating {len(sequence_scenes)} scenes with {model_name} ===")
    results = []
    for index, scene in enumerate(sequence_scenes):
        scene_number = scene["scene_number"]
        spec = SCENES[index]
        # Use standard videos directory
        out_dir = VIDEOS_DIR / f"benchmark_{model_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = generate_video(
            image_path=scene["image_path"],
            prompt=spec["video_prompt"],
            model_name=model_name,
            output_dir=str(out_dir),
            output_basename=f"scene_{scene_number}",
            seed=SEED,
            **BENCHMARK_VIDEO_PARAMS[model_name],
        )
        results.append((scene_number, result))
        print(f"Scene {scene_number}: {result['video_path']}")
    return results


def join_sequence(model_name: str):
    """Concatenate the 3 scene clips for a model into one sequence video.

    TODO: Implement this step - currently tracked in ROADMAP for later
    implementation. Requires all 3 scene MP4 clips to exist in
    /outputs/videos/<model_name>/, then uses ffmpeg concat demuxer
    to join them into a single sequence video.

    Returns:
        None (not implemented yet).
    """
    print(f"WARNING: join_sequence not yet implemented for {model_name}")
    print("This step is tracked in the ROADMAP for future implementation.")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", action="store_true", help="generate the 3 scene images")
    parser.add_argument("--videos", action="store_true", help="animate scenes with all i2v models")
    parser.add_argument("--join", action="store_true", help="join clips into a sequence video per model")
    parser.add_argument("--all", action="store_true", help="run all stages")
    args = parser.parse_args()

    if args.all or args.scenes:
        generate_scene_images()
    if args.all or args.videos:
        for model_name in AVAILABLE_VIDEO_MODELS:
            try:
                animate_scenes(model_name)
            except Exception as e:  # noqa: BLE001
                print(f"ERROR animating with {model_name}: {e}")
    if args.all or args.join:
        for model_name in AVAILABLE_VIDEO_MODELS:
            try:
                join_sequence(model_name)
            except Exception as e:  # noqa: BLE001
                print(f"ERROR joining {model_name}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()