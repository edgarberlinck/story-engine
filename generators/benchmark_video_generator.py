#!/usr/bin/env python3
"""
Benchmark suite for image-to-video generation models.

Benchmark 1 — "Yamu":
  1. Character: Yamu, a Brazilian indian (dark hair, tall, strong body,
     face paint, feathers in his hair).
  2. Scene: Yamu riding a horse, holding a long bow and arrow, ready to
     shoot a tiger (face-validated against his reference).
  3. Video: Yamu killing the tiger with a long bow arrow, generated with
     EVERY i2v model. >= 4 seconds, >= 720p, named benchmark_<model>.

Benchmark 2 — "Tribe meeting":
  1. Character: Richard Morton, a tall 40yo Swedish archeologist, blonde,
     green eyes, normal body.
  2. Character: Cristal, an old lady, Shaman of Yamu's tribe.
  3. Scene: Cristal in the center of a house, Richard in front of her and
     Yamu right in the back (all three face-validated).
  4. Video: Cristal talking to Richard — same specs as benchmark 1.

Face recognition is REQUIRED: scenes are regenerated until every character
is verified, and the benchmark aborts if verification is impossible.

Outputs follow the project structure (mock project 'test_project'):
  outputs/test_project/characters/<name>/reference.png
  outputs/test_project/scenes/scene_<n>/scene.png
  outputs/test_project/scenes/scene_<n>/out/benchmark_<model>.mp4 (+ metrics)
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generators.image_engine import generate_character, get_character
from generators.video_engine import create_validated_scene
from generators.video_generator import AVAILABLE_VIDEO_MODELS, generate_video
from utils.project_paths import scene_out_dir

# Characters used across benchmarks.
CHARACTERS = {
    "Yamu": (
        "Portrait of Yamu, a Brazilian indian man, dark hair, tall, strong "
        "muscular body. He wears traditional indian painting on his face and "
        "has feathers in his hair. Photorealistic, detailed face"
    ),
    "Richard Morton": (
        "Portrait of Richard Morton, a tall 40 year old Swedish archeologist "
        "man, blonde hair, green eyes, normal average body build (not strong, "
        "not skinny), wearing practical field clothes. Photorealistic, "
        "detailed face"
    ),
    "Cristal": (
        "Portrait of Cristal, a very old lady, the Shaman of a Brazilian "
        "indian tribe, wrinkled face, wise eyes, traditional tribal shaman "
        "clothing and ornaments. Photorealistic, detailed face"
    ),
}

# Per-model overrides to guarantee >= 4 seconds and >= 720p.
# Frame counts respect each model's constraints (Wan/Hunyuan: 4k+1 frames).
#   wan22_i2v:          81 frames @ 16 fps = 5.06 s, 1280x720
#   hunyuan_video_i2v: 61 frames @ 15 fps = 4.07 s, 1280x720
BENCHMARK_VIDEO_PARAMS = {
    "wan22_i2v": {"width": 1280, "height": 720, "num_frames": 81, "fps": 16},
    "hunyuan_video_i2v": {"width": 1280, "height": 720, "num_frames": 61, "fps": 15},
}


def ensure_character(name: str):
    """Create a benchmark character if it does not exist yet."""
    if get_character(name) is None:
        print(f"Generating benchmark character '{name}'...")
        generate_character(name, CHARACTERS[name])
    else:
        print(f"Character '{name}' already exists; reusing reference.")


def generate_benchmark_videos(scene: dict, video_prompt: str) -> dict:
    """Animate a validated scene with every i2v model (benchmark specs)."""
    out_dir = scene_out_dir(scene["scene_number"])
    results = {}
    for model_name in AVAILABLE_VIDEO_MODELS:
        try:
            print(f"\n--- Generating video with {model_name} ---")
            results[model_name] = generate_video(
                image_path=scene["image_path"],
                prompt=video_prompt,
                model_name=model_name,
                output_dir=str(out_dir),
                output_basename=f"benchmark_{model_name}",
                seed=42,
                **BENCHMARK_VIDEO_PARAMS[model_name],
            )
        except Exception as e:
            print(f"Failed to generate video with {model_name}: {e}")
            results[model_name] = None
    return results


def run_benchmark_1() -> dict:
    """Yamu hunting a tiger."""
    print("\n########## Benchmark 1: Yamu ##########")
    ensure_character("Yamu")

    scene_prompt = (
        "Yamu riding a horse, holding a long bow and arrow, ready to shoot "
        "a tiger"
    )
    video_prompt = (
        "Yamu killing a tiger with a long bow arrow, dramatic action, the "
        "arrow flies and strikes the tiger, cinematic motion"
    )

    print(f"\n=== Generating scene: {scene_prompt} ===")
    scene = create_validated_scene(
        scene_prompt,
        character_name="Yamu",
        seed=42,
        require_verification=True,
    )
    return generate_benchmark_videos(scene, video_prompt)


def run_benchmark_2() -> dict:
    """Yamu and Richard Morton meeting Cristal, the tribe shaman."""
    print("\n########## Benchmark 2: Tribe meeting ##########")
    for name in ("Yamu", "Richard Morton", "Cristal"):
        ensure_character(name)

    scene_prompt = (
        "Yamu and Richard Morton talking to Cristal. Cristal is in the "
        "center of a house, Richard Morton is in front of her and Yamu is "
        "right in the back"
    )
    video_prompt = (
        "Cristal talking to Richard Morton, natural conversation, subtle "
        "gestures and facial expressions, cinematic motion"
    )

    print(f"\n=== Generating scene: {scene_prompt} ===")
    scene = create_validated_scene(
        scene_prompt,
        character_names=["Yamu", "Richard Morton", "Cristal"],
        seed=42,
        require_verification=True,
    )
    return generate_benchmark_videos(scene, video_prompt)


def print_summary(all_results: dict):
    print("\n=== Benchmark Results Summary ===")
    for benchmark_name, results in all_results.items():
        print(f"\n{benchmark_name}:")
        for model_name, result in results.items():
            if result:
                m = result["metrics"]
                seconds = m["num_frames"] / m["fps"]
                print(f"  {model_name}: {result['video_path']} "
                      f"({m['width']}x{m['height']}, {seconds:.1f}s, "
                      f"{m['duration_ms']} ms, {m['peak_memory_mb']} MB)")
            else:
                print(f"  {model_name}: FAILED")


def main():
    """Run all image-to-video benchmarks across all i2v models."""
    print("=== Image-to-Video Generation Benchmark Suite ===")
    print(f"Models: {', '.join(AVAILABLE_VIDEO_MODELS)} (default: wan22_i2v)")

    all_results = {}
    try:
        all_results["Benchmark 1 (Yamu vs tiger)"] = run_benchmark_1()
        all_results["Benchmark 2 (Tribe meeting)"] = run_benchmark_2()

        print_summary(all_results)
        print("\nBenchmark suite completed!")
        print("Compare scene_*/out/benchmark_<model>.mp4 and the matching "
              "*_benchmark_metrics.json files to pick the best model.")
    except Exception as e:
        print(f"Error during benchmark: {e}")
        if all_results:
            print_summary(all_results)


if __name__ == "__main__":
    main()
