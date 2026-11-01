#!/usr/bin/env python3
"""
Image-to-video benchmark: a random character walking.

Scene: a single character walks along a quiet street. The same validated
scene image and seed are animated with each benchmarked i2v model
(see models.py IMAGE_TO_VIDEO_MODELS).

To avoid overloading the machine, models run strictly ONE AT A TIME:
each pipeline is loaded, used for one clip, released (cleanup_pipeline) and
garbage-collected before the next starts. A failure in one model does not
abort the others. Pass model names as arguments to run a subset, e.g.
    python generators/benchmark_video_generator.py cosmos3_nano_i2v

Outputs (project BENCHMARK_PROJECT):
  outputs/<project>/scenes/scene_<n>/out/benchmark_<model>.mp4 (+ metrics)
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generators.image_generator import cleanup_pipeline
from generators.image_engine import generate_character, get_character
from generators.video_engine import create_validated_scene
from generators.video_generator import (
    AVAILABLE_VIDEO_MODELS,
    generate_video,
)
from utils.project_paths import scene_out_dir

BENCHMARK_PROJECT = "Test_ui"

CHARACTER_NAME = "Walker"
CHARACTER_PROMPT = (
    "Full-body portrait of Walker, a young person with short dark hair, "
    "wearing a green jacket, jeans and sneakers. Photorealistic, detailed face"
)

SCENE_PROMPT = (
    "Walker, standing on a quiet tree-lined street in soft morning light, "
    "full body visible, realistic anatomy, cinematic composition."
)

VIDEO_PROMPT = (
    "Walker walks forward along the street with a natural relaxed gait, "
    "arms swinging gently, steady tracking camera, realistic motion"
)

# Resolution, frame count and fps are owned by
# video_generator.MODEL_GENERATION_PARAMS; the benchmark passes only the
# conditioning image, prompt, model name, output location and a fixed seed.
# The scene image is generated at the models' shared 832x480 size.
SCENE_SIZE = {"width": 832, "height": 480}


def ensure_character(name: str = CHARACTER_NAME, project: str = BENCHMARK_PROJECT):
    """Create the benchmark character if it does not exist yet."""
    if get_character(name, project) is None:
        print(f"Generating benchmark character '{name}'...")
        generate_character(name, CHARACTER_PROMPT, project=project)
    else:
        print(f"Character '{name}' already exists; reusing reference.")


def generate_benchmark_videos(
    scene: dict,
    video_prompt: str,
    project: str = BENCHMARK_PROJECT,
    models=None,
) -> dict:
    """Animate a validated scene with each i2v model, strictly one at a time.

    generate_video unloads its pipeline before returning; the explicit
    gc/cache cleanup below guarantees memory is released between models.

    Model-specific generation parameters (resolution, frame count, fps,
    guidance) are owned by video_generator.MODEL_GENERATION_PARAMS; this
    benchmark passes only the conditioning image, motion prompt, model name,
    output location and a fixed seed, and lets video_generator apply each
    model's native parameters.

    Each raw clip is post-processed by the dependency-free video enhancer
    (temporal smoothing + light denoise + a high-quality low-CRF MKV). The
    enhanced path and its metrics are stored on the result.
    """
    out_dir = scene_out_dir(scene["scene_number"], project)
    results = {}
    for model_name in models or AVAILABLE_VIDEO_MODELS:
        try:
            print(f"\n--- Generating video with {model_name} ---")
            results[model_name] = generate_video(
                image_path=scene["image_path"],
                prompt=video_prompt,
                model_name=model_name,
                output_dir=str(out_dir),
                output_basename=f"benchmark_{model_name}",
                seed=42,
            )
            # Post-generation enhancement of the raw clip. Dependency-free
            # (numpy/scipy/imageio); failures degrade to the raw clip.
            try:
                from generators.video_enhancer import enhance_video

                r = results[model_name]
                enhanced = enhance_video(
                    r["video_path"],
                    output_dir=str(out_dir),
                    output_basename=f"benchmark_{model_name}",
                    fps=r["metrics"].get("fps", 25.0),
                )
                r["enhanced_video_path"] = enhanced["video_path"]
                r["enhanced_metrics"] = enhanced["metrics"]
            except Exception as e:
                print(f"Enhancement skipped for {model_name}: {e}")
        except Exception as e:
            print(f"Failed to generate video with {model_name}: {e}")
            results[model_name] = None
        finally:
            cleanup_pipeline(None)
    return results


def run_walking_benchmark(project: str = BENCHMARK_PROJECT, models=None) -> dict:
    """Benchmark: a random character walking."""
    print("\n########## Benchmark: Character walking ##########")
    ensure_character(CHARACTER_NAME, project)

    print(f"\n=== Generating scene: {SCENE_PROMPT[:80]}... ===")
    # Face verification is non-fatal: the point is to see whether a video
    # can be generated.
    scene = create_validated_scene(
        SCENE_PROMPT,
        character_names=[CHARACTER_NAME],
        seed=42,
        project=project,
        require_verification=False,
        **SCENE_SIZE,
    )
    return generate_benchmark_videos(scene, VIDEO_PROMPT, project, models)


def print_summary(all_results: dict):
    print("\n=== Benchmark Results Summary ===")
    for benchmark_name, results in all_results.items():
        print(f"\n{benchmark_name}:")
        for model_name, result in results.items():
            if result:
                m = result["metrics"]
                seconds = m["num_frames"] / m["fps"]
                print(
                    f"  {model_name}: {result['video_path']} "
                    f"({m['width']}x{m['height']}, {seconds:.1f}s, "
                    f"{m['duration_ms']} ms, {m['peak_memory_mb']} MB)"
                )
            else:
                print(f"  {model_name}: FAILED")


def main(argv=None):
    """Run the walking benchmark on the requested (default: all) models."""
    requested = list(sys.argv[1:] if argv is None else argv) or None
    if requested:
        unknown = [m for m in requested if m not in AVAILABLE_VIDEO_MODELS]
        if unknown:
            sys.exit(
                f"Unknown model(s): {', '.join(unknown)}. "
                f"Available: {', '.join(AVAILABLE_VIDEO_MODELS)}"
            )
    print("=== Image-to-Video Generation Benchmark ===")
    print(
        f"Models: {', '.join(requested or AVAILABLE_VIDEO_MODELS)} "
        f"(run sequentially, one at a time)"
    )

    all_results = {}
    try:
        all_results["Character walking"] = run_walking_benchmark(models=requested)
        print_summary(all_results)
        print("\nBenchmark completed!")
    except Exception as e:
        print(f"Error during benchmark: {e}")
        if all_results:
            print_summary(all_results)


if __name__ == "__main__":
    main()
