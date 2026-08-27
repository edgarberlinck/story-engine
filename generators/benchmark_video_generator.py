#!/usr/bin/env python3
"""
Definitive image-to-video benchmark: the café conversation.

Scene:
  A cinematic two-character conversation. Nikita and Roger sit at a small
  table in a quiet, stylish café during the morning. Nikita (long curly red
  hair, friendly) is on the LEFT; Roger (bald, dark-skinned, muscular,
  calm) is on the RIGHT. Both visible from the waist up, facing each other,
  faces clear and unobstructed, medium cinematic shot, warm morning light.

Dialogue (drives the future TTS + lip-sync + music stage):
  Nikita: "Good morning, Roger."
  Roger:  "Good morning to you too, Nikita."
  Background: quiet, relaxed café ambience.

Benchmark rules:
   - The characters live in the 'Test_ui' project, so the benchmark runs
     against that project (see BENCHMARK_PROJECT).
   - Face verification is SOFT: both characters are checked for presence in
     the scene, but a non-match or an inconclusive check does not abort it
     (the point of this benchmark is to see whether a video can be
     generated, not to gate it on perfect face detection).
   - The same validated scene is animated with EVERY i2v model.
    - Every video is >= 4 s, seeded identically, named
     benchmark_<model>.mp4, with a metrics JSON next to it. The surviving
     model (ltx_video_095_i2v) is run at its LTX-native 704x512 / 161f / 25fps
     = 6.44s; the original ">= 720p" bar was dropped because the 64 GB run that
     could hit 720p (Wan 2.2 A14B) OOMs — see docs/image-to-video.md §5.

Outputs follow the project structure (project 'Test_ui'):
  outputs/Test_ui/characters/<name>/reference.png
  outputs/Test_ui/scenes/scene_<n>/scene.png
  outputs/Test_ui/scenes/scene_<n>/out/benchmark_<model>.mp4 (+ metrics)
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generators.image_engine import generate_character, get_character
from generators.video_engine import create_validated_scene
from generators.video_generator import (
    AVAILABLE_VIDEO_MODELS,
    DEFAULT_VIDEO_MODEL,
    generate_video,
)
from utils.project_paths import scene_out_dir

# Nikita and Roger live in the 'Test_ui' project (their reference images and
# scenes), NOT the default 'test_project'. Run the benchmark against the
# project that actually owns the characters.
BENCHMARK_PROJECT = "Test_ui"

# Characters used by the benchmark. Portraits drive character generation;
# the scene prompt only references them by name.
CHARACTERS = {
    "Nikita": (
        "Portrait of Nikita, a young woman with long curly red hair and fair "
        "skin, natural friendly expression, warm eyes. Photorealistic, "
        "detailed face"
    ),
    "Roger": (
        "Portrait of Roger, a bald dark-skinned man with a muscular build "
        "and a calm friendly expression. Photorealistic, detailed face"
    ),
}

# The definitive benchmark scene: both characters in one frame, positioned
# explicitly, medium cinematic shot, warm morning café light.
SCENE_PROMPT = (
    "Nikita and Roger sitting together at a small table in a quiet, stylish "
    "café during the morning. Nikita is sitting on the left side of the "
    "image, she has long curly red hair and a natural friendly expression, "
    "she is looking toward Roger. Roger is sitting on the right side of the "
    "image, he is a bald dark-skinned man with a muscular build and a calm "
    "friendly expression, he is looking toward Nikita. Both characters are "
    "clearly visible from the waist up, sitting naturally and facing each "
    "other, their faces clearly visible and unobstructed. The composition "
    "leaves enough visual space around both characters for subtle natural "
    "movements during a conversation. Warm morning light enters through the "
    "windows, the café environment is realistic but not visually "
    "distracting, soft background details, natural shadows, coherent "
    "lighting and realistic perspective. Medium cinematic shot, both "
    "characters framed together in the same image. Nikita and Roger appear "
    "naturally integrated into the same environment with coherent body "
    "proportions, lighting, shadows and perspective. Photorealistic, "
    "cinematic composition, realistic facial features, natural body "
    "posture, subtle storytelling atmosphere"
)

# Video prompt: describes the natural conversation motion for the i2v
# animation (scene content stays in the conditioning image).
VIDEO_PROMPT = (
    "Nikita and Roger having a natural morning conversation, subtle gestures "
    "and facial expressions, gentle head movements, cinematic motion"
)

# Dialogue and background sound for the audio stage (TTS, lip sync and
# music are the next pipeline phase; captured here as the benchmark spec).
DIALOGUE = [
    {"character": "Nikita", "line": "Good morning, Roger."},
    {"character": "Roger", "line": "Good morning to you too, Nikita."},
]
BACKGROUND_SOUND = "quiet and relaxed background ambience"

# Per-model overrides to guarantee >= 4 seconds at the model's native resolution.
# Frame counts respect each model's latent constraints.
#   ltx_video_095_i2v: 161 frames @ 25 fps = 6.44 s, 704x512 (LTX native;
#   VAE temporal stride 8 -> num_frames == 1 (mod 8)). The original >= 720p bar
#   was dropped: only Wan 2.2 A14B could output 720p but it OOMs on 64 GB macOS.
# (Benchmark decision 2026-08: ltx_video_095_i2v is the surviving i2v model;
#  Wan 2.2 I2V A14B OOMs on 64 GB macOS; HunyuanVideo-I2V dropped.)

BENCHMARK_VIDEO_PARAMS = {
    "ltx_video_095_i2v": {"width": 704, "height": 512, "num_frames": 161, "fps": 25},   # LTX native
}


def ensure_character(name: str, project: str = BENCHMARK_PROJECT):
    """Create a benchmark character if it does not exist yet."""
    if get_character(name, project) is None:
        print(f"Generating benchmark character '{name}'...")
        generate_character(name, CHARACTERS[name], project=project)
    else:
        print(f"Character '{name}' already exists; reusing reference.")


def generate_benchmark_videos(scene: dict, video_prompt: str,
project: str = BENCHMARK_PROJECT) -> dict:
    """Animate a validated scene with every i2v model (benchmark specs)."""
    out_dir = scene_out_dir(scene["scene_number"], project)
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
                **BENCHMARK_VIDEO_PARAMS.get(model_name, {}),
            )
        except Exception as e:
            print(f"Failed to generate video with {model_name}: {e}")
            results[model_name] = None
    return results


def run_cafe_conversation_benchmark(project: str = BENCHMARK_PROJECT) -> dict:
    """The definitive benchmark: Nikita and Roger talking in a café."""
    print("\n########## Benchmark: Café conversation (Nikita & Roger) ##########")
    for name in ("Nikita", "Roger"):
        ensure_character(name, project)

    print(f"\n=== Generating scene: {SCENE_PROMPT[:80]}... ===")
    # Face verification is checked but NON-FATAL (require_verification=False):
    # an inconclusive/non-matching check must not abort the benchmark — the
    # point is to see whether a video can be generated.
    scene = create_validated_scene(
        SCENE_PROMPT,
        character_names=["Nikita", "Roger"],
        seed=42,
        project=project,
        require_verification=False,
     )

    print("\n=== Dialogue for the audio stage ===")
    for d in DIALOGUE:
        print(f"     {d['character']}: \"{d['line']}\"")
    print(f"  Background: {BACKGROUND_SOUND}")

    return generate_benchmark_videos(scene, VIDEO_PROMPT, project)


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
    """Run the image-to-video benchmark across all i2v models."""
    print("=== Image-to-Video Generation Benchmark Suite ===")
    print(f"Models: {', '.join(AVAILABLE_VIDEO_MODELS)} (default: {DEFAULT_VIDEO_MODEL})")

    all_results = {}
    try:
        all_results["Café conversation (Nikita & Roger)"] = run_cafe_conversation_benchmark()

        print_summary(all_results)
        print("\nBenchmark suite completed!")
        print("Compare scene_*/out/benchmark_<model>.mp4 and the matching "
              "*_benchmark_metrics.json files to pick the winning i2v model.")
    except Exception as e:
        print(f"Error during benchmark: {e}")
        if all_results:
            print_summary(all_results)


if __name__ == "__main__":
    main()
