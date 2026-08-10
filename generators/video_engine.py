#!/usr/bin/env python3
"""
Video engine: high-level orchestration for scene-to-video generation.

Workflow:
  1. Generate a character reference (image_engine.generate_character).
  2. Generate a scene image that mentions the character by name.
  3. Verify via face recognition that the character is in the scene;
     regenerate with a new seed if not.
  4. Animate the validated scene with image-to-video models. While
     benchmarking, the same scene is rendered with ALL available models
     (default single-model generation uses Wan).

Outputs follow the project structure:
  outputs/<project>/scenes/scene_<n>/scene.png
  outputs/<project>/scenes/scene_<n>/out/<videos + metrics>
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generators.image_engine import (
    generate_character,
    get_character,
    generate_scene,
    verify_character_in_scene,
)
from generators.video_generator import (
    AVAILABLE_VIDEO_MODELS,
    DEFAULT_VIDEO_MODEL,
    generate_video,
)
from utils.project_paths import DEFAULT_PROJECT, scene_out_dir

MAX_SCENE_ATTEMPTS = 3


def create_validated_scene(
    prompt: str,
    character_name: str = None,
    character_names: list = None,
    project: str = DEFAULT_PROJECT,
    seed: int = 42,
    max_attempts: int = MAX_SCENE_ATTEMPTS,
    require_verification: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Generate a scene and verify the presence of every listed character
    via face recognition, regenerating with new seeds on failure.

    Args:
        character_name: Single character to verify (convenience).
        character_names: List of characters to verify (all must appear).
        require_verification: If True, an inconclusive face check (e.g.
            face_recognition unavailable) or exhausted attempts raise
            RuntimeError instead of accepting the scene.

    Returns the scene dict (see image_engine.generate_scene) with an extra
    'character_verified' key (True/False/None).
    """
    names = list(character_names or [])
    if character_name:
        names.append(character_name)

    characters = []
    for name in names:
        character = get_character(name, project)
        if character is None:
            raise ValueError(
                f"Unknown character '{name}'. "
                "Generate it first with generate_character()."
            )
        characters.append(character)

    scene = None
    verified: Optional[bool] = None
    scene_number = None

    for attempt in range(1, max_attempts + 1):
        attempt_seed = seed + (attempt - 1)
        print(f"\n--- Scene attempt {attempt}/{max_attempts} (seed={attempt_seed}) ---")
        scene = generate_scene(
            prompt,
            project=project,
            scene_number=scene_number,
            seed=attempt_seed,
            **kwargs,
        )
        scene_number = scene["scene_number"]  # reuse the same scene folder

        if not characters:
            verified = None
            break

        # Every character must be found; None (inconclusive) is tracked
        results = {
            c["name"]: verify_character_in_scene(c, scene["image_path"])
            for c in characters
        }
        if any(r is None for r in results.values()):
            if require_verification:
                raise RuntimeError(
                    "Face verification is required but inconclusive for: "
                    + ", ".join(n for n, r in results.items() if r is None)
                )
            print("Character verification inconclusive; accepting scene.")
            verified = None
            break
        if all(results.values()):
            print(f"All characters verified in scene: {', '.join(results)}")
            verified = True
            break
        missing = [n for n, r in results.items() if not r]
        verified = False
        print(f"Characters NOT found in scene: {', '.join(missing)}; regenerating...")

    if verified is False and require_verification:
        raise RuntimeError(
            f"Could not verify all characters in scene after {max_attempts} attempts."
        )

    scene["character_verified"] = verified
    return scene


def animate_scene(
    scene: Dict[str, Any],
    model_name: str = DEFAULT_VIDEO_MODEL,
    project: str = DEFAULT_PROJECT,
    seed: int = 42,
    **overrides: Any,
) -> Dict[str, Any]:
    """Generate a video for a scene using a single i2v model (default: wan).

    Videos are written to outputs/<project>/scenes/scene_<n>/out/.
    """
    out_dir = scene_out_dir(scene["scene_number"], project)
    return generate_video(
        image_path=scene["image_path"],
        prompt=scene.get("enriched_prompt", scene["prompt"]),
        model_name=model_name,
        output_dir=str(out_dir),
        output_basename=f"scene_{scene['scene_number']}_{model_name}",
        seed=seed,
        **overrides,
    )


def benchmark_scene_video(
    scene: Dict[str, Any],
    project: str = DEFAULT_PROJECT,
    seed: int = 42,
) -> Dict[str, Any]:
    """Animate the same scene with ALL available i2v models for benchmarking.

    Returns a dict mapping model name -> generation result (or None on
    failure). Metrics JSON files are saved next to each video in out/.
    """
    results: Dict[str, Any] = {}
    print(f"\n=== Video benchmark for scene_{scene['scene_number']} ===")
    for model_name in AVAILABLE_VIDEO_MODELS:
        try:
            print(f"\n--- Generating video with {model_name} ---")
            results[model_name] = animate_scene(
                scene, model_name=model_name, project=project, seed=seed
            )
        except Exception as e:
            print(f"Failed to generate video with {model_name}: {e}")
            results[model_name] = None
    return results


def main():
    """Demo/benchmark run using the mock 'test_project'."""
    character_name = "Richard Morton"
    character_prompt = (
        "Portrait of Richard Morton, a Swedish archeologist, a tall blonde "
        "man in his mid-40s with a medium build, wearing practical field "
        "clothes, photorealistic, detailed face"
    )
    scene_prompt = (
        f"{character_name} is entering a cave, he is staring at the wall and "
        "has a glimpse of various ancient drawings"
    )

    print("=== Video Generation Benchmark ===")

    if get_character(character_name) is None:
        generate_character(character_name, character_prompt)
    else:
        print(f"Character '{character_name}' already exists; reusing reference.")

    scene = create_validated_scene(scene_prompt, character_name=character_name)
    results = benchmark_scene_video(scene)

    print("\n=== Benchmark Results Summary ===")
    for model_name, result in results.items():
        if result:
            m = result["metrics"]
            print(f"  {model_name}: {result['video_path']} "
                  f"({m['duration_ms']} ms, {m['peak_memory_mb']} MB)")
        else:
            print(f"  {model_name}: FAILED")


if __name__ == "__main__":
    main()
