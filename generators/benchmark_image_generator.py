#!/usr/bin/env python3
"""
Benchmark suite for image generation models.
Runs a set of representative tasks (characters, scenes, objects, interactions)
across all available diffusion models so the best model can be chosen per task.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generators.image_generator import (
    AVAILABLE_DIFFUSION_MODELS,
    generate_images,
    setup_model_directories,
)

from generators.text_generator import generate_prompt_with_llm


def benchmark_models(prompt, num_images=1, task_name=None, seed=42):
    """Generate images using all available models for benchmarking.

    Args:
        prompt (str): Description of the image to generate
        num_images (int): Number of images to generate with each model (default: 1)
        task_name (str): Optional benchmark task name used in output filenames
        seed (int): Random seed shared across models for a fair comparison

    Returns:
        dict: Dictionary mapping model names to lists of generated image file paths
    """
    results = {}
    print(f"Starting benchmark for prompt: '{prompt}'")

    # Iterate through all available diffusion models
    for model_name in AVAILABLE_DIFFUSION_MODELS.keys():
        try:
            print(f"\n--- Generating with {model_name} ---")
            files = generate_images(
                prompt, num_images, model_name, seed=seed, task_name=task_name
            )
            results[model_name] = files
        except Exception as e:
            print(f"Failed to generate with {model_name}: {e}")
            results[model_name] = []

    return results


def main():
    """Run a multi-task benchmark suite across all diffusion models.

    Tasks are designed to compare model quality per category (characters,
    scenes, objects, interactions) so faster models can be chosen for the
    tasks they handle well.
    """
    print("=== Image Generation Benchmark Suite ===")
    setup_model_directories()

    # Base character description, reused across tasks for consistency.
    character = (
        "A Swedish archeologist, a tall blonde man in his mid-40s with a "
        "medium build, wearing practical field clothes"
    )

    # (task_name, short description) pairs. task_name is used in filenames
    # and metrics files so results can be compared per task.
    benchmark_tasks = [
        ("char_base", f"Portrait of {character}, neutral expression"),
        ("char_smiling", f"Portrait of {character}, smiling warmly"),
        ("scene_forest_morning", "A dense forest during the morning, soft "
                                 "sunlight filtering through the trees, mist"),
        ("scene_forest_night", "The same dense forest at night, moonlight, "
                               "dark atmosphere, fireflies"),
        ("objects_still_life", "A still life of simple objects on a wooden "
                               "table: colorful balls, forks, spoons, a cup"),
        ("char_interaction", f"{character} carefully brushing dust off an "
                             "ancient artifact at an excavation site"),
    ]

    all_results = {}

    try:
        for task_name, description in benchmark_tasks:
            print(f"\n=== Task: {task_name} ===")

            # Use a text generation model to enrich the prompt
            prompt = generate_prompt_with_llm(description)

            all_results[task_name] = benchmark_models(
                prompt, num_images=1, task_name=task_name, seed=42
            )

        # Show results summary
        print("\n=== Benchmark Results Summary ===")
        for task_name, task_results in all_results.items():
            print(f"\nTask: {task_name}")
            for model_name, files in task_results.items():
                print(f"  {model_name}: {len(files)} image(s) generated")

        print("\nBenchmark suite completed successfully!")
        print("Compare outputs/<task>_<model>_benchmark.png and the matching "
              "*_benchmark_metrics.json files to pick the best model per task.")

    except Exception as e:
        print(f"Error during generation: {e}")


if __name__ == "__main__":
    main()
