#!/usr/bin/env python3
"""
Enhanced image generation script with organized model structure and multiple image support.
This script demonstrates how to use the organized models directory structure.
"""

import os
import sys
import re
import time
from pathlib import Path
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, FluxPipeline, Flux2Pipeline
import torch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import model constants
from models import DIFFUSION_MODELS, MODEL_PATHS, DEFAULT_MODEL_CONFIGS, TEXT_GENERATION_MODELS, get_model_config
from utils.model_metrics import ModelMetrics

# Define the available diffusion models for image generation
AVAILABLE_DIFFUSION_MODELS = {
    "stable_diffusion_v1_5": "runwayml/stable-diffusion-v1-5",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "flux_klein": "black-forest-labs/FLUX.2-klein-4B",
    "flux_dev": "black-forest-labs/FLUX.1-dev"
}

def resolve_model_path(model_type, model_name, hub_id):
    """Resolve a model to its locally installed path (from `make install`).

    Args:
        model_type (str): Key in MODEL_PATHS ('diffusion', 'text_generation', ...)
        model_name (str): Local directory name of the model
        hub_id (str): Hugging Face hub repo id used as a fallback

    Returns:
        str: Local path if the model was installed, otherwise the hub id.
    """
    local_path = os.path.join(str(project_root), MODEL_PATHS[model_type], model_name)
    if os.path.isdir(local_path) and os.listdir(local_path):
        return local_path
    print(f"Warning: {model_name} not found locally at {local_path}. "
          f"Run 'make install' first. Falling back to hub download: {hub_id}")
    return hub_id

def setup_model_directories():
    """Ensure all required model directories exist."""
    model_dirs = [
        "models/diffusion",
        "models/segmentation",
        "models/text_generation"
    ]
    
    for dir_path in model_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Ensured directory exists: {dir_path}")
        
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)
    print("Ensured outputs directory exists")

def generate_filename_from_prompt(prompt):
    """Generate a concise filename from the prompt using text generation models.
    
    Args:
        prompt (str): Description of the image to generate
        
    Returns:
        str: Generated filename (without extension, max 20 characters)
    """
    # Handle empty or None prompt
    if not prompt:
        return "generated_image"
    
    # Clean and normalize the prompt for filename creation
    clean_prompt = re.sub(r'[^\w\s-]', '', prompt.lower())
    clean_prompt = re.sub(r'[-\s]+', '_', clean_prompt)
    
    # Take first meaningful words to form a concise filename  
    words = [w for w in clean_prompt.split('_') if w and len(w) >= 2]
    
    # Use first two words or the entire cleaned prompt (limited to 20 chars)
    if len(words) >= 2:
        filename = '_'.join(words[:2])
    else:
        filename = clean_prompt
    
    # Ensure it starts with a letter or number and is at least 3 characters
    if filename and not filename[0].isalnum():
        filename = "img_" + filename
        
    # Trim to max 20 characters for conciseness  
    filename = filename[:20]
    
    # If the result is empty after all processing, return a default
    if not filename:
        filename = "generated_image"
    
    print(f"Generated concise filename from prompt '{prompt}': {filename}")
    
    return filename

def generate_prompt_with_llm(description, model_name="phi3_mini"):
    """Use a local text generation model to expand a short description
    into a rich image-generation prompt.

    Args:
        description (str): Short description of what should be in the image
        model_name (str): Text generation model key from TEXT_GENERATION_MODELS

    Returns:
        str: An enriched prompt (falls back to the raw description on failure)
    """
    try:
        from transformers import pipeline as hf_pipeline
        from transformers.utils import logging as hf_logging

        # Only surface real errors from transformers (hides benign
        # generation-config and tokenizer warnings).
        hf_logging.set_verbosity_error()

        model_id = resolve_model_path(
            "text_generation", model_name, TEXT_GENERATION_MODELS[model_name]
        )
        device, torch_dtype = get_model_config("text_generation")

        generator = hf_pipeline(
            "text-generation",
            model=model_id,
            dtype=torch_dtype,
            device=device,
        )

        # Configure generation on the model's generation_config to avoid
        # deprecated mixing of a generation_config with per-call kwargs.
        generator.model.generation_config.max_new_tokens = 120
        generator.model.generation_config.do_sample = False
        generator.model.generation_config.temperature = None
        generator.model.generation_config.top_p = None
        generator.model.generation_config.top_k = None

        instruction = (
            "Rewrite the following image description as a single detailed "
            "prompt for a text-to-image diffusion model. Reply with only the "
            f"prompt, no explanations.\n\nDescription: {description}\n\nPrompt:"
        )

        result = generator(
            instruction,
            return_full_text=False,
        )
        enriched = result[0]["generated_text"].strip().split("\n")[0].strip()

        # Free memory before diffusion models load
        del generator

        if enriched:
            print(f"LLM-enriched prompt: {enriched}")
            return enriched
    except Exception as e:
        print(f"Prompt enrichment failed ({e}); using raw description.")

    return description

def generate_images(
    prompt, 
    num_images=1, 
    model_name="flux_dev", 
    seed=42,
    steps=30,
    cfg=7.5,
    width=1024,
    height=1024,
    task_name=None
):
    """Generate images using the specified diffusion model.
    
    Args:
        prompt (str): Description of the image to generate
        num_images (int): Number of images to generate (default: 1)
        model_name (str): Name of the diffusion model to use (default: flux_dev)
        seed (int): Random seed for reproducibility (default: 42)
        steps (int): Number of inference steps (default: 30)
        cfg (float): Classifier-free guidance scale (default: 7.5)
        width (int): Image width (default: 1024)
        height (int): Image height (default: 1024)
        task_name (str): Optional benchmark task name used as the filename base
        
    Returns:
        list: List of paths to generated image files
    """
    print(f"Generating {num_images} image(s) for prompt: '{prompt}'")
    
    # Setup directories 
    setup_model_directories()
    
    # Get the actual model path from available models
    if model_name not in DIFFUSION_MODELS:
        raise ValueError(f"Unsupported diffusion model: {model_name}")
        
    model_path = resolve_model_path("diffusion", model_name, DIFFUSION_MODELS[model_name])
    print(f"Using diffusion model: {model_name}")
    print(f"Model path: {model_path}")
    
    # Load the pipeline for image generation
    try:
        # Resolve device/dtype from DEFAULT_MODEL_CONFIGS (mps > cuda > cpu)
        device, torch_dtype = get_model_config("diffusion")

        # Determine pipeline type based on model name
        if model_name == "sdxl":
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                safety_checker=None
            )
        elif model_name == "flux_dev":
            # FLUX.1 models use FluxPipeline
            pipe = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype
            )
        elif "flux" in model_name.lower():
            # FLUX.2 models (e.g. Klein) use Flux2Pipeline
            pipe = Flux2Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                safety_checker=None
            )

        pipe = pipe.to(device)
        
        # Set up generation parameters 
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Generate the image(s)
        metrics_tracker = ModelMetrics()
        metrics_tracker.start_timer()
        
        if model_name == "flux_dev":
            # FLUX Dev has specific requirements for generation
            images = pipe(
                prompt, 
                generator=generator,
                num_inference_steps=steps,
                guidance_scale=cfg,
                width=width,
                height=height,
                max_sequence_length=512  # Standard for FLUX models
            )
        elif "flux" in model_name.lower():
            # For other Flux models like Klein
            images = pipe(
                prompt, 
                generator=generator,
                num_inference_steps=steps,
                guidance_scale=cfg,
                width=width,
                height=height
            )
        else:
            # Standard SD models
            images = pipe(
                prompt, 
                num_images=num_images,
                num_inference_steps=steps,
                guidance_scale=cfg,
                width=width,
                height=height,
                generator=generator
            )
            
        metrics_tracker.end_timer()
        
        generated_images = []
        
        # Save each image with metadata
        filename_base = task_name if task_name else generate_filename_from_prompt(prompt)
        for i, image in enumerate(images.images):
            output_path = f"outputs/{filename_base}_{model_name}_benchmark.png"
            image.save(output_path)
            generated_images.append(output_path)
            print(f"Image {i+1} saved to: {output_path}")
            
        # Record and save metrics
        metrics_tracker.record_generation(
            model_name, prompt, seed, steps, cfg, width, height, output_path
        )
        metrics_filename = metrics_tracker.save_metrics(filename_base, model_name)
        print(f"Metrics saved to: {metrics_filename}")
        
    except Exception as e:
        print(f"Error during image generation: {e}")
        raise
        
    return generated_images

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

def generate_image(prompt, model_name=None, amount=1, output_filename=None):
    """Generate one or more images based on the given prompt.
    
    Args:
        prompt (str): Description of the image to generate
        model_name (str): Name of the diffusion model to use (defaults to stable_diffusion_v1_5)
        amount (int): Number of images to generate with the same prompt (default: 1)
        output_filename (str): Optional base filename (if None, will be auto-generated from prompt)
    
    Returns:
        list: List of paths to the generated image files
    """
    print(f"Generating {amount} image(s) for prompt: '{prompt}'")
    
    # Get model name with defaults
    if model_name is None:
        model_name = "stable_diffusion_v1_5"
    
    # Get model path from constants
    if model_name in DIFFUSION_MODELS:
        model_path = os.path.join(MODEL_PATHS["diffusion"], model_name)
        print(f"Using diffusion model: {DIFFUSION_MODELS[model_name]}")
    else:
        model_path = "models/diffusion/stable-diffusion-v1-5"
        print(f"Using default model: stable-diffusion-v1-5")
        
    print(f"Model path: {model_path}")
    
    # Handle multiple image generation
    generated_files = []
    
    # Use the appropriate pipeline based on model type
    try:
        device, torch_dtype = get_model_config("diffusion")

        if model_name == "flux_dev":
            pipe = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype
            )
        elif "flux" in model_name.lower():
            pipe = Flux2Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype
            )
        else:
            # Use StableDiffusionPipeline for other models
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype
            )

        pipe = pipe.to(device)
        
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        print("Falling back to default pipeline...")
        # Fallback to basic StableDiffusionPipeline
        try:
            device, torch_dtype = get_model_config("diffusion")
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch_dtype
            )
            pipe = pipe.to(device)
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            raise ValueError("Could not load any pipeline for image generation")
    
    for i in range(amount):
        # Generate filename if not provided or make unique filenames when generating multiple images
        if output_filename is None:
            base_filename = generate_filename_from_prompt(prompt)
        else:
            base_filename = output_filename
        
        # If generating multiple, append counter to filename
        if amount > 1:
            filename_suffix = f"_{i+1:03d}" 
            filename = f"{base_filename}{filename_suffix}"
        else:
            filename = base_filename
            
        # Ensure safe filename (remove invalid characters)
        filename = re.sub(r'[^\w\-_\.]', '', filename)
        if not filename:
            filename = "generated_image"
        
        # Final output path
        output_path = f"outputs/{filename}.png"
        print(f"Image {i+1} will be saved to: {output_path}")
        
        try:
            # Generate image using the pipeline
            image = pipe(prompt).images[0]
            image.save(output_path)
            print(f"Image {i+1} generated successfully!")
            generated_files.append(output_path)
            
        except Exception as e:
            print(f"Error generating image {i+1}: {e}")
            # Create a placeholder file for failed generations
            with open(output_path, "w") as f:
                f.write("Failed to generate image")
            generated_files.append(output_path)
    
    print(f"Generated {len(generated_files)} image(s)")
    return generated_files

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