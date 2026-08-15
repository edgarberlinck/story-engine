#!/usr/bin/env python3
"""
Enhanced image generation script with organized model structure and multiple image support.
This script demonstrates how to use the organized models directory structure.
"""

import gc
import os
import sys
import re
import time
from pathlib import Path

# Avoid HuggingFace tokenizers spawning fork-based parallelism (leaks
# semaphores on macOS and triggers resource_tracker warnings at shutdown).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    FluxPipeline,
    Flux2Pipeline,
    Flux2KleinPipeline,
    Flux2KleinKVPipeline,
)
import torch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import model constants
from models import DIFFUSION_MODELS, MODEL_PATHS, DEFAULT_MODEL_CONFIGS, TEXT_GENERATION_MODELS, get_model_config
from utils.model_metrics import ModelMetrics
from utils.configuration_manager import setup_model_directories

# Define the available diffusion models for image generation
AVAILABLE_DIFFUSION_MODELS = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "flux_dev": "black-forest-labs/FLUX.1-dev"
}

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





def cleanup_pipeline(pipe):
    """Release a diffusion pipeline and free accelerator memory.

    Deleting the pipeline and clearing device caches ensures torch/HF
    resources (including multiprocessing semaphores) are released before
    interpreter shutdown, avoiding resource_tracker leak warnings.
    """
    if pipe is not None:
        del pipe
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    pipe = None
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
        elif model_name == "flux_klein":
            # FLUX.2 Klein 4B uses the Flux2KleinPipeline (text-only generation).
            pipe = Flux2KleinPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype
            )
        else:
            # Default to standard pipeline for any other case (shouldn't happen now)
            pipe = StableDiffusionXLPipeline.from_pretrained(
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
        elif model_name == "flux_klein":
            # FLUX.2 Klein text-only generation (Flux2KleinPipeline supports guidance_scale).
            images = pipe(
                prompt,
                generator=generator,
                num_inference_steps=steps,
                guidance_scale=cfg,
                width=width,
                height=height,
                num_images_per_prompt=num_images,
                max_sequence_length=512
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
    finally:
        cleanup_pipeline(pipe)
        pipe = None

    return generated_images

def benchmark_models(*args, **kwargs):
    """Deprecated: moved to generators.benchmark_image_generator."""
    from generators.benchmark_image_generator import benchmark_models as _bm
    print("Warning: benchmark_models moved to generators.benchmark_image_generator; "
          "please update your imports.")
    return _bm(*args, **kwargs)

def generate_image(prompt, model_name=None, amount=1, output_filename=None):
    """Generate one or more images based on the given prompt.
    
    Args:
        prompt (str): Description of the image to generate
        model_name (str): Name of the diffusion model to use (defaults to sdxl)
        amount (int): Number of images to generate with the same prompt (default: 1)
        output_filename (str): Optional base filename (if None, will be auto-generated from prompt)
    
    Returns:
        list: List of paths to the generated image files
    """
    print(f"Generating {amount} image(s) for prompt: '{prompt}'")
    
    # Get model name with defaults
    if model_name is None:
        model_name = "sdxl"
    
    # Get model path from constants
    if model_name in DIFFUSION_MODELS:
        model_path = os.path.join(MODEL_PATHS["diffusion"], model_name)
        print(f"Using diffusion model: {DIFFUSION_MODELS[model_name]}")
    else:
        model_path = "models/diffusion/stable-diffusion-xl-base-1.0"
        print(f"Using default model: stable-diffusion-xl-base-1.0")
        
    print(f"Model path: {model_path}")
    
    # Handle multiple image generation
    generated_files = []
    
    # Use the appropriate pipeline based on model type
    pipe = None
    try:
        device, torch_dtype = get_model_config("diffusion")

        if model_name == "flux_dev":
            pipe = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype
            )
        elif model_name == "flux_klein":
            pipe = Flux2KleinPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype
            )
        else:
            # For sdxl and other models
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                safety_checker=None
            )

        pipe = pipe.to(device)
        
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        print("Falling back to default pipeline...")
        # Fallback to basic StableDiffusionXLPipeline (SDXL)  
        try:
            device, torch_dtype = get_model_config("diffusion")
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch_dtype,
                safety_checker=None
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

    cleanup_pipeline(pipe)
    pipe = None

    print(f"Generated {len(generated_files)} image(s)")
    return generated_files

def main():
    """Generate a single image from a command-line prompt."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate images from a text prompt")
    parser.add_argument("prompt", nargs="?", default="A beautiful sunset over the ocean",
                        help="Text prompt describing the image")
    parser.add_argument("--model", default=None,
                        help="Diffusion model to use (default: sdxl)")
    parser.add_argument("--amount", type=int, default=1,
                        help="Number of images to generate (default: 1)")
    parser.add_argument("--output", default=None,
                        help="Base output filename (default: derived from prompt)")
    args = parser.parse_args()

    setup_model_directories()
    generate_image(args.prompt, model_name=args.model,
                   amount=args.amount, output_filename=args.output)

if __name__ == "__main__":
    main()