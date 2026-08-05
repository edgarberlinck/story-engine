#!/usr/bin/env python3
"""
Enhanced image generation script with organized model structure and multiple image support.
This script demonstrates how to use the organized models directory structure.
"""

import os
import sys
import re
from pathlib import Path
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers import AutoPipelineForText2Image
import torch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import model constants
from models import DIFFUSION_MODELS, MODEL_PATHS, DEFAULT_MODEL_CONFIGS, TEXT_GENERATION_MODELS

# Define the available diffusion models for image generation
AVAILABLE_DIFFUSION_MODELS = {
    "stable_diffusion_v1_5": "runwayml/stable-diffusion-v1-5",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "flux_klein_base_9b_fp8": "black-forest-labs/FLUX.2-klein-base-9b-fp8",
    "flux_dev": "black-forest-labs/FLUX.1-dev"
}

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

def generate_images(prompt, num_images=1, model_name="stable_diffusion_v1_5"):
    """Generate image(s) from a text prompt using the specified diffusion model.
    
    Args:
        prompt (str): Description of the image to generate
        num_images (int): Number of images to generate (default: 1)
        model_name (str): Name of the model to use (default: "stable_diffusion_v1_5")
        
    Returns:
        list: List of generated image file paths
    """
    print(f"Generating {num_images} image(s) for prompt: '{prompt}'")
    
    # Setup directories 
    setup_model_directories()
    
    # Get the actual model path from available models
    if model_name not in DIFFUSION_MODELS:
        raise ValueError(f"Unsupported diffusion model: {model_name}")
        
    model_path = DIFFUSION_MODELS[model_name]
    print(f"Using diffusion model: {model_name}")
    print(f"Model path: {model_path}")
    
    # Load the pipeline for image generation
    try:
        # Determine pipeline type based on model name
        if model_name == "sdxl":
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                safety_checker=None
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                safety_checker=None
            )
        
        # Set up device (use MPS for M5 Pro, CUDA if available, otherwise CPU)
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        pipe = pipe.to(device)
        
        # Generate the image(s)
        images = pipe(prompt, num_images=num_images)
        generated_images = []
        
        # Save each image
        filename_base = generate_filename_from_prompt(prompt)
        for i, image in enumerate(images.images):
            output_path = f"outputs/{filename_base}_{model_name}_benchmark.png"
            image.save(output_path)
            generated_images.append(output_path)
            print(f"Image {i+1} will be saved to: {output_path}")
            
    except Exception as e:
        print(f"Error during image generation: {e}")
        raise
        
    return generated_images

def benchmark_models(prompt, num_images=1):
    """Generate images using all available models for benchmarking.
    
    Args:
        prompt (str): Description of the image to generate
        num_images (int): Number of images to generate with each model (default: 1)
        
    Returns:
        dict: Dictionary mapping model names to lists of generated image file paths
    """
    results = {}
    print(f"Starting benchmark for prompt: '{prompt}'")
    
    # Iterate through all available diffusion models
    for model_name in AVAILABLE_DIFFUSION_MODELS.keys():
        try:
            print(f"\n--- Generating with {model_name} ---")
            files = generate_images(prompt, num_images, model_name)
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
        if "flux" in model_name.lower():
            # Use AutoPipelineForText2Image for Flux models
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                variant="fp16"
            )
        else:
            # Use StableDiffusionPipeline for other models
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16
            )
            
        # Set up device (use MPS for M5 Pro, CUDA if available, otherwise CPU)
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        pipe = pipe.to(device)
        
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        print("Falling back to default pipeline...")
        # Fallback to basic StableDiffusionPipeline
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
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
    """Main function to demonstrate the script."""
    print("=== Enhanced Image Generation Script ===")
    setup_model_directories()
    
    # Example usage with your requested prompt
    prompt = "Goku playing volleyball with Rod Stewart on a sunny beach, vibrant colors, dynamic action, cinematic lighting"
    
    try:
        # Generate images using all models for benchmarking
        print("\n--- Benchmarking All Models ---")
        benchmark_results = benchmark_models(prompt, num_images=1)
        
        # Show results summary
        print("\n=== Benchmark Results Summary ===")
        for model_name, files in benchmark_results.items():
            count = len(files)
            print(f"{model_name}: {count} image(s) generated")
            
        print("\nBenchmark completed successfully!")
        
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()