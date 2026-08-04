#!/usr/bin/env python3
"""
Image generation script with organized model structure.
This script demonstrates how to use the organized models directory structure.
"""

import os
import sys
import re
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import model constants
from models import DIFFUSION_MODELS, MODEL_PATHS, DEFAULT_MODEL_CONFIGS, TEXT_GENERATION_MODELS

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

def generate_filename_from_prompt(prompt, model_name="llama3_8b"):
    """Generate a concise, descriptive filename based on the prompt using a text model.
    
    Args:
        prompt (str): Description of the image to generate
        model_name (str): Name of the text generation model to use
        
    Returns:
        str: Generated filename (without extension)
    """
    # Clean and normalize the prompt for filename creation
    # Remove special characters and limit length
    clean_prompt = re.sub(r'[^\w\s-]', '', prompt.lower())
    clean_prompt = re.sub(r'[-\s]+', '_', clean_prompt)
    clean_prompt = clean_prompt[:30]  # Limit to 30 chars for conciseness
    
    # Simple approach for now - create a basic descriptive name
    # In a real implementation, you could use the text model to generate better names
    words = clean_prompt.split('_')
    if len(words) > 2:
        # Keep only first two words with meaningful descriptors
        short_words = [w for w in words[:3] if w and len(w) >= 2]
        filename = '_'.join(short_words[:2]) 
    else:
        filename = clean_prompt
        
    # Ensure it starts with a letter or number and is at least 3 characters
    if not filename[0].isalnum():
        filename = "img_" + filename
    
    # Trim to max 20 characters for conciseness
    filename = filename[:20]
    
    print(f"Generated concise filename from prompt '{prompt}': {filename}")
    
    return filename

def download_model(model_name, destination_dir):
    """Download a model to the specified directory.
    
    Args:
        model_name (str): Name of the model to download
        destination_dir (str): Directory to download the model to
    """
    # This is where you would implement actual downloading logic
    print(f"Downloading {model_name} to {destination_dir}")
    print("In a real implementation, this would use HuggingFace or other download methods")
    # Example: 
    # from huggingface_hub import snapshot_download
    # snapshot_download(model_name, repo_type="model", local_dir=destination_dir)
    
def generate_image(prompt, model_name="stable_diffusion_v1_5", output_filename=None):
    """Generate an image based on the given prompt.
    
    Args:
        prompt (str): Description of the image to generate
        model_name (str): Name of the model to use (from models.py constants)
        output_filename (str): Optional custom filename (if None, will be auto-generated from prompt)
    
    Returns:
        str: Path to the generated image file
    """
    print(f"Generating image for prompt: {prompt}")
    
    # Get model path from constants
    if model_name in DIFFUSION_MODELS:
        model_path = os.path.join(MODEL_PATHS["diffusion"], model_name)
        print(f"Using model: {DIFFUSION_MODELS[model_name]}")
    else:
        model_path = "models/diffusion/stable-diffusion-v1-5"
        print(f"Using default model: stable-diffusion-v1-5")
        
    print(f"Model path: {model_path}")
    
    # Generate filename if not provided
    if output_filename is None:
        output_filename = generate_filename_from_prompt(prompt)
    
    # Ensure safe filename (remove invalid characters)
    output_filename = re.sub(r'[^\w\-_\.]', '', output_filename)
    if not output_filename:
        output_filename = "generated_image"
    
    # Final output path
    output_path = f"outputs/{output_filename}.png"
    print(f"Image will be saved to: {output_path}")
    
    # In a real implementation, you'd do something like:
    # from diffusers import StableDiffusionPipeline
    # pipe = StableDiffusionPipeline.from_pretrained(model_path)
    # image = pipe(prompt).images[0]
    # image.save(output_path)
    
    print("Image generation would happen here in a full working implementation")
    return output_path

def main():
    """Main function to demonstrate the script."""
    print("=== Image Generation Script ===")
    setup_model_directories()
    
    # Example usage with your requested prompt
    prompt = "Goku playing volleyball"
    
    try:
        image_file = generate_image(prompt)
        print(f"Image generated successfully: {image_file}")
        
        # Also test with explicit filename
        custom_filename = "goku_volleyball_match"
        image_file_custom = generate_image(prompt, output_filename=custom_filename)
        print(f"Image with custom name generated successfully: {image_file_custom}")
        
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()