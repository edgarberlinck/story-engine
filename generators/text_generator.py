#!/usr/bin/env python3
"""
Text generation utilities for story-engine project.
This module provides functions for generating and manipulating text prompts.
"""

import os
import sys
import re
from pathlib import Path

# Avoid HuggingFace tokenizers spawning fork-based parallelism (leaks
# semaphores on macOS and triggers resource_tracker warnings at shutdown).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import model constants
from models import TEXT_GENERATION_MODELS, get_model_config
from utils.model_metrics import ModelMetrics

def resolve_model_path(model_type, model_name, hub_id):
    """Resolve a model to its locally installed path (from `make install`).

    Args:
        model_type (str): Key in MODEL_PATHS ('diffusion', 'text_generation', ...)
        model_name (str): Local directory name of the model
        hub_id (str): Hugging Face hub repo id used as a fallback

    Returns:
        str: Local path if the model was installed, otherwise the hub id.
    """
    from models import MODEL_PATHS
    
    local_path = os.path.join(str(project_root), MODEL_PATHS[model_type], model_name)
    if os.path.isdir(local_path) and os.listdir(local_path):
        return local_path
    print(f"Warning: {model_name} not found locally at {local_path}. "
          f"Run 'make install' first. Falling back to hub download: {hub_id}")
    return hub_id

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
    generator = None
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

        if enriched:
            print(f"LLM-enriched prompt: {enriched}")
            return enriched
    except Exception as e:
        print(f"Prompt enrichment failed ({e}); using raw description.")
    finally:
        # Free memory (and multiprocessing resources) before diffusion
        # models load.
        from generators.image_generator import cleanup_pipeline
        cleanup_pipeline(generator)
        generator = None

    return description