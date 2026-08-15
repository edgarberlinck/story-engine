#!/usr/bin/env python3
"""
Text generation utilities for story-engine project.
This module provides functions for generating and manipulating text prompts.
"""

import os
import sys
import re
from pathlib import Path
from typing import Optional

# Avoid HuggingFace tokenizers spawning fork-based parallelism (leaks
# semaphores on macOS and triggers resource_tracker warnings at shutdown).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import model constants
from models import TEXT_GENERATION_MODELS, get_model_config
from utils.model_metrics import ModelMetrics

# Imported at module scope (rather than lazily inside each function) so it is
# a stable, patchable module attribute for tests. `transformers` exposes
# `pipeline` via a lazy __getattr__, which makes `from transformers import
# pipeline` inside a function immune to `patch("transformers.pipeline")`;
# binding it here avoids that ambiguity.
try:
    from transformers import pipeline as hf_pipeline
    from transformers.utils import logging as hf_logging
    _HF_AVAILABLE = True
except Exception:  # pragma: no cover - transformers is a core dependency
    hf_pipeline = None
    hf_logging = None
    _HF_AVAILABLE = False

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


def generate_text_with_llm(
    prompt: str,
    model_name: str = "phi3_mini",
    max_new_tokens: int = 150,
    temperature: Optional[float] = None,
) -> Optional[str]:
    """Run a general-purpose completion against a local text-generation model.

    Unlike `generate_prompt_with_llm` (which is hardcoded to rewrite an image
    description into a single-line diffusion prompt), this returns the model's
    raw completion for an arbitrary instruction prompt. This is what the scene
    planner needs for structured planning (Stage A context resolution, Stage B
    decomposition, Stage C compression, Stage D strategy selection) where the
    caller expects JSON or specific structured output.

    Args:
        prompt: The full instruction prompt to send to the model (passed
            verbatim, NOT wrapped in the image-rewrite template).
        model_name: Key in TEXT_GENERATION_MODELS.
        max_new_tokens: Maximum tokens to generate.
        temperature: Optional sampling temperature (None = greedy).

    Returns:
        The model's raw completion string, or None on any failure.
    """
    generator = None
    try:
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

        generator.model.generation_config.max_new_tokens = max_new_tokens
        generator.model.generation_config.do_sample = temperature is not None
        generator.model.generation_config.temperature = temperature
        generator.model.generation_config.top_p = None
        generator.model.generation_config.top_k = None

        # Instruct models (e.g. Phi-3-mini-instruct) require their chat
        # template to interpret a bare prompt as an instruction; without it
        # they continue with plausible-but-irrelevant text instead of obeying
        # "return JSON only". Apply the template when one is available, and
        # only ask for the assistant turn so return_full_text=False yields
        # just the completion.
        tokenizer = getattr(generator, "tokenizer", None)
        send_prompt = prompt
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            try:
                messages = [{"role": "user", "content": prompt}]
                send_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                send_prompt = prompt

        # Pass generation knobs directly to the call: the pipeline applies its
        # own defaults (e.g. a max_new_tokens cap) that are NOT controlled by
        # mutating `model.generation_config`, so relying on that silently
        # truncates structured output. Explicit call args are authoritative.
        result = generator(
            send_prompt,
            return_full_text=False,
            max_new_tokens=max_new_tokens,
            do_sample=temperature is not None,
            temperature=temperature,
            top_p=None,
            top_k=None,
        )
        text = result[0]["generated_text"].strip()
        if text:
            return text
    except Exception as e:
        print(f"LLM completion failed ({e}); returning None.")
    finally:
        from generators.image_generator import cleanup_pipeline
        cleanup_pipeline(generator)
        generator = None

    return None