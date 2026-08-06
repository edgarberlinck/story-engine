#!/usr/bin/env python3
"""
Model constants and configuration for the story-engine project.
This file standardizes model references for image generation and other components.

Note: This file primarily focuses on models used for image generation, but can be extended 
to include other types of models as the project expands.
"""

import torch

# Diffusion Models (Primary for Image Generation)
DIFFUSION_MODELS = {
    "stable_diffusion_v1_5": "runwayml/stable-diffusion-v1-5",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0", 
    "flux_klein": "black-forest-labs/FLUX.2-klein-4B",
    "flux_dev": "black-forest-labs/FLUX.1-dev"
}

# Segmentation Models (Used for image analysis and processing)
SEGMENTATION_MODELS = {
    "detr_resnet_50_panoptic": "facebook/detr-resnet-50-panoptic"
}

# Text Generation Models (for naming images based on prompts)
TEXT_GENERATION_MODELS = {
    "phi3_mini": "microsoft/Phi-3-mini-4k-instruct",
    "gemma_2b": "google/gemma-2b"
}

# Model paths (relative to project root)
MODEL_PATHS = {
    "diffusion": "models/diffusion",
    "segmentation": "models/segmentation",
    "text_generation": "models/text_generation"
}

# Default model configurations
DEFAULT_MODEL_CONFIGS = {
    "diffusion": {
        "dtype": "float16",
        "device": "cpu"
    },
    "segmentation": {
        "dtype": "float32", 
        "device": "cpu"
    },
    "text_generation": {
        "dtype": "float16",
        "device": "cpu"
    }
}

# Model metadata
MODEL_METADATA = {
    "stable_diffusion_v1_5": {
        "name": "Stable Diffusion v1.5",
        "type": "diffusion",
        "size": "4GB",
        "description": "General purpose text-to-image generation model",
        "repo_id": "runwayml/stable-diffusion-v1-5"
    },
    "sdxl": {
        "name": "Stable Diffusion XL",
        "type": "diffusion",
        "size": "5GB",
        "description": "Next generation text-to-image generation model with improved detail",
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0"
    },
    "flux_klein": {
        "name": "FLUX.2 Klein 4B",
        "type": "diffusion",
        "size": "13GB",
        "description": "High quality text-to-image generation model (diffusers format, open access)",
        "repo_id": "black-forest-labs/FLUX.2-klein-4B"
    },
    "flux_dev": {
        "name": "FLUX.1 Dev",
        "type": "diffusion",
        "size": "34GB",
        "description": "Latest text-to-image generation model (gated: requires accepting license on Hugging Face)",
        "repo_id": "black-forest-labs/FLUX.1-dev"
    },
    "detr_resnet_50_panoptic": {
        "name": "DETR ResNet-50 Panoptic",
        "type": "segmentation", 
        "size": "1.2GB",
        "description": "Panoptic segmentation model for image analysis",
        "repo_id": "facebook/detr-resnet-50-panoptic"
    },
    "phi3_mini": {
        "name": "Phi-3 Mini",
        "type": "text_generation",
        "size": "3GB",
        "description": "Compact language model for image naming",
        "repo_id": "microsoft/Phi-3-mini-4k-instruct"
    },
    "gemma_2b": {
        "name": "Gemma 2B",
        "type": "text_generation",
        "size": "2GB",
        "description": "Small efficient language model for image naming",
        "repo_id": "google/gemma-2b"
    }
}

def get_model_config(model_type):
    """
    Get the appropriate device and torch_dtype configuration for a given model type.
    
    Args:
        model_type (str): Type of model ('diffusion', 'segmentation', 'text_generation')
        
    Returns:
        tuple: (device, torch_dtype) - device string and torch dtype
    """
    # Resolve device (mps > cuda > cpu)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # Get the default config for this model type
    config = DEFAULT_MODEL_CONFIGS.get(model_type, {})
    dtype = config.get("dtype", "float32")
    
    # Map string dtype to actual torch dtype
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
        
    return device, torch_dtype