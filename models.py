#!/usr/bin/env python3
"""
Model constants and configuration for the story-engine project.
This file standardizes model references for image generation and other components.

Note: This file primarily focuses on models used for image generation, but can be extended 
to include other types of models as the project expands.
"""

# Diffusion Models (Primary for Image Generation)
DIFFUSION_MODELS = {
    "stable_diffusion_v1_5": "runwayml/stable-diffusion-v1-5",
    "flux_klein_base_9b_fp8": "black-forest-labs/FLUX.2-klein-base-9b-fp8"
}

# Segmentation Models (Used for image analysis and processing)
SEGMENTATION_MODELS = {
    "detr_resnet_50_panoptic": "facebook/detr-resnet-50-panoptic"
}

# Model paths (relative to project root)
MODEL_PATHS = {
    "diffusion": "models/diffusion",
    "segmentation": "models/segmentation"
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
    }
}

# Model metadata
MODEL_METADATA = {
    "stable_diffusion_v1_5": {
        "name": "Stable Diffusion v1.5",
        "type": "diffusion",
        "size": "4GB",
        "description": "General purpose text-to-image generation model"
    },
    "flux_klein_base_9b_fp8": {
        "name": "FLUX.2 Klein Base 9B FP8", 
        "type": "diffusion",
        "size": "4GB",
        "description": "High quality text-to-image generation model"
    },
    "detr_resnet_50_panoptic": {
        "name": "DETR ResNet-50 Panoptic",
        "type": "segmentation", 
        "size": "1.2GB",
        "description": "Panoptic segmentation model for image analysis"
    }
}