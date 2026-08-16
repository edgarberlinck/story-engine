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
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0", 
    "flux_dev": "black-forest-labs/FLUX.1-dev",
    "flux_klein": "black-forest-labs/FLUX.2-klein-4B"
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

# Image-to-Video Models (for animating generated images)
# Policy: the benchmark suite decides which i2v model survives; the losing
# model is removed from this registry once the comparison is final.
IMAGE_TO_VIDEO_MODELS = {
    "wan22_i2v": "Wan-AI/Wan2.2-I2V-A14B",
    "hunyuan_video_i2v": "tencent/HunyuanVideo-I2V"
}

# Text-to-Speech Models (spoken dialogue for scenes)
# Voice cloning/design is a capability of the TTS models below (Qwen3-TTS
# CustomVoice == voice clone), so no separate voice registry is kept.
# source "huggingface" => downloadable by `make install`; source "cloud" =>
# API-driven model, identified by its provider model id (needs API key).
TEXT_TO_SPEECH_MODELS = {
    "qwen3_tts": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "qwen3_tts_base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}

# Lip Sync Models (animate the speaking character's mouth to match audio)
LIP_SYNC_MODELS = {
    "latentsync_1_6": "ByteDance/LatentSync-1.6",
}

# Music Generation Models (background music for scenes)
MUSIC_GENERATION_MODELS = {
    "musicgen_medium": "facebook/musicgen-medium",
}

# Model paths (relative to project root)
MODEL_PATHS = {
    "diffusion": "models/diffusion",
    "segmentation": "models/segmentation",
    "text_generation": "models/text_generation",
    "image_to_video": "models/image_to_video",
    "text_to_speech": "models/text_to_speech",
    "lip_sync": "models/lip_sync",
    "music_generation": "models/music_generation"
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
    },
    "image_to_video": {
        "dtype": "bfloat16",
        "device": "cpu"
    },
    "text_to_speech": {
        "dtype": "float16",
        "device": "cpu"
    },
    "lip_sync": {
        "dtype": "float16",
        "device": "cpu"
    },
    "music_generation": {
        "dtype": "float16",
        "device": "cpu"
    }
}

# Model metadata
MODEL_METADATA = {
    "sdxl": {
        "name": "Stable Diffusion XL",
        "type": "diffusion",
        "size": "5GB",
        "description": "Next generation text-to-image generation model with improved detail",
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0"
    },
    "flux_dev": {
        "name": "FLUX.1 Dev",
        "type": "diffusion",
        "size": "34GB",
        "description": "Latest text-to-image generation model (gated: requires accepting license on Hugging Face)",
        "repo_id": "black-forest-labs/FLUX.1-dev"
    },
    "flux_klein": {
        "name": "FLUX.2 Klein 4B",
        "type": "diffusion",
        "size": "4B params",
        "description": "FLUX.2 Klein 4B text-to-image model; supports image-reference conditioning via Flux2KleinKVPipeline",
        "repo_id": "black-forest-labs/FLUX.2-klein-4B"
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
    },
    "wan22_i2v": {
        "name": "Wan 2.2 I2V A14B",
        "type": "image_to_video",
        "size": "~60GB",
        "description": "Wan 2.2 image-to-video MoE model (A14B) for high-quality video generation from images",
        "repo_id": "Wan-AI/Wan2.2-I2V-A14B"
    },
    "hunyuan_video_i2v": {
        "name": "HunyuanVideo I2V",
        "type": "image_to_video",
        "size": "~40GB",
        "description": "Tencent HunyuanVideo image-to-video generation model",
        "repo_id": "tencent/HunyuanVideo-I2V"
    },
    "qwen3_tts": {
        "name": "Qwen3-TTS CustomVoice 1.7B",
        "type": "text_to_speech",
        "size": "~4.2GB",
        "description": "Qwen3-TTS 12Hz 1.7B CustomVoice: expressive TTS with voice cloning from reference audio (Apache-2.0)",
        "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "source": "huggingface"
    },
    "qwen3_tts_base": {
        "name": "Qwen3-TTS Base 0.6B",
        "type": "text_to_speech",
        "size": "~2.3GB",
        "description": "Qwen3-TTS 12Hz 0.6B Base: lightweight text-to-speech backbone (Apache-2.0)",
        "repo_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "source": "huggingface"
    },
    "latentsync_1_6": {
        "name": "LatentSync 1.6",
        "type": "lip_sync",
        "size": "~9GB weights / 18GB VRAM",
        "description": "ByteDance diffusion lip sync, 512x512, highest visual fidelity (needs strong GPU)",
        "repo_id": "ByteDance/LatentSync-1.6",
        "source": "huggingface"
    },
    "musicgen_medium": {
        "name": "MusicGen Medium",
        "type": "music_generation",
        "size": "~11.1GB",
        "description": "Meta MusicGen medium, controllable text-to-music (MIT)",
        "repo_id": "facebook/musicgen-medium",
        "source": "huggingface"
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