# Story Engine Project

## Overview
This project provides an organized system for managing machine learning models, primarily focused on image generation with diffusion models and segmentation models.

## Features
- Organized model directory structure (models/diffusion, models/segmentation)
- Standardized model constants and configurations
- Automated model installation script
- Image generation capabilities using Hugging Face models

## Prerequisites
Before running the project, you need to install and authenticate with the Hugging Face CLI:

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Authenticate with Hugging Face
hf auth login
```

## Getting Started

### 1. Install Dependencies
```bash
make install
```

### 2. Download Models (if not already done)
```bash
make models
```

### 3. Generate Images
```bash
make generate
```

## Project Structure

```
.
├── Makefile              # Automation commands
├── models.py             # Model constants and configurations
├── scripts/
│   ├── install.py        # Model installation script
│   └── image_generator.py # Image generation script
└── models/               # Model storage directories
    ├── diffusion/        # Diffusion models (text-to-image)
    └── segmentation/     # Segmentation models
```

## Model Constants

The `models.py` file defines the following model categories:

### Diffusion Models
- `stable_diffusion_v1_5`: runwayml/stable-diffusion-v1-5
- `flux_klein_base_9b_fp8`: black-forest-labs/FLUX.2-klein-base-9b-fp8

### Segmentation Models  
- `detr_resnet_50_panoptic`: facebook/detr-resnet-50-panoptic

## Automation Commands
The Makefile provides the following commands:

- `make install` - Install dependencies and download models
- `make models` - Download models only
- `make generate` - Generate an image with default prompt
- `make clean` - Clean model directories
- `make help` - Show all available commands

## Extensibility
The project is designed to be easily extensible for other types of ML models beyond image generation. The directory structure and constants format allow for adding new categories like:
- `models/text_generation`
- `models/audio_processing` 
- `models/translation`

## Troubleshooting
If you encounter authentication issues with Hugging Face models:
1. Run `hf auth login` to authenticate
2. Ensure your account has access to the required model repositories
3. Check your internet connection for large model downloads (4GB+ models may time out)
