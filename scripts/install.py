#!/usr/bin/env python3
"""
Model installation script for the story-engine project.
This script downloads all required models to the organized model directories.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import model constants
from models import DIFFUSION_MODELS, SEGMENTATION_MODELS

def install_models():
    """Install all required models for the project."""
    print("=== Installing Models ===")
    print("This may take some time depending on your internet connection and the size of the models.")
    print("This will take a shit load of disk space and, not less importantly, time. Grab a coffee, or two, or three. This is going to take a while.")
    # Use model constants from models.py instead of hardcoded values
    models_to_install = {
        "diffusion": list(DIFFUSION_MODELS.keys()),
        "segmentation": list(SEGMENTATION_MODELS.keys())
    }
    
    try:
        # Import required libraries
        from huggingface_hub import snapshot_download
        import shutil
        
        for category, model_list in models_to_install.items():
            print(f"\nInstalling {category} models:")
            category_dir = f"models/{category}"
            os.makedirs(category_dir, exist_ok=True)
            
            for model_name in model_list:
                # Get the actual Hugging Face repo ID from constants
                if category == "diffusion" and model_name in DIFFUSION_MODELS:
                    repo_id = DIFFUSION_MODELS[model_name]
                elif category == "segmentation" and model_name in SEGMENTATION_MODELS:
                    repo_id = SEGMENTATION_MODELS[model_name]
                else:
                    repo_id = model_name  # Fallback to direct name
                    
                print(f"  - {model_name} ({repo_id})")
                try:
                    # Download the model
                    download_path = snapshot_download(
                        repo_id=repo_id,
                        revision="main",
                        local_dir=category_dir,
                        local_dir_use_symlinks=False
                    )
                    print(f"    ✓ Downloaded to: {download_path}")
                except Exception as e:
                    print(f"    ✗ Failed to download {model_name}: {e}")
        
        print("\n=== Installation Complete ===")
        print("Models have been installed in the 'models' directory.")
        print("You can now use the image generator script.")
        
    except ImportError:
        print("Error: huggingface_hub library not found.")
        print("Please install it with: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"Installation failed with error: {e}")
        sys.exit(1)

def verify_models():
    """Verify that required models are installed."""
    print("\n=== Verifying Models ===")
    
    required_dirs = ["models/diffusion", "models/segmentation"]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            print(f"{dir_path}: {len(files)} items found")
            for f in files[:3]:  # Show first 3 items
                print(f"  - {f}")
            if len(files) > 3:
                print(f"  ... and {len(files) - 3} more")
        else:
            print(f"{dir_path}: Directory does not exist")

if __name__ == "__main__":
    install_models()
    verify_models()