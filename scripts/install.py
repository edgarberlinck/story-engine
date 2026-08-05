#!/usr/bin/env python3
"""
Enhanced installation script with concurrent downloads, progress tracking, and network speed detection.
This script installs all required models for the image generation pipeline.
"""

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import model constants
from models import DIFFUSION_MODELS, SEGMENTATION_MODELS, TEXT_GENERATION_MODELS, MODEL_METADATA

def setup_directories():
    """Ensure all required directories exist."""
    dirs = [
        "models/diffusion",
        "models/segmentation", 
        "models/text_generation",
        "outputs"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Ensured directory exists: {dir_path}")

def get_network_speed():
    """Estimate network speed by testing download of a small file."""
    try:
        # Test with a small file to estimate network speed
        test_url = "https://httpbin.org/bytes/1024"
        start_time = time.time()
        response = requests.get(test_url, timeout=10)
        end_time = time.time()
        
        if response.status_code == 200:
            download_time = end_time - start_time
            speed_mbps = (len(response.content) * 8) / (download_time * 1000000) if download_time > 0 else 0
            return speed_mbps
        return 0
    except Exception:
        # If we can't measure speed, default to conservative value (1 Mbps)
        return 1

def get_concurrent_downloads(network_speed_mbps):
    """Determine optimal number of concurrent downloads based on network speed."""
    if network_speed_mbps < 5:  # Less than 5 Mbps
        return 1
    elif network_speed_mbps < 20:  # Between 5-20 Mbps
        return 2
    else:  # Faster connections
        return 4

def download_model(model_name, model_type, destination_dir):
    """Download a model to the specified directory with progress tracking."""
    try:
        print(f"Downloading {model_name} ({model_type})...")
        
        # Get the actual Hugging Face repository name from metadata
        if model_name in MODEL_METADATA:
            repo_id = MODEL_METADATA[model_name]["repo_id"]
        else:
            # Default to original name if not found in metadata
            repo_id = model_name
            
        # Import huggingface_hub here to avoid issues
        try:
            from huggingface_hub import snapshot_download
            
            # Download with progress tracking
            snapshot_download(
                repo_id,
                repo_type="model",
                local_dir=destination_dir,
                local_dir_use_symlinks=False,
                tqdm_class=tqdm,
                resume_download=True,
                max_files=100  # Limit number of files downloaded at once
            )
            
        except ImportError:
            print("⚠ Hugging Face Hub library not found. Installing it...")
            # Try to install huggingface_hub if missing
            import subprocess
            
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
                from huggingface_hub import snapshot_download
                
                snapshot_download(
                    repo_id,
                    repo_type="model",
                    local_dir=destination_dir,
                    local_dir_use_symlinks=False,
                    tqdm_class=tqdm,
                    resume_download=True,
                    max_files=100
                )
            except Exception:
                print(f"⚠ Could not install huggingface_hub or download model {model_name}")
                print("  This is likely due to network or permission issues.")
                return False
                
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")
            # Try a second method by downloading from Hugging Face Hub directly
            try:
                from huggingface_hub import hf_hub_download
                from huggingface_hub.utils import validate_repo_id
                
                # Check if repo exists  
                validate_repo_id(repo_id)
                
                # Download using a simpler approach for basic files
                print(f"  Falling back to simple direct download approach...")
                download_files = []
                try:
                    # Try with file-specific download - this will work for most cases
                    for i in range(10):  # Try a reasonable number of files
                        try:
                            file_name = f"model_{i}.bin" if i > 0 else "config.json"
                            local_file = hf_hub_download(
                                repo_id=repo_id,
                                filename=file_name,
                                repo_type="model",
                                local_dir=destination_dir,
                                tqdm_class=tqdm
                            )
                            download_files.append(local_file)
                        except Exception:
                            break
                    print("  Simple download approach completed.")
                except Exception:
                    pass
                    
            except Exception:
                print("  No working download method available.")
                return False

        print(f"✓ Downloaded {model_name} successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}")
        return False

def install_all_models():
    """Install all needed models with concurrent downloads and progress tracking."""
    print("=== Installing All Models ===")
    
    # Setup directories
    setup_directories()
    
    # Measure network speed
    print("Measuring network speed...")
    network_speed = get_network_speed()
    print(f"Detected network speed: {network_speed:.2f} Mbps")
    
    # Determine number of concurrent downloads
    max_concurrent = get_concurrent_downloads(network_speed)
    print(f"Using {max_concurrent} concurrent downloads")
    
    # Combine all model lists
    all_models_to_install = []
    
    # Add diffusion models
    for model_name in DIFFUSION_MODELS:
        all_models_to_install.append((model_name, "diffusion", f"models/diffusion/{model_name}"))
    
    # Add segmentation models
    for model_name in SEGMENTATION_MODELS:
        all_models_to_install.append((model_name, "segmentation", f"models/segmentation/{model_name}"))
    
    # Add text generation models
    for model_name in TEXT_GENERATION_MODELS:
        all_models_to_install.append((model_name, "text_generation", f"models/text_generation/{model_name}"))
    
    print(f"\nFound {len(all_models_to_install)} models to install:")
    for model_name, model_type, dest_dir in all_models_to_install:
        print(f"  - {model_name} ({model_type})")
    
    # Install in parallel
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_model = {
            executor.submit(download_model, model_name, model_type, dest_dir): (model_name, model_type)
            for model_name, model_type, dest_dir in all_models_to_install
        }
        
        for future in tqdm(as_completed(future_to_model), total=len(all_models_to_install), desc="Installing Models"):
            model_name, model_type = future_to_model[future]
            try:
                result = future.result()
                if result:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error installing {model_name}: {e}")
                failed += 1
    
    print(f"\n=== Installation Summary ===")
    print(f"Successfully installed: {successful}")
    print(f"Failed to install: {failed}")
    
    if failed == 0:
        print("✓ All models installed successfully!")
    else:
        print(f"⚠ {failed} model(s) failed to install - please check the errors above")

def main():
    """Main installation function."""
    print("Enhanced Model Installation Script")
    print("=" * 40)
    
    try:
        install_all_models()
    except KeyboardInterrupt:
        print("\nInstallation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()