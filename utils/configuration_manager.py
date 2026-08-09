#!/usr/bin/env python3
"""
Configuration management utilities for story-engine project.
This module provides functions for managing model directories and configurations.
"""

import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
os.sys.path.insert(0, str(project_root))

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