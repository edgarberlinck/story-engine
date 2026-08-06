#!/usr/bin/env python3
"""
Simple unit tests that focus only on core functionality without heavy processing.
"""

import unittest
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from generators.image_generator import (
    setup_model_directories,
    generate_filename_from_prompt
)
from models import DIFFUSION_MODELS, MODEL_PATHS


class TestImageGenerator(unittest.TestCase):
    
    def test_setup_model_directories(self):
        """Test that model directories are properly set up."""
        # This just tests the function can be called 
        try:
            setup_model_directories()
            self.assertTrue(True)  # If we reach here, no exception was raised
        except Exception as e:
            self.fail(f"setup_model_directories should not raise an exception: {e}")
    
    def test_generate_filename_from_prompt(self):
        """Test that filenames are generated correctly from prompts."""
        # Test basic prompt
        filename = generate_filename_from_prompt("Goku playing volleyball")
        self.assertIsInstance(filename, str)
        self.assertGreater(len(filename), 0)
        self.assertTrue(filename.isalnum() or '_' in filename or '-' in filename)
        
        # Test with special characters
        filename = generate_filename_from_prompt("Hello, World! This is a test.")
        self.assertIsInstance(filename, str)
        self.assertGreater(len(filename), 0)
        
        # Test empty prompt
        filename = generate_filename_from_prompt("")
        self.assertIsInstance(filename, str)
        
    def test_model_constants_exist(self):
        """Test that model constants are properly defined."""
        self.assertIsNotNone(DIFFUSION_MODELS)
        self.assertIsNotNone(MODEL_PATHS)
        
        # Check that required models exist (only SDXL and FLUX Dev now)
        self.assertIn("sdxl", DIFFUSION_MODELS)
        self.assertIn("flux_dev", DIFFUSION_MODELS)
        
        # Check that model paths are defined
        self.assertIn("diffusion", MODEL_PATHS)
        self.assertIn("segmentation", MODEL_PATHS)
        self.assertIn("text_generation", MODEL_PATHS)
        
    def test_filename_generation_edge_cases(self):
        """Test edge cases in filename generation."""
        # Test very long prompt
        long_prompt = "A" * 100 + " B" * 100
        filename = generate_filename_from_prompt(long_prompt)
        self.assertIsInstance(filename, str)
        self.assertLessEqual(len(filename), 20)  # Should be truncated
        
        # Test prompt with only special characters  
        filename = generate_filename_from_prompt("!@#$%^&*()")
        self.assertIsInstance(filename, str)
        # Note: We allow empty results but ensure it's a string

    def test_output_directory_exists(self):
        """Test that outputs directory is created."""
        setup_model_directories()
        self.assertTrue(os.path.exists("outputs"))


if __name__ == '__main__':
    unittest.main()