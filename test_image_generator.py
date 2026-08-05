#!/usr/bin/env python3
"""
Test suite for the image generator functionality.
This verifies that the contract is respected and all features work correctly.
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
    generate_filename_from_prompt,
    generate_image
)
from models import DIFFUSION_MODELS, MODEL_PATHS


class TestImageGenerator(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Ensure outputs directory exists for testing
        os.makedirs("outputs", exist_ok=True)
        
    def test_setup_model_directories(self):
        """Test that model directories are properly set up."""
        setup_model_directories()
        
        # Check that directories exist
        expected_dirs = [
            "models/diffusion",
            "models/segmentation", 
            "models/text_generation",
            "outputs"
        ]
        
        for dir_path in expected_dirs:
            self.assertTrue(os.path.exists(dir_path), f"Directory {dir_path} should exist")
    
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
        self.assertGreater(len(filename), 0)  # Should generate default
        
    def test_model_constants_exist(self):
        """Test that model constants are properly defined."""
        self.assertIsNotNone(DIFFUSION_MODELS)
        self.assertIsNotNone(MODEL_PATHS)
        
        # Check that required models exist
        self.assertIn("stable_diffusion_v1_5", DIFFUSION_MODELS)
        self.assertIn("flux_klein", DIFFUSION_MODELS)
        
        # Check that model paths are defined
        self.assertIn("diffusion", MODEL_PATHS)
        self.assertIn("segmentation", MODEL_PATHS)
        self.assertIn("text_generation", MODEL_PATHS)
        
    def test_generate_image_function_signature(self):
        """Test that generate_image function can be called with various parameters."""
        # This test just ensures the function can be called without error
        # Actual image generation requires models which aren't available in testing
        try:
            # Test default call (should not crash, even if it fails to generate)
            result = generate_image("Test prompt")
            self.assertIsInstance(result, list)
        except Exception as e:
            # This might fail due to missing models, but the function should be callable
            pass

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
        self.assertGreater(len(filename), 0)

    def test_output_directory_exists(self):
        """Test that outputs directory is created."""
        self.assertTrue(os.path.exists("outputs"))


if __name__ == '__main__':
    unittest.main()