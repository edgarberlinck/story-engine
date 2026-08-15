"""
Tests for token budget management and CLIP limit compliance.
"""

import unittest
from utils.token_budget import (
    TokenBudgetManager,
    count_tokens,
    strip_generation_instructions,
    build_token_aware_scene_prompt
)


class TestTokenBudget(unittest.TestCase):
    
    def test_count_tokens_basic(self):
        """Test basic token counting."""
        text = "Hello world"
        count = count_tokens(text)
        self.assertGreater(count, 0)
        self.assertLess(count, 20)
    
    def test_strip_generation_instructions_pose(self):
        """Test stripping pose instructions."""
        text = "standing upright, entire body visible, feet touching the ground"
        cleaned = strip_generation_instructions(text)
        
        self.assertNotIn("standing", cleaned.lower())
        self.assertNotIn("entire body", cleaned.lower())
        self.assertNotIn("feet", cleaned.lower())
    
    def test_strip_generation_instructions_framing(self):
        """Test stripping framing instructions."""
        text = "full length wide shot, camera far from subject, not a portrait"
        cleaned = strip_generation_instructions(text)
        
        self.assertNotIn("wide shot", cleaned.lower())
        self.assertNotIn("camera far", cleaned.lower())
        self.assertNotIn("portrait", cleaned.lower())
    
    def test_strip_generation_instructions_quality(self):
        """Test stripping quality modifiers."""
        text = "high detail, detailed face, good lighting, photorealistic"
        cleaned = strip_generation_instructions(text)
        
        self.assertNotIn("high detail", cleaned.lower())
        self.assertNotIn("detailed face", cleaned.lower())
    
    def test_strip_preserves_appearance(self):
        """Test that appearance attributes are preserved."""
        text = "muscular man, short bald black hair, dark skin tone, brown eyes"
        cleaned = strip_generation_instructions(text)
        
        self.assertIn("muscular", cleaned.lower())
        self.assertIn("bald", cleaned.lower())
        self.assertIn("dark skin", cleaned.lower())
    
    def test_token_budget_limits(self):
        """Test token budget enforcement."""
        manager = TokenBudgetManager(max_tokens=77)
        
        base_prompt = "Nikita is playing guitar, Roger is playing drums on stage"
        characters = [
            {"name": "Nikita", "prompt": "muscular man, short black hair, dark skin"},
            {"name": "Roger", "prompt": "tall woman, long blonde hair, fair skin"}
        ]
        
        result_prompt, stats = manager.build_scene_prompt(base_prompt, characters)
        
        # Check tokens are within budget
        self.assertLessEqual(stats['total_tokens_estimated'], 77 * 1.1)  # Allow 10% buffer
        
    def test_token_budget_priority(self):
        """Test that high priority items are preserved."""
        manager = TokenBudgetManager(max_tokens=30)  # Very strict
        
        base_prompt = "A beautiful forest scene with trees and sunlight"
        characters = [
            {"name": "Hero1", "prompt": "tall man with blue eyes, black hair"},
            {"name": "Hero2", "prompt": "short woman with brown eyes, red hair"},
            {"name": "Hero3", "prompt": "average person with gray hair"}
        ]
        
        result_prompt, stats = manager.build_scene_prompt(base_prompt, characters)
        
        # Base prompt should be included (highest priority)
        self.assertIn("forest", result_prompt.lower())
        # Should drop some characters due to budget
        self.assertGreater(stats['items_dropped'], 0)
    
    def test_build_token_aware_prompt(self):
        """Test token-aware prompt building."""
        base_prompt = "Characters on stage performing"
        characters = [
            {"name": "Nikita", "prompt": "muscular male, age 30-40, short bald black hair, dark skin"},
            {"name": "Roger", "prompt": "athletic male, age 25-35, brown hair, light skin"}
        ]
        
        prompt, stats = build_token_aware_scene_prompt(base_prompt, characters)
        
        self.assertIn("Nikita", prompt)
        self.assertIn("Roger", prompt)
        self.assertIn("stage", prompt.lower())
        
    def test_character_description_extraction(self):
        """Test character description extraction with attributes."""
        manager = TokenBudgetManager()
        
        char_prompt = "ultra realistic man, age 30-40, muscular build, short bald black hair"
        attributes = {
            "age": "Adult",
            "body_type": "Muscular", 
            "hair_color": "Black",
            "hair_length": "Bald",
            "skin_tone": "Dark"
        }
        
        desc, priority = manager.prioritize_character_description(char_prompt, attributes)
        
        # Should prefer attributes over prompt parsing
        self.assertIn("muscular", desc.lower())
        self.assertEqual(priority, 1)


if __name__ == "__main__":
    unittest.main()
