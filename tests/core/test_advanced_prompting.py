"""
Tests for Phase 3 advanced prompting features.
"""

import unittest
from core.advanced_prompting import (
    AdvancedPromptingEngine,
    CharacterStyleToken,
    StyleBlend,
    build_advanced_scene_prompt
)


class TestAdvancedPrompting(unittest.TestCase):
    
    def test_build_per_character_tokens(self):
        """Test building per-character style tokens."""
        characters = [
            {
                "name": "Hero1",
                "prompt": "ultra realistic man, athletic build"
            },
            {
                "name": "Hero2",
                "prompt": "anime style woman, slender build"
            }
        ]
        
        tokens = AdvancedPromptingEngine.build_per_character_tokens(characters)
        
        self.assertEqual(len(tokens), 2)
        self.assertEqual(tokens[0].name, "Hero1")
        self.assertEqual(tokens[1].name, "Hero2")
        self.assertIsNotNone(tokens[0].style_id)
        self.assertIsNotNone(tokens[1].style_id)
    
    def test_build_negative_prompt_realistic(self):
        """Test negative prompt building for realistic style."""
        negative = AdvancedPromptingEngine._build_negative_prompt(
            "ultra_realistic", "athletic build"
        )
        
        self.assertIn("anime", negative.lower())
        self.assertIn("cartoon", negative.lower())
    
    def test_build_negative_prompt_anime(self):
        """Test negative prompt building for anime style."""
        negative = AdvancedPromptingEngine._build_negative_prompt(
            "anime", "slender build"
        )
        
        self.assertIn("photorealistic", negative.lower())
        self.assertIn("ultra realistic", negative.lower())
    
    def test_build_weighted_scene_prompt(self):
        """Test building weighted scene prompt."""
        tokens = [
            CharacterStyleToken(
                name="Hero1",
                style_id="ultra_realistic",
                appearance_prompt="athletic build, brown hair",
                negative_prompt="anime, cartoon",
                weight=1.0
            )
        ]
        
        positive, negative = AdvancedPromptingEngine.build_weighted_scene_prompt(
            "Two heroes in a forest",
            tokens
        )
        
        self.assertIn("Two heroes in a forest", positive)
        self.assertIn("Hero1", positive)
        self.assertIn("anime", negative.lower())
    
    def test_blend_styles(self):
        """Test style blending."""
        blend = AdvancedPromptingEngine.blend_styles(
            "ultra_realistic",
            "cinematic",
            ratio=0.3
        )
        
        self.assertEqual(blend.primary_style, "ultra_realistic")
        self.assertEqual(blend.secondary_style, "cinematic")
        self.assertEqual(blend.blend_ratio, 0.3)
    
    def test_build_blended_prompt(self):
        """Test building blended style prompt."""
        attributes = {
            "gender": "male",
            "age": "30",
            "body_type": "athletic"
        }
        
        blend = StyleBlend(
            primary_style="ultra_realistic",
            secondary_style="cinematic",
            blend_ratio=0.5
        )
        
        prompt = AdvancedPromptingEngine.build_blended_prompt(attributes, blend)
        
        self.assertIn("male", prompt.lower())
        self.assertIn("athletic", prompt.lower())
    
    def test_recommend_model_single_style(self):
        """Test model recommendation for single style."""
        model = AdvancedPromptingEngine.recommend_model(["ultra_realistic"])
        self.assertIn(model, ["flux_dev", "sdxl"])
    
    def test_recommend_model_mixed_styles(self):
        """Test model recommendation for mixed styles."""
        model = AdvancedPromptingEngine.recommend_model([
            "ultra_realistic",
            "anime"
        ])
        self.assertEqual(model, "flux_dev")
    
    def test_analyze_style_compatibility_single(self):
        """Test compatibility analysis for single style."""
        analysis = AdvancedPromptingEngine.analyze_style_compatibility(["ultra_realistic"])
        
        self.assertTrue(analysis["compatible"])
        self.assertEqual(analysis["score"], 1.0)
    
    def test_analyze_style_compatibility_conflicting(self):
        """Test compatibility analysis for conflicting styles."""
        analysis = AdvancedPromptingEngine.analyze_style_compatibility([
            "ultra_realistic",
            "anime"
        ])
        
        self.assertFalse(analysis["compatible"])
        self.assertLess(analysis["score"], 0.5)
    
    def test_analyze_style_compatibility_compatible(self):
        """Test compatibility analysis for compatible styles."""
        analysis = AdvancedPromptingEngine.analyze_style_compatibility([
            "ultra_realistic",
            "cinematic"
        ])
        
        # Same family should be compatible
        self.assertTrue(analysis["compatible"])
    
    def test_build_advanced_scene_prompt_basic(self):
        """Test advanced scene prompt building."""
        characters = [
            {"name": "Hero1", "prompt": "ultra realistic man"},
            {"name": "Hero2", "prompt": "anime style woman"}
        ]
        
        positive, negative = build_advanced_scene_prompt(
            "Two heroes talking",
            characters
        )
        
        self.assertIn("Two heroes talking", positive)
        self.assertIn("Hero1", positive)
        self.assertIn("Hero2", positive)


if __name__ == "__main__":
    unittest.main()
