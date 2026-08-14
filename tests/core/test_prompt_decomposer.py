"""
Tests for prompt decomposition and attribute-based generation.
"""

import unittest
from core.prompt_decomposer import (
    decompose_character_prompt,
    build_appearance_prompt,
    extract_appearance_from_stored_prompt,
    should_use_scene_style_override
)


class TestPromptDecomposition(unittest.TestCase):
    
    def test_decompose_realistic_prompt(self):
        """Test decomposition of realistic style prompt."""
        prompt = "ultra realistic man, athletic build, short brown hair, highly detailed skin texture, natural facial features"
        prefix, modifiers, appearance = decompose_character_prompt(prompt)
        
        self.assertEqual(prefix.lower(), "ultra realistic")
        self.assertIsNotNone(modifiers)
        self.assertIn("highly detailed", modifiers)
        self.assertIn("athletic build", appearance)
        self.assertIn("short brown hair", appearance)
    
    def test_decompose_manga_prompt(self):
        """Test decomposition of manga style prompt."""
        prompt = "manga style woman, slender build, long black hair, black and white manga illustration"
        prefix, modifiers, appearance = decompose_character_prompt(prompt)
        
        self.assertEqual(prefix.lower(), "manga style")
        self.assertIsNotNone(modifiers)
        self.assertIn("black and white", modifiers.lower())
        self.assertIn("slender build", appearance)
        self.assertIn("long black hair", appearance)
    
    def test_decompose_no_style(self):
        """Test decomposition when no style detected."""
        prompt = "man with brown hair and blue eyes"
        prefix, modifiers, appearance = decompose_character_prompt(prompt)
        
        self.assertIsNone(prefix)
        self.assertIsNone(modifiers)
        self.assertEqual(appearance, prompt)
    
    def test_build_appearance_prompt_man(self):
        """Test building appearance-only prompt for man."""
        attributes = {
            "age": "Adult",
            "ethnicity": "European",
            "skin_tone": "Fair",
            "body_type": "Athletic",
            "height": "Tall",
            "hair_color": "Brown",
            "hair_length": "Short",
            "eye_color": "Blue"
        }
        
        prompt = build_appearance_prompt("man", attributes)
        
        self.assertIn("adult", prompt.lower())
        self.assertIn("athletic body", prompt.lower())
        self.assertIn("brown hair", prompt.lower())
        self.assertIn("blue eyes", prompt.lower())
        # Should not contain style modifiers
        self.assertNotIn("ultra realistic", prompt.lower())
        self.assertNotIn("photographic realism", prompt.lower())
    
    def test_build_appearance_prompt_woman(self):
        """Test building appearance-only prompt for woman."""
        attributes = {
            "age": "Young Adult",
            "body_type": "Slim",
            "hair_color": "Black",
            "eye_color": "Green"
        }
        
        prompt = build_appearance_prompt("woman", attributes)
        
        self.assertIn("young adult", prompt.lower())
        self.assertIn("slim body", prompt.lower())
        self.assertIn("black hair", prompt.lower())
        self.assertIn("green eyes", prompt.lower())
    
    def test_extract_appearance_from_stored(self):
        """Test extracting appearance from stored character prompt."""
        prompt = "ultra realistic man, athletic build, short brown hair"
        appearance = extract_appearance_from_stored_prompt(prompt)
        
        self.assertIn("athletic build", appearance)
        self.assertIn("short brown hair", appearance)
        # Style should be removed
        self.assertNotIn("ultra realistic", appearance.lower() or "")
    
    def test_should_use_scene_style_override_conflicts(self):
        """Test detection of when to use style override."""
        from core.style_conflict import find_style_conflicts
        
        characters = [
            {"name": "RealHero", "prompt": "ultra realistic man"},
            {"name": "AnimeHero", "prompt": "anime style woman"}
        ]
        
        # Mock the actual conflict detection
        conflicts = find_style_conflicts(characters)
        self.assertEqual(len(conflicts), 1)
    
    def test_should_use_scene_style_override_no_conflict(self):
        """Test no override needed for compatible styles."""
        from core.style_conflict import find_style_conflicts
        
        characters = [
            {"name": "Hero1", "prompt": "ultra realistic man"},
            {"name": "Hero2", "prompt": "cinematic woman"}
        ]
        
        conflicts = find_style_conflicts(characters)
        self.assertEqual(len(conflicts), 0)


if __name__ == "__main__":
    unittest.main()
