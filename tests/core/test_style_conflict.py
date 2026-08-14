"""
Tests for style conflict detection.
"""

import unittest
from core.style_conflict import (
    detect_character_style,
    find_style_conflicts,
    StyleConflict
)
from core.character_attributes import CHARACTER_STYLES


class TestStyleConflictDetection(unittest.TestCase):
    
    def test_detect_character_style_ultra_realistic(self):
        """Test detection of ultra realistic style from prompt."""
        prompt = "ultra realistic man, athletic build, short brown hair, highly detailed skin texture"
        style_id = detect_character_style(prompt)
        self.assertEqual(style_id, "ultra_realistic")
    
    def test_detect_character_style_manga(self):
        """Test detection of manga style from prompt."""
        prompt = "manga style woman, slender build, long black hair, black and white manga illustration"
        style_id = detect_character_style(prompt)
        self.assertEqual(style_id, "manga")
    
    def test_detect_character_style_prefix_priority(self):
        """Test that longer prefixes win over shorter ones."""
        # 'ultra realistic' should be detected, not just 'realistic'
        prompt = "ultra realistic man, athletic build"
        style_id = detect_character_style(prompt)
        self.assertEqual(style_id, "ultra_realistic")
    
    def test_detect_character_style_no_match(self):
        """Test detection with no style match."""
        prompt = "man with brown hair"
        style_id = detect_character_style(prompt)
        self.assertIsNone(style_id)
    
    def test_find_style_conflicts_empty_list(self):
        """No conflicts with empty list."""
        conflicts = find_style_conflicts([])
        self.assertEqual(len(conflicts), 0)
    
    def test_find_style_conflicts_single_character(self):
        """No conflicts with single character."""
        characters = [
            {"name": "Hero", "prompt": "ultra realistic man"}
        ]
        conflicts = find_style_conflicts(characters)
        self.assertEqual(len(conflicts), 0)
    
    def test_find_style_conflicts_same_family(self):
        """No conflicts when characters share same family."""
        characters = [
            {"name": "Hero1", "prompt": "ultra realistic man"},
            {"name": "Hero2", "prompt": "cinematic woman"}
        ]
        conflicts = find_style_conflicts(characters)
        self.assertEqual(len(conflicts), 0)
    
    def test_find_style_conflicts_realistic_anime(self):
        """Detect conflict between realistic and anime styles."""
        characters = [
            {"name": "RealHero", "prompt": "ultra realistic man"},
            {"name": "AnimeHero", "prompt": "anime style woman"}
        ]
        conflicts = find_style_conflicts(characters)
        self.assertEqual(len(conflicts), 1)
        conflict = conflicts[0]
        self.assertEqual(conflict.family_a, "realistic")
        self.assertEqual(conflict.family_b, "anime_manga")
        self.assertIn("RealHero", conflict.characters_a)
        self.assertIn("AnimeHero", conflict.characters_b)
    
    def test_find_style_conflicts_message(self):
        """Test conflict message generation."""
        conflict = StyleConflict(
            family_a="realistic",
            family_b="anime_manga",
            characters_a=["RealHero"],
            characters_b=["AnimeHero"]
        )
        msg = conflict.message()
        self.assertIn("RealHero", msg)
        self.assertIn("AnimeHero", msg)
        self.assertIn("realistic", msg)
        self.assertIn("anime manga", msg)
    
    def test_find_style_conflicts_bridge_families(self):
        """Test that bridge families don't trigger conflicts."""
        characters = [
            {"name": "RealHero", "prompt": "ultra realistic man"},
            {"name": "PainterlyHero", "prompt": "digital painting of a woman"}
        ]
        conflicts = find_style_conflicts(characters)
        # realistic and painterly are bridged, so no conflict
        self.assertEqual(len(conflicts), 0)
    
    def test_find_style_conflicts_ambiguous_family(self):
        """Test that ambiguous families are excluded."""
        characters = [
            {"name": "RealHero", "prompt": "ultra realistic man"},
            {"name": "AbstractHero", "prompt": "abstract man"}
        ]
        conflicts = find_style_conflicts(characters)
        # abstract is ambiguous, should be excluded
        self.assertEqual(len(conflicts), 0)
    
    def test_find_style_conflicts_multiple(self):
        """Test multiple conflicts with three characters."""
        characters = [
            {"name": "RealHero", "prompt": "ultra realistic man"},
            {"name": "AnimeHero", "prompt": "anime style woman"},
            {"name": "ComicHero", "prompt": "cartoon man"}
        ]
        conflicts = find_style_conflicts(characters)
        # Should have 2 conflicts: Real vs Anime, Real vs Comic
        self.assertEqual(len(conflicts), 2)


if __name__ == "__main__":
    unittest.main()
