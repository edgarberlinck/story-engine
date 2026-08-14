#!/usr/bin/env python3
"""
Integration test for style conflict detection with scene generation.
"""

import sys
sys.path.insert(0, '/Users/edgarberlinck/code/story-engine')

from core.style_conflict import detect_character_style, find_style_conflicts


def test_integration():
    print("=== Integration Test: Style Conflict Detection ===")
    
    # Simulate characters with different styles
    characters = [
        {
            "name": "RealHero",
            "prompt": "ultra realistic man, athletic build, short brown hair, highly detailed skin texture, natural facial features, realistic proportions"
        },
        {
            "name": "AnimeHero", 
            "prompt": "manga style woman, slender build, long black hair, black and white manga illustration, detailed ink lines"
        }
    ]
    
    print("\nTest 1: Detect styles from character prompts")
    for char in characters:
        style_id = detect_character_style(char["prompt"])
        print(f"  {char['name']}: detected style_id={style_id}")
    
    print("\nTest 2: Find conflicts between characters")
    conflicts = find_style_conflicts(characters)
    print(f"  Found {len(conflicts)} conflict(s)")
    
    for conflict in conflicts:
        msg = conflict.message()
        print(f"  Conflict: {msg}")
    
    if len(conflicts) == 1 and conflicts[0].family_a == "realistic" and conflicts[0].family_b == "anime_manga":
        print("\n✓ Integration test PASSED")
        return True
    else:
        print("\n❌ Integration test FAILED")
        return False


if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
