#!/usr/bin/env python3
"""
Integration test for Phase 2 architectural improvements.
"""

import sys
sys.path.insert(0, '/Users/edgarberlinck/code/story-engine')

from core.prompt_decomposer import (
    decompose_character_prompt,
    build_appearance_prompt,
    extract_appearance_from_stored_prompt,
    should_use_scene_style_override
)
from core.style_conflict import find_style_conflicts


def test_phase2_integration():
    print("=== Phase 2 Integration Test ===")
    
    # Simulate characters with conflicting styles
    characters = [
        {
            "name": "RealHero",
            "prompt": "ultra realistic man, athletic build, short brown hair, highly detailed skin texture, natural facial features, realistic proportions, photographic realism"
        },
        {
            "name": "AnimeHero", 
            "prompt": "manga style woman, slender build, long black hair, black and white manga illustration, detailed ink lines, expressive facial features"
        }
    ]
    
    print("\n1. Detecting style conflicts...")
    conflicts = find_style_conflicts(characters)
    print(f"   Found {len(conflicts)} conflict(s)")
    for conflict in conflicts:
        print(f"   - {conflict.message()}")
    
    print("\n2. Decomposing character prompts...")
    for char in characters:
        prefix, modifiers, appearance = decompose_character_prompt(char["prompt"])
        print(f"   {char['name']}:")
        print(f"     Style: {prefix}")
        print(f"     Appearance: {appearance[:80]}...")
    
    print("\n3. Building appearance-only prompts...")
    for char in characters:
        # Simulate attributes extraction
        if "RealHero" in char["name"]:
            attributes = {
                "age": "Adult",
                "body_type": "Athletic",
                "hair_color": "Brown",
                "hair_length": "Short"
            }
        else:
            attributes = {
                "age": "Young Adult", 
                "body_type": "Slim",
                "hair_color": "Black",
                "hair_length": "Long"
            }
        
        appearance_prompt = build_appearance_prompt("man" if "Hero" in char["name"] else "woman", attributes)
        print(f"   {char['name']} appearance: {appearance_prompt}")
    
    print("\n4. Testing style override decision...")
    should_override = should_use_scene_style_override(characters)
    print(f"   Should use appearance-only mode: {should_override}")
    
    if len(conflicts) == 1 and should_override:
        print("\n✓ Phase 2 Integration Test PASSED")
        return True
    else:
        print("\n❌ Phase 2 Integration Test FAILED")
        return False


if __name__ == "__main__":
    success = test_phase2_integration()
    sys.exit(0 if success else 1)
