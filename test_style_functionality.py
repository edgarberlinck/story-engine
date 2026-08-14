#!/usr/bin/env python3
"""
Direct test of the character attributes system to verify the new STYLE functionality.
"""

import sys
import os

# Add the project root to Python path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.character_attributes import (
    CHARACTER_TYPES, 
    CHARACTER_STYLES,
    build_character_prompt
)

def test_style_attributes():
    print("Testing Style attribute functionality:")
    
    # Test 1: Check that we have the right structure
    print(f"Total character types: {len(CHARACTER_TYPES)}")
    print(f"Total styles: {len(CHARACTER_STYLES)}")
    
    # Test 2: Verify all character types include STYLE
    for char_type, attrs in CHARACTER_TYPES.items():
        style_found = False
        for category in attrs["categories"]:
            if category["name"] == "Style":
                style_found = True
                break
        if not style_found:
            print(f"ERROR: Character type '{char_type}' missing 'Style' category")
            return False
        print(f"✓ Character type '{char_type}' has Style category")
    
    # Test 3: Check a few styles exist with proper modifiers
    test_styles = ['Ultra Realistic', 'Anime', 'Fantasy Art']
    for style in test_styles:
        if style.lower().replace(" ", "_") in CHARACTER_STYLES:
            print(f"✓ Style '{style}' exists with modifiers")
        else:
            print(f"ERROR: Style '{style}' not found (tried {style.lower().replace(' ', '_')})")
            return False
    
    # Test 4: Validate prompt building for different scenarios
    print("\nTesting prompt building...")
    
    try:
        # Basic character with style
        prompt = build_character_prompt(
            char_type="man",
            style_id="anime",
            attributes={
                "gender": "male",
                "age_range": "20-30",
                "body_type": "athletic",
                "hair_type": "curly",
                "hair_color": "brown",
                "hair_length": "long",
                "skin_tone": "medium",
                "eye_color": "green",
                "clothing": "casual",
                "mood": "confident",
                "style": "anime"
            }
        )
        print("✓ Basic prompt build successful")
        print(f"Result: {prompt[:100]}...")
        
        # Test with different style
        prompt2 = build_character_prompt(
            char_type="woman", 
            style_id="photorealistic",
            attributes={
                "gender": "female",
                "age_range": "30-40",
                "body_type": "slender",
                "hair_type": "straight",
                "hair_color": "black",
                "hair_length": "medium",
                "skin_tone": "light",
                "eye_color": "brown",
                "clothing": "formal",
                "mood": "serious",
                "style": "photorealistic"
            }
        )
        print("✓ Photorealistic prompt build successful")
        print(f"Result: {prompt2[:100]}...")
        
    except Exception as e:
        print(f"ERROR in prompt building: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Verify styles are working correctly 
    anime_style = CHARACTER_STYLES["anime"]
    if anime_style and "detailed anime style" in anime_style["modifiers"]:
        print(f"✓ Found style 'anime' with modifiers: {anime_style['modifiers'][:50]}...")
    else:
        print("ERROR: Could not find style 'anime'")
        return False
        
    print("\nAll tests passed! ✅")
    return True

if __name__ == "__main__":
    test_style_attributes()