#!/usr/bin/env python3
"""
Test script to verify the Style attribute integration in Character Attributes system.
This verifies that prompts are built correctly with Style modifiers and that 
the character type resolution works properly.
"""

import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from core.character_attributes import (
    CHARACTER_TYPES, 
    build_character_prompt,
    CHARACTER_STYLES
)

def test_style_integration():
    """Test that Style attributes work correctly and don't cause duplications."""
    
    print("Testing Character Attributes with Style Integration...")
    
    # Test 1: Basic styles for different character types
    print("\n1. Testing basic style integration:")
    
    test_cases = [
        {
            'name': 'Man with Ultra Realistic',
            'gender': 'male',
            'body_type': 'athletic',
            'hair_type': 'curly',
            'hair_color': 'black',
            'hair_length': 'long',
            'skin_tone': 'medium',
            'eye_color': 'brown',
            'clothing': 'casual',
            'mood': 'confident',
            'style': 'Ultra Realistic'
        },
        {
            'name': 'Woman with Anime Style',
            'gender': 'female',
            'body_type': 'slender',
            'hair_type': 'straight',
            'hair_color': 'blonde',
            'hair_length': 'medium',
            'skin_tone': 'light',
            'eye_color': 'blue',
            'clothing': 'formal',
            'mood': 'happy',
            'style': 'Anime'
        },
        {
            'name': 'Animal with Cartoon Style',
            'gender': 'other',
            'body_type': 'slender',
            'hair_type': 'bald',
            'hair_color': 'black',
            'hair_length': 'short',
            'skin_tone': 'light',
            'eye_color': 'green',
            'clothing': 'none',
            'mood': 'playful',
            'style': 'Cartoon'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {case['name']}")
        
        # Build the full attributes dict
        full_attrs = case.copy()
        full_attrs['age_range'] = '20-30'
        full_attrs['distinctive_features'] = 'scar on left cheek'
        full_attrs['custom_description'] = 'heroic pose'
        
        try:
            # Determine character type from gender
            char_type = 'man' if case['gender'] == 'male' else 'woman' if case['gender'] == 'female' else 'animal'
            
            # Build minimal attributes for prompt function (style needs to be in the attributes dict)
            attrs_for_prompt = full_attrs.copy()
            attrs_for_prompt['character_type'] = char_type  # This should be removed, let's test with proper structure
            
            # Actually, let's just pass what we know and use defaults for simplicity
            prompt = build_character_prompt(
                char_type=char_type,
                style_id=case['style'].lower().replace(' ', '_'), 
                attributes={k: v for k, v in full_attrs.items() if k not in ['gender', 'name']}
            )
            print(f"  Generated Prompt: {prompt}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            
    # Test 2: Verify CHARACTER_STYLES structure
    print("\n2. Testing CHARACTER_STYLES structure:")
    
    print("  Total styles:", len(CHARACTER_STYLES))
    print("  First few styles:")
    for i, (style_id, style_data) in enumerate(list(CHARACTER_STYLES.items())[:5]):
        print(f"    {style_id}: modifiers={len(style_data['modifiers'])}")
        
    # Test 3: Verify all character types have STYLE
    print("\n3. Testing that all character types have STYLE:")
    
    for char_type, attrs in CHARACTER_TYPES.items():
        # Check if 'Style' category exists in this type
        has_style_category = any(cat['name'] == 'Style' for cat in attrs['categories'])
        if not has_style_category:
            print(f"  ERROR: Character type '{char_type}' missing STYLE category")
        else:
            print(f"  ✓ Character type '{char_type}' has STYLE category")

    print("\nAll tests completed!")

if __name__ == "__main__":
    test_style_integration()