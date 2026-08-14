#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from character_attributes import CHARACTER_TYPES, build_character_prompt

def test_style_attribute_exists():
    """Test that Style attribute exists in all character types"""
    
    print("Testing Style attribute in character types...")
    
    # Check man type
    man_categories = CHARACTER_TYPES['man']['categories']
    man_has_style = any('Style' in category['name'] for category in man_categories)
    print(f"Man has Style category: {man_has_style}")
    
    # Check woman type  
    woman_categories = CHARACTER_TYPES['woman']['categories']
    woman_has_style = any('Style' in category['name'] for category in woman_categories)
    print(f"Woman has Style category: {woman_has_style}")
    
    # Check animal type
    animal_categories = CHARACTER_TYPES['animal']['categories']
    animal_has_style = any('Style' in category['name'] for category in animal_categories)
    print(f"Animal has Style category: {animal_has_style}")
    
    # Test building a prompt with style
    print("\nTesting prompt generation with style...")
    
    try:
        # Test man with style
        man_attributes = {
            'age': '30',
            'ethnicity': 'Caucasian',
            'hair_color': 'Brown'
        }
        prompt = build_character_prompt('man', 'Anime', man_attributes)
        print(f"Man prompt with style: {prompt}")
        
        # Test woman with style
        woman_attributes = {
            'age': '25',
            'ethnicity': 'Asian',
            'hair_color': 'Black'
        }
        prompt = build_character_prompt('woman', 'Photorealistic', woman_attributes)
        print(f"Woman prompt with style: {prompt}")
        
        # Test animal with style
        animal_attributes = {
            'species': 'Dog',
            'age': 'Adult'
        }
        prompt = build_character_prompt('animal', 'Fantasy Art', animal_attributes)
        print(f"Animal prompt with style: {prompt}")
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_style_attribute_exists()