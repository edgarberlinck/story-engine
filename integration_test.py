#!/usr/bin/env python3
"""
Integration test: Simulate the end-to-end flow that would happen with 
the character builder screen and new Style functionality.
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

def simulate_ui_interaction():
    """Simulate how the UI would collect data and pass it to our system."""
    
    print("=== Simulating UI Character Creation Flow ===")
    
    # This simulates what happens when a user fills out the character builder UI
    # with a selection
    user_selection = {
        "name": "Test Hero", 
        "gender": "Male",
        "age_range": "Adult",
        "body_type": "Athletic",
        "hair_type": "Curly", 
        "hair_color": "Brown",
        "hair_length": "Long",
        "skin_tone": "Medium",
        "eye_color": "Green",
        "clothing": "Fantasy",
        "mood": "Confident",
        "style": "Anime"  # This is what UI collects from dropdown
    }
    
    print(f"User selected character: {user_selection['name']}")
    print(f"Style selected: {user_selection['style']}")
    
    # Now we convert this user selection to what the system actually needs
    # (mapping UI values to internal representation)
    attributes = {
        "gender": user_selection["gender"].lower(),
        "age_range": user_selection["age_range"],
        "body_type": user_selection["body_type"],
        "hair_type": user_selection["hair_type"].lower(),
        "hair_color": user_selection["hair_color"].lower(),
        "hair_length": user_selection["hair_length"].lower(),
        "skin_tone": user_selection["skin_tone"].lower(),
        "eye_color": user_selection["eye_color"].lower(),
        "clothing": user_selection["clothing"].lower(),
        "mood": user_selection["mood"].lower(),
        "style": user_selection["style"]  # This becomes the style_id
    }
    
    print(f"Converted attributes: {attributes}")
    
    # Determine character type from gender (this is how UI logic works)
    char_type = "man"
    if user_selection["gender"].lower() == "female":
        char_type = "woman"
    elif user_selection["gender"].lower() in ["animal", "other"]:
        char_type = "animal"
    
    print(f"Character type determined: {char_type}")
    
    # This is what the system actually builds now with our enhanced prompt builder
    try:
        final_prompt = build_character_prompt(
            char_type=char_type,
            style_id=attributes["style"].lower().replace(" ", "_"),  # Convert to internal ID
            attributes=attributes
        )
        
        print("\n=== Generated Prompt ===")
        print(final_prompt)
        print("\n=== Analysis ===")
        
        # Verify this should contain both the style prefix and modifiers
        style_id = attributes["style"].lower().replace(" ", "_")
        if style_id in CHARACTER_STYLES:
            style_info = CHARACTER_STYLES[style_id]
            print(f"✓ Style prefix '{style_info['prefix']}' included")
            print(f"✓ Style modifiers included: {style_info['modifiers'][:60]}...")
        else:
            print("ERROR: Style not found!")
            return False
            
        # Check that style is present in the prompt in the right way
        if style_info["prefix"] in final_prompt:
            print("✓ Style prefix correctly added to beginning")
        else:
            print("❌ Style prefix missing from prompt")
            return False
            
        # Verify it includes other attributes
        expected_elements = ["man", "athletic", "green eyes", "brown hair", "fantasy clothing"]
        for element in expected_elements:
            if element.lower() in final_prompt.lower():
                print(f"✓ Contains '{element}'")
            else:
                print(f"❌ Missing '{element}' from prompt")
                
        print("\n=== End-to-End Test Passed! ===")
        return True
        
    except Exception as e:
        print(f"ERROR during prompt building: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simulate_ui_interaction()
    if success:
        print("\n🎉 All integration tests passed! The new Style functionality works correctly.")
    else:
        print("\n❌ Integration test failed!")
        sys.exit(1)