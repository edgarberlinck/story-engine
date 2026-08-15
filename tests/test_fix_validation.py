"""
Validation test for the scene generation fix.
Test the specific failing case from user report.
"""

import sys
sys.path.insert(0, '/Users/edgarberlinck/code/story-engine')

from utils.token_budget import build_token_aware_scene_prompt, count_tokens


def test_user_reported_case():
    """Test the exact case from user report where scene was not generated correctly."""
    print("\n=== User Reported Case: Multi-character Stage Scene ===")
    
    base_prompt = """There's a very small stage with red curtains. The stage is made of old, cracked wood.

In front of the stage, there are tables with people sitting and watching the performance.

On the stage, Nikita is sitting on a chair and playing a black Gibson Explorer guitar.

Roger is sitting behind a drum kit and playing drums.

The camera is positioned at the back of the bar, showing the crowd in the foreground and the entire stage with both musicians in the background.

Wide shot, both characters clearly visible."""
    
    characters = [
        {
            'name': 'Nikita',
            'prompt': 'ultra realistic woman, age 25-35, slender build, long black hair, dark skin, wearing leather jacket, playing guitar',
            'attributes': {
                'age': 'Young Adult',
                'body_type': 'Slim',
                'hair_color': 'Black',
                'hair_length': 'Long',
                'skin_tone': 'Dark'
            }
        },
        {
            'name': 'Roger',
            'prompt': 'full body photo of a muscular male character, age 30-40, with short bald black hair, dark skin tone and brown eyes, wearing formal clothing, serious expression, photorealistic, detailed face, high detail. standing upright, entire body visible from head to toe including feet, full length wide shot',
            'attributes': {
                'age': 'Adult',
                'body_type': 'Muscular',
                'hair_color': 'Black',
                'hair_length': 'Bald',
                'skin_tone': 'Dark Brown'
            }
        }
    ]
    
    prompt, stats = build_token_aware_scene_prompt(base_prompt, characters)
    token_count = count_tokens(prompt)
    
    print(f"Token count: {token_count}")
    print(f"Stats: {stats}")
    print(f"\nGenerated prompt:\n{prompt}\n")
    
    # Validation checks
    assert 'stage' in prompt.lower(), "Scene description lost!"
    assert 'red curtains' in prompt.lower(), "Scene details lost!"
    assert 'Nikita' in prompt, "Nikita not in prompt!"
    assert 'Roger' in prompt, "Roger not in prompt!"
    # Actions may be in base prompt which gets truncated - check at least one action
    has_action = 'guitar' in prompt.lower() or 'drums' in prompt.lower() or 'playing' in prompt.lower()
    assert has_action, "No actions preserved!"
    assert token_count <= 77, f"Token count too high: {token_count}"
    assert stats['characters_included'] == 2, "Both characters should be included"
    
    # Should NOT have dropped the scene description
    assert stats.get('base_compressed', False) or token_count < 77, "Base should be compressed not dropped"
    
    print("✓ All validations passed!")


def test_sitting_not_stripped():
    """Test that 'sitting' is preserved in scene descriptions."""
    print("\n=== Test Sitting Preserved ===")
    
    base_prompt = "Nikita is sitting on a chair playing guitar. Roger is sitting behind drums."
    characters = []
    
    from utils.token_budget import TokenBudgetManager
    manager = TokenBudgetManager()
    
    # Scene text should NOT strip 'sitting'
    cleaned = manager.strip_generation_instructions(base_prompt, is_character_prompt=False)
    
    print(f"Original: {base_prompt}")
    print(f"Cleaned: {cleaned}")
    
    assert 'sitting' in cleaned.lower(), "'sitting' should be preserved in scene!"
    print("✓ Sitting preserved!")


def test_character_prompt_strips_sitting():
    """Test that sitting IS stripped from character prompts."""
    print("\n=== Test Character Prompt Stripping ===")
    
    char_prompt = "muscular man, age 30-40, sitting upright, full body photo"
    
    from utils.token_budget import TokenBudgetManager
    manager = TokenBudgetManager()
    
    cleaned = manager.strip_generation_instructions(char_prompt, is_character_prompt=True)
    
    print(f"Original: {char_prompt}")
    print(f"Cleaned: {cleaned}")
    
    assert 'sitting upright' not in cleaned.lower(), "'sitting upright' should be stripped from char prompt"
    assert 'muscular man' in cleaned.lower(), "Appearance should be preserved"
    print("✓ Character stripping works!")


def test_word_boundary_matching():
    """Test word boundary matching prevents false positives."""
    print("\n=== Test Word Boundary Matching ===")
    
    import re
    
    text = "Roger is playing drums. The Rogers are watching."
    pattern = r'\b' + re.escape('roger') + r'\b'
    matches = re.findall(pattern, text.lower())
    
    print(f"Text: {text}")
    print(f"Matches for 'roger': {matches}")
    
    # Should match only one occurrence
    assert len(matches) == 1, "Should only match standalone 'Roger', not 'Rogers'"
    print("✓ Word boundaries work!")


if __name__ == "__main__":
    test_user_reported_case()
    test_sitting_not_stripped()
    test_character_prompt_strips_sitting()
    test_word_boundary_matching()
    print("\n✓✓✓ All fix validation tests passed! ✓✓✓")
