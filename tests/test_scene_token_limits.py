"""
Integration test for scene generation with token limits.
Tests the actual prompt construction flow.
"""

import sys
sys.path.insert(0, '/Users/edgarberlinck/code/story-engine')

from utils.token_budget import build_token_aware_scene_prompt, count_tokens


def test_single_character_scene():
    """Test single character scene token limits."""
    print("\n=== Single Character Scene Test ===")
    
    base_prompt = "Roger is sitting behind a drum kit playing drums"
    characters = [
        {
            'name': 'Roger',
            'prompt': 'full body photo of a muscular male character, age 30-40, with short bald black hair, dark skin tone and brown eyes, wearing formal clothing, serious expression, photorealistic, detailed face, high detail, good lighting. standing upright, entire body visible from head to toe including feet, full length wide shot, camera far from subject, feet touching the ground visible in frame, not a portrait, not a close-up'
        }
    ]
    
    prompt, stats = build_token_aware_scene_prompt(base_prompt, characters)
    token_count = count_tokens(prompt)
    
    print(f"Base prompt: {base_prompt}")
    print(f"Generated prompt: {prompt[:200]}...")
    print(f"Token count: {token_count}")
    print(f"Stats: {stats}")
    
    # Should be within reasonable limits
    assert token_count < 100, f"Token count too high: {token_count}"
    assert 'Roger' in prompt, "Character name missing"
    assert 'drum kit' in prompt.lower(), "Action missing"
    assert 'standing upright' not in prompt.lower(), "Generation instruction should be stripped"
    assert 'full body photo' not in prompt.lower(), "Generation instruction should be stripped"


def test_two_character_scene():
    """Test two character scene token limits."""
    print("\n=== Two Character Scene Test ===")
    
    base_prompt = "Nikita is playing a black Gibson Explorer guitar. Roger is playing drums. Both are on a small stage with red curtains."
    characters = [
        {
            'name': 'Nikita',
            'prompt': 'ultra realistic woman, age 25-35, slender build, long black hair, dark skin, wearing black clothing, playing guitar'
        },
        {
            'name': 'Roger',
            'prompt': 'full body photo of a muscular male character, age 30-40, short bald black hair, dark skin tone, brown eyes, wearing formal clothing, standing upright, entire body visible, photorealistic'
        }
    ]
    
    prompt, stats = build_token_aware_scene_prompt(base_prompt, characters)
    token_count = count_tokens(prompt)
    
    print(f"Generated prompt: {prompt}")
    print(f"Token count: {token_count}")
    
    assert 'Nikita' in prompt
    assert 'Roger' in prompt
    assert 'guitar' in prompt.lower()
    assert 'drums' in prompt.lower()
    assert token_count < 100


def test_long_scene_description():
    """Test very long scene description with prioritization."""
    print("\n=== Long Scene Description Test ===")
    
    base_prompt = """
    Nikita is sitting on a chair playing a black Gibson Explorer guitar. 
    Roger is sitting behind a drum kit playing drums.
    They are performing on a small stage with red curtains.
    The stage is made of old cracked wood.
    The camera is positioned at the back of a crowded bar.
    The audience is visible in the foreground, cheering and dancing.
    There are neon lights illuminating the stage.
    The atmosphere is energetic and lively.
    Smoke fills the air from previous performances.
    """
    
    characters = [
        {'name': 'Nikita', 'prompt': 'woman, age 25-35, long black hair, dark skin'},
        {'name': 'Roger', 'prompt': 'man, age 30-40, bald head, brown eyes'}
    ]
    
    prompt, stats = build_token_aware_scene_prompt(base_prompt, characters)
    token_count = count_tokens(prompt)
    
    print(f"Token count: {token_count}")
    print(f"Items dropped: {stats['items_dropped']}")
    print(f"Prompt preview: {prompt[:200]}...")
    
    # Should prioritize critical info
    assert 'Nikita' in prompt
    assert 'Roger' in prompt
    assert token_count < 100


def test_character_generation_instructions_stripped():
    """Test that character generation instructions are removed."""
    print("\n=== Generation Instructions Stripping Test ===")
    
    char_prompt = """
    full body photo of a muscular male character, age 30-40,
    with short bald black hair, dark skin tone and brown eyes,
    wearing formal clothing, serious expression, photorealistic,
    detailed face, high detail, good lighting.
    standing upright, entire body visible from head to toe including feet,
    full length wide shot, camera far from subject,
    feet touching the ground visible in frame,
    not a portrait, not a close-up
    """
    
    from utils.token_budget import TokenBudgetManager
    manager = TokenBudgetManager()
    cleaned = manager.strip_generation_instructions(char_prompt)
    
    print(f"Original: {char_prompt[:100]}...")
    print(f"Cleaned: {cleaned}")
    
    # Should remove instructions
    assert 'standing upright' not in cleaned.lower()
    assert 'entire body visible' not in cleaned.lower()
    assert 'full length wide shot' not in cleaned.lower()
    assert 'not a portrait' not in cleaned.lower()
    # Should keep attributes
    assert 'muscular' in cleaned.lower() or 'male' in cleaned.lower()


if __name__ == "__main__":
    test_single_character_scene()
    test_two_character_scene()
    test_long_scene_description()
    test_character_generation_instructions_stripped()
    print("\n✓ All tests passed!")
