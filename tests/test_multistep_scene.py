"""
Test multi-step scene generation.
"""

import sys
sys.path.insert(0, '/Users/edgarberlinck/code/story-engine')

from core.scene_composer import SceneComposer


def test_scene_decomposition():
    """Test scene decomposition into layers."""
    print("\n=== Test Scene Decomposition ===")
    
    description = """There's a very small stage with red curtains. The stage is made of old, cracked wood.

In front of the stage, there are tables with people sitting and watching the performance.

On the stage, Nikita is sitting on a chair and playing a black Gibson Explorer guitar.

Roger is sitting behind a drum kit and playing drums.

The camera is positioned at the back of the bar, showing the crowd in the foreground and the entire stage with both musicians in the background.

Wide shot, both characters clearly visible."""
    
    characters = [
        {'name': 'Nikita', 'prompt': 'woman, long black hair'},
        {'name': 'Roger', 'prompt': 'man, bald head'}
    ]
    
    plan = SceneComposer.decompose_scene(description, characters)
    
    print(f"Base environment: {plan.base_environment[:100]}...")
    print(f"Number of layers: {len(plan.layers)}")
    
    for layer in plan.layers:
        print(f"  - {layer.name}: {layer.prompt[:80]}...")
    
    assert len(plan.layers) >= 3, "Should have base + 2 characters"
    assert 'stage' in plan.base_environment.lower(), "Base should mention stage"
    assert any('Nikita' in l.name for l in plan.layers), "Should have Nikita layer"
    assert any('Roger' in l.name for l in plan.layers), "Should have Roger layer"
    
    print("✓ Decomposition works!")


def test_incremental_prompts():
    """Test incremental prompt building."""
    print("\n=== Test Incremental Prompts ===")
    
    description = "A stage with red curtains. Nikita playing guitar. Roger playing drums."
    characters = [
        {'name': 'Nikita', 'prompt': 'woman'},
        {'name': 'Roger', 'prompt': 'man'}
    ]
    
    plan = SceneComposer.decompose_scene(description, characters)
    steps = SceneComposer.build_incremental_prompts(plan)
    
    print(f"Number of steps: {len(steps)}")
    for name, prompt in steps:
        print(f"  Step '{name}': {prompt[:60]}...")
    
    assert len(steps) >= 3, "Should have at least 3 steps"
    assert 'base' in steps[0][0], "First step should be base"
    
    print("✓ Incremental prompts work!")


def test_user_example():
    """Test the exact user example."""
    print("\n=== Test User Example ===")
    
    description = """There's a very small stage with red curtains. The stage is made of old, cracked wood.

In front of the stage, there are tables with people sitting and watching the performance.

On the stage, Nikita is sitting on a chair and playing a black Gibson Explorer guitar.

Roger is sitting behind a drum kit and playing drums.

The camera is positioned at the back of the bar, showing the crowd in the foreground and the entire stage with both musicians in the background.

Wide shot, both characters clearly visible."""
    
    characters = [
        {
            'name': 'Nikita',
            'prompt': 'ultra realistic woman, age 25-35, slender build, long black hair',
            'attributes': {'age': 'Young Adult', 'body_type': 'Slim'}
        },
        {
            'name': 'Roger',
            'prompt': 'full body photo of muscular male, age 30-40, short bald black hair',
            'attributes': {'age': 'Adult', 'body_type': 'Muscular'}
        }
    ]
    
    plan = SceneComposer.decompose_scene(description, characters)
    
    print(f"✓ Base environment extracted: {plan.base_environment[:80]}...")
    print(f"✓ Characters identified: {[c['name'] for c in plan.characters]}")
    print(f"✓ Layers created: {len(plan.layers)}")
    
    # Verify we can build prompts without token limits
    from utils.token_budget import count_tokens
    
    steps = SceneComposer.build_incremental_prompts(plan)
    for name, prompt in steps:
        tokens = count_tokens(prompt)
        print(f"  Step '{name}': {tokens} tokens")
        assert tokens < 100, f"Step {name} too long: {tokens} tokens"
    
    print("✓ User example works with multi-step!")


if __name__ == "__main__":
    test_scene_decomposition()
    test_incremental_prompts()
    test_user_example()
    print("\n✓✓✓ All multi-step tests passed! ✓✓✓")
