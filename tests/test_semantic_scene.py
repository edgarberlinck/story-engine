"""
Test semantic three-layer scene decomposition.
"""

import sys
sys.path.insert(0, '/Users/edgarberlinck/code/story-engine')

from core.scene_semantics import (
    SemanticSceneExtractor,
    CharacterIdentity,
    CharacterSceneData,
    ScenePromptSpec
)
from core.prompt_builder import SemanticPromptBuilder


def test_three_layer_structure():
    """Test the three-layer semantic structure."""
    print("\n=== Test Three-Layer Structure ===")
    
    scene_description = """A small stage inside an old bar with red curtains and cracked wooden floors.
Several tables are positioned between the camera and the stage, with people sitting and watching the performance.
The camera is positioned at the back of the bar in a wide shot, showing the audience, the entire stage and both musicians.

On stage, Nikita is positioned on the left, playing a black Gibson Explorer.
Roger is positioned on the right, sitting behind a drum kit."""

    characters = [
        {
            'name': 'Nikita',
            'prompt': 'Young woman, long curly red hair, fair skin, blue eyes, fantasy clothing, mysterious expression'
        },
        {
            'name': 'Roger',
            'prompt': 'Muscular dark-skinned man with short bald hair, casual clothes'
        }
    ]
    
    # Extract semantic spec
    spec = SemanticSceneExtractor.extract_scene_layers(
        scene_description,
        characters
    )
    
    print(f"\nScene: {spec.scene.environment[:80]}...")
    print(f"Characters: {[c.name for c in spec.characters]}")
    
    for char in spec.characters:
        print(f"\n  {char.name}:")
        print(f"    Identity: {char.identity.to_concise_description()}")
        print(f"    Position: {char.position}")
        print(f"    Clothing: {char.clothing}")
        print(f"    Action: {char.action}")
    
    assert len(spec.characters) == 2
    assert spec.characters[0].name == 'Nikita'
    print("\n✓ Three-layer structure works!")


def test_explicit_identity_association():
    """Test explicit identity association requirement."""
    print("\n=== Test Explicit Identity Association ===")
    
    identity = CharacterIdentity(
        gender='young woman',
        age_range='25-35',
        hair_color='red',
        hairstyle='curly long',
        skin_tone='fair'
    )
    
    char_data = CharacterSceneData(
        name='Nikita',
        identity=identity,
        position='left side of the stage',
        clothing='black suit',
        action='playing a black Gibson Explorer'
    )
    
    builder = SemanticPromptBuilder()
    prompt = builder.build_explicit_identity_prompt(
        ScenePromptSpec(
            scene=None,
            characters=[char_data],
            style="photorealistic"
        )
    )
    
    print(f"\nGenerated prompt:\n{prompt}")
    
    # Must explicitly associate identity with name
    assert 'Nikita' in prompt
    assert 'young woman' in prompt or 'young' in prompt
    assert 'long curly red hair' in prompt or 'red hair' in prompt
    assert 'left side of the stage' in prompt or 'left' in prompt
    assert 'playing' in prompt
    
    # Must NOT be generic like "two musicians"
    assert 'two musicians' not in prompt.lower()
    
    print("\n✓ Explicit identity association works!")


def test_fantasy_clothing_filtered():
    """Test that irrelevant clothing is filtered out."""
    print("\n=== Test Fantasy Clothing Filtered ===")
    
    # Character with fantasy clothing
    char_record = {
        'name': 'Nikita',
        'prompt': 'Young woman, long curly red hair, fair skin, blue eyes, FANTASY CLOTHING, mysterious expression'
    }
    
    scene_desc = "Nikita is wearing a black suit and playing guitar on stage."
    
    # Extract identity
    identity = SemanticSceneExtractor._extract_identity(char_record, scene_desc)
    
    print(f"\nOriginal prompt had: fantasy clothing")
    print(f"Extracted identity: {identity.to_concise_description()}")
    
    # Identity should have hair/skin but NOT fantasy clothing
    identity_str = identity.to_concise_description().lower()
    assert 'red' in identity_str or 'hair' in identity_str
    assert 'fantasy' not in identity_str
    
    print("\n✓ Fantasy clothing filtered out!")


def test_three_layers_separate():
    """Test that three layers are properly separated."""
    print("\n=== Test Three Layers Separate ===")
    
    spec = ScenePromptSpec(
        scene=None,
        characters=[
            CharacterSceneData(
                name='Nikita',
                identity=CharacterIdentity(gender='woman', hair_color='red'),
                position='left',
                clothing='black suit',
                action='playing guitar'
            )
        ],
        style="photorealistic"
    )
    
    builder = SemanticPromptBuilder()
    layers = builder.build_layered_prompts(spec)
    
    print(f"\nComposition: {layers['composition']}")
    print(f"\nIdentities: {layers['identities']}")
    print(f"\nAppearances/Actions: {layers['appearances_actions']}")
    print(f"\nFull Explicit: {layers['full_explicit'][:100]}...")
    
    # Layers should be different
    assert layers['composition'] != layers['identities']
    assert 'Nikita' in layers['full_explicit']
    
    print("\n✓ Three layers properly separated!")


def test_user_example():
    """Test exact user example."""
    print("\n=== Test User Example ===")
    
    scene_desc = """There's a very small stage with red curtains. The stage is made of old, cracked wood.
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
            'prompt': 'full body photo of muscular male, age 30-40, short bald black hair'
        }
    ]
    
    spec = SemanticSceneExtractor.extract_scene_layers(scene_desc, characters)
    
    print(f"\nExtracted {len(spec.characters)} characters")
    
    for char in spec.characters:
        print(f"\n{char.name}:")
        print(f"  Identity: {char.identity.to_concise_description()}")
        print(f"  Position: {char.position}")
        print(f"  Clothing: {char.clothing}")
        print(f"  Action: {char.action}")
    
    # Build explicit prompt
    builder = SemanticPromptBuilder()
    explicit_prompt = builder.build_explicit_identity_prompt(spec)
    
    print(f"\n{'='*70}")
    print("FULL EXPLICIT PROMPT:")
    print(f"{'='*70}")
    print(explicit_prompt)
    print(f"{'='*70}")
    
    assert 'Nikita' in explicit_prompt
    assert 'Roger' in explicit_prompt
    assert 'stage' in explicit_prompt.lower()
    
    print("\n✓ User example works!")


if __name__ == "__main__":
    test_three_layer_structure()
    test_explicit_identity_association()
    test_fantasy_clothing_filtered()
    test_three_layers_separate()
    test_user_example()
    
    print("\n" + "="*70)
    print("✓✓✓ All semantic scene tests passed! ✓✓✓")
    print("="*70)
