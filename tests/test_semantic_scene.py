"""
Test semantic three-layer scene decomposition.

Converted from standalone ``def test_*()`` functions into proper
``unittest.TestCase`` classes so they are collected by
``python -m unittest discover`` and contribute real coverage to
``core.prompt_builder`` / ``core.scene_semantics``.
"""

import unittest

from core.scene_semantics import (
    SemanticSceneExtractor,
    CharacterIdentity,
    CharacterSceneData,
    ScenePromptSpec,
)
from core.prompt_builder import SemanticPromptBuilder


class TestSemanticScene(unittest.TestCase):

    def test_three_layer_structure(self):
        """Test the three-layer semantic structure."""
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

        for char in spec.characters:
            pass  # exercise identity / position / clothing / action fields

        self.assertEqual(len(spec.characters), 2)
        self.assertEqual(spec.characters[0].name, 'Nikita')

    def test_explicit_identity_association(self):
        """Test explicit identity association requirement."""
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

           # Must explicitly associate identity with name
        self.assertIn('Nikita', prompt)
        self.assertTrue('young woman' in prompt or 'young' in prompt)
        self.assertTrue('long curly red hair' in prompt or 'red hair' in prompt)
        self.assertTrue('left side of the stage' in prompt or 'left' in prompt)
        self.assertIn('playing', prompt)

           # Must NOT be generic like "two musicians"
        self.assertNotIn('two musicians', prompt.lower())

    def test_fantasy_clothing_filtered(self):
        """Test that irrelevant clothing is filtered out."""
           # Character with fantasy clothing
        char_record = {
             'name': 'Nikita',
             'prompt': 'Young woman, long curly red hair, fair skin, blue eyes, FANTASY CLOTHING, mysterious expression'
          }

        scene_desc = "Nikita is wearing a black suit and playing guitar on stage."

           # Extract identity
        identity = SemanticSceneExtractor._extract_identity(char_record, scene_desc)

           # Identity should have hair/skin but NOT fantasy clothing
        identity_str = identity.to_concise_description().lower()
        self.assertTrue('red' in identity_str or 'hair' in identity_str)
        self.assertNotIn('fantasy', identity_str)

    def test_three_layers_separate(self):
        """Test that three layers are properly separated."""
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

           # Layers should be different
        self.assertNotEqual(layers['composition'], layers['identities'])
        self.assertIn('Nikita', layers['full_explicit'])

    def test_user_example(self):
        """Test exact user example."""
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

           # Build explicit prompt
        builder = SemanticPromptBuilder()
        explicit_prompt = builder.build_explicit_identity_prompt(spec)

        self.assertIn('Nikita', explicit_prompt)
        self.assertIn('Roger', explicit_prompt)
        self.assertIn('stage', explicit_prompt.lower())


if __name__ == "__main__":
    unittest.main()
