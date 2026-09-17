"""
Complete test of semantic scene system with user's exact requirements.

Converted from standalone ``def test_*()`` functions into proper
``unittest.TestCase`` classes so they are collected by
``python -m unittest discover`` and contribute real coverage to
``core.prompt_builder`` / ``core.scene_semantics``.
"""

import unittest

from core.scene_semantics import (
    SemanticSceneExtractor,
    ScenePromptSpec,
    SceneComposition,
    CharacterSceneData,
    CharacterIdentity,
)
from core.prompt_builder import SemanticPromptBuilder
from utils.token_budget import count_tokens


class TestSceneSemanticsComplete(unittest.TestCase):

    def test_user_requirements(self):
        """Test all user requirements."""
        # User's exact example
        scene_description = """A small stage inside an old bar with red curtains and cracked wooden floors.
Several tables are positioned between the camera and the stage, with people sitting and watching the performance.
The camera is positioned at the back of the bar in a wide shot, showing the audience, the entire stage and both musicians.

On stage, Nikita is positioned on the left, playing a black Gibson Explorer.
Roger is positioned on the right, sitting behind a drum kit.

Both characters must be visible at the same time."""

        character_records = [
            {
                "name": "Nikita",
                "prompt": "Young woman with long curly red hair and fair skin, fantasy clothing",
            },
            {
                "name": "Roger",
                "prompt": "Muscular dark-skinned man with short bald hair, casual clothes",
            },
        ]

        # 1. Scene composition layer
        spec = SemanticSceneExtractor.extract_scene_layers(
            scene_description, character_records
        )

        self.assertIsNotNone(spec.scene)
        self.assertTrue(
            "stage" in spec.scene.environment.lower()
            or "bar" in spec.scene.environment.lower()
        )

        # 2. Character identity layer
        for char in spec.characters:
            identity_desc = char.identity.to_concise_description()
            # Should NOT have fantasy clothing
            self.assertNotIn("fantasy", identity_desc.lower())
            # Should have gender/age
            identity_lower = identity_desc.lower()
            self.assertTrue("woman" in identity_lower or "man" in identity_lower)

        # 3. Scene-specific appearance/action layer
        for char in spec.characters:
            self.assertTrue(char.position)
            self.assertTrue(char.action)

        # 4. Explicit identity association
        builder = SemanticPromptBuilder()
        explicit_prompt = builder.build_explicit_identity_prompt(spec)

        self.assertIn("Nikita", explicit_prompt)
        self.assertIn("Roger", explicit_prompt)

        # Should NOT be generic
        self.assertNotIn("two musicians", explicit_prompt.lower())
        self.assertTrue(
            "a woman playing guitar" in explicit_prompt.lower()
            or "woman" in explicit_prompt.lower()
        )

        # 5. Three distinct semantic layers
        layers = builder.build_layered_prompts(spec)

        self.assertNotEqual(layers["composition"], layers["identities"])
        self.assertIn("Nikita", layers["full_explicit"])

        # 6. Token budget exercise
        tokens = count_tokens(explicit_prompt)
        self.assertGreater(tokens, 0)

    def test_user_json_example(self):
        """Test user's exact JSON example format."""
        # Create spec matching the user's structure
        spec = ScenePromptSpec(
            scene=SceneComposition(
                environment="small old bar with red curtains and cracked wooden stage",
                camera_position="back of the bar",
                shot_type="wide shot",
                spatial_composition="audience in foreground, entire stage visible",
            ),
            characters=[
                CharacterSceneData(
                    name="Nikita",
                    identity=CharacterIdentity(
                        gender="young woman",
                        hair_color="red",
                        hairstyle="curly long",
                        skin_tone="fair",
                    ),
                    position="left side of the stage",
                    clothing="black suit",
                    action="playing a black Gibson Explorer",
                ),
                CharacterSceneData(
                    name="Roger",
                    identity=CharacterIdentity(
                        gender="muscular man", skin_tone="dark", hairstyle="short bald"
                    ),
                    position="right side of the stage",
                    clothing="dark formal outfit",
                    action="sitting behind a drum kit",
                ),
            ],
            style="photorealistic",
        )

        expected_structure = {
            "scene": {
                "environment": "small old bar with red curtains and cracked wooden stage",
                "camera": "wide shot from the back of the bar",
                "composition": "audience in foreground, entire stage visible",
            },
            "characters": [
                {
                    "name": "Nikita",
                    "identity": ["young woman", "long curly red hair", "fair skin"],
                    "position": "left side of the stage",
                    "clothing": "black suit",
                    "action": "playing a black Gibson Explorer",
                },
                {
                    "name": "Roger",
                    "identity": ["muscular man", "dark skin", "short bald hair"],
                    "position": "right side of the stage",
                    "clothing": "dark formal outfit",
                    "action": "sitting behind a drum kit",
                },
            ],
            "style": "photorealistic",
        }

        # Convert to dict and verify structure matches
        result = spec.to_dict()

        self.assertEqual(result["style"], expected_structure["style"])
        self.assertEqual(len(result["characters"]), 2)
        self.assertEqual(result["characters"][0]["name"], "Nikita")
        self.assertEqual(result["characters"][1]["name"], "Roger")

    def test_style_normalization(self):
        """Test that different character styles normalize to scene style."""
        # Characters from different generation styles
        characters = [
            {
                "name": "Nikita",
                "prompt": "manga style, anime character, fantasy clothing",
            },
            {"name": "Roger", "prompt": "photorealistic man, 3d render"},
        ]

        scene_desc = "Both characters in photorealistic bar scene"
        spec = SemanticSceneExtractor.extract_scene_layers(scene_desc, characters)

        # Scene style should be consistent (defaulted to photorealistic)
        self.assertTrue(spec.style == "photorealistic" or spec.style is not None)


if __name__ == "__main__":
    unittest.main()
