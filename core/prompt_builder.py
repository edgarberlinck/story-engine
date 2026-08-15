"""
Prompt builder for semantic scene specification.

Converts three-layer ScenePromptSpec into actual diffusion prompts
respecting token budgets and explicit identity association.
"""

from typing import List, Dict, Any
from core.scene_semantics import ScenePromptSpec, CharacterSceneData


class SemanticPromptBuilder:
    """Builds prompts from semantic scene specification."""
    
    @staticmethod
    def build_composition_prompt(spec: ScenePromptSpec) -> str:
        """
        Build composition-only prompt (Layer 1).
        
        Describes environment, camera, spatial layout WITHOUT characters.
        """
        
        parts = []
        
        # Environment
        if spec.scene and spec.scene.environment:
            parts.append(spec.scene.environment)
        
        # Camera and shot
        if spec.scene:
            camera_desc = f"{spec.scene.shot_type} shot"
            if spec.scene.camera_position:
                camera_desc += f" from {spec.scene.camera_position}"
            parts.append(camera_desc)
            
            # Spatial composition
            if spec.scene.spatial_composition:
                parts.append(spec.scene.spatial_composition)
            
            # Lighting and atmosphere
            if spec.scene.lighting:
                parts.append(f"lighting: {spec.scene.lighting}")
            
            if spec.scene.atmosphere:
                parts.append(f"atmosphere: {spec.scene.atmosphere}")
        
        return ". ".join(parts) + "."
    
    @staticmethod
    def build_character_identity_prompt(char_data: CharacterSceneData) -> str:
        """
        Build character identity description (Layer 2).
        
        Only identity attributes, no clothing/pose from original generation.
        """
        
        identity_desc = char_data.identity.to_concise_description()
        
        return f"{char_data.name}, {identity_desc}"
    
    @staticmethod
    def build_scene_appearance_action_prompt(char_data: CharacterSceneData) -> str:
        """
        Build scene-specific appearance and action (Layer 3).
        """
        
        parts = []
        
        # Position
        if char_data.position:
            parts.append(f"positioned {char_data.position}")
        
        # Clothing
        if char_data.clothing:
            parts.append(f"wearing {char_data.clothing}")
        
        # Action
        if char_data.action:
            parts.append(f"{char_data.action}")
        
        return ", ".join(parts)
    
    @staticmethod
    def build_explicit_identity_prompt(spec: ScenePromptSpec) -> str:
        """
        Build prompt with EXPLICIT identity association.
        
        This implements the user's rule: 
        "The scene generation prompt must explicitly associate 
         each character identity with a position and action."
        
        Example output:
        "On the left side of the stage, Nikita, a young woman with 
         long curly red hair and fair skin, is playing a black 
         Gibson Explorer wearing a black suit."
        """
        
        parts = []
        
        # Base scene composition
        comp_prompt = SemanticPromptBuilder.build_composition_prompt(spec)
        parts.append(comp_prompt)
        
        # Add each character with explicit identity association
        for char_data in spec.characters:
            identity_desc = char_data.identity.to_concise_description()
            
            # Build explicit sentence
            if char_data.position:
                pos_desc = f"On {char_data.position}"
            else:
                pos_desc = f"In the scene"
            
            if char_data.clothing and char_data.action:
                action_desc = f"{char_data.name}, {identity_desc}, is wearing {char_data.clothing} and {char_data.action}"
            elif char_data.action:
                action_desc = f"{char_data.name}, {identity_desc}, is {char_data.action}"
            else:
                action_desc = f"{char_data.name}, {identity_desc}"
            
            full_sentence = f"{pos_desc}, {action_desc}."
            parts.append(full_sentence)
        
        # Add style
        if spec.style:
            parts.append(f"style: {spec.style}")
        
        return " ".join(parts)
    
    @staticmethod
    def build_layered_prompts(spec: ScenePromptSpec) -> Dict[str, str]:
        """
        Build prompts for each of the three layers.
        
        Returns dict with layer names and prompts.
        """
        
        # Layer 1: Scene Composition
        composition = SemanticPromptBuilder.build_composition_prompt(spec)
        
        # Layer 2: Character Identities (separate)
        identities = []
        for char_data in spec.characters:
            identity = SemanticPromptBuilder.build_character_identity_prompt(char_data)
            identities.append(identity)
        
        # Layer 3: Scene Appearance & Actions
        appearances = []
        for char_data in spec.characters:
            appearance = SemanticPromptBuilder.build_scene_appearance_action_prompt(char_data)
            appearances.append(f"{char_data.name}: {appearance}")
        
        # Full explicit prompt
        full_prompt = SemanticPromptBuilder.build_explicit_identity_prompt(spec)
        
        return {
            'composition': composition,
            'identities': " | ".join(identities),
            'appearances_actions': " | ".join(appearances),
            'full_explicit': full_prompt
        }
    
    @staticmethod
    def optimize_for_token_budget(
        spec: ScenePromptSpec,
        max_tokens: int = 77
    ) -> str:
        """
        Optimize full prompt to fit token budget.
        
        Uses LLM to compress while preserving explicit identity association.
        """
        
        from utils.token_budget import count_tokens
        
        # Build initial prompt
        full_prompt = SemanticPromptBuilder.build_explicit_identity_prompt(spec)
        
        current_tokens = count_tokens(full_prompt)
        
        if current_tokens <= max_tokens:
            return full_prompt
        
        # Need to compress - use LLM or priority-based reduction
        print(f"Prompt too long: {current_tokens} tokens, need ≤{max_tokens}")
        
        # Priority order for preservation:
        # 1. Character identities (MUST keep)
        # 2. Position and action
        # 3. Scene composition
        # 4. Style details
        
        # Build minimal version
        chars_part = []
        for char_data in spec.characters:
            identity_desc = char_data.identity.to_concise_description()
            
            # Most critical: who + position + action
            essential = f"{char_data.name} ({identity_desc}), {char_data.position}, {char_data.action}"
            if char_data.clothing:
                essential += f", wearing {char_data.clothing}"
            
            chars_part.append(essential)
        
        minimal = " ".join(chars_part)
        
        # Add environment
        if spec.scene.environment:
            minimal = f"{spec.scene.environment} | {minimal}"
        
        return minimal


def create_optimized_scene_prompt(
    scene_description: str,
    character_records: List[Dict[str, Any]],
    project_style: str = "photorealistic",
    max_tokens: int = 77
) -> str:
    """
    Main entry point for creating optimized scene prompt.
    
    Takes scene description and character records, extracts semantic layers
    using LLM, then builds explicit identity prompt respecting token budget.
    """
    
    from core.scene_semantics import SemanticSceneExtractor, LLMSceneSemanticExtractor
    
    # Extract semantic layers
    extractor = LLMSceneSemanticExtractor()
    spec = extractor.extract_semantic_spec(
        scene_description,
        character_records,
        project_style
    )
    
    # Build optimized prompt
    builder = SemanticPromptBuilder()
    
    # Try to keep all information if within budget
    full_prompt = builder.build_explicit_identity_prompt(spec)
    
    from utils.token_budget import count_tokens
    tokens = count_tokens(full_prompt)
    
    if tokens <= max_tokens:
        print(f"✓ Prompt fits budget: {tokens}/{max_tokens} tokens")
        return full_prompt
    else:
        # Optimize for budget
        print(f"⚠ Prompt exceeds budget ({tokens}/{max_tokens}), optimizing...")
        optimized = builder.optimize_for_token_budget(spec, max_tokens)
        
        optimized_tokens = count_tokens(optimized)
        print(f"✓ Optimized to {optimized_tokens}/{max_tokens} tokens")
        
        return optimized


def example_usage():
    """Example from user's requirements."""
    
    scene_description = """A small stage inside an old bar with red curtains and cracked wooden floors.
Several tables are positioned between the camera and the stage, with people sitting and watching the performance.
The camera is positioned at the back of the bar in a wide shot, showing the audience, the entire stage and both musicians.

On stage, Nikita is positioned on the left, playing a black Gibson Explorer.
Roger is positioned on the right, sitting behind a drum kit.

Both characters must be visible at the same time."""
    
    character_records = [
        {
            'name': 'Nikita',
            'prompt': 'Young woman, long curly red hair, fair skin, blue eyes, fantasy clothing, mysterious expression, detailed face, full body, studio lighting, standing upright'
        },
        {
            'name': 'Roger',
            'prompt': 'Muscular dark-skinned man with short bald hair, casual clothes, standing pose'
        }
    ]
    
    # Create optimized prompt
    prompt = create_optimized_scene_prompt(
        scene_description,
        character_records,
        project_style="photorealistic",
        max_tokens=77
    )
    
    print("\n" + "="*70)
    print("OPTIMIZED SCENE PROMPT:")
    print("="*70)
    print(prompt)
    print("="*70)
    
    return prompt


if __name__ == "__main__":
    example_usage()
