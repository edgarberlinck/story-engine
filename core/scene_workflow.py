"""
High-level scene generation workflow with LLM orchestration.

This module provides the main entry point for generating scenes using
LLM-based planning and multi-step composition.

As of the character-asset + deterministic composition fallback design
(`docs/story-engine-reference-conditioned-scene-design.md`), the heavy
lifting is delegated to `core/scene_pipeline.py`, which selects between
reference-conditioned, single-pass, progressive (prompt-only), and
asset-composition strategies. This function is kept as a thin,
backward-compatible wrapper so existing callers/tests don't need to change.
"""

from typing import List, Dict, Any, Optional

from generators.image_engine import generate_scene


def generate_scene_with_llm_orchestration(
    prompt: str,
    project: str = "default",
    scene_number: Optional[int] = None,
    characters: Optional[List[Dict]] = None,
    use_multi_step: bool = True,
    enforce_token_budget: bool = True,
    model: str = "flux_dev",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generate scene using LLM orchestration for planning and decomposition.

    This is the recommended workflow when:
    - Multiple characters with complex descriptions
    - Context mismatch (e.g., fantasy character in modern setting)
    - Token budget concerns with detailed prompts

    Args:
        prompt: Scene description
        project: Project name
        scene_number: Optional specific scene number
        characters: Optional explicit character list (auto-detected if None)
        use_multi_step: Whether to allow multi-step (progressive/
            asset-composition) strategies for multi-character scenes. If
            False, always uses direct single-pass generation.
        enforce_token_budget: Keep prompts within CLIP limits (passed through
            to the single-pass path).
        model: Generation model
        seed: Random seed

    Returns:
        Scene generation result with planning metadata.
    """

    from services.database.character_service import character_service

    if characters is None:
        characters = character_service.find_characters_in_text(prompt, project)

    if not characters or not use_multi_step:
        print("Using direct single-pass generation" if not characters else
              "Multi-step disabled, using direct single-pass generation")
        return generate_scene(
            prompt=prompt,
            project=project,
            scene_number=scene_number,
            model=model,
            seed=seed,
            enforce_token_budget=enforce_token_budget,
        )

    print(f"\n{'='*60}")
    print(f"Scene pipeline orchestration - {len(characters)} character(s) detected")
    print(f"{'='*60}\n")

    from core.scene_pipeline import generate_scene_pipeline

    result = generate_scene_pipeline(
        prompt=prompt,
        project=project,
        scene_number=scene_number,
        characters=characters,
        model=model,
        seed=seed,
    )
    result['orchestration_type'] = 'scene_pipeline'
    return result


def enhance_scene_with_context_awareness(
    prompt: str,
    project: str,
    characters: List[Dict]
) -> str:
    """
    Enhance scene prompt with context-aware character handling.
    
    This handles cases where character generation style doesn't match
    scene context (e.g., fantasy clothing in bar scene).
    
    Uses LLM to determine what should be preserved vs adapted.
    """
    
    from core.scene_planner import stage_a_context_resolution
    
    # Resolve characters against scene context
    resolved = stage_a_context_resolution(prompt, characters)
    
    # Build enhanced prompt with context-aware adjustments
    enhanced = prompt
    for char_res in resolved:
        if char_res.presentation_decision == 'REPLACE':
            print(f"Note: {char_res.name} presentation adapted for scene")
            # Add scene-appropriate guidance
            enhanced += f"\n{char_res.name} wearing scene-appropriate attire"
    
    return enhanced


# Example usage
if __name__ == "__main__":
    # Example from user:
    # "Nikita was generated using Fantasy Clothing, but she's dressed differently in the scene"
    
    scene_desc = """There's a very small stage with red curtains. 
In front of stage, tables with people watching.
Nikita is sitting on chair playing black Gibson Explorer guitar.
Roger is behind drums.
Camera from back of bar."""
    
    characters = [
        {
            'name': 'Nikita',
            'prompt': 'fantasy art style, woman with long black hair, ornate elven armor, glowing runes'
        },
        {
            'name': 'Roger',
            'prompt': 'man, bald head, casual clothing'
        }
    ]
    
    # Generate with LLM orchestration
    result = generate_scene_with_llm_orchestration(
        prompt=scene_desc,
        project="demo",
        characters=characters,
        use_multi_step=True
    )
    
    print("\nGeneration complete!")
    print(f"Result: {result}")
