import enum
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING

# Import existing functionality
from generators.image_generator import generate_images, AVAILABLE_DIFFUSION_MODELS
from models import DIFFUSION_MODELS, MODEL_PATHS
from services.database.character_service import character_service
from utils.project_paths import (
    DEFAULT_PROJECT,
    character_dir,
    scene_dir,
    next_scene_number,
    slugify,
)
from utils.face_check import character_appears_in_image
from utils.scene_logger import scene_logging

if TYPE_CHECKING:
    from abc import ABC, abstractmethod


class GenerationType(str, enum.Enum):
    """Enumeration of supported generation types."""
    CHARACTER = "character"
    ENVIRONMENT = "environment"


class CharacterReferenceStore:
    """Abstract base class for storing and retrieving character visual references.
    
    This interface is designed to support identity consistency for character generation.
    Implementations will be provided in future steps.
    """
    
    def has_reference(self, character_id: str) -> bool:
        """Check whether a visual reference exists for the given character.
        
        Args:
            character_id: Unique identifier for the character
            
        Returns:
            True if a reference exists, False otherwise
        """
        raise NotImplementedError
    
    def get_reference(self, character_id: str) -> Optional[str]:
        """Get the visual reference path for the given character.
        
        Args:
            character_id: Unique identifier for the character
            
        Returns:
            Path to the reference image or None if not found
        """
        raise NotImplementedError


class NullCharacterReferenceStore(CharacterReferenceStore):
    """Default implementation that always returns no references."""
    
    def has_reference(self, character_id: str) -> bool:
        return False
    
    def get_reference(self, character_id: str) -> Optional[str]:
        return None


# Model selection policy - encapsulated for future flexibility
_MODEL_BY_TYPE: Dict[GenerationType, str] = {
    GenerationType.CHARACTER: "flux_dev",
    GenerationType.ENVIRONMENT: "sdxl"
}


def generate(
    prompt: str,
    type: GenerationType,
    character_id: Optional[str] = None,
    reference_store: CharacterReferenceStore = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """Generate an image based on the specified type and parameters.
    
    For character generation, this function may use identity-consistency mechanisms.
    For environment generation, it uses the model appropriate for environments.
    
    Args:
        prompt: The text prompt to guide image generation
        type: The type of generation ("character" or "environment")
        character_id: Optional identifier for character to maintain consistency
        reference_store: Storage mechanism for visual references (defaults to null store)
        **kwargs: Additional parameters forwarded to the backend (seed, steps, cfg, etc.)
        
    Returns:
        Dictionary containing generated image information
    """
    # Validate the generation type
    if type not in GenerationType:
        valid_types = ", ".join([t.value for t in GenerationType])
        raise ValueError(f"Invalid type '{type}'. Valid types are: {valid_types}")
        
    # Use default reference store if none provided
    if reference_store is None:
        reference_store = NullCharacterReferenceStore()
    
    # Select the appropriate model based on generation type
    model_name = _MODEL_BY_TYPE[type]
    
    # Get model path for the selected model (this mimics how generate_image does it)
    model_path = MODEL_PATHS["diffusion"] + "/" + model_name
    
    # For characters, check if there's a reference to maintain consistency
    image_kwargs = kwargs.copy()  # Create a copy to avoid modifying original
    
    if type == GenerationType.CHARACTER and character_id is not None:
        # Check for existing reference
        if reference_store.has_reference(character_id):
            reference_path = reference_store.get_reference(character_id)
            # TODO: Implement actual identity preservation mechanism (IP-Adapter, etc.)
            # This would involve passing the reference image to the pipeline
            # For now, we're just preparing the architecture
    
    # Generate the image using the selected model and parameters
    return generate_images(
        prompt=prompt,
        model_name=model_name,
        model_path=model_path,
        **image_kwargs
    )


# ---------------------------------------------------------------------------
# Character & scene high-level API
# ---------------------------------------------------------------------------

DEFAULT_CHARACTER_MODEL = "flux_dev"


def generate_character(
    name: str,
    prompt: str,
    model: str = DEFAULT_CHARACTER_MODEL,
    project: str = DEFAULT_PROJECT,
    seed: int = 42,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Generate a character reference image and persist it in the database.

    The reference image is stored under
    outputs/<project>/characters/<name>/reference.png and the character's
    prompt, seed and model are saved so scenes can reference it by name.

    Returns:
        The stored character record (dict).
    """
    print(f"Generating character '{name}' with model {model}")
    files = generate_images(
        prompt=prompt,
        model_name=model,
        seed=seed,
        task_name=f"character_{slugify(name)}",
        **kwargs,
    )

    # Move the generated image into the project character folder
    target_dir = character_dir(name, project)
    reference_path = str(target_dir / "reference.png")
    shutil.move(files[0], reference_path)
    print(f"Character reference saved to: {reference_path}")

    character_service.save_character(
        name=name,
        prompt=prompt,
        seed=seed,
        model=model,
        reference_image=reference_path,
        project=project,
    )
    return character_service.get_character(name, project)


def get_character(name: str, project: str = DEFAULT_PROJECT) -> Optional[Dict[str, Any]]:
    """Fetch a stored character reference by name.

    Returns None if the character is unknown. If the record exists but its
    reference image is missing on disk (stale record, e.g. after cleaning
    the outputs folder), the reference is regenerated from the stored
    prompt/seed/model and the database record is updated.
    """
    character = character_service.get_character(name, project)
    if character is None:
        return None
    reference = character.get("reference_image")
    if not reference or not Path(reference).is_file():
        print(
            f"Reference image missing for character '{name}' ({reference}); "
            "regenerating from stored prompt."
        )
        return generate_character(
            name,
            character["prompt"],
            model=character.get("model") or DEFAULT_CHARACTER_MODEL,
            project=project,
            seed=character.get("seed", 42),
        )
    return character


def _enrich_scene_prompt(prompt: str, project: str, use_appearance_only: bool = False) -> str:
    """Inject stored character descriptions for any character named in the
    prompt, so callers never need to resend the character prompt.
    
    Args:
        prompt: Base scene prompt
        project: Project name
        use_appearance_only: If True, use appearance-only attributes without style
                              to avoid style conflicts in multi-character scenes
    """
    from utils.token_budget import build_token_aware_scene_prompt, count_tokens
    
    characters = character_service.find_characters_in_text(prompt, project)
    
    # Use token-aware prompt building for better CLIP compatibility
    if len(characters) >= 1:
        token_aware_prompt, stats = build_token_aware_scene_prompt(
            base_prompt=prompt,
            characters=[{'name': c['name'], 'prompt': c.get('prompt', ''), 
                        'attributes': c.get('attributes')} for c in characters]
        )
        
        # Log token usage
        print(f"Token budget: {stats['total_tokens_estimated']}/{stats['max_tokens']} tokens")
        if stats['items_dropped'] > 0:
            print(f"Warning: Dropped {stats['items_dropped']} items due to token budget")
            for dropped in stats['dropped_details']:
                print(f"  - Dropped {dropped['category']} (priority {dropped['priority']}): {dropped['text']}")
        
        # If token-aware prompt is significantly better, use it
        original_tokens = count_tokens(prompt)
        if len(characters) > 1 or original_tokens > 60:
            print(f"Using token-aware prompt construction")
            return token_aware_prompt
    
    # Fallback to original enrichment logic
    enriched = prompt
    for character in characters:
        if use_appearance_only:
            from core.prompt_decomposer import extract_appearance_from_stored_prompt
            appearance = extract_appearance_from_stored_prompt(character['prompt'])
            # Prefer stored attributes if available
            if 'attributes' in character and character['attributes']:
                from core.prompt_decomposer import build_appearance_prompt
                # Determine character type from stored prompt or attributes
                char_type = "man"  # default, could infer from attributes
                appearance = build_appearance_prompt(char_type, character['attributes'])
            
            enriched += (
                f"\n{character['name']} appearance: {appearance}"
            )
        else:
            enriched += (
                f"\n{character['name']} appearance: {character['prompt']}"
            )
        print(f"Scene references known character: {character['name']}")
    return enriched


def detect_scene_style_conflicts(prompt: str, project: str = DEFAULT_PROJECT) -> list:
    """Phase 1 mitigation: detect when characters referenced in a scene
    prompt use incompatible visual styles.
    
    This is non-blocking — just returns warnings for UI display. See
    docs/scene-generation-caveats.md for details.
    """
    from core.style_conflict import detect_scene_style_conflicts as detect_conflicts
    
    conflicts = detect_conflicts(prompt, project)
    for conflict in conflicts:
        print(f"WARNING: {conflict.message()}")
    return [c.message() for c in conflicts]


@scene_logging(scene_name_arg="scene_number", prompt_arg="prompt")
def generate_scene(
    prompt: str,
    project: str = DEFAULT_PROJECT,
    scene_number: int = None,
    model: str = DEFAULT_CHARACTER_MODEL,
    seed: int = 42,
    use_style_mediation: bool = True,
    use_advanced_prompting: bool = True,
    enforce_token_budget: bool = True,
    use_asset_pipeline: bool = True,
    enable_refinement: bool = True,
    refinement_strength: float = 0.25,
    refinement_model: str = "sdxl",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Generate a scene image inside the project scene folder structure.

    Characters mentioned by name in the prompt are resolved from the
    database and their stored descriptions are appended to the prompt.

    Returns:
        dict with 'scene_number', 'image_path', 'prompt', 'seed', 'model',
        and optionally 'style_warnings'.
    """
    from utils.token_budget import count_tokens
    
    if scene_number is None:
        scene_number = next_scene_number(project)
    target_dir = scene_dir(scene_number, project)

    # Find characters in prompt
    characters = character_service.find_characters_in_text(prompt, project)
    
    # Phase 1: Detect style conflicts (non-blocking warning)
    style_warnings = detect_scene_style_conflicts(prompt, project)

    # Multi-character scenes: delegate to the asset-composition pipeline
    # (background text2img -> per-character assets -> segmentation ->
    # deterministic composition), retained as fallback for non-reference
    # models, per docs/story-engine-reference-conditioned-scene-design.md §2.
    if use_asset_pipeline and len(characters) >= 2:
        from core.scene_pipeline import generate_scene_pipeline
        print(f"INFO: {len(characters)} characters detected, delegating to asset-composition pipeline")
        result = generate_scene_pipeline(
            prompt=prompt,
            project=project,
            scene_number=scene_number,
            characters=characters,
            model=model,
            seed=seed,
            enable_refinement=enable_refinement,
            refinement_strength=refinement_strength,
            refinement_model=refinement_model,
        )
        if style_warnings:
            result.setdefault("style_warnings", style_warnings)
        return result

    # Phase 2: Style mediation - use appearance-only for conflicting styles
    use_appearance_only = False
    if use_style_mediation and len(characters) >= 2:
        from core.prompt_decomposer import should_use_scene_style_override
        use_appearance_only = should_use_scene_style_override(characters, project)
        
        if use_appearance_only and style_warnings:
            print(f"INFO: Using appearance-only mode for scene to avoid style conflicts")

    # Phase 3: Advanced prompting with per-character style tokens
    enriched_prompt = prompt
    negative_prompt = None
    model_recommendation = None
    
    if use_advanced_prompting and len(characters) >= 2:
        from core.advanced_prompting import (
            build_advanced_scene_prompt,
            AdvancedPromptingEngine
        )
        
        # Build advanced prompt with per-character tokens
        adv_prompt, neg_prompt = build_advanced_scene_prompt(
            prompt, characters, use_advanced_techniques=True
        )
        enriched_prompt = adv_prompt
        negative_prompt = neg_prompt
        
        # Recommend model based on style combination
        style_ids = []
        from core.style_conflict import detect_character_style
        for char in characters:
            style_id = detect_character_style(char.get("prompt", ""))
            if style_id:
                style_ids.append(style_id)
        
        if style_ids:
            recommended_model = AdvancedPromptingEngine.recommend_model(style_ids)
            model_recommendation = {
                "recommended": recommended_model,
                "current": model,
                "use_recommended": recommended_model != model
            }
            
            if recommended_model != model:
                print(f"INFO: Recommended model for style combination: {recommended_model} "
                      f"(currently using {model})")
        
        # Use appearance-only mode if severe conflicts detected
        if use_appearance_only:
            enriched_prompt = _enrich_scene_prompt(prompt, project, use_appearance_only)
    else:
        enriched_prompt = _enrich_scene_prompt(prompt, project, use_appearance_only)

    # Phase 4: Token budget enforcement (CLIP limit protection)
    if enforce_token_budget and len(characters) >= 1:
        from utils.token_budget import build_token_aware_scene_prompt
        
        base_enriched_prompt = enriched_prompt
        token_aware_prompt, stats = build_token_aware_scene_prompt(
            base_prompt=prompt,
            characters=[{'name': c['name'], 'prompt': c.get('prompt', ''), 
                        'attributes': c.get('attributes')} for c in characters]
        )
        
        # Only use token-aware prompt if it reduces tokens or is within budget
        original_token_count = count_tokens(base_enriched_prompt)
        new_token_count = count_tokens(token_aware_prompt)
        
        print(f"Token analysis: Original={original_token_count}, Token-aware={new_token_count}")
        
        if new_token_count < original_token_count or stats['total_tokens_estimated'] > 70:
            print(f"INFO: Using token-aware prompt (saved {original_token_count - new_token_count} tokens)")
            enriched_prompt = token_aware_prompt
            
            # Add token stats to result
            if 'token_stats' not in locals():
                pass

    # Phase 5: (removed) The old prompt-only MultiStepSceneGenerator branch
    # was replaced by the asset-composition pipeline delegation above. The
    # progressive strategy is now only used as an internal fallback inside
    # core/scene_pipeline.py.

    files = generate_images(
        prompt=enriched_prompt,
        model_name=model,
        seed=seed,
        task_name=f"scene_{scene_number}",
        **kwargs,
    )

    image_path = str(target_dir / "scene.png")
    shutil.move(files[0], image_path)
    print(f"Scene {scene_number} image saved to: {image_path}")

    # Add token statistics to result for debugging
    from utils.token_budget import count_tokens
    final_token_count = count_tokens(enriched_prompt)
    
    result = {
        "scene_number": scene_number,
        "image_path": image_path,
        "prompt": prompt,
        "enriched_prompt": enriched_prompt,
        "seed": seed,
        "model": model,
        "appearance_only_mode": use_appearance_only,
        "token_count": final_token_count,
        "token_limit": 77,
    }
    
    if style_warnings:
        result["style_warnings"] = style_warnings
    
    if negative_prompt:
        result["negative_prompt"] = negative_prompt
    
    if model_recommendation:
        result["model_recommendation"] = model_recommendation
    
    return result


def verify_character_in_scene(
    character: Dict[str, Any], scene_image_path: str
) -> Optional[bool]:
    """Check via face recognition whether a character appears in a scene.

    Args:
        character: Character record (from get_character/generate_character).
        scene_image_path: Path to the generated scene image.

    Returns:
        True/False for a definite answer, None if the check is inconclusive
        (e.g. face_recognition is not installed).
    """
    reference = character.get("reference_image")
    if not reference:
        print(f"Character '{character.get('name')}' has no reference image")
        return None
    return character_appears_in_image(reference, scene_image_path)