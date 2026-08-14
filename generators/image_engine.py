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


def _enrich_scene_prompt(prompt: str, project: str) -> str:
    """Inject stored character descriptions for any character named in the
    prompt, so callers never need to resend the character prompt."""
    enriched = prompt
    for character in character_service.find_characters_in_text(prompt, project):
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


def generate_scene(
    prompt: str,
    project: str = DEFAULT_PROJECT,
    scene_number: int = None,
    model: str = DEFAULT_CHARACTER_MODEL,
    seed: int = 42,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Generate a scene image inside the project scene folder structure.

    Characters mentioned by name in the prompt are resolved from the
    database and their stored descriptions are appended to the prompt.

    Returns:
        dict with 'scene_number', 'image_path', 'prompt', 'seed', 'model',
        and optionally 'style_warnings'.
    """
    if scene_number is None:
        scene_number = next_scene_number(project)
    target_dir = scene_dir(scene_number, project)

    # Phase 1: Detect style conflicts (non-blocking warning)
    style_warnings = detect_scene_style_conflicts(prompt, project)

    enriched_prompt = _enrich_scene_prompt(prompt, project)
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

    result = {
        "scene_number": scene_number,
        "image_path": image_path,
        "prompt": prompt,
        "enriched_prompt": enriched_prompt,
        "seed": seed,
        "model": model,
    }
    
    if style_warnings:
        result["style_warnings"] = style_warnings
    
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
