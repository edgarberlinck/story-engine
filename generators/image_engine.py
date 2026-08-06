import enum
from typing import Optional, Dict, Any, TYPE_CHECKING

# Import existing functionality
from generators.image_generator import generate_images, AVAILABLE_DIFFUSION_MODELS
from models import DIFFUSION_MODELS, MODEL_PATHS

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