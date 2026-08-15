"""
Advanced prompting techniques for multi-character scene generation.

Phase 3 enhancements:
1. Per-character style tokens with negative prompts
2. Style blending support
3. Model selection logic based on style combinations
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from core.character_attributes import CHARACTER_STYLES, STYLE_FAMILIES


@dataclass
class CharacterStyleToken:
    """Represents per-character style token for advanced prompting."""
    name: str
    style_id: Optional[str]
    appearance_prompt: str
    negative_prompt: str = ""
    weight: float = 1.0


@dataclass
class StyleBlend:
    """Represents a blended style configuration."""
    primary_style: str
    secondary_style: str
    blend_ratio: float  # 0.0 to 1.0, how much secondary blends in


class AdvancedPromptingEngine:
    """Advanced prompting techniques for handling multi-character scenes."""
    
    # Style compatibility weights for blending
    STYLE_COMPATIBILITY_WEIGHTS = {
        ("realistic", "painterly"): 0.7,
        ("realistic", "semi_realistic"): 0.9,
        ("cinematic", "photorealistic"): 0.8,
        ("anime", "manga"): 0.95,
        ("comic_book", "cartoon"): 0.85,
        ("3d_animation", "pixar_like"): 0.9,
    }
    
    # Model recommendations for style combinations
    MODEL_RECOMMENDATIONS = {
        "realistic": ["flux_dev"],
        "anime_manga": ["sdxl", "flux_dev"],
        "comic_cartoon": ["sdxl"],
        "3d_stylized": ["sdxl", "flux_dev"],
        "painterly": ["sdxl"],
        "sketch": ["sdxl"],
    }
    
    @classmethod
    def build_per_character_tokens(cls, characters: List[Dict]) -> List[CharacterStyleToken]:
        """Build per-character style tokens with negative prompts.
        
        This implements advanced prompting where each character gets
        specific style control through weighted tokens and negative prompts.
        """
        tokens = []
        
        from core.prompt_decomposer import decompose_character_prompt
        
        for char in characters:
            name = char.get("name", "")
            prompt = char.get("prompt", "")
            
            # Decompose to get style and appearance
            style_prefix, style_modifiers, appearance = decompose_character_prompt(prompt)
            
            # Detect style ID
            from core.style_conflict import detect_character_style
            style_id = detect_character_style(prompt)
            
            # Build negative prompt to prevent style bleeding
            negative_prompt = cls._build_negative_prompt(style_id, appearance)
            
            token = CharacterStyleToken(
                name=name,
                style_id=style_id,
                appearance_prompt=appearance or prompt,
                negative_prompt=negative_prompt,
                weight=1.0
            )
            tokens.append(token)
        
        return tokens
    
    @classmethod
    def _build_negative_prompt(cls, style_id: Optional[str], appearance: str) -> str:
        """Build negative prompt to prevent style bleeding between characters."""
        if not style_id:
            return ""
        
        style_info = CHARACTER_STYLES.get(style_id, {})
        style_prefix = style_info.get("prefix", "").lower()
        
        # Build negative prompts based on style family
        family = STYLE_FAMILIES.get(style_id, "")
        
        negative_parts = []
        
        if family == "realistic":
            negative_parts.extend([
                "anime style", "cartoon", "comic book", "manga style",
                "stylized proportions", "exaggerated features"
            ])
        elif family in ["anime_manga", "comic_cartoon"]:
            negative_parts.extend([
                "photorealistic", "ultra realistic", "realistic skin texture",
                "natural lighting", "photographic realism"
            ])
        elif family == "3d_stylized":
            negative_parts.extend([
                "photorealistic", "anime style", "manga style",
                "painterly", "sketchy"
            ])
        
        return ", ".join(negative_parts)
    
    @classmethod
    def build_weighted_scene_prompt(
        cls,
        base_prompt: str,
        tokens: List[CharacterStyleToken],
        scene_style: Optional[str] = None
    ) -> Tuple[str, str]:
        """Build weighted scene prompt with per-character style control.
        
        Returns tuple of (positive_prompt, negative_prompt)
        """
        positive_parts = [base_prompt]
        negative_parts = []
        
        # Add character-specific prompts with weights
        for token in tokens:
            char_section = f"{token.name}: {token.appearance_prompt}"
            if token.weight < 1.0:
                char_section += f" (weight: {token.weight})"
            positive_parts.append(char_section)
            
            if token.negative_prompt:
                negative_parts.extend(token.negative_prompt.split(", "))
        
        # Add scene-level style if provided
        if scene_style and scene_style in CHARACTER_STYLES:
            style_info = CHARACTER_STYLES[scene_style]
            positive_parts.insert(0, f"{style_info['prefix']}, {style_info['modifiers']}")
        
        positive_prompt = "\n".join(positive_parts)
        negative_prompt = ", ".join(set(negative_parts))
        
        return positive_prompt, negative_prompt
    
    @classmethod
    def blend_styles(
        cls,
        style_id_1: str,
        style_id_2: str,
        ratio: float = 0.5
    ) -> StyleBlend:
        """Create a blended style configuration.
        
        Args:
            style_id_1: Primary style
            style_id_2: Secondary style
            ratio: Blend ratio (0.0 = only primary, 1.0 = only secondary)
        
        Returns:
            StyleBlend configuration
        """
        return StyleBlend(
            primary_style=style_id_1,
            secondary_style=style_id_2,
            blend_ratio=max(0.0, min(1.0, ratio))
        )
    
    @classmethod
    def build_blended_prompt(
        cls,
        base_attributes: Dict[str, str],
        blend: StyleBlend
    ) -> str:
        """Build prompt with blended styles.
        
        Combines two styles with controlled interpolation.
        """
        style1 = CHARACTER_STYLES.get(blend.primary_style, {})
        style2 = CHARACTER_STYLES.get(blend.secondary_style, {})
        
        # Interpolate style prefixes
        prefix1 = style1.get("prefix", "")
        prefix2 = style2.get("prefix", "")
        
        if blend.blend_ratio < 0.5:
            primary_prefix = prefix1
            secondary_prefix = prefix2
            weight_primary = 1.0 - blend.blend_ratio * 2
            weight_secondary = blend.blend_ratio * 2
        else:
            primary_prefix = prefix2
            secondary_prefix = prefix1
            weight_primary = (1.0 - blend.blend_ratio) * 2
            weight_secondary = blend.blend_ratio
        
        # Build blended style description
        if weight_secondary > 0.01:
            style_desc = f"{primary_prefix} with hints of {secondary_prefix}"
        else:
            style_desc = primary_prefix
        
        parts = [style_desc]
        
        # Add attributes
        from core.character_attributes import build_character_prompt
        # Build appearance without style
        attrs_copy = base_attributes.copy()
        if "style" in attrs_copy:
            del attrs_copy["style"]
        
        appearance = f"{base_attributes.get('gender', 'person')}, "
        if base_attributes.get("age"):
            appearance += f"{base_attributes['age']} "
        if base_attributes.get("body_type"):
            appearance += f"{base_attributes['body_type']} build "
        
        parts.append(appearance.strip())
        
        # Add style modifiers with blending
        modifiers1 = style1.get("modifiers", "")
        modifiers2 = style2.get("modifiers", "")
        
        if modifiers1 and modifiers2:
            blended_modifiers = f"{modifiers1}, {modifiers2}"
        elif modifiers1:
            blended_modifiers = modifiers1
        elif modifiers2:
            blended_modifiers = modifiers2
        
        if blended_modifiers:
            parts.append(blended_modifiers)
        
        return ", ".join(parts)
    
    @classmethod
    def recommend_model(cls, style_ids: List[str]) -> str:
        """Recommend best model for given style combination.
        
        Analyzes style families and recommends optimal diffusion model.
        """
        if not style_ids:
            return "flux_dev"
        
        # Get unique families
        families = set()
        for style_id in style_ids:
            family = STYLE_FAMILIES.get(style_id, "")
            if family:
                families.add(family)
        
        # Check compatibility and recommend
        if len(families) == 1:
            family = list(families)[0]
            models = cls.MODEL_RECOMMENDATIONS.get(family, ["flux_dev"])
            return models[0]
        
        # Multiple families - choose based on primary style
        # Prefer flux_dev for mixed realistic + artistic combos
        if "realistic" in families or "anime_manga" in families:
            return "flux_dev"
        
        return "sdxl"
    
    @classmethod
    def analyze_style_compatibility(cls, style_ids: List[str]) -> Dict:
        """Analyze compatibility between multiple styles.
        
        Returns analysis with compatibility scores and recommendations.
        """
        if len(style_ids) < 2:
            return {
                "compatible": True,
                "score": 1.0,
                "recommendation": "Single style - no conflicts"
            }
        
        families = []
        for style_id in style_ids:
            family = STYLE_FAMILIES.get(style_id, "")
            if family and family != "ambiguous":
                families.append(family)
        
        # Calculate compatibility
        scores = []
        pairs_checked = 0
        
        # Check if all styles are from the same family
        if len(set(families)) == 1:
            return {
                "compatible": True,
                "score": 1.0,
                "families": families,
                "recommendation": "All styles from same family - excellent compatibility"
            }
        
        for i in range(len(families)):
            for j in range(i + 1, len(families)):
                pair = (families[i], families[j])
                reverse_pair = (families[j], families[i])
                
                weight = cls.STYLE_COMPATIBILITY_WEIGHTS.get(pair) or \
                        cls.STYLE_COMPATIBILITY_WEIGHTS.get(reverse_pair, 0.3)
                scores.append(weight)
                pairs_checked += 1
        
        avg_score = sum(scores) / len(scores) if scores else 1.0
        
        return {
            "compatible": avg_score > 0.5,
            "score": avg_score,
            "families": families,
            "recommendation": cls._get_compatibility_recommendation(avg_score)
        }
    
    @classmethod
    def _get_compatibility_recommendation(cls, score: float) -> str:
        """Get recommendation based on compatibility score."""
        if score >= 0.8:
            return "Excellent compatibility - styles will blend well"
        elif score >= 0.6:
            return "Good compatibility - minor adjustments recommended"
        elif score >= 0.4:
            return "Moderate compatibility - use appearance-only mode"
        else:
            return "Poor compatibility - strongly consider style harmonization"


def build_advanced_scene_prompt(
    base_prompt: str,
    characters: List[Dict],
    scene_style: Optional[str] = None,
    use_advanced_techniques: bool = True
) -> Tuple[str, str]:
    """Build advanced scene prompt with per-character style control.
    
    This is the main entry point for Phase 3 enhanced prompting.
    """
    if not use_advanced_techniques or len(characters) < 2:
        # Fallback to standard enrichment
        from generators.image_engine import _enrich_scene_prompt
        enriched = _enrich_scene_prompt(base_prompt, "", "")
        return enriched, ""
    
    tokens = AdvancedPromptingEngine.build_per_character_tokens(characters)
    positive_prompt, negative_prompt = AdvancedPromptingEngine.build_weighted_scene_prompt(
        base_prompt, tokens, scene_style
    )
    
    return positive_prompt, negative_prompt
