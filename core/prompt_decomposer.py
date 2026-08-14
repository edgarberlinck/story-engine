"""
Prompt decomposition for separating appearance attributes from style in character prompts.

This module enables Phase 2 architectural refactor: extract appearance-only descriptions
from stored prompts for use in multi-character scenes, preventing style leakage.
"""

import re
from typing import Dict, Tuple, Optional

from core.character_attributes import CHARACTER_STYLES


def decompose_character_prompt(prompt: str) -> Tuple[Optional[str], Optional[str], str]:
    """Decompose a character prompt into style prefix, style modifiers, and appearance.
    
    Args:
        prompt: Full character prompt with style embedded
        
    Returns:
        Tuple of (style_prefix, style_modifiers, appearance_description)
        If style cannot be detected, returns (None, None, original_prompt)
    """
    if not prompt:
        return None, None, ""
    
    text = prompt.strip()
    
    # Find style prefix at the beginning
    best_style_id = None
    best_prefix_len = 0
    style_prefix = None
    
    for style_id, style in CHARACTER_STYLES.items():
        prefix = style["prefix"].lower()
        if text.lower().startswith(prefix):
            if len(prefix) > best_prefix_len:
                best_style_id = style_id
                best_prefix_len = len(prefix)
                style_prefix = style["prefix"]
    
    if not best_style_id:
        # Try to find style prefix anywhere in prompt
        for style_id, style in CHARACTER_STYLES.items():
            prefix = style["prefix"].lower()
            pattern = r'\b' + re.escape(prefix) + r'\b'
            if re.search(pattern, text.lower()):
                if len(prefix) > best_prefix_len:
                    best_style_id = style_id
                    best_prefix_len = len(prefix)
                    style_prefix = style["prefix"]
    
    if not best_style_id:
        # No style detected, return prompt as appearance-only
        return None, None, text
    
    style_info = CHARACTER_STYLES[best_style_id]
    style_modifiers = style_info["modifiers"]
    
    # Extract appearance by removing style elements
    appearance = text
    
    # Remove style prefix from start
    if style_prefix:
        pattern = r'^' + re.escape(style_prefix) + r'\s*'
        appearance = re.sub(pattern, '', appearance, flags=re.IGNORECASE)
    
    # Remove style modifiers from end
    if style_modifiers:
        modifier_parts = [m.strip() for m in style_modifiers.split(',')]
        for part in modifier_parts:
            pattern = r',?\s*' + re.escape(part) + r'\s*,?'
            appearance = re.sub(pattern, '', appearance, flags=re.IGNORECASE)
    
    # Clean up appearance
    appearance = re.sub(r'[, ]+,', ',', appearance)
    appearance = re.sub(r',\s*$', '', appearance)
    appearance = re.sub(r'^\s*,', '', appearance)
    appearance = appearance.strip(' ,')
    
    return style_prefix, style_modifiers, appearance


def build_appearance_prompt(char_type: str, attributes: Dict[str, str], custom_description: str = "") -> str:
    """Build an appearance-only prompt without style modifiers.
    
    This is used for scene enrichment where we want only physical/appearance
    attributes, without style instructions that could conflict.
    """
    from core.character_attributes import build_character_prompt, _SUBJECT_KEYS, get_categories
    
    # Build subject line
    from core.character_attributes import _build_subject
    subject = _build_subject(char_type, attributes)
    
    parts = [subject]
    
    # Add physical attributes only (skip style attribute)
    for category in get_categories(char_type):
        for attr in category["attributes"]:
            if attr["key"] in _SUBJECT_KEYS or attr["key"] == "style":
                continue
            value = attributes.get(attr["key"], "")
            if not value or value in attr["skip"]:
                continue
            parts.append(attr["template"].format(value.lower()))
    
    # Add custom description
    custom_description = custom_description.strip()
    if custom_description:
        parts.append(custom_description)
    
    return ", ".join(parts)


def extract_appearance_from_stored_prompt(prompt: str, char_type: str = "man") -> str:
    """Extract appearance-only description from a stored character prompt.
    
    This is the key function for Phase 2: take an existing stored prompt
    with embedded style and return only the appearance attributes.
    
    Args:
        prompt: Stored character prompt with style embedded
        char_type: Character type for context (man/woman/animal)
        
    Returns:
        Appearance-only description without style modifiers
    """
    _, _, appearance = decompose_character_prompt(prompt)
    
    if appearance and len(appearance) > 10:
        return appearance
    
    # Fallback: return prompt as-is if we can't decompose it well
    return prompt


def should_use_scene_style_override(scene_characters: list, project: str = "test_project") -> bool:
    """Determine if we should use scene-level style override for multi-character scenes.
    
    Returns True if characters have conflicting styles that would benefit from
    using appearance-only descriptions.
    """
    from core.style_conflict import find_style_conflicts
    
    conflicts = find_style_conflicts(scene_characters)
    return len(conflicts) > 0
