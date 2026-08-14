"""
Phase 1 mitigation: detect conflicting visual styles across characters
referenced in a scene prompt. See docs/scene-generation-caveats.md

This module provides style conflict detection without blocking generation,
serving as immediate mitigation while Phase 2 architectural refactor is planned.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.character_attributes import CHARACTER_STYLES, STYLE_FAMILIES, \
    INCOMPATIBLE_FAMILY_PAIRS, FAMILY_BRIDGES


def detect_character_style(character_prompt: str) -> Optional[str]:
    """Best-effort match of a stored character prompt back to a style_id.

    Matches on the style 'prefix' text (e.g. 'ultra realistic', 'manga
    style'), preferring the *longest* matching prefix to avoid false
    positives (e.g. 'realistic' being a substring of 'ultra realistic').
    """
    if not character_prompt:
        return None
    
    text = character_prompt.lower()
    best_match: Optional[str] = None
    best_len = 0
    for style_id, style in CHARACTER_STYLES.items():
        prefix = style["prefix"].lower()
        if prefix in text and len(prefix) > best_len:
            best_match = style_id
            best_len = len(prefix)
    return best_match


@dataclass
class StyleConflict:
    family_a: str
    family_b: str
    characters_a: List[str]
    characters_b: List[str]

    def message(self) -> str:
        names_a = ", ".join(self.characters_a)
        names_b = ", ".join(self.characters_b)
        return (
            f"Style conflict detected: {names_a} use a "
            f"'{self.family_a.replace('_', ' ')}' style while {names_b} use "
            f"'{self.family_b.replace('_', ' ')}'. Mixing these styles in one "
            f"scene commonly produces hybrid/aberrant rendering "
            f"(see docs/scene-generation-caveats.md)."
        )


def find_style_conflicts(
    characters: List[Dict[str, str]]
) -> List[StyleConflict]:
    """Given character records (each needs 'name' and 'prompt'), detect
    incompatible style-family combinations.

    Returns an empty list when there's nothing to warn about (0 or 1
    characters, styles undetected, or families compatible/ambiguous).
    """
    if len(characters) < 2:
        return []

    by_family: Dict[str, List[str]] = {}
    for c in characters:
        style_id = detect_character_style(c.get("prompt", ""))
        if style_id is None:
            continue
        family = STYLE_FAMILIES.get(style_id)
        if not family or family == "ambiguous":
            continue
        by_family.setdefault(family, []).append(c["name"])

    families = list(by_family.keys())
    conflicts: List[StyleConflict] = []
    for i in range(len(families)):
        for j in range(i + 1, len(families)):
            pair = frozenset({families[i], families[j]})
            if pair in FAMILY_BRIDGES:
                continue
            if pair in INCOMPATIBLE_FAMILY_PAIRS:
                conflicts.append(
                    StyleConflict(
                        family_a=families[i],
                        family_b=families[j],
                        characters_a=by_family[families[i]],
                        characters_b=by_family[families[j]],
                    )
                )
    return conflicts


def detect_scene_style_conflicts(prompt: str, project: str = "test_project") -> List[StyleConflict]:
    """Detect style conflicts for characters referenced in a scene prompt.
    
    Args:
        prompt: Scene prompt text
        project: Project name
        
    Returns:
        List of StyleConflict objects
    """
    from services.database.character_service import character_service
    
    characters = character_service.find_characters_in_text(prompt, project)
    return find_style_conflicts(characters)
