"""Project-scoped output path helpers.

Enforces the canonical output layout:

    outputs/<project>/characters/<character_name>/...
    outputs/<project>/scenes/scene_<n>/<scene image>
    outputs/<project>/scenes/scene_<n>/out/<videos + metrics>
"""

import os
import re
from pathlib import Path

OUTPUTS_ROOT = Path("outputs")
DEFAULT_PROJECT = "test_project"


def slugify(name: str) -> str:
    """Turn an arbitrary name into a safe directory/file name."""
    slug = re.sub(r"[^\w\s-]", "", name.strip())
    slug = re.sub(r"[-\s]+", "_", slug)
    return slug or "unnamed"


def project_dir(project: str = DEFAULT_PROJECT) -> Path:
    path = OUTPUTS_ROOT / slugify(project)
    path.mkdir(parents=True, exist_ok=True)
    return path


def character_dir(character_name: str, project: str = DEFAULT_PROJECT) -> Path:
    path = project_dir(project) / "characters" / slugify(character_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def scenes_dir(project: str = DEFAULT_PROJECT) -> Path:
    path = project_dir(project) / "scenes"
    path.mkdir(parents=True, exist_ok=True)
    return path


def next_scene_number(project: str = DEFAULT_PROJECT) -> int:
    """Return the next available scene number (scene_1, scene_2, ...)."""
    root = scenes_dir(project)
    numbers = []
    for entry in root.iterdir():
        match = re.fullmatch(r"scene_(\d+)", entry.name)
        if match and entry.is_dir():
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def scene_dir(scene_number: int = None, project: str = DEFAULT_PROJECT) -> Path:
    """Get (and create) the directory for a scene.

    If scene_number is None, allocates the next available one.
    """
    if scene_number is None:
        scene_number = next_scene_number(project)
    path = scenes_dir(project) / f"scene_{scene_number}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def scene_out_dir(scene_number: int, project: str = DEFAULT_PROJECT) -> Path:
    """Directory where generated videos for a scene are stored."""
    path = scene_dir(scene_number, project) / "out"
    path.mkdir(parents=True, exist_ok=True)
    return path
