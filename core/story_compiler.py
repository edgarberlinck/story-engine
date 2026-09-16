"""Story compiler: turn written chapters (story markup) into audio scenes.

This is the bridge between the writing layer (ported from writing-tools:
chapters with per-language content) and the audiobook pipeline: everything
you write can be turned into audio (plan §5).

Compiling a chapter:
1. Parses its story markup (XML-like and/or bracket shorthand).
2. Converts each parsed scene into an AudioSceneRepresentation.
3. Persists the representations (reusing the chapter's previously assigned
   scene numbers when recompiling, so a rewrite updates scenes in place
   instead of duplicating them).
4. Records the chapter -> scene-numbers mapping for traceability.

The result feeds the existing scene editor, LLM assistance, and per-line
audio preview untouched.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from core.story_markup import parse_story_markup, scenes_to_representations, ParseIssue
from services.audio_scene_service import audio_scene_service
from services.database.manuscript_service import manuscript_service


@dataclass
class CompileResult:
    scene_numbers: List[int] = field(default_factory=list)
    issues: List[ParseIssue] = field(default_factory=list)
    error: Optional[str] = None


def _next_scene_number(project: str) -> int:
    """First scene number after everything already in the project."""
    existing = audio_scene_service.list_representations(project)
    if not existing:
        return 1
    return max(e["scene_number"] for e in existing) + 1


def compile_chapter(
    project: str,
    chapter_id: int,
    locale: str = "en",
) -> CompileResult:
    """Compile one chapter (in one language) into persisted audio scenes."""
    chapter = manuscript_service.get_chapter(chapter_id)
    if not chapter:
        return CompileResult(error=f"Chapter {chapter_id} not found")

    content = manuscript_service.get_content(chapter_id, locale)
    if not content.strip():
        return CompileResult(error=f"Chapter has no {locale} content to compile")

    scenes, issues = parse_story_markup(content)
    if not scenes:
        return CompileResult(issues=issues, error="No scenes found in the markup")

    # Reuse this chapter's previous scene numbers when recompiling so a
    # rewrite updates in place; extend with fresh numbers if it grew.
    previous = chapter["compiled_scenes"].get(locale, [])
    numbers: List[int] = list(previous[: len(scenes)])
    next_free = _next_scene_number(project)
    while len(numbers) < len(scenes):
        numbers.append(next_free)
        next_free += 1

    reps = scenes_to_representations(
        scenes,
        scene_id_prefix=f"ch{chapter_id}_{locale}",
        start_number=1,
    )
    for rep, number in zip(reps, numbers):
        audio_scene_service.save_representation(project, rep.scene_id, number, rep)

    manuscript_service.set_compiled_scenes(chapter_id, locale, numbers)
    return CompileResult(scene_numbers=numbers, issues=issues)


def compile_project(project: str, locale: str = "en") -> List[CompileResult]:
    """Compile every chapter of a project, in order."""
    results = []
    for chapter in manuscript_service.list_chapters(project):
        results.append(compile_chapter(project, chapter["id"], locale))
    return results
