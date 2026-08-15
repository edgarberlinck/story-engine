"""
Scene generation logging utilities.

Creates logs/scenes/<scene name>/log.txt and captures all stdout/stderr
during scene generation, plus original and generated prompts.
"""

from __future__ import annotations

import inspect
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Any, Optional


class _Tee:
    """Write to multiple streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self) -> None:
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


class SceneLogger:
    """
    Context manager that captures all prints for a scene generation run.

    Log path: logs/scenes/<scene_name>/log.txt
    """

    def __init__(self, scene_number: int, project: Optional[str] = None):
        # Build directory name. Keep it simple and deterministic.
        root = Path("logs") / "scenes"
        if project:
            safe_proj = self._slugify(project)
            log_dir = root / safe_proj / f"scene_{scene_number}"
        else:
            log_dir = root / f"scene_{scene_number}"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path: Path = log_dir / "log.txt"
        self._file = open(self.log_path, "a", encoding="utf-8")
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self._started_at = datetime.now()
        self._project = project

    @staticmethod
    def _slugify(name: str) -> str:
        import re
        slug = re.sub(r"[^\w\s-]", "", name.strip())
        slug = re.sub(r"[-\s]+", "_", slug)
        return slug or "unnamed"

    def __enter__(self) -> "SceneLogger":
        # Header for this run
        header = (
            "\n"
            + "=" * 80
            + f"\nScene generation started: {self._started_at.isoformat()}\n"
            + "=" * 80
            + "\n"
        )
        self._file.write(header)
        self._file.flush()

        sys.stdout = _Tee(self._orig_stdout, self._file)
        sys.stderr = _Tee(self._orig_stderr, self._file)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore stdout/stderr
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

        finished_at = datetime.now()
        footer = (
            "\n"
            + "=" * 80
            + f"\nScene generation finished: {finished_at.isoformat()}\n"
        )
        if exc_type:
            footer += f"ERROR: {exc_type.__name__}: {exc_val}\n"
        footer += "=" * 80 + "\n"
        self._file.write(footer)
        self._file.flush()
        self._file.close()

    # Helper methods to log prompts without going through stdout tee
    def log_section(self, title: str, content: str) -> None:
        sep = "\n" + "-" * 80 + f"\n{title}\n" + "-" * 80 + "\n"
        self._file.write(sep)
        self._file.write(content)
        if not content.endswith("\n"):
            self._file.write("\n")
        self._file.write("\n")
        self._file.flush()


def scene_logging(*, scene_name_arg: str, prompt_arg: str) -> Callable:
    """Decorator to log scene generation runs with all prints, prompts, and metadata.
    
    Args:
        scene_name_arg: Name of the argument that contains the scene name or number
        prompt_arg: Name of the argument that contains the prompt to be logged
        
    Returns:
        Decorator function that wraps the target function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            # Get arguments by name
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            scene_number = bound.arguments.get(scene_name_arg)
            original_prompt = bound.arguments.get(prompt_arg)
            project = bound.arguments.get('project', None)
            
            # Resolve auto-assigned scene numbers up front so the log lands in
            # logs/scenes/<project>/scene_<N>/ instead of scene_pending/.
            if scene_number is None:
                try:
                    from utils.project_paths import next_scene_number
                    scene_number = next_scene_number(project) if project else next_scene_number()
                    # Pass the resolved number through to the wrapped function
                    # so it uses the same scene directory.
                    bound.arguments[scene_name_arg] = scene_number
                except Exception:
                    scene_number = None
            
            temp_scene_num = scene_number if scene_number is not None else "pending"
            
            # Create logger and context for capturing prints
            with SceneLogger(temp_scene_num, project) as logger:
                # Log original prompt before generation
                if original_prompt:
                    logger.log_section("ORIGINAL PROMPT", str(original_prompt))
                
                try:
                    result = func(*bound.args, **bound.kwargs)
                    
                    # After generation, get actual scene number if it changed
                    final_scene_number = None
                    if isinstance(result, dict):
                        final_scene_number = result.get('scene_number')
                        
                        # Log generated prompts from result
                        if 'strategy' in result:
                            logger.log_section("STRATEGY", str(result['strategy']))
                        if 'enriched_prompt' in result:
                            logger.log_section("GENERATED ENRICHED PROMPT", str(result['enriched_prompt']))
                        if 'prompt' in result and result['prompt'] != original_prompt:
                            logger.log_section("FINAL PROMPT USED", str(result['prompt']))
                        if 'negative_prompt' in result:
                            logger.log_section("NEGATIVE PROMPT", str(result['negative_prompt']))
                    
                    # If scene number was auto-assigned, move log to correct location
                    if final_scene_number and temp_scene_num == "pending":
                        # Re-log with correct scene number (simplified - just continue)
                        pass
                    
                    return result
                    
                except Exception as e:
                    logger.log_section("GENERATION ERROR", f"{type(e).__name__}: {e}")
                    raise
                    
        return wrapper
    return decorator
