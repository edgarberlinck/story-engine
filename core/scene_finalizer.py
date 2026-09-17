"""Finalize one scene into a persistent, fully mixed WAV.

Reuses the audiobook exporter's scene renderer (speech layout + background
beds with ducking) but writes the mix to a stable location:

    outputs/<project>/scenes/scene_<n>/audio/scene_final.wav

Fragment WAVs are content-hash cached, so previously accepted/pre-generated
fragments are reused and only missing ones are generated.
"""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional

from core import audiobook_exporter as _exporter
from services.audio_scene_service import audio_scene_service
from utils.project_paths import scene_dir

logger = logging.getLogger(__name__)

FINAL_FILENAME = "scene_final.wav"


def final_scene_path(project: str, scene_number: int) -> Path:
    """Where the finalized scene mix lives."""
    return scene_dir(scene_number, project) / "audio" / FINAL_FILENAME


def finalize_scene(
    project: str,
    scene_number: int,
    representation=None,
    progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """Render one scene (speech + ducked sound/music beds) to scene_final.wav.

    Uses cached fragment WAVs where present and generates any missing ones.
    Raises on missing representation, missing ffmpeg, or an empty scene.
    """
    report = progress or (lambda msg: logger.info("%s", msg))

    rep = representation or audio_scene_service.get_representation(
        project, scene_number
    )
    if rep is None:
        raise ValueError(
            f"No audio scene representation for scene {scene_number} "
            f"in project '{project}'"
        )

    _exporter._ffmpeg()  # raise early if ffmpeg is missing

    out_path = final_scene_path(project, scene_number)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report(f"Finalizing scene {scene_number}: {rep.title}")
    with tempfile.TemporaryDirectory(prefix="scene_final_") as tmp:
        result = _exporter.ExportResult()
        mixed = _exporter._render_scene(
            project,
            scene_number,
            rep,
            Path(tmp),
            include_cues=True,
            result=result,
            report=report,
        )
        if mixed is None:
            raise RuntimeError(f"Scene {scene_number} has no renderable segments")
        shutil.copy(str(mixed), str(out_path))

    report(f"Finalized: {out_path}")
    return out_path
