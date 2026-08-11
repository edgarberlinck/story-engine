"""
Scene management core for UI.
"""

import re
from typing import List, Dict, Optional
from services.database.scene_service import scene_service
from generators.image_engine import generate_scene
from utils.project_paths import DEFAULT_PROJECT, OUTPUTS_ROOT


class SceneManager:
    def __init__(self):
        self.service = scene_service

    def list_scenes(self, project: str) -> List[Dict]:
        self._sync_from_filesystem(project)
        return self.service.list_scenes(project)

    def _sync_from_filesystem(self, project: str) -> None:
        """Import scenes that exist on disk but are missing from the DB.

        Older runs wrote images to outputs/<project>/scenes/scene_<n>/
        without recording them in SQLite.
        """
        scenes_root = OUTPUTS_ROOT / project / "scenes"
        if not scenes_root.is_dir():
            return

        known = {s["scene_number"] for s in self.service.list_scenes(project)}

        for scene_path in scenes_root.iterdir():
            match = re.fullmatch(r"scene_(\d+)", scene_path.name)
            if not match or not scene_path.is_dir():
                continue
            number = int(match.group(1))
            if number in known:
                continue
            # Find the scene image (canonical name is scene.png)
            image = scene_path / "scene.png"
            if not image.is_file():
                pngs = sorted(scene_path.glob("*.png"))
                if not pngs:
                    continue
                image = pngs[0]
            self.service.save_scene(
                project=project,
                scene_number=number,
                prompt="(imported from filesystem)",
                image_path=str(image),
            )

    def create_scene(self, project: str, prompt: str,
                     scene_number: Optional[int] = None,
                     model: str = "flux_dev",
                     seed: int = 42) -> Dict:
        result = generate_scene(
            prompt=prompt,
            project=project,
            scene_number=scene_number,
            model=model,
            seed=seed
        )
        # Save metadata
        saved = self.service.save_scene(
            project=project,
            scene_number=result["scene_number"],
            prompt=result["prompt"],
            image_path=result["image_path"],
            seed=seed,
            model=model
        )
        return saved

    def get_scene(self, project: str, scene_number: int) -> Optional[Dict]:
        return self.service.get_scene(project, scene_number)


scene_manager = SceneManager()
