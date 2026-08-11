"""
Project management core for UI.
"""

import shutil
from pathlib import Path
from typing import List, Dict, Optional
from services.database.project_service import project_service
from utils.project_paths import OUTPUTS_ROOT


class ProjectManager:
    def __init__(self):
        self.service = project_service

    def list_projects(self) -> List[Dict]:
        return self.service.list_projects()

    def create_project(self, name: str, description: str = "") -> str:
        project_id = self.service.create_project(name, description)
        # Ensure output directory exists
        Path(OUTPUTS_ROOT / name.replace(" ", "_")).mkdir(parents=True, exist_ok=True)
        return project_id

    def update_project(self, project_id: str, name: Optional[str] = None, description: Optional[str] = None) -> bool:
        return self.service.update_project(project_id, name, description)

    def delete_project(self, project_id: str) -> bool:
        project = self.service.get_project(project_id)
        if not project:
            return False
        # Delete from DB
        success = self.service.delete_project(project_id)
        if success:
            # Remove output folder
            project_dir = OUTPUTS_ROOT / project["name"].replace(" ", "_")
            if project_dir.exists():
                shutil.rmtree(project_dir)
        return success

    def get_project(self, project_id: str) -> Optional[Dict]:
        return self.service.get_project(project_id)


project_manager = ProjectManager()
