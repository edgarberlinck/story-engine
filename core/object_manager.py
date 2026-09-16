"""
Object/Artifact management core for UI.

Thin manager over ObjectService following the CharacterManager pattern:
UI-facing methods take (project, ...) and delegate to the service layer.
"""

from typing import List, Dict, Optional, Any

from services.database.object_service import object_service


class ObjectManager:
    def __init__(self):
        self.obj_service = object_service

    def list_objects(self, project: str) -> List[Dict]:
        return self.obj_service.list_objects(project)

    def get_object(self, project: str, name: str) -> Optional[Dict]:
        return self.obj_service.get_object(name, project)

    def create_object(
        self,
        project: str,
        name: str,
        obj_type: str = "artifact",
        description: str = "",
        owner: Optional[str] = None,
        visual_identity: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> int:
        return self.obj_service.save_object(
            name=name,
            obj_type=obj_type,
            description=description,
            owner=owner,
            visual_identity=visual_identity,
            properties=properties,
            project=project,
        )

    def delete_object(self, project: str, name: str) -> bool:
        return self.obj_service.delete_object(name, project)

    def find_objects_in_text(self, project: str, text: str) -> List[Dict]:
        return self.obj_service.find_objects_in_text(text, project)


object_manager = ObjectManager()
