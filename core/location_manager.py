"""
Location management core for UI.

Thin manager over LocationService following the CharacterManager pattern:
UI-facing methods take (project, ...) and delegate to the service layer.
"""

from typing import List, Dict, Optional, Any

from services.database.location_service import location_service


class LocationManager:
    def __init__(self):
        self.loc_service = location_service

    def list_locations(self, project: str) -> List[Dict]:
        return self.loc_service.list_locations(project)

    def get_location(self, project: str, name: str) -> Optional[Dict]:
        return self.loc_service.get_location(name, project)

    def create_location(
        self,
        project: str,
        name: str,
        loc_type: str = "place",
        description: str = "",
        parent_location: Optional[str] = None,
        visual_identity: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> int:
        return self.loc_service.save_location(
            name=name,
            loc_type=loc_type,
            description=description,
            parent_location=parent_location,
            visual_identity=visual_identity,
            properties=properties,
            project=project,
        )

    def delete_location(self, project: str, name: str) -> bool:
        return self.loc_service.delete_location(name, project)

    def update_state(
        self,
        project: str,
        name: str,
        new_state: str,
        objects_present: Optional[List[str]] = None,
        characters_associated: Optional[List[str]] = None,
    ) -> bool:
        return self.loc_service.update_state(
            name, project, new_state, objects_present, characters_associated
        )

    def find_locations_in_text(self, project: str, text: str) -> List[Dict]:
        return self.loc_service.find_locations_in_text(text, project)


location_manager = LocationManager()
