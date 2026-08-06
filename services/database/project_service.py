"""
Project Service for managing projects in the story engine.
This service handles creation, updating, listing, searching, and retrieval of projects.
"""

from datetime import datetime
from typing import List, Dict, Optional, Any
from services.database.database_service import db_service

class ProjectService:
    def __init__(self):
        self.db = db_service
    
    def create_project(self, name: str, description: str = None) -> str:
        """
        Create a new project
        
        Args:
            name (str): The name of the project
            description (str, optional): A brief description of the project
            
        Returns:
            str: The ID of the created project
        """
        return self.db.create_project(name, description)
    
    def update_project(self, project_id: str, name: str = None, description: str = None) -> bool:
        """
        Update an existing project
        
        Args:
            project_id (str): The ID of the project to update
            name (str, optional): The new name for the project
            description (str, optional): The new description for the project
            
        Returns:
            bool: True if the project was updated successfully, False otherwise
        """
        return self.db.update_project(project_id, name, description)
    
    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a project by its ID
        
        Args:
            project_id (str): The ID of the project to retrieve
            
        Returns:
            dict or None: Project data if found, None otherwise
        """
        return self.db.get_project(project_id)
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """
        List all projects
        
        Returns:
            list: A list of all projects, sorted by creation date (newest first)
        """
        return self.db.list_projects()
    
    def search_projects(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for projects by name or description
        
        Args:
            query (str): The search query
            
        Returns:
            list: A list of projects matching the search query
        """
        return self.db.search_projects(query)

# Create a singleton instance
project_service = ProjectService()