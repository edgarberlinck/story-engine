import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Any


class DatabaseService:
    def __init__(self, db_path: str = "story_engine.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the database with required tables"""
        from services.database.migrations import migrate_database
        migrate_database(self.db_path)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create projects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def create_project(self, name: str, description: str = None) -> str:
        """Create a new project"""
        project_id = f"project_{int(datetime.now().timestamp())}"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO projects (id, name, description, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """,
            (project_id, name, description, datetime.now(), datetime.now()),
        )

        conn.commit()
        conn.close()

        return project_id

    def update_project(
        self, project_id: str, name: str = None, description: str = None
    ) -> bool:
        """Update an existing project"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get current values
        cursor.execute(
            "SELECT name, description FROM projects WHERE id = ?", (project_id,)
        )
        result = cursor.fetchone()

        if not result:
            conn.close()
            return False

        current_name, current_description = result
        updated_name = name if name is not None else current_name
        updated_description = (
            description if description is not None else current_description
        )

        cursor.execute(
            """
            UPDATE projects 
            SET name = ?, description = ?, updated_at = ?
            WHERE id = ?
        """,
            (updated_name, updated_description, datetime.now(), project_id),
        )

        conn.commit()
        rows_affected = conn.total_changes
        conn.close()

        return rows_affected > 0

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get a project by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        result = cursor.fetchone()

        conn.close()

        if result:
            columns = [description[0] for description in cursor.description]
            return dict(zip(columns, result))

        return None

    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM projects ORDER BY created_at DESC")
        results = cursor.fetchall()

        columns = [description[0] for description in cursor.description]
        projects = [dict(zip(columns, row)) for row in results]

        conn.close()

        return projects

    def search_projects(self, query: str) -> List[Dict[str, Any]]:
        """Search projects by name or description"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM projects 
            WHERE name LIKE ? OR description LIKE ?
            ORDER BY created_at DESC
        """,
            (f"%{query}%", f"%{query}%"),
        )

        results = cursor.fetchall()

        columns = [description[0] for description in cursor.description]
        projects = [dict(zip(columns, row)) for row in results]

        conn.close()

        return projects

    def delete_project(self, project_id: str) -> bool:
        """Delete a project and related data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        # Cascade delete related data
        cursor.execute("DELETE FROM character_versions WHERE project = ?", (project_id,))
        cursor.execute("DELETE FROM scenes WHERE project = ?", (project_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        return deleted


# Create a singleton instance
db_service = DatabaseService()
