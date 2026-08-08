# Project Service Documentation

## Overview

The project service is a database-backed system that provides create, read, update, and search functionality for managing projects within the Story Engine. It's designed to be scalable, reliable and easy to integrate with other components of the system.

## Core Features

### 1. Create Projects
- Create new projects with unique identifiers
- Projects automatically receive timestamps for creation and updates
- All project data is stored persistently in SQLite database

### 2. Read Projects
- Get individual projects by ID (returns `None` if not found)
- List all projects (newest first)
- Search projects by name or description (case-insensitive, partial match)

### 3. Update Projects  
- Modify existing project information (name, description, or both)
- Updates maintain proper timestamps
- Returns `False` if the project doesn't exist (no exception raised)

> **Note:** There is currently no delete operation. Projects are persistent once created.

## Technical Implementation

### Database Schema
The service uses a SQLite database (`story_engine.db`) with a `projects` table structured as follows:
```
id (TEXT) - Unique identifier for each project
name (TEXT) - Project name 
description (TEXT) - Human-readable description
created_at (TIMESTAMP) - Creation timestamp
updated_at (TIMESTAMP) - Last updated timestamp
```

### Service Location
- Implementation: `services/database/project_service.py`
- Database service: `services/database/database_service.py`
- Database file: `story_engine.db` (created automatically)

## Usage Examples

### Creating a Project
```python
from services.database.project_service import project_service

# Create a new project
project_id = project_service.create_project(
    name="My Awesome Story",
    description="A story about space exploration"
)
print(f"Created project ID: {project_id}")
```

### Retrieving Projects
```python
# Get specific project
project = project_service.get_project(project_id)

# List all projects  
projects = project_service.list_projects()

# Search for projects
results = project_service.search_projects("space")
```

### Updating a Project
```python
# Update project information
success = project_service.update_project(
    project_id="project_123456",
    name="Updated Story Title", 
    description="New story description"
)
```

## Integration Points

The project service integrates seamlessly with:
- Database system via `DatabaseService` singleton pattern  
- Existing image generation components
- File system operations for project storage
- Test infrastructure and validation

## Configuration Requirements

### Prerequisites
- Python 3.14 or higher
- Required packages installed (`pip install -r requirements.txt`)
- SQLite database file location accessible  

### Runtime Environment
- Database file `story_engine.db` auto-created on first use
- No external dependencies required for operation  
- Thread-safe implementation using singleton pattern

## Development Setup

### Installation
1. Clone the repository
2. Install required packages: `pip install -r requirements.txt`
3. Run tests to verify installation: `make test`

### Testing
The project is covered by the repository test suite:
- Run all tests with `make test`
- Auto-run tests on changes with `make watch`

## Error Handling

The service behaves as follows:
- `get_project` returns `None` when a project is not found (no exception)
- `update_project` returns `False` when the project doesn't exist
- `create_project` requires a name (database enforces NOT NULL); description is optional
- Database connection failures raise standard `sqlite3` exceptions

## Future Enhancements

Consider these improvements for advanced usage:
- Delete operation for removing projects
- Version control for projects
- Project tagging and categorization
- Export/import functionality
- Advanced search filters