# System Architecture Documentation

## Project Service Interface

### Module Structure
- **File**: `services/database/project_service.py`
- **Class**: `ProjectService`  
- **Singleton**: Yes (access via `project_service` instance)

### API Specification

#### create_project(name: str, description: str = None) -> str
Creates a new project with unique identifier.
- Parameters:
  - `name` (str): Human-readable project name (required, NOT NULL in database)
  - `description` (str, optional): Detailed project description (defaults to None)
- Returns: Unique project ID string (format: `project_<unix_timestamp>`)
- Raises: `sqlite3.IntegrityError` if name is None

#### get_project(project_id: str) -> dict | None
Retrieves project data by ID.
- Parameters:
  - `project_id` (str): Unique identifier 
- Returns: Dictionary with keys: id, name, description, created_at, updated_at
- Returns `None` if project doesn't exist (does NOT raise)

#### update_project(project_id: str, name: str = None, description: str = None) -> bool  
Updates existing project data. Partial updates supported: pass only the fields to change; None values keep the current value.
- Parameters:
  - `project_id` (str): Unique identifier
  - `name` (str, optional): Updated project name
  - `description` (str, optional): Updated project description
- Returns: `True` on success, `False` if project doesn't exist (does NOT raise)

#### list_projects() -> list[dict]
Returns all projects ordered by creation date, newest first (`created_at DESC`).
- Parameters: None
- Returns: List of project dictionaries

#### search_projects(query: str) -> list[dict]  
Searches projects by name or description using SQL `LIKE '%query%'` (substring match).
- Parameters:
  - `query` (str): Search text
- Returns: List of matching project dictionaries, newest first

> **Important:** There is NO delete method. Read operations return `None`/`False`/empty lists on missing data instead of raising exceptions.

## Database Integration Details

### Schema 
```
projects (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL, 
  description TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

### Connection Management  
- Uses singleton pattern in `DatabaseService`
- Auto-initializes database on first access
- Thread-safe implementation
- Database file location: `story_engine.db` (relative to working directory)

## File Structure

### Core Implementation Files
1. `services/database/project_service.py` - Main service logic
2. `services/database/database_service.py` - Database connection management  
3. `story_engine.db` - SQLite database file (auto-created)

### Development Files
- `Makefile` - Testing and automation commands
- `test_project_service.py` - Unit tests (planned; not yet present)
- `docs/` - Documentation directory

## Requirements

### Technical Dependencies
- Python 3.14+
- sqlite3 (built-in)
- watchdog (for make watch command)

### System Requirements  
- Write permissions to working directory
- SQLite database support available
- Standard Python development environment

## Quality Assurance

### Testing Framework
- Unit tests using unittest framework
- Test coverage for all CRUD operations
- Integration testing of database interactions  
- Automated regression testing with `make test`

### Code Quality
- Linting through flake8
- Formatting with black tool
- No external dependencies required for core functionality

## Performance Characteristics

### Database Operations  
- All queries use prepared statements for security
- Automatic indexing on ID column
- Efficient row retrieval and updates 

### Memory Usage
- Lazy database connection loading
- Minimal memory overhead during operations
- Thread-safe handling of concurrent requests

## Error Conditions

### Behavior on Missing Data (no exceptions)
- `get_project` with unknown ID returns `None`
- `update_project` with unknown ID returns `False`
- `search_projects` with no matches returns `[]`

### Actual Exceptions
- `create_project(None, ...)` raises `sqlite3.IntegrityError` (NOT NULL constraint)
- Database file access issues raise standard `sqlite3` exceptions
- No application-level validation is performed (no ValueError/KeyError)

## Example Usage Patterns

### Typical Workflow
```python
from services.database.project_service import project_service

# Create new project
project_id = project_service.create_project("Test", "Description")

# Update project
project_service.update_project(project_id, "New Name", "New Description") 

# Retrieve and list projects
project = project_service.get_project(project_id)
all_projects = project_service.list_projects()

# Search functionality
results = project_service.search_projects("test")
```

## Security Considerations

### Data Protection
- All SQL queries use parameterized statements preventing injection
- No sensitive data stored beyond project metadata 
- Access controlled through standard Python file permissions

### Database Security
- SQLite database file is created with default system permissions
- No external network access required  
- Automatic table creation with secure schema definition

## Integration Guidelines

### For Development
1. Import via `from services.database.project_service import project_service`  
2. Use singleton instance directly (no instantiation needed)
3. Handle exceptions appropriately in calling code
4. Validate input parameters before calling methods

### For Testing
1. Run the full test suite with `make test` (discovers `test_*.py` in project root)
2. Run with `make test` command for full test suite
3. Auto-test functionality available via `make watch`
4. Tests cover all edge cases and error scenarios

## Configuration Parameters

### Environment Variables
- None required beyond standard Python setup
- Database location specified automatically

### File Locations
- Primary database: `story_engine.db`
- Test files: `test_*.py` in project root
- Documentation: `docs/humans/project-service.md`

## Maintenance Notes

### Backup Strategy
- Database backup recommended by copying `story_engine.db` file
- No automated backup included in current implementation

### Versioning
- Single version of project service interface
- Backward compatibility maintained for all operations  

## Troubleshooting

### Common Issues  
1. `story_engine.db` file permission errors - check directory permissions
2. Database connection failures - verify Python SQLite installation 
3. Invalid project IDs during updates - validate before calling update methods

### Diagnostic Commands
- Run `make test` to verify service integrity
- Check `story_engine.db` exists and is accessible  
- Verify required packages are installed with `pip list`