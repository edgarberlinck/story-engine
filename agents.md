# Story Engine - Agent Documentation

## Project Structure Overview

This project follows a modular architecture that organizes code into distinct functional areas:

### Core Directories:
- `services/database/` - Database connectivity and project management services
- `infrastructure/database/` - Database connection infrastructure
- `models/` - Data models and definitions
- `utils/` - Utility functions and helpers
- `generators/` - Generation components (image, text, etc.)
- `scripts/` - Supporting scripts

## Testing Approach 

### Automated Testing:
- All code changes must pass unit tests before being committed
- Tests are run via `make test` command in the Makefile
- Tests cover all core functionality including project service operations

### Quality Assurance:
- The Makefile includes comprehensive commands for testing, linting, and formatting
- Running `make test` ensures all existing tests pass without errors
- Continuous integration can be enabled from this base structure

## Makefile Commands

The project includes a Makefile with the following key commands:

### Basic Testing:
- `make test` - Run all unit tests in the project
- `make watch` - Watch for code changes and automatically re-run tests

### Development Tools:
- `make install` - Install project dependencies
- `make clean` - Clean build artifacts and temporary files  
- `make format` - Format code with Black
- `make lint` - Lint code with Flake8
- `make check` - Run both linting and testing

## Project Service Implementation

The project service provides complete CRUD functionality:

### Features:
- Create projects with unique identifiers
- Update existing project information  
- Retrieve individual projects by ID
- List all projects in chronological order
- Search projects by name or description

### Database:
- Uses SQLite for persistence (no external dependencies)
- Data stored in `story_engine.db` file
- All project data includes id, name, description, and timestamps