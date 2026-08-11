# Story Engine UI Overview

## Features
- Projects: create, update, delete, list
- Characters: generate multiple variants, select default look, view history
- Scenes: create scenes with prompt enrichment from characters

## Architecture
- Desktop app: PySide6/Qt, local-only
- Core layer: `core/project_manager.py`, `core/character_manager.py`, `core/scene_manager.py`
- Workers: background generation via ThreadPoolExecutor
- DB: SQLite with migrations for character_versions and scenes

## Running
```bash
make ui
```

## CLI
```bash
make cli
python cli/main.py list-projects
```
