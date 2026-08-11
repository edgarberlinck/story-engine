"""
CLI interface for Story Engine using Typer.
"""

import typer
from core.project_manager import project_manager
from core.character_manager import character_manager
from core.scene_manager import scene_manager

app = typer.Typer()


@app.command()
def list_projects():
    projects = project_manager.list_projects()
    for p in projects:
        typer.echo(f"{p['id']}: {p['name']} - {p.get('description','')}")


@app.command()
def new_project(name: str, description: str = ""):
    pid = project_manager.create_project(name, description)
    typer.echo(f"Created project {pid}")


@app.command()
def delete_project(project_id: str):
    if project_manager.delete_project(project_id):
        typer.echo(f"Deleted project {project_id}")
    else:
        typer.echo(f"Project not found {project_id}")


@app.command()
def list_characters(project: str = typer.Option("test_project", "--project", "-p")):
    chars = character_manager.list_characters(project)
    for c in chars:
        typer.echo(f"{c['name']}: {c.get('prompt','')[:50]}")


@app.command()
def generate_character(project: str, name: str, prompt: str, variants: int = 3):
    versions = character_manager.generate_versions(project, name, prompt, num_versions=variants)
    typer.echo(f"Generated {len(versions)} versions for {name}")


if __name__ == "__main__":
    app()
