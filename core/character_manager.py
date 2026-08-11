"""
Character management core for UI with versioning support.
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from services.database.character_service import character_service
from services.database.character_version_service import character_version_service
from utils.project_paths import character_dir, slugify
from generators.image_engine import generate_character


class CharacterManager:
    def __init__(self):
        self.char_service = character_service
        self.version_service = character_version_service

    def list_characters(self, project: str) -> List[Dict]:
        return self.char_service.list_characters(project)

    def get_character(self, project: str, name: str) -> Optional[Dict]:
        return self.char_service.get_character(name, project)

    def delete_character(self, project: str, name: str) -> bool:
        """Delete a character, its versions and its files."""
        success = self.char_service.delete_character(name, project)
        # Remove version rows
        import sqlite3
        conn = sqlite3.connect(self.version_service.db_path)
        conn.execute(
            "DELETE FROM character_versions WHERE project = ? AND character_name = ?",
            (project, name),
        )
        conn.commit()
        conn.close()
        # Remove files
        char_path = character_dir(name, project)
        if char_path.exists():
            shutil.rmtree(char_path, ignore_errors=True)
        return success

    def generate_versions(self, project: str, name: str, prompt: str,
                          model: str = "flux_dev", num_versions: int = 3,
                          seed_start: int = 42) -> List[Dict]:
        """Generate multiple versions of a character."""
        versions = []
        char_path = character_dir(name, project)
        versions_dir = char_path / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)

        # Save manifest
        manifest_path = char_path / "manifest.json"

        for i in range(num_versions):
            seed = seed_start + i * 1000
            # Generate image
            from generators.image_generator import generate_images
            files = generate_images(
                prompt=prompt,
                model_name=model,
                seed=seed,
                task_name=f"character_{slugify(name)}_v{i+1}"
            )
            version_path = versions_dir / f"v_{i+1}.png"
            # Move generated file
            if files and len(files) > 0:
                shutil.move(files[0], str(version_path))
            else:
                # Create placeholder
                version_path.touch()

            # Save to DB
            version = self.version_service.add_version(
                project=project,
                character_name=name,
                prompt=prompt,
                seed=seed,
                model=model,
                image_path=str(version_path)
            )
            versions.append(version)

            # Set first as default
            if i == 0:
                self.version_service.set_default(project, name, version["version"])

        # Update manifest
        manifest = {
            "name": name,
            "project": project,
            "versions": [v["version"] for v in versions]
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)

        # Update main character record to point to default
        default_version = self.version_service.get_default(project, name)
        if default_version:
            self.char_service.save_character(
                name=name,
                prompt=prompt,
                seed=default_version["seed"],
                model=model,
                reference_image=default_version["image_path"],
                project=project
            )

        return versions

    def list_versions(self, project: str, character_name: str) -> List[Dict]:
        return self.version_service.list_versions(project, character_name)

    def set_default_version(self, project: str, character_name: str, version: int) -> bool:
        success = self.version_service.set_default(project, character_name, version)
        if success:
            # Update reference image symlink/copy
            default = self.version_service.get_default(project, character_name)
            if default:
                ref_path = character_dir(character_name, project) / "reference.png"
                # Copy file to reference location for compatibility
                import shutil
                shutil.copy2(default["image_path"], str(ref_path))
                # Update character service record
                self.char_service.save_character(
                    name=character_name,
                    prompt=default["prompt"],
                    seed=default["seed"],
                    model=default["model"],
                    reference_image=str(ref_path),
                    project=project
                )
        return success


character_manager = CharacterManager()
