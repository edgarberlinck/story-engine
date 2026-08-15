"""
Multi-step scene composition engine.

Generates complex scenes in stages:
1. Base environment
2. Character addition (one at a time)
3. Final refinement

This avoids CLIP token limits and improves generation quality.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import shutil


@dataclass
class SceneLayer:
    """Represents one layer in scene composition."""
    name: str
    prompt: str
    mask_description: Optional[str] = None
    priority: int = 1


@dataclass
class SceneCompositionPlan:
    """Plan for building a scene incrementally."""
    base_environment: str
    characters: List[Dict]
    camera_setup: str
    layers: List[SceneLayer]


class SceneComposer:
    """Decomposes complex scenes into incremental generation steps."""
    
    @staticmethod
    def decompose_scene(description: str, characters: List[Dict]) -> SceneCompositionPlan:
        """Break down scene description into composition layers."""
        
        # Extract environment elements
        env_elements = []
        char_actions = []
        
        lines = [l.strip() for l in description.split('\n') if l.strip()]
        
        # Identify what is environment vs characters
        for line in lines:
            lower = line.lower()
            has_char = any(c['name'].lower() in lower for c in characters)
            
            if has_char and ('playing' in lower or 'sitting' in lower or 'behind' in lower):
                char_actions.append(line)
            elif 'stage' in lower or 'bar' in lower or 'crowd' in lower or 'tables' in lower:
                env_elements.append(line)
            elif 'camera' in lower or 'wide shot' in lower:
                env_elements.append(line)
        
        # Build base environment prompt
        base_env = ". ".join(env_elements[:3])  # First few env elements
        
        # Create layers
        layers = []
        
        # Layer 1: Base environment
        layers.append(SceneLayer(
            name="base_environment",
            prompt=base_env or "A bar with stage and audience",
            priority=1
        ))
        
        # Layer 2+: Characters one at a time
        for i, char in enumerate(characters):
            char_name = char['name']

            # Find action for this character
            action = ""
            for act in char_actions:
                if char_name.lower() in act.lower():
                    action = act
                    break

            # IMPORTANT: never inject the full stored character prompt here.
            # Stored prompts contain generation-time artifacts (fantasy
            # clothing, studio lighting, "standing upright", "not a
            # portrait", plain backgrounds, ...) that corrupt scene steps.
            # Use only the character name + the scene-description snippet.
            layer_prompt = f"{char_name}. {action}".strip()

            layers.append(SceneLayer(
                name=f"character_{char_name}",
                prompt=layer_prompt,
                mask_description=f"Area where {char_name} should appear",
                priority=2 + i
            ))
        
        return SceneCompositionPlan(
            base_environment=base_env,
            characters=characters,
            camera_setup="Wide shot, both characters clearly visible",
            layers=layers
        )
    
    @staticmethod
    def build_incremental_prompts(plan: SceneCompositionPlan) -> List[Tuple[str, str]]:
        """Build prompts for each generation step."""
        
        steps = []
        
        # Step 1: Base environment
        base_prompt = plan.base_environment
        if plan.camera_setup:
            base_prompt += f". {plan.camera_setup}"
        
        steps.append(("base", base_prompt))
        
        # Subsequent steps: Add characters to existing scene
        for layer in plan.layers[1:]:
            # Each step builds on previous
            step_prompt = f"Continue the scene. {layer.prompt}"
            steps.append((layer.name, step_prompt))
        
        return steps


class MultiStepSceneGenerator:
    """Generates scenes incrementally with character addition."""
    
    def __init__(self):
        from generators.image_engine import generate_images
        self.generate_images = generate_images
    
    def generate_scene_incrementally(
        self,
        description: str,
        characters: List[Dict],
        project: str,
        scene_number: int,
        model: str = "flux_dev",
        seed: int = 42
    ) -> Dict:
        """Generate scene in multiple steps."""
        
        from utils.project_paths import scene_dir
        from core.scene_composer import SceneComposer
        
        # Plan the composition
        plan = SceneComposer.decompose_scene(description, characters)
        steps = SceneComposer.build_incremental_prompts(plan)
        
        target_dir = scene_dir(scene_number, project)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        previous_image = None
        generated_files = []
        
        for i, (step_name, prompt) in enumerate(steps):
            print(f"\n[Step {i+1}/{len(steps)}] Generating: {step_name}")
            print(f"Prompt: {prompt[:100]}...")
            
            # For first step, generate from scratch
            if i == 0:
                files = self.generate_images(
                    prompt=prompt,
                    model_name=model,
                    seed=seed + i,
                    task_name=f"scene_{scene_number}_step{i}"
                )
                current_image = files[0]
            
            # For subsequent steps, we would ideally use inpainting
            # For now, generate with modified prompt that references previous
            else:
                enhanced_prompt = f"{prompt} Maintaining the existing composition and style."
                files = self.generate_images(
                    prompt=enhanced_prompt,
                    model_name=model,
                    seed=seed + i,
                    task_name=f"scene_{scene_number}_step{i}"
                )
                current_image = files[0]
            
            generated_files.append(current_image)
            
            # Copy to working file
            step_path = target_dir / f"step_{i}_{step_name}.png"
            shutil.copy(current_image, step_path)
            previous_image = str(step_path)
        
        # Return final result
        final_path = target_dir / "scene.png"
        shutil.copy(generated_files[-1], final_path)
        
        return {
            "scene_number": scene_number,
            "image_path": str(final_path),
            "prompt": description,
            "steps_generated": len(steps),
            "step_files": generated_files
        }
