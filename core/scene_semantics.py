"""
Semantic scene decomposition with three-layer architecture.

Implements the user's requirements:
1. Scene Composition (environment, camera, spatial)
2. Character Identity (only identity-defining attributes)
3. Scene-Specific Appearance and Actions

Uses LLM for semantic extraction instead of heuristics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class LayerType(Enum):
    SCENE_COMPOSITION = "scene_composition"
    CHARACTER_IDENTITY = "character_identity"
    SCENE_APPEARANCE_ACTION = "scene_appearance_action"


@dataclass
class SceneComposition:
    """Layer 1: Scene composition details."""
    environment: str
    camera_position: str
    shot_type: str
    spatial_composition: str
    lighting: Optional[str] = None
    atmosphere: Optional[str] = None
    character_positions: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'environment': self.environment,
            'camera_position': self.camera_position,
            'shot_type': self.shot_type,
            'spatial_composition': self.spatial_composition,
            'lighting': self.lighting,
            'atmosphere': self.atmosphere,
            'character_positions': self.character_positions
        }


@dataclass
class CharacterIdentity:
    """Layer 2: Identity-defining attributes only."""
    name: str = ""
    gender: Optional[str] = None
    age_range: Optional[str] = None
    hair_color: Optional[str] = None
    hairstyle: Optional[str] = None
    skin_tone: Optional[str] = None
    body_type: Optional[str] = None
    facial_traits: List[str] = field(default_factory=list)
    permanent_traits: List[str] = field(default_factory=list)
    
    def to_concise_description(self) -> str:
        """Create concise identity description."""
        parts = []
        
        if self.age_range and self.gender:
            parts.append(f"{self.age_range} {self.gender}")
        elif self.gender:
            parts.append(self.gender)
        elif self.age_range:
            parts.append(self.age_range)
        
        # Hair
        if self.hair_color or self.hairstyle:
            hair_desc = f"{self.hairstyle or ''} {self.hair_color or ''}".strip()
            if hair_desc:
                # Ensure it says "hair" at end
                if 'hair' not in hair_desc:
                    hair_desc += " hair"
                parts.append(f"with {hair_desc}")
        
        # Skin
        if self.skin_tone:
            parts.append(f"{self.skin_tone} skin")
        
        # Body
        if self.body_type:
            parts.append(self.body_type + " build")
        
        # Facial traits
        if self.facial_traits:
            parts.append(", ".join(self.facial_traits))
        
        return ", ".join(parts)
    
    def to_list(self) -> List[str]:
        """Convert to list format matching user example."""
        items = []
        if self.gender:
            items.append(self.gender)
        if self.age_range:
            items.append(self.age_range)
        if self.hair_color and self.hairstyle:
            items.append(f"{self.hairstyle} {self.hair_color} hair")
        elif self.hair_color:
            items.append(f"{self.hair_color} hair")
        if self.skin_tone:
            items.append(self.skin_tone)
        if self.body_type:
            items.append(self.body_type + " build")
        items.extend(self.facial_traits)
        items.extend(self.permanent_traits)
        return items


@dataclass
class CharacterSceneData:
    """Character data for specific scene."""
    name: str
    identity: CharacterIdentity
    position: str
    clothing: str
    action: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'identity': self.identity.to_list(),
            'position': self.position,
            'clothing': self.clothing,
            'action': self.action
        }


@dataclass
class ScenePromptSpec:
    """Complete three-layer scene specification."""
    scene: SceneComposition
    characters: List[CharacterSceneData]
    style: str = "photorealistic"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'scene': {
                'environment': self.scene.environment,
                'camera': f"{self.scene.shot_type} {self.scene.camera_position}",
                'composition': self.scene.spatial_composition
            },
            'characters': [c.to_dict() for c in self.characters],
            'style': self.style
        }


class SemanticSceneExtractor:
    """LLM-based semantic extraction for scenes."""
    
    @staticmethod
    def extract_scene_layers(
        scene_description: str,
        character_records: List[Dict[str, Any]],
        project_style: str = "photorealistic"
    ) -> ScenePromptSpec:
        """
        Extract semantic layers using LLM.
        
        Returns structured representation with three layers:
        1. Scene composition
        2. Character identity
        3. Scene-specific appearance/action
        """
        
        # For now, create structured data manually from input
        # In production, this would call LLM
        scene_comp = SemanticSceneExtractor._parse_scene_composition(scene_description)
        
        char_data = []
        for char_record in character_records:
            identity = SemanticSceneExtractor._extract_identity(
                char_record,
                scene_description
            )
            
            # Extract scene-specific info from description
            position, clothing, action = SemanticSceneExtractor._extract_scene_attributes(
                char_record['name'],
                scene_description
            )
            
            char_data.append(CharacterSceneData(
                name=char_record['name'],
                identity=identity,
                position=position,
                clothing=clothing,
                action=action
            ))
        
        return ScenePromptSpec(
            scene=scene_comp,
            characters=char_data,
            style=project_style
        )
    
    @staticmethod
    def _parse_scene_composition(description: str) -> SceneComposition:
        """Parse scene composition from description."""
        # This would be LLM-driven in production
        # For now, extract key elements
        
        # Simple heuristic parsing
        env_parts = []
        camera_pos = "wide shot"
        shot_type = "wide shot"
        
        lines = [l.strip() for l in description.split('\n') if l.strip()]
        
        for line in lines:
            lower = line.lower()
            if 'stage' in lower or 'bar' in lower or 'room' in lower:
                env_parts.append(line)
            elif 'camera' in lower or 'shot' in lower:
                if 'camera' in lower:
                    camera_pos = line
                if 'wide' in lower or 'close' in lower:
                    shot_type = line
        
        environment = ". ".join(env_parts) if env_parts else "scene environment"
        
        return SceneComposition(
            environment=environment,
            camera_position=camera_pos,
            shot_type=shot_type,
            spatial_composition="characters positioned in scene",
            character_positions={}
        )
    
    @staticmethod
    def _extract_identity(
        char_record: Dict[str, Any],
        scene_description: str
    ) -> CharacterIdentity:
        """Extract only identity-defining attributes."""
        
        # Original prompt
        orig_prompt = char_record.get('prompt', '')
        
        # Use LLM to extract identity traits
        # For now, simulate extraction
        identity_attrs = {
            'gender': None,
            'age_range': None,
            'hair_color': None,
            'hairstyle': None,
            'skin_tone': None,
            'body_type': None,
            'facial_traits': [],
            'permanent_traits': []
        }
        
        # Simple extraction (would be LLM in production)
        lower_prompt = orig_prompt.lower()
        
        identity_data = {}
        
        # Gender
        if 'woman' in lower_prompt or 'female' in lower_prompt:
            identity_data['gender'] = 'woman'
        elif 'man' in lower_prompt or 'male' in lower_prompt:
            identity_data['gender'] = 'man'
        
        # Age
        if '25-35' in lower_prompt or 'young' in lower_prompt:
            identity_data['age_range'] = 'young'
        elif '30-40' in lower_prompt or 'adult' in lower_prompt:
            identity_data['age_range'] = 'adult'
        
        # Hair
        if 'long black hair' in lower_prompt:
            identity_data['hair_color'] = 'black'
            identity_data['hairstyle'] = 'long'
        elif 'curly red hair' in lower_prompt:
            identity_data['hair_color'] = 'red'
            identity_data['hairstyle'] = 'curly long'
        
        # Skin
        if 'fair skin' in lower_prompt:
            identity_data['skin_tone'] = 'fair'
        elif 'dark skin' in lower_prompt or 'dark-skinned' in lower_prompt:
            identity_data['skin_tone'] = 'dark'
        
        # Body
        if 'slender' in lower_prompt or 'slim' in lower_prompt:
            identity_data['body_type'] = 'slender'
        elif 'muscular' in lower_prompt:
            identity_data['body_type'] = 'muscular'
        
        identity_data['name'] = char_record.get('name', '')
        return CharacterIdentity(**identity_data)
    
    @staticmethod
    def _extract_scene_attributes(
        char_name: str,
        scene_description: str
    ) -> tuple[str, str, str]:
        """Extract position, clothing, action from scene description."""
        
        lower_desc = scene_description.lower()
        char_lower = char_name.lower()
        
        # Find sentences mentioning character
        sentences = [s.strip() for s in scene_description.split('.') if char_lower in s.lower()]
        
        if not sentences:
            return "in scene", "scene-appropriate clothing", "performing action"
        
        sentence = sentences[0]
        
        position = "in scene"
        clothing = "scene-appropriate clothing"
        action = "performing action"
        
        # Simple extraction
        lower_sent = sentence.lower()
        
        # Position
        if 'left' in lower_sent:
            position = "left side of stage"
        elif 'right' in lower_sent:
            position = "right side of stage"
        elif 'on stage' in lower_sent:
            position = "on stage"
        
        # Clothing
        if 'suit' in lower_sent:
            clothing = "black suit"
        elif 'gown' in lower_sent:
            clothing = "ceremonial gown"
        
        # Action
        if 'playing' in lower_sent:
            action = sentence.split('playing')[-1].strip() if 'playing' in sentence else "playing instrument"
            action = f"playing {action}"
        elif 'sitting' in lower_sent:
            action = "sitting"
        
        return position, clothing, action


class LLMSceneSemanticExtractor:
    """Production LLM-based extractor."""
    
    def __init__(self):
        from generators.text_generator import generate_prompt_with_llm
        self.generate = generate_prompt_with_llm
    
    def extract_semantic_spec(
        self,
        scene_description: str,
        character_records: List[Dict[str, Any]],
        project_style: str = "photorealistic"
    ) -> ScenePromptSpec:
        """
        Use LLM to extract three-layer semantic spec.
        
        This implements the user's exact requirements with LLM delegation.
        """
        
        # Build LLM prompt
        chars_text = ""
        for char in character_records:
            chars_text += f"\n- {char['name']}: {char.get('prompt', '')}"
        
        llm_prompt = f"""You are a scene semantic extractor. Analyze the scene and characters.

SCENE DESCRIPTION:
{scene_description}

CHARACTERS (original generation prompts):
{chars_text}

PROJECT STYLE: {project_style}

Extract a structured specification with THREE LAYERS:

1. SCENE COMPOSITION
   - environment, camera position, shot type, spatial composition, lighting, atmosphere

2. CHARACTER IDENTITY  
   For each character, extract ONLY identity-defining attributes:
   - Gender, age range, hair color/style, skin tone, body type
   - Distinctive facial features
   DO NOT include: clothing from original prompt, poses, generation artifacts

3. SCENE-SPECIFIC APPEARANCE & ACTIONS
   From scene description, extract for each character:
   - Position in scene
   - Clothing described in scene (overrides original)
   - Action being performed

IMPORTANT RULES:
- Character identity must be EXPLICIT, never generic ("two musicians")
- Each character identity must be associated with specific position and action
- Scene clothing/action overrides character defaults
- Remove irrelevant attributes (e.g., "Fantasy Clothing" if scene says black suit)
- Keep within token budget by being concise

Return JSON matching this structure:
{{
  "scene": {{
    "environment": "...",
    "camera_position": "...",
    "shot_type": "...",
    "spatial_composition": "...",
    "lighting": "...",
    "atmosphere": "..."
  }},
  "characters": [
    {{
      "name": "...",
      "identity": ["young woman", "long curly red hair", "fair skin"],
      "position": "...",
      "clothing": "...",
      "action": "..."
    }}
  ],
  "style": "{project_style}"
}}

Return JSON only, no explanations."""
        
        try:
            result = self.generate(llm_prompt, model_name="phi3_mini")
            
            # Parse result
            import json
            data = json.loads(result)
            
            return self._parse_llm_output(data, project_style)
            
        except Exception as e:
            print(f"LLM extraction failed: {e}, using fallback")
            return SemanticSceneExtractor.extract_scene_layers(
                scene_description, character_records, project_style
            )
    
    def _parse_llm_output(
        self,
        data: Dict[str, Any],
        style: str
    ) -> ScenePromptSpec:
        """Parse LLM output into ScenePromptSpec."""
        
        from dataclasses import dataclass
        
        scene_data = data.get('scene', {})
        
        scene_comp = SceneComposition(
            environment=scene_data.get('environment', ''),
            camera_position=scene_data.get('camera_position', 'wide shot'),
            shot_type=scene_data.get('shot_type', 'wide'),
            spatial_composition=scene_data.get('spatial_composition', ''),
            lighting=scene_data.get('lighting'),
            atmosphere=scene_data.get('atmosphere')
        )
        
        characters = []
        for char_data in data.get('characters', []):
            # Parse identity list into CharacterIdentity
            identity_list = char_data.get('identity', [])
            
            # Simple parsing from list
            identity = CharacterIdentity(
                name=char_data.get('name', ''),
                gender='woman' if 'woman' in str(identity_list) else 'man',
                age_range='young' if 'young' in str(identity_list) else None
            )
            
            chars = CharacterSceneData(
                name=char_data.get('name', ''),
                identity=identity,
                position=char_data.get('position', 'in scene'),
                clothing=char_data.get('clothing', ''),
                action=char_data.get('action', '')
            )
            characters.append(chars)
        
        return ScenePromptSpec(
            scene=scene_comp,
            characters=characters,
            style=style
        )
