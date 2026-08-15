"""
LLM-driven prompt optimization for token budget management.

Implements the user's requirement to use LLMs instead of heuristics
for managing token budgets and compressing prompts intelligently.
"""

from typing import List, Dict, Any, Optional
from core.scene_semantics import ScenePromptSpec


class LLMTokenOptimizer:
    """Optimizes prompts using LLM while preserving semantics."""
    
    def __init__(self):
        from generators.text_generator import generate_prompt_with_llm
        self.generate = generate_prompt_with_llm
    
    def optimize_to_budget(
        self,
        spec: ScenePromptSpec,
        max_tokens: int = 77
    ) -> str:
        """
        Optimize prompt to fit token budget using LLM.
        
        The LLM intelligently compresses while preserving:
        - Character identities (sacred)
        - Positions and actions
        - Essential scene composition
        """
        
        from core.prompt_builder import SemanticPromptBuilder
        from utils.token_budget import count_tokens
        
        # Build full explicit prompt
        builder = SemanticPromptBuilder()
        full_prompt = builder.build_explicit_identity_prompt(spec)
        
        current_tokens = count_tokens(full_prompt)
        
        if current_tokens <= max_tokens:
            print(f"✓ Prompt within budget: {current_tokens}/{max_tokens}")
            return full_prompt
        
        print(f"⚠ Optimizing prompt: {current_tokens} → {max_tokens} tokens")
        
        # Create LLM optimization request
        spec_json = self._spec_to_json(spec)
        
        llm_prompt = f"""You must compress this scene prompt to fit {max_tokens} CLIP tokens.

Current prompt has ~{current_tokens} tokens. Your goal is semantic preservation, not just truncation.

SCENE SPECIFICATION:
{spec_json}

CURRENT PROMPT:
{full_prompt}

INSTRUCTIONS:
1. Keep ALL character identities EXPLICIT with names
2. Preserve positions and actions
3. Keep essential scene composition (environment, camera)
4. Remove non-essential adjectives and elaboration
5. Merge related phrases
6. Make it concise but complete

CRITICAL RULES:
- Never use generic placeholders like "two musicians"
- Always associate character identity with name: "Nikita, young woman with red hair"
- Preserve position-action pairs: "on left side playing guitar"
- Keep scene context for coherence

Return only the compressed prompt, no explanations."""

        try:
            optimized = self.generate(llm_prompt, model_name="phi3_mini")
            
            if optimized:
                new_tokens = count_tokens(optimized)
                print(f"✓ LLM optimized to {new_tokens} tokens")
                
                # Verify token count
                if new_tokens <= max_tokens:
                    return optimized
        
        except Exception as e:
            print(f"LLM optimization failed: {e}")
        
        # Fallback to priority-based compression
        return self._fallback_compression(spec, max_tokens)
    
    def _spec_to_json(self, spec: ScenePromptSpec) -> str:
        """Convert spec to compact JSON for LLM."""
        import json
        
        data = {
            'scene': {
                'environment': spec.scene.environment if spec.scene else '',
                'camera': f"{spec.scene.shot_type} from {spec.scene.camera_position}" if spec.scene else ''
            },
            'characters': [
                {
                    'name': c.name,
                    'identity': c.identity.to_concise_description(),
                    'position': c.position,
                    'clothing': c.clothing,
                    'action': c.action
                }
                for c in spec.characters
            ],
            'style': spec.style
        }
        
        return json.dumps(data, indent=2)
    
    def _fallback_compression(
        self,
        spec: ScenePromptSpec,
        max_tokens: int
    ) -> str:
        """Priority-based compression fallback."""
        
        from utils.token_budget import count_tokens
        
        parts = []
        
        # Essential scene elements (minimal)
        if spec.scene and spec.scene.environment:
            env_short = spec.scene.environment.split('.')[0][:100]
            parts.append(env_short)
        
        # Characters with identity
        for char in spec.characters:
            identity_parts = []
            
            # Keep name + essential identity
            identity_desc = char.identity.to_concise_description()
            if 'woman' in identity_desc or 'man' in identity_desc:
                gender_part = 'woman' if 'woman' in identity_desc else 'man'
                hair_part = ''
                
                for word in identity_desc.split():
                    if word in ['red', 'black', 'blonde', 'brown'] and 'hair' in identity_desc:
                        hair_part = f"{word} hair"
                        break
                
                identity_parts.append(f"{char.name}, {gender_part}")
                if hair_part:
                    identity_parts.append(hair_part)
            else:
                identity_parts.append(char.name)
            
            # Position + action
            pos_act = []
            if char.position:
                pos_act.append(char.position)
            if char.action:
                pos_act.append(char.action)
            
            parts.append(f"{', '.join(identity_parts)}: {', '.join(pos_act)}")
        
        result = ". ".join(parts) + f". Style: {spec.style}"
        
        # Ensure within budget
        tokens = count_tokens(result)
        if tokens > max_tokens:
            # Very aggressive truncation
            from utils.token_budget import TokenBudgetManager
            manager = TokenBudgetManager()
            result = manager.truncate_to_tokens(result, max_tokens)
        
        return result


def test_optimization():
    """Test optimization with example."""
    
    from core.scene_semantics import ScenePromptSpec, SceneComposition, CharacterSceneData, CharacterIdentity
    
    # Create test spec
    spec = ScenePromptSpec(
        scene=SceneComposition(
            environment="Small old bar with red curtains and cracked wooden stage. Several tables with people watching.",
            camera_position="back of bar",
            shot_type="wide",
            spatial_composition="audience foreground, stage background"
        ),
        characters=[
            CharacterSceneData(
                name='Nikita',
                identity=CharacterIdentity(gender='young woman', hair_color='black', hairstyle='long'),
                position='left side of stage',
                clothing='black suit',
                action='playing black Gibson Explorer guitar'
            ),
            CharacterSceneData(
                name='Roger',
                identity=CharacterIdentity(gender='man', body_type='muscular'),
                position='right side of stage',
                clothing='dark formal outfit',
                action='sitting behind drum kit playing drums'
            )
        ],
        style='photorealistic'
    )
    
    # Test prompt building
    from core.prompt_builder import SemanticPromptBuilder
    builder = SemanticPromptBuilder()
    
    full_prompt = builder.build_explicit_identity_prompt(spec)
    
    from utils.token_budget import count_tokens
    tokens = count_tokens(full_prompt)
    
    print(f"\nFull prompt ({tokens} tokens):")
    print(full_prompt[:300] + "...")
    
    # Optimize
    optimizer = LLMTokenOptimizer()
    optimized = optimizer.optimize_to_budget(spec, max_tokens=77)
    
    opt_tokens = count_tokens(optimized)
    print(f"\nOptimized prompt ({opt_tokens} tokens):")
    print(optimized)
    
    return optimized


if __name__ == "__main__":
    test_optimization()
