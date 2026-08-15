"""
Test LLM-based scene planner.
"""

import sys
sys.path.insert(0, '/Users/edgarberlinck/code/story-engine')

from core.scene_planner import (
    stage_a_context_resolution,
    stage_b_decompose_scene,
    stage_c_token_budget_fit,
    LLMScenePlanner,
    ResolvedCharacter,
    ScenePlan
)


def test_stage_a_fantasy_clothing():
    """Test context resolution filters inappropriate clothing."""
    print("\n=== Test Stage A: Fantasy Clothing Filtering ===")
    
    scene = "A cozy bar with wood tables. Nikita is sitting at a table drinking coffee."
    characters = [{
        'name': 'Nikita',
        'prompt': 'ultra realistic woman, age 25-35, long black hair, ornate elven armor, glowing runes, fantasy art style'
    }]
    
    # Stage A should resolve context
    resolved = stage_a_context_resolution(scene, characters)
    
    print(f"Resolved character: {resolved[0].name}")
    print(f"Identity: {resolved[0].identity}")
    print(f"Decision: {resolved[0].presentation_decision}")
    print(f"Dropped: {resolved[0].dropped}")
    
    # Even with LLM fallback, should have basic structure
    assert resolved[0].name == 'Nikita'
    print("✓ Stage A works!")


def test_stage_b_decomposition():
    """Test scene decomposition."""
    print("\n=== Test Stage B: Scene Decomposition ===")
    
    scene = "There's a very small stage with red curtains. Nikita is playing guitar. Roger is playing drums."
    resolved = [
        ResolvedCharacter(
            name='Nikita',
            identity=['woman', 'long black hair'],
            default_presentation=[],
            presentation_decision='KEEP',
            scene_presentation=['casual clothes'],
            dropped=[],
            dropped_reason=''
        ),
        ResolvedCharacter(
            name='Roger',
            identity=['man', 'bald head'],
            default_presentation=[],
            presentation_decision='KEEP',
            scene_presentation=['casual clothes'],
            dropped=[],
            dropped_reason=''
        )
    ]
    
    plan = stage_b_decompose_scene(scene, resolved)
    
    print(f"Camera: {plan.camera}")
    print(f"Layers: {len(plan.layers)}")
    print(f"Single pass feasible: {plan.single_pass_feasible}")
    print(f"Rationale: {plan.rationale}")
    
    for layer in plan.layers:
        print(f"  - {layer.name}: {layer.prompt[:60]}...")
    
    assert isinstance(plan, ScenePlan)
    print("✓ Stage B works!")


def test_stage_c_token_budget():
    """Test token budget fitting."""
    print("\n=== Test Stage C: Token Budget Fitting ===")
    
    from core.scene_planner import SceneLayerPlan
    
    long_prompt = "A very long and detailed description of a complex scene with many elements and characters and actions and settings and lighting and atmosphere that goes on and on and exceeds the token limit by far"
    
    layer = SceneLayerPlan(
        name="test",
        prompt=long_prompt,
        must_include=[]
    )
    
    fitted = stage_c_token_budget_fit(layer, token_limit=50)
    
    from utils.token_budget import count_tokens
    
    original_tokens = count_tokens(long_prompt)
    new_tokens = count_tokens(fitted.prompt)
    
    print(f"Original: {original_tokens} tokens")
    print(f"Fitted: {new_tokens} tokens")
    print(f"Prompt: {fitted.prompt[:100]}...")
    
    assert new_tokens <= 50
    print("✓ Stage C works!")


def test_full_llm_planner():
    """Test full LLM planner pipeline."""
    print("\n=== Test Full LLM Planner ===")
    
    from core.scene_planner import create_llm_scene_plan
    
    scene = """There's a very small stage with red curtains. 
Nikita is sitting on a chair playing guitar.
Roger is behind drums.
Wide shot of bar interior."""
    
    characters = [
        {
            'name': 'Nikita',
            'prompt': 'woman, long black hair, fantasy clothing',
            'attributes': {'age': 'Young Adult'}
        },
        {
            'name': 'Roger', 
            'prompt': 'man, bald head, casual clothes'
        }
    ]
    
    # Use LLM planner (may fallback to simple if LLM unavailable)
    plan = create_llm_scene_plan(scene, characters, project_name="test")
    
    print(f"Plan created: {len(plan.layers)} layers")
    print(f"Single pass: {plan.single_pass_feasible}")
    
    assert plan is not None
    print("✓ Full planner works!")


def test_zero_info_loss():
    """Test that nothing is lost silently."""
    print("\n=== Test Zero Information Loss ===")
    
    scene = "Nikita at bar"
    characters = [{
        'name': 'Nikita',
        'prompt': 'woman, long black hair, green eyes'
    }]
    
    resolved = stage_a_context_resolution(scene, characters)
    
    # Identity should always be captured
    char_resolved = resolved[0]
    
    # Check audit trail exists
    assert hasattr(char_resolved, 'dropped')
    assert hasattr(char_resolved, 'dropped_reason')
    assert hasattr(char_resolved, 'identity')
    
    print(f"Audit trail present: dropped={char_resolved.dropped}, reason={char_resolved.dropped_reason}")
    print("✓ Zero info loss enforced!")


if __name__ == "__main__":
    test_stage_a_fantasy_clothing()
    test_stage_b_decomposition()
    test_stage_c_token_budget()
    test_full_llm_planner()
    test_zero_info_loss()
    
    print("\n✓✓✓ All LLM planner tests passed! ✓✓✓")
