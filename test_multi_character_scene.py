#!/usr/bin/env python3
"""
Test multi-character scene generation with Nikita and Roger.
Uses the new progressive asset-composition pipeline.
"""

from core.scene_planner import LLMScenePlanner
from core.character_asset_generator import generate_character_assets, build_character_asset_prompt
from services.database.character_service import character_service

project = "Test_ui"

# Scene from investigation doc
scene_description = """There's a stage, very small with red curtains. The stage is made of old cracked wood. In front of this stage there's some tables with people. On stage we have Nikita wearing a black suit, she is sitting in a chair playing a black Gibson Explorer guitar and Roger playing drums. The camera is positioned in the back of the bar, so we can see the crowd, the stage and the band."""

print("=" * 80)
print("Testing Multi-Character Scene Generation")
print("=" * 80)

# Find characters
characters = character_service.find_characters_in_text(scene_description, project)
print(f"\nFound {len(characters)} characters:")
for c in characters:
    print(f"  - {c['name']}: {c.get('prompt', '')[:100]}...")

# Test LLM planning
print("\n--- Stage A/B: LLM Planning ---")
planner = LLMScenePlanner(use_llm=False)  # Use fallback for test
plan = planner.plan_scene(scene_description, characters)

print(f"\nPlan strategy feasible: {plan.single_pass_feasible}")
print(f"Camera: {plan.camera}")
print(f"Rationale: {plan.rationale}")
print(f"Layers: {len(plan.layers)}")
for layer in plan.layers:
    print(f"  - {layer.name}: {layer.prompt[:80]}...")

# Test character asset prompt building
print("\n--- Testing Character Asset Prompt Building ---")
from core.scene_planner import stage_a_context_resolution

resolved = stage_a_context_resolution(scene_description, characters)
print(f"\nResolved {len(resolved)} characters:")
for rc in resolved:
    print(f"  - {rc.name}")
    print(f"    Identity: {rc.identity}")
    print(f"    Scene presentation: {rc.scene_presentation}")

# Build asset prompts
from core.character_asset_generator import build_character_asset_prompt
print("\n--- Asset Prompts ---")
for rc in resolved:
    prompt = build_character_asset_prompt(rc, "photorealistic")
    print(f"\n{rc.name}:")
    print(f"  Prompt length: {len(prompt)} chars")
    print(f"  Tokens: ~{len(prompt.split())}")
    print(f"  Preview: {prompt[:200]}...")

print("\n" + "=" * 80)
print("Test complete - pipeline modules loaded successfully!")
print("=" * 80)
