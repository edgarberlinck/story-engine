"""
Smoke test for the character-asset + deterministic composition pipeline.

Exercises core/scene_pipeline.py end-to-end using the real Nikita/Roger
characters already stored in the Test_ui project database, but with the
diffusion (`generate_images`) and LLM (`generate_prompt_with_llm`) calls
mocked out so the test runs in seconds instead of requiring real GPU/CPU
inference. This validates the orchestration, segmentation, and composition
logic itself (the parts this task actually adds), not diffusion model
quality.

Run with: .venv/bin/python scripts/test_scene_pipeline_smoke.py
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw


PROJECT = "Test_ui"
SCENE_PROMPT = (
    "There's a stage, very small with red curtains. The stage is made of "
    "old cracked wood. In front of this stage there's some tables with "
    "people. On stage we have Nikita wearing a black suit, she is sitting "
    "in a chair playing a black Gibson Explorer guitar and Roger playing "
    "drums. The camera is positioned in the back of the bar, so we can see "
    "the crowd, the stage and the band."
)


def _fake_person_image(seed: int, color) -> Path:
    """Create a plain-gray-background image with a colored 'person' blob,
    simulating a character asset with an easily-segmentable background."""
    img = Image.new("RGB", (512, 1024), (200, 200, 200))  # plain gray bg
    draw = ImageDraw.Draw(img)
    draw.ellipse([156, 150, 356, 400], fill=color)  # head
    draw.rectangle([156, 400, 356, 900], fill=color)  # body
    out_dir = Path("/tmp/scene_pipeline_smoke")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"fake_{seed}.png"
    img.save(path)
    return path


def _fake_background_image() -> Path:
    img = Image.new("RGB", (1024, 1024), (60, 20, 20))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 700, 1024, 1024], fill=(90, 60, 30))  # "stage floor"
    out_dir = Path("/tmp/scene_pipeline_smoke")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "fake_background.png"
    img.save(path)
    return path


_call_count = {"n": 0}


def fake_generate_images(prompt, model_name="flux_dev", seed=42, task_name=None, **kwargs):
    _call_count["n"] += 1
    print(f"[mock generate_images] task={task_name} seed={seed} prompt[:60]={prompt[:60]!r}")
    if task_name and "background" in task_name:
        return [str(_fake_background_image())]
    if task_name and "Nikita" in task_name:
        return [str(_fake_person_image(seed, (200, 50, 50)))]
    if task_name and "Roger" in task_name:
        return [str(_fake_person_image(seed, (50, 50, 200)))]
    return [str(_fake_person_image(seed, (100, 100, 100)))]


def fake_generate_text_with_llm(llm_prompt: str, model_name="phi3_mini", **kwargs):
    """Return canned JSON responses depending on which stage is calling."""
    if "You resolve character descriptions" in llm_prompt:
        # Stage A: context resolution
        return json.dumps([
            {
                "name": "Nikita",
                "identity": ["young woman", "long curly red hair", "fair skin"],
                "default_presentation": ["fantasy clothing"],
                "presentation_decision": "REPLACE",
                "scene_presentation": ["black suit"],
                "dropped": ["fantasy clothing"],
                "dropped_reason": "scene specifies black suit"
            },
            {
                "name": "Roger",
                "identity": ["muscular man", "dark skin", "short bald hair"],
                "default_presentation": ["formal clothing"],
                "presentation_decision": "KEEP",
                "scene_presentation": ["dark formal outfit"],
                "dropped": [],
                "dropped_reason": ""
            }
        ])
    if "Produce JSON with layers" in llm_prompt:
        # Stage B: decomposition
        return json.dumps({
            "camera": "wide shot from back of bar",
            "layers": [
                {
                    "name": "base_environment",
                    "prompt": "small stage with red curtains, cracked wood floor, tables with people, empty stage",
                    "must_include": ["stage", "curtains"],
                    "region_hint": "full frame"
                },
                {
                    "name": "character_Nikita",
                    "prompt": "Nikita, young woman, long curly red hair, fair skin, black suit, playing guitar",
                    "must_include": ["red hair"],
                    "region_hint": "left"
                },
                {
                    "name": "character_Roger",
                    "prompt": "Roger, muscular man, dark skin, short bald hair, dark formal outfit, playing drums",
                    "must_include": ["bald"],
                    "region_hint": "right"
                }
            ],
            "single_pass_feasible": False,
            "rationale": "two characters with distinct positions and identity constraints"
        })
    # Stage C: compression fallback (should rarely trigger given short canned prompts)
    return llm_prompt


def main():
    with patch("generators.image_engine.generate_images", side_effect=fake_generate_images), \
         patch("core.scene_planner.generate_text_with_llm", side_effect=fake_generate_text_with_llm):

        from core.scene_pipeline import generate_scene_pipeline

        result = generate_scene_pipeline(
            prompt=SCENE_PROMPT,
            project=PROJECT,
            model="sdxl",
            seed=42,
        )

    print("\n=== RESULT ===")
    print(json.dumps({k: v for k, v in result.items() if k != "assets"}, indent=2, default=str))
    print(f"\nGenerated image at: {result.get('image_path')}")
    assert Path(result["image_path"]).is_file(), "final composed image missing"
    print(f"\nMock generate_images call count: {_call_count['n']}")
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
