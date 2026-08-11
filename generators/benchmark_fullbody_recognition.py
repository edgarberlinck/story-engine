#!/usr/bin/env python3
"""
Benchmark suite for face recognition accuracy — FULL BODY variant.

Same methodology as generators/benchmark_face_recognition.py, but both
the character references and the scenes are generated in FULL BODY
framing (head to toe) instead of portrait framing. This measures how
well the face verification pipeline performs when faces are much
smaller in the frame.

  1. Self check   — the character's reference image must match itself.
  2. Scene check  — a full body scene featuring the character is
                    generated and the character must be found in it.
  3. Cross check  — the character is compared against every OTHER
                    character's scene; matches here are false positives.

Model: flux_dev — Project: test_project_3

Outputs:
  outputs/test_project_3/characters/<name>/reference.png
  outputs/test_project_3/scenes/scene_<n>/scene.png
  outputs/test_project_3/face_benchmark/report.md
  outputs/test_project_3/face_benchmark/report.json
"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generators.benchmark_face_recognition import (
    NUM_CHARACTERS,
    RANDOM_SEED,
    FIRST_NAMES,
    LAST_NAMES,
    GENDERS,
    AGES,
    ETHNICITIES,
    HAIR,
    FEATURES,
    pct,
    summarize,
)
from generators.image_engine import generate_character, get_character, generate_scene
from utils.face_check import character_appears_in_image, is_face_check_available
from utils.project_paths import project_dir

MODEL = "flux_dev"
PROJECT = "test_project_3"

# Body type pool: explicit physique descriptions push the model towards
# actually rendering the whole body instead of a portrait crop.
BODY_TYPES = [
    "skinny and tall",
    "short and overweight",
    "very attractive with a stunning physique",
    "strong and muscular",
    "short and slim",
    "tall and athletic",
    "average build with a slight belly",
    "curvy and fit",
]

# Action scene templates: the SAME character performing different things,
# always framed with the entire body visible.
SCENE_TEMPLATES = [
    "{name} running at full speed down a street, dynamic full body action shot, entire body visible from head to toe",
    "{name} in an epic action scene dodging an explosion, full body wide shot, entire body in frame",
    "{name} jumping high in the air over an obstacle, full body captured mid-jump, head to toe visible",
    "{name} fighting in hand-to-hand combat, dynamic fighting stance, full body visible from head to toe",
    "{name} climbing a rocky cliff, whole body visible in a wide action shot",
    "{name} dancing energetically in a plaza, full body in frame from head to toe",
    "{name} kicking a ball on a field, dynamic full body sports shot, entire body visible",
    "{name} sprinting away from a collapsing bridge, cinematic full body wide shot, head to toe in frame",
]


def build_random_characters(rng: random.Random, count: int):
    """Create `count` random (name, prompt) FULL BODY character definitions."""
    characters = []
    used_names = set()
    while len(characters) < count:
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        if name in used_names:
            continue
        used_names.add(name)
        prompt = (
            f"Full body photo of {name}, a {rng.choice(ETHNICITIES)} "
            f"{rng.choice(GENDERS)}, {rng.choice(AGES)}, "
            f"{rng.choice(BODY_TYPES)}, "
            f"{rng.choice(HAIR)}, {rng.choice(FEATURES)}. "
            "Standing upright, ENTIRE body visible from head to toe including "
            "feet, front facing, photorealistic, detailed face, good lighting, "
            "full length wide shot, camera far from subject, feet touching the "
            "ground visible in frame, not a portrait, not a close-up"
        )
        characters.append({"name": name, "prompt": prompt})
    return characters


def run_benchmark():
    """Run the full body face recognition benchmark and return results."""
    if not is_face_check_available():
        raise RuntimeError(
            "face_recognition is not installed; this benchmark requires it."
        )

    rng = random.Random(RANDOM_SEED)
    definitions = build_random_characters(rng, NUM_CHARACTERS)

    entries = []
    for i, definition in enumerate(definitions):
        name, prompt = definition["name"], definition["prompt"]
        print(f"\n=== Character {i + 1}/{len(definitions)}: {name} ===")

        character = get_character(name, PROJECT)
        if character is None:
            character = generate_character(
                name, prompt, model=MODEL, project=PROJECT,
                seed=RANDOM_SEED + i,
            )

        # Full body scene featuring this character
        scene_prompt = rng.choice(SCENE_TEMPLATES).format(name=name)
        scene = generate_scene(
            scene_prompt, project=PROJECT, model=MODEL,
            seed=RANDOM_SEED + 100 + i,
        )

        reference = character["reference_image"]

        # 1. Self check: reference must match itself
        self_check = character_appears_in_image(reference, reference)

        # 2. Scene check: character must be found in their own scene
        scene_check = character_appears_in_image(reference, scene["image_path"])

        entries.append({
            "name": name,
            "prompt": prompt,
            "reference_image": reference,
            "scene_prompt": scene_prompt,
            "scene_image": scene["image_path"],
            "self_check": self_check,
            "scene_check": scene_check,
        })

    # 3. Cross checks: character vs every OTHER character's scene
    cross_checks = []
    for a in entries:
        for b in entries:
            if a["name"] == b["name"]:
                continue
            match = character_appears_in_image(
                a["reference_image"], b["scene_image"]
            )
            cross_checks.append({
                "character": a["name"],
                "scene_of": b["name"],
                "scene_image": b["scene_image"],
                "match": match,
            })

    return {"characters": entries, "cross_checks": cross_checks}


def write_report(results: dict, summary: dict) -> Path:
    """Write Markdown + JSON reports for manual review. Returns md path."""
    report_dir = project_dir(PROJECT) / "face_benchmark"
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "report.json"
    with open(json_path, "w") as f:
        json.dump({"model": MODEL, "project": PROJECT, "framing": "full_body",
                   "summary": summary, **results}, f, indent=2)

    def check(v):
        return {True: "PASS", False: "FAIL", None: "INCONCLUSIVE"}[v]

    lines = [
        "# Face Recognition Benchmark Report (Full Body)",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Model: **{MODEL}** — Project: **{PROJECT}** — Framing: **full body**",
        "",
        "## Summary",
        "",
        f"- Characters tested: **{summary['total_characters']}**",
        f"- Self check (reference matches itself): "
        f"**{pct(summary['self_check_pass'], summary['self_check_total'])}** "
        f"({summary['self_check_pass']}/{summary['self_check_total']})",
        f"- Scene check (character found in own scene): "
        f"**{pct(summary['scene_check_pass'], summary['scene_check_total'])}** "
        f"({summary['scene_check_pass']}/{summary['scene_check_total']})",
        f"- False positives (character 'found' in someone else's scene): "
        f"**{pct(summary['false_positives'], summary['cross_check_total'])}** "
        f"({summary['false_positives']}/{summary['cross_check_total']})",
        "",
        "## Characters",
        "",
        "| Character | Self check | Scene check | Reference | Scene |",
        "|---|---|---|---|---|",
    ]
    for e in results["characters"]:
        lines.append(
            f"| {e['name']} | {check(e['self_check'])} | "
            f"{check(e['scene_check'])} | {e['reference_image']} | "
            f"{e['scene_image']} |"
        )

    lines += [
        "",
        "### Details (for manual review)",
        "",
    ]
    for e in results["characters"]:
        lines += [
            f"#### {e['name']}",
            "",
            f"- Character prompt: {e['prompt']}",
            f"- Scene prompt: {e['scene_prompt']}",
            f"- Reference image: `{e['reference_image']}`",
            f"- Scene image: `{e['scene_image']}`",
            f"- Self check: {check(e['self_check'])}",
            f"- Scene check: {check(e['scene_check'])}",
            "",
        ]

    false_positives = [c for c in results["cross_checks"] if c["match"]]
    lines += ["## False positives", ""]
    if false_positives:
        lines += ["| Character | Wrongly found in scene of | Scene |",
                  "|---|---|---|"]
        lines += [
            f"| {c['character']} | {c['scene_of']} | {c['scene_image']} |"
            for c in false_positives
        ]
    else:
        lines.append("None.")
    lines.append("")

    md_path = report_dir / "report.md"
    md_path.write_text("\n".join(lines))
    return md_path


def main():
    print("=== Face Recognition Benchmark (Full Body) ===")
    print(f"Characters: {NUM_CHARACTERS} (seed={RANDOM_SEED})")
    print(f"Model: {MODEL} — Project: {PROJECT}")

    try:
        results = run_benchmark()
        summary = summarize(results)
        md_path = write_report(results, summary)

        print("\n=== Face Recognition Benchmark Summary (Full Body) ===")
        print(f"  Self check:      "
              f"{pct(summary['self_check_pass'], summary['self_check_total'])}")
        print(f"  Scene check:     "
              f"{pct(summary['scene_check_pass'], summary['scene_check_total'])}")
        print(f"  False positives: "
              f"{pct(summary['false_positives'], summary['cross_check_total'])}")
        print(f"\nReport for manual review: {md_path}")
    except Exception as e:
        print(f"Error during benchmark: {e}")


if __name__ == "__main__":
    main()
