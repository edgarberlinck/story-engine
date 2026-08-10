#!/usr/bin/env python3
"""
Benchmark suite for face recognition accuracy.

Generates a batch of random characters (varied gender, age, ethnicity,
looks), then measures how well the face verification pipeline performs:

  1. Self check   — the character's reference image must match itself
                    (sanity check of detection + encoding).
  2. Scene check  — a new scene featuring the character is generated and
                    the character must be found in it (true positive rate).
  3. Cross check  — the character is compared against every OTHER
                    character's scene; matches here are false positives.

A human-readable Markdown report (plus raw JSON) is written with every
image path so the results can be reviewed manually.

Outputs:
  outputs/test_project/characters/<name>/reference.png
  outputs/test_project/scenes/scene_<n>/scene.png
  outputs/test_project/face_benchmark/report.md
  outputs/test_project/face_benchmark/report.json
"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generators.image_engine import generate_character, get_character, generate_scene
from utils.face_check import character_appears_in_image, is_face_check_available
from utils.project_paths import project_dir

NUM_CHARACTERS = 8
RANDOM_SEED = 42

# Attribute pools for random character generation.
FIRST_NAMES = ["Aiyra", "Bruno", "Chen", "Dalia", "Erik", "Fatima", "Gustavo",
               "Hana", "Igor", "Jamila", "Kauan", "Leila", "Mateus", "Naomi"]
LAST_NAMES = ["Silva", "Larsson", "Okafor", "Tanaka", "Morales", "Petrov",
              "Nakamura", "Costa", "Haddad", "Johansson"]
GENDERS = ["man", "woman"]
AGES = ["young adult in their 20s", "adult in their 30s",
        "middle-aged in their 50s", "elderly in their 70s"]
ETHNICITIES = ["Brazilian indigenous", "Scandinavian", "East Asian",
               "West African", "Middle Eastern", "Latin American",
               "Mediterranean"]
HAIR = ["short dark hair", "long blonde hair", "curly brown hair",
        "gray hair", "braided black hair", "red hair"]
FEATURES = ["gentle smile", "serious expression", "freckles",
            "strong jawline", "round face", "thin face", "expressive eyes"]
SCENE_TEMPLATES = [
    "{name} walking through a busy market street",
    "{name} sitting by a campfire at night",
    "{name} standing on a hill looking at the horizon",
    "{name} reading a book in an old library",
    "{name} riding a horse across a field",
    "{name} cooking in a rustic kitchen",
]


def build_random_characters(rng: random.Random, count: int):
    """Create `count` random (name, prompt) character definitions."""
    characters = []
    used_names = set()
    while len(characters) < count:
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        if name in used_names:
            continue
        used_names.add(name)
        prompt = (
            f"Portrait of {name}, a {rng.choice(ETHNICITIES)} "
            f"{rng.choice(GENDERS)}, {rng.choice(AGES)}, "
            f"{rng.choice(HAIR)}, {rng.choice(FEATURES)}. "
            "Photorealistic, detailed face, front facing, good lighting"
        )
        characters.append({"name": name, "prompt": prompt})
    return characters


def pct(hits: int, total: int) -> str:
    return f"{(100.0 * hits / total):.1f}%" if total else "n/a"


def run_benchmark():
    """Run the face recognition benchmark and return the results dict."""
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

        character = get_character(name)
        if character is None:
            character = generate_character(name, prompt, seed=RANDOM_SEED + i)

        # Scene featuring this character
        scene_prompt = rng.choice(SCENE_TEMPLATES).format(name=name)
        scene = generate_scene(scene_prompt, seed=RANDOM_SEED + 100 + i)

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


def summarize(results: dict) -> dict:
    entries = results["characters"]
    cross = results["cross_checks"]

    self_conclusive = [e for e in entries if e["self_check"] is not None]
    scene_conclusive = [e for e in entries if e["scene_check"] is not None]
    cross_conclusive = [c for c in cross if c["match"] is not None]

    return {
        "total_characters": len(entries),
        "self_check_pass": sum(1 for e in self_conclusive if e["self_check"]),
        "self_check_total": len(self_conclusive),
        "scene_check_pass": sum(1 for e in scene_conclusive if e["scene_check"]),
        "scene_check_total": len(scene_conclusive),
        "false_positives": sum(1 for c in cross_conclusive if c["match"]),
        "cross_check_total": len(cross_conclusive),
    }


def write_report(results: dict, summary: dict) -> Path:
    """Write Markdown + JSON reports for manual review. Returns md path."""
    report_dir = project_dir() / "face_benchmark"
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "report.json"
    with open(json_path, "w") as f:
        json.dump({"summary": summary, **results}, f, indent=2)

    def check(v):
        return {True: "PASS", False: "FAIL", None: "INCONCLUSIVE"}[v]

    lines = [
        "# Face Recognition Benchmark Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
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
    print("=== Face Recognition Benchmark ===")
    print(f"Characters: {NUM_CHARACTERS} (seed={RANDOM_SEED})")

    try:
        results = run_benchmark()
        summary = summarize(results)
        md_path = write_report(results, summary)

        print("\n=== Face Recognition Benchmark Summary ===")
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
