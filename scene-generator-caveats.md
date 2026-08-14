# Task: Plan a Fix for Multi-Character Scene Generation

I want you to analyze the current Story Engine codebase and create a detailed implementation plan for fixing an issue related to generating scenes containing multiple characters.

## Important

Do **not** implement anything yet.

Your task is to:

1. Investigate the current implementation.
2. Understand how scene prompts are constructed.
3. Identify the most likely cause of the issue described below.
4. Propose a clean architectural solution.
5. Create a detailed implementation plan that can later be used to guide another model or developer.

Do not modify the code.

---

# Background

Story Engine allows users to create characters with different attributes and visual styles.

Characters can then be used inside scenes.

Each character may contain attributes such as:

- Type
- Appearance
- Hair
- Clothing
- Physical characteristics
- Other descriptive attributes
- Style

The `Style` attribute can include values such as:

- Ultra Realistic
- Manga
- Comic
- Animation
- Other visual styles

The system currently generates images using diffusion models.

---

# The Problem

I discovered a problem when generating a scene containing multiple characters with different visual styles.

For example:

```text
Character 1:
Style: Ultra Realistic

Character 2:
Style: Manga 

The final artfact must be a file calles scene-generation-caveats.md