# Story Engine - Multi-Character Scene Generation Investigation and Planning

I need you to investigate and plan a solution for a problem in the Story Engine scene generation pipeline.

Do not immediately start implementing.

First, understand the current architecture, investigate the available models and generation capabilities, identify what is currently possible, and then create a proper implementation plan.

---

# Context

Story Engine is a local AI-powered application that generates characters and scenes using diffusion models.

Characters can be generated independently and have their own visual identity. A character may contain attributes such as gender, age, hair, skin tone, body characteristics, distinctive visual features, original clothing, and original generation style.

Scenes can contain one or multiple characters. The scene itself can define new clothing, actions, poses, positions, environment, camera angle, lighting, composition, and visual style.

This means that the original character generation prompt cannot simply be injected into the scene generation prompt.

Some information from the original character definition is important for identity preservation, while other information becomes irrelevant or is explicitly overridden by the scene.

For example, Nikita may originally have been generated using fantasy clothing. However, the scene may explicitly say that Nikita is wearing a black suit. In this case, the original "fantasy clothing" information is irrelevant and must not influence the scene.

---

# The Original Problem

We initially tried to generate a scene containing two characters:

- Nikita
- Roger

The scene prompt was:

> There's a stage, very small with red curtains. The stage is made of old cracked wood. In front of this stage there's some tables with people. On stage we have Nikita wearing a black suit, she is sitting in a chair playing a black Gibson Explorer guitar and Roger playing drums. The camera is positioned in the back of the bar, so we can see the crowd, the stage and the band.

The initial pipeline attempted to inject character information into the scene prompt.

This created a token budget problem. The generated prompt exceeded the CLIP context limit used by the diffusion pipeline.

For example, we received warnings such as:

```text
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (166 > 77).

Running this sequence through the model will result in indexing errors.

The following part of your input was truncated because CLIP can only handle sequences up to 77 tokens.
```

The character descriptions were consuming a large amount of the available prompt budget.

For example, Roger's injected character prompt contained information similar to:

```text
background Roger:
full body photo of a muscular male character,
age 30-40,
with short bald black hair,
dark skin tone and brown eyes,
wearing formal clothing,
serious expression,
photorealistic,
detailed face,
high detail,
good lighting,
standing upright,
entire body visible from head to toe including feet,
full length wide shot,
camera far from subject,
feet touching the ground visible in frame,
not a portrait,
not a close-up
```

Much of this information was irrelevant to the actual scene, including:

- Studio-like framing instructions
- Standing upright
- Full body instructions
- Feet touching the ground
- Original pose
- Original camera instructions
- Original environment
- Other generation-specific modifiers

These should not necessarily be carried into a scene generation prompt.

---

# First Attempt: Prompt Optimization

We decided that instead of blindly injecting the entire original character prompt, the pipeline should preserve only identity-defining attributes.

The idea was to separate information into three conceptual layers.

## 1. Scene Composition

This contains:

- Environment
- Camera position
- Shot type
- Lighting
- Composition
- Spatial relationships
- Audience/background
- Character positions

## 2. Character Identity

This should contain only attributes necessary to recognize and differentiate the character.

For example:

### Nikita

- Young woman
- Long curly red hair
- Fair skin

### Roger

- Muscular man
- Dark skin
- Short bald hair

## 3. Scene-Specific Appearance and Actions

The scene overrides the original character generation state.

For example:

### Nikita

- Wearing a black suit
- Playing a black Gibson Explorer
- Positioned on the left side of the stage

### Roger

- Wearing a dark formal outfit
- Sitting behind a drum kit
- Positioned on the right side of the stage

The principle is:

> Character identity should be preserved, but scene-specific instructions should override original character generation details.

---

# Optimized Prompt Test

We tested the following prompt:

```text
A small stage inside an old bar with red curtains and a floor made of old, cracked wood.

The camera is positioned at the back of the bar in a wide cinematic shot. In the foreground, several tables with people sitting and watching the performance are visible. The entire stage and both musicians must be visible.

On the left side of the stage is Nikita, a young woman with long curly red hair and fair skin. She is wearing a black suit and playing a black Gibson Explorer guitar.

On the right side of the stage is Roger, a muscular dark-skinned man with short bald hair. He is wearing a dark formal outfit and sitting behind a drum kit, playing the drums.

Nikita and Roger are two distinct characters performing together on the same stage. Keep both characters clearly visible and visually distinct.

Photorealistic, cinematic composition, detailed environment, natural stage lighting, wide shot showing the audience, the entire stage, Nikita, and Roger.
```

This solved some of the token-related problems and improved the scene composition.

However, the generated result still failed to correctly preserve both character identities.

The model generated a visually plausible scene with two musicians, but the musicians were essentially generic characters rather than clearly recognizable versions of Nikita and Roger.

This is the current problem.

---

# Important Finding

At this point, we should not assume that the problem can be solved simply by writing a longer or more detailed prompt.

The optimized prompt already explicitly described the environment, camera position, audience, stage, Nikita's identity, position, clothing and action, Roger's identity, position, clothing and action, and the requirement that both characters must be visible and distinct.

Despite this, the diffusion model still failed to reliably preserve the relationships between:

```text
Nikita
    -> red curly hair
    -> fair skin
    -> black suit
    -> black Gibson Explorer
    -> left side of the stage

Roger
    -> dark skin
    -> short bald hair
    -> formal clothing
    -> drum kit
    -> right side of the stage
```

The model appears capable of understanding these concepts individually. However, it may not reliably preserve all entity relationships when everything is generated in a single diffusion pass.

Increasing prompt complexity may make this worse rather than better.

---

# New Hypothesis: Progressive Scene Generation

We should investigate whether complex scenes should be generated progressively instead of attempting to generate everything in a single pass.

The conceptual pipeline would be:

```text
Scene
    |
    +-- Environment
    |
    +-- Camera / Composition
    |
    +-- Background / Crowd
    |
    +-- Main Character 1
    |
    +-- Main Character 2
    |
    +-- Main Character N
    |
    +-- Final Refinement
```

The idea is to construct the scene step by step.

---

# Proposed Generation Flow

## Step 1: Generate the Base Environment

Generate the environment and composition without the main characters.

Example:

```text
A small stage inside an old bar with red curtains and a floor made of old, cracked wood.

The camera is positioned at the back of the bar in a wide cinematic shot.

Several tables with people sitting and watching the performance are visible in the foreground.

The entire stage is clearly visible.

The stage is currently empty.
```

The goal of this step is to establish:

- Environment
- Camera
- Lighting
- Composition
- Audience
- Spatial layout

No main characters should be generated at this stage.

---

## Step 2: Add Nikita

Starting from the generated environment, investigate whether an image editing, image-to-image, inpainting, masking, or another appropriate technique can insert Nikita into the scene.

Example:

```text
Add Nikita to the left side of the stage.

Nikita is a young woman with long curly red hair and fair skin.

She is wearing a black suit and playing a black Gibson Explorer guitar.

Preserve the existing environment, camera angle, audience, lighting, and composition.

Do not modify the rest of the image.
```

---

## Step 3: Add Roger

Starting from the result of the previous step:

```text
Add Roger to the right side of the stage.

Roger is a muscular dark-skinned man with short bald hair.

He is wearing a dark formal outfit and sitting behind a drum kit, playing the drums.

Preserve Nikita, the environment, the audience, camera angle, lighting, and existing composition.

Do not replace or modify existing characters.
```

---

## Step 4: Optional Refinement

If necessary, run a final refinement pass.

The purpose is not to regenerate the scene.

The refinement should improve:

- Lighting consistency
- Shadows
- Character/environment integration
- Visual coherence
- Obvious artifacts

The refinement must preserve:

- Character identities
- Character positions
- Character actions
- Overall composition

---

# Character Reference Investigation

Another important possibility is using the previously generated character image as a visual reference.

Please investigate whether the currently available models or pipelines support mechanisms such as:

- Image-to-image
- Reference images
- Identity preservation
- IP-Adapter
- InstantID
- ControlNet
- Inpainting
- Masked generation
- Other compatible conditioning mechanisms

The goal is not to assume that any specific technology is available.

Investigate what already exists in the project and what the selected models actually support.

If character reference images can be used, this may significantly improve identity preservation compared with relying only on textual descriptions.

---

# LLM Orchestration

We already have LLM models available in the pipeline.

We should use them to reason about and orchestrate scene generation instead of implementing complex hard-coded heuristics.

The LLM can analyze:

1. The scene description.
2. The selected diffusion model.
3. The available token budget.
4. The characters involved.
5. The original character definitions.
6. The scene-specific instructions.
7. The available generation capabilities.

The LLM should determine:

- Whether single-pass generation is sufficient.
- Whether progressive generation is necessary.
- Which character attributes are essential.
- Which original character attributes are irrelevant.
- Which attributes are overridden by the scene.
- How the prompt should be compressed.
- Which generation strategy is appropriate.

For example, before generation, an LLM could produce a structured plan:

```json
{
  "strategy": "progressive_generation",
  "scene": {
    "environment": "small old bar with red curtains and cracked wooden stage",
    "camera": "wide shot from the back of the bar",
    "composition": "audience in foreground, entire stage visible"
  },
  "characters": [
    {
      "name": "Nikita",
      "identity": [
        "young woman",
        "long curly red hair",
        "fair skin"
      ],
      "position": "left side of the stage",
      "clothing": "black suit",
      "action": "playing a black Gibson Explorer"
    },
    {
      "name": "Roger",
      "identity": [
        "muscular man",
        "dark skin",
        "short bald hair"
      ],
      "position": "right side of the stage",
      "clothing": "dark formal outfit",
      "action": "sitting behind a drum kit"
    }
  ],
  "style": "photorealistic"
}
```

This is only an example of the planning stage.

Do not blindly implement this exact structure if the existing architecture suggests a better design.

---

# Style Rules

The scene must have one coherent rendering style.

Character-specific generation styles must not automatically override the selected scene style.

For example:

- Nikita may originally have been generated as manga.
- Roger may originally have been generated photorealistically.

If the scene style is photorealistic, both characters should be rendered photorealistically while preserving their recognizable identity.

The principle is:

```text
Scene style controls HOW everything is rendered.

Character identity controls WHO the character is.
```

Original character style should not dominate scene generation unless explicitly requested.

---

# Token Budget

We must not blindly concatenate:

```text
Scene Prompt
+
Character 1 Full Prompt
+
Character 2 Full Prompt
+
Character 3 Full Prompt
...
```

This will quickly exceed the context limitations of models such as CLIP-based diffusion pipelines.

Instead:

1. Extract semantic information.
2. Remove irrelevant information.
3. Apply scene overrides.
4. Preserve identity-defining features.
5. Generate a compact prompt appropriate for the selected model.

The LLM should be used for semantic compression when appropriate.

We do not need to create complex hard-coded heuristics for every possible attribute.

The important rule is:

> We must not lose important information, but we also must not blindly carry irrelevant information into the scene.

---

# Investigation Requirements

Before implementation, investigate:

## Current Project

- Existing scene generation architecture.
- Current diffusion models.
- Existing image generation pipelines.
- Available image editing capabilities.
- Existing prompt construction logic.
- Existing LLM orchestration capabilities.

## Model Capabilities

Determine which available models support:

- Text-to-image
- Image-to-image
- Inpainting
- Masking
- Reference images
- Identity preservation
- Control mechanisms
- Multi-stage generation

## Architecture

Evaluate whether the best approach is:

1. Improved single-pass generation.
2. Progressive generation.
3. Image editing/inpainting.
4. Reference-image conditioning.
5. A hybrid strategy.
6. Different strategies depending on scene complexity and model capabilities.

---

# Expected Outcome

Do not start by implementing random changes.

First:

1. Investigate the current architecture.
2. Identify the available capabilities.
3. Analyze the failure mode.
4. Determine whether the issue is primarily prompt-related or model-capability-related.
5. Propose the best sustainable architecture.

Then create a detailed implementation plan.

The Story Engine should eventually be capable of deciding how to generate a scene based on its complexity.

Simple scenes may use:

```text
Single-pass generation
```

More complex scenes may use:

```text
Scene planning
    ->
Base environment generation
    ->
Character insertion
    ->
Character insertion
    ->
Refinement
```

Or another strategy if the investigation shows that a better mechanism is available.

---

# Core Principle

The current approach assumes:

> Generate everything correctly in one shot.

The architecture should investigate a more flexible approach:

> Build the scene progressively when the complexity exceeds what the selected model can reliably generate in a single pass.

The system should not blindly apply the same generation strategy to every scene.

Use the available LLM models to help with planning, semantic extraction, prompt optimization, and generation orchestration.

Avoid unnecessary hard-coded heuristics when an LLM can reason about the semantic structure of the scene.

The goal is a sustainable architecture that can evolve as new models and generation capabilities become available.

---

# Alternative Strategy: Generate Character Assets First and Compose the Scene

We should also investigate a potentially more reliable strategy.

Instead of asking a diffusion model to generate the entire complex scene in one pass, we can reuse the existing Character Generator to generate each character specifically for the scene.

The current problem is that a single diffusion prompt may need to simultaneously handle:

```text
Environment
+
Camera
+
Audience
+
Stage
+
Character 1 identity
+
Character 1 clothing
+
Character 1 pose
+
Character 1 action
+
Character 1 props
+
Character 2 identity
+
Character 2 clothing
+
Character 2 pose
+
Character 2 action
+
Character 2 props
+
Lighting
+
Composition
```

This creates both a prompt-budget problem and an entity-binding problem.

A different strategy would be to split the problem into independent responsibilities.

## Proposed Pipeline

```text
Scene Request
      |
      v
Scene Planner / LLM
      |
      +------------------------------+
      |                              |
      v                              v
Generate Background           Generate Character Assets
      |                              |
      |                              +--> Character 1
      |                              |    identity
      |                              |    scene clothing
      |                              |    pose
      |                              |    action
      |                              |    props
      |                              |    framing
      |                              |
      |                              +--> Character 2
      |                                   identity
      |                                   scene clothing
      |                                   pose
      |                                   action
      |                                   props
      |                                   framing
      |                             
      +---------------+--------------+
                      |
                      v
              Scene Composition
                      |
                      v
           Optional AI Refinement
                      |
                      v
                 Final Scene
```

The important idea is that the Character Generator should not only generate the original canonical version of a character.

It should also be capable of generating scene-specific character assets.

For example:

```json
{
  "character": "Nikita",
  "identity": {
    "gender": "woman",
    "hair": "long curly red hair",
    "skin": "fair"
  },
  "scene_context": {
    "clothing": "black suit",
    "pose": "sitting on a chair",
    "action": "playing a black Gibson Explorer guitar",
    "camera": "wide shot",
    "view": "full body"
  }
}
```

The Character Generator would then generate Nikita already prepared for the requested scene.

Roger would be generated independently using the same approach.

## Character Asset Generation

For a scene, each character asset should preserve:

- Core visual identity
- Distinctive physical characteristics
- Scene-specific clothing
- Pose
- Action
- Required props
- Appropriate camera/framing for composition

The character asset should NOT preserve irrelevant details from the original character generation.

For example, if the original character was generated standing in fantasy clothing, but the scene requires the character to sit and wear a black suit, the scene-specific requirements override those original generation details.

Conceptually:

```text
Canonical Character
        +
Scene Requirements
        =
Scene-Specific Character Asset
```

Examples:

```text
Nikita standing
Nikita sitting
Nikita running
Nikita playing guitar
Nikita looking surprised
Nikita walking through a door
```

All of these should attempt to preserve the same character identity while allowing pose, action, clothing, and context to change.

## Background Isolation and Transparency

Investigate whether the available generation stack can generate character assets with:

- Transparent backgrounds
- Simple removable backgrounds
- Easily segmentable backgrounds

If native transparent generation is not supported, investigate whether background removal or segmentation can be performed after generation.

The desired result is a character asset that can be treated as an independent visual layer.

For example:

```text
Nikita
  |
  +-- Correct identity
  +-- Correct clothing
  +-- Correct pose
  +-- Correct action
  +-- Correct props
  +-- Isolated / transparent background
```

The same approach would apply to Roger and any additional characters.

## Deterministic Scene Composition

Once the environment and character assets exist independently, the scene can be assembled using a deterministic composition layer.

This layer could be responsible for:

- Positioning characters
- Scaling characters
- Layer ordering
- Foreground/background placement
- Basic perspective adjustments
- Cropping
- Placement relative to scene landmarks
- Shadows
- Blending

This is important because these responsibilities do not necessarily need to be delegated to a diffusion model.

Traditional deterministic image composition may provide more reliable control over spatial placement than asking a diffusion model to infer all spatial relationships from text.

For example:

```text
Background Scene
      |
      +--> Place Nikita on left side
      |
      +--> Scale Nikita based on scene perspective
      |
      +--> Place Roger on right side
      |
      +--> Scale Roger based on scene perspective
      |
      +--> Define layer ordering
      |
      +--> Add or approximate shadows
      |
      v
Composed Scene
```

This would solve part of the multi-character binding problem because the diffusion model would no longer be responsible for deciding where every character belongs.

The composition system would explicitly control placement.

## Optional AI Refinement

After deterministic composition, an optional AI refinement step may improve:

- Lighting consistency
- Shadows
- Integration between character and environment
- Texture consistency
- Visual coherence
- Minor artifacts

However, refinement should be optional.

It must not be treated as a mandatory step that regenerates the entire scene.

The system should preserve the composed structure as much as possible.

The refinement stage should receive explicit constraints to preserve:

- Character identity
- Character pose
- Clothing
- Props
- Character positions
- Scene composition

A conceptual refinement instruction could be:

```text
Integrate the characters naturally into the environment while preserving their identity, pose, clothing, instruments, positions, and overall composition.
```

If refinement introduces unacceptable changes, the deterministic composition should still remain usable as the final result.

## Investigation Requirements for This Strategy

Before implementation, investigate:

### Character Generator

- Can the existing Character Generator accept scene-specific pose and action requirements?
- Can it preserve identity while changing pose and clothing?
- Can it generate full-body assets suitable for composition?
- Can it use existing character images as references?

### Background Removal

- Is transparent generation available?
- Is there an existing segmentation or background removal pipeline?
- Which available local models or tools can isolate generated characters?

### Composition

Investigate the most appropriate local composition approach.

Determine whether the project can support:

- Layer-based image composition
- Scaling
- Positioning
- Basic perspective adjustments
- Layer ordering
- Shadows and blending

The implementation should prefer deterministic operations whenever deterministic control is sufficient.

### Refinement

Investigate whether any available model supports refining a composed image while preserving:

- Structure
- Character identity
- Pose
- Object placement

## Architectural Principle

The Story Engine should not assume that every complex visual problem must be solved by a single diffusion prompt.

A complex scene can be treated as a composition of independently generated visual assets.

The diffusion model may be strongest at generating:

```text
One character
+
One pose
+
One action
+
A controlled visual context
```

A deterministic composition system may be stronger at:

```text
Where the character goes
+
How large the character is
+
What appears in front or behind
+
How multiple characters coexist spatially
```

AI refinement can then be used selectively where generative capabilities provide actual value.

## Relationship to Progressive Generation

This strategy should be evaluated alongside the previously proposed progressive generation approach.

Possible strategies include:

1. Single-pass generation for simple scenes.
2. Progressive inpainting or image editing for moderately complex scenes.
3. Character asset generation plus deterministic composition for scenes requiring strong spatial control and multiple characters.
4. Hybrid pipelines combining composition with optional AI refinement.

The Scene Planner should eventually be capable of selecting the appropriate strategy based on:

- Number of characters
- Scene complexity
- Required identity preservation
- Available model capabilities
- Selected visual style
- Required spatial precision

## Core Principle

Instead of always asking:

> Can the diffusion model generate the entire scene correctly?

The system should also consider:

> Which parts of the scene should be generated, and which parts should be composed deterministically?

The goal is not to force one generation strategy to solve every problem.

The goal is to build a flexible visual production pipeline where each component is used for the type of work it performs most reliably.

