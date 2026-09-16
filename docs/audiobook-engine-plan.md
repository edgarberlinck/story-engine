# Story Engine — Audiobook & AI Story Engine

## Context

The video-generation experiment showed that image generation and character identity work well, while local video generation is currently limited by hardware and model constraints.

The primary direction should therefore shift from video generation to an **AI-powered audiobook and audio-story generation engine**.

Preserve and expand the existing character-generation and voice-generation capabilities.

Do not immediately implement everything. First inspect the current architecture and produce an implementation plan.

---

## 1. Characters, Objects and Artifacts

The system already supports generated characters.

Extend the same concept to **objects and artifacts**. Objects should be first-class story entities, just like characters.

Examples:

- Nikita's sword
- Roger's bag
- Edgar's car keys
- A specific car
- A magical artifact
- A photograph
- A recurring item belonging to a character

Objects need persistent identity and attributes.

Example:

```json
{
  "id": "nikita_sword",
  "name": "Astra",
  "type": "weapon",
  "description": "A long silver sword with a dark leather grip.",
  "owner": "nikita",
  "visual_identity": "...",
  "properties": {
    "material": "steel",
    "color": "silver"
  }
}
```

The image generator must recognize objects in the same conceptual way it recognizes characters.

If the same object appears in multiple scenes, its visual identity and characteristics must remain consistent.

The architecture should support reusable entities such as:

```text
Character
Object
Artifact
Location
```

Inspect the existing character-generation system and extend it rather than creating a parallel implementation.

---

## 1.1 Locations and Places

Locations should also be first-class story entities.

A story frequently returns to the same places, so locations need persistent identity and characteristics in the same way characters and objects do.

Examples:

- The Central Square
- Roger's Bedroom
- Nikita's Apartment
- The family house
- A specific restaurant
- A city
- The city of Azeroth
- A fictional kingdom
- A street
- A forest
- A spaceship

Locations may exist at different levels of scale and may contain other locations.

For example:

```text
Azeroth
  └── Capital City
       └── Central Square
            └── The Old Café
```

A location should have persistent information such as:

- Name
- Type
- Description
- Visual identity
- Parent location
- Child locations
- Geographic/contextual information where applicable
- Important characteristics
- Objects normally present
- Characters associated with the location

Example:

```json
{
  "id": "central_square",
  "name": "Central Square",
  "type": "public_square",
  "description": "A large stone plaza surrounded by old buildings.",
  "parent_location": "capital_city",
  "visual_identity": "...",
  "properties": {
    "architecture": "medieval",
    "surface": "stone",
    "atmosphere": "busy"
  }
}
```

The image generator must be able to recognize locations in the same way it recognizes characters and objects.

If a story returns to the same location in multiple scenes, the system should provide the location's persistent visual identity and characteristics to the image generator.

The location may change over time, but those changes should be explicit.

For example:

```text
Central Square
  Initial state:
    clean
    populated
    daytime

  Later state:
    damaged
    abandoned
    nighttime
```

The underlying location identity remains the same while its state can change according to the timeline.

Locations should therefore participate in the story state and timeline.

The system should distinguish between:

```text
Location Identity
```

and:

```text
Location State
```

This is important for continuity.

A scene should be able to reference:

```text
Characters
Objects
Location
Location State
Timeline Position
```

rather than requiring the LLM to reconstruct the entire environment from prose every time.

---

## 2. Story Engine

Create a proper story engine responsible for constructing the content of each scene.

The engine must understand:

- Scene
- Characters present
- Objects present
- Timeline position
- Events
- Actions
- Dialogue
- Narration
- Emotional state
- Tone
- Context

The engine should determine who is speaking.

Example:

```text
Scene:
Nikita enters the room.

Narrator:
"Nikita entered the room and immediately noticed the broken window."

Nikita:
"What happened here?"

Roger:
"I don't know. I heard a noise a few minutes ago."
```

Whenever no character is speaking, the narrator handles the narration.

### Narrator configuration

The narrator must be configurable.

The user should be able to:

- Select a dedicated narrator voice.
- Select a generated narrator voice.
- Select one of the existing characters as the narrator.

Examples:

```text
Narrator → Dedicated Narrator Voice
Narrator → Nikita's Voice
Narrator → Roger's Voice
```

This must be a project/story-level configuration.

---

## 3. Emotion, Feeling and Vocal Tone

Dialogue must contain performance information, not just text.

Example:

```json
{
  "character": "Nikita",
  "text": "Where are you going?",
  "emotion": "angry",
  "tone": "confrontational",
  "intensity": 0.8,
  "delivery": "fast and forceful"
}
```

Emotional state should be derived from scene context.

Examples:

Running:
```text
emotion: stressed
delivery: breathless
pace: fast
```

Angry:
```text
emotion: angry
delivery: tense, forceful
pace: fast
```

Sad:
```text
emotion: sad
delivery: quiet, slow
```

Afraid:
```text
emotion: frightened
delivery: shaky, breathless
```

Calm:
```text
emotion: calm
delivery: relaxed
```

Use only capabilities actually supported by the selected TTS/voice-generation model. Do not invent model capabilities.

The Story Engine should translate narrative context into controls supported by the selected backend.

---

## 4. Timeline

The story must maintain an explicit timeline.

The **user is responsible for directing the timeline**.

The system may assist and detect inconsistencies, but must not silently rewrite or reorder the user's timeline.

The user should be able to define:

- Chapter order
- Scene order
- Time
- Duration
- Relative temporal relationships
- Events
- Character presence
- Object state
- Timeline anchors

Example:

```text
Chapter 1
  Scene 1
    Monday — 08:00

  Scene 2
    Monday — 08:15

Chapter 2
  Scene 1
    Monday — 19:30
```

The Story Engine may warn about inconsistencies, but the user remains the authority over the timeline.

Timeline data must therefore be explicit project data, not merely inferred context.

---

## 5. Audio Scene Generation

Before generating audio, the engine must create an intermediate structured representation of every scene.

This representation should describe:

- Scene
- Narration
- Dialogue
- Speaker
- Emotion
- Tone
- Intensity
- Delivery
- Voice
- Timing/order
- Sound effects where applicable
- Music where applicable
- Characters present
- Objects present

Example:

```json
{
  "scene_id": "scene_001",
  "title": "The Morning",
  "timeline": {
    "day": 1,
    "time": "08:30"
  },
  "segments": [
    {
      "type": "narration",
      "speaker": "narrator",
      "text": "Nikita entered the kitchen and looked toward Roger.",
      "emotion": "neutral",
      "tone": "descriptive"
    },
    {
      "type": "dialogue",
      "speaker": "nikita",
      "text": "Good morning, Roger.",
      "emotion": "warm",
      "tone": "friendly",
      "intensity": 0.3,
      "delivery": "calm"
    },
    {
      "type": "dialogue",
      "speaker": "roger",
      "text": "Good morning, Nikita.",
      "emotion": "calm",
      "tone": "friendly",
      "intensity": 0.2,
      "delivery": "relaxed"
    }
  ]
}
```

The user must be able to inspect and modify this representation before audio generation.

The user should be able to change:

- Text
- Speaker
- Voice
- Emotion
- Tone
- Intensity
- Delivery
- Scene order
- Narration

The user may ask an LLM for assistance, for example:

> Make Nikita sound more irritated.

> Rewrite this dialogue so Roger sounds afraid but tries to hide it.

> Make the narration more cinematic.

The LLM should modify the structured scene representation rather than directly generating final audio.

Workflow:

```text
Story
  ↓
Story Engine
  ↓
Scene Representation
  ↓
User Review / Editing
  ↓
Optional LLM Assistance
  ↓
Approved Scene
  ↓
Voice Generation
  ↓
Audio Assembly
  ↓
Final Audio
```

---

## 6. Audio Generation

The audio-generation layer must be modular.

Different voice-generation/TTS models expose different capabilities.

Use a capability-based architecture.

Possible capabilities:

```text
voice_generation
tts
emotion_control
prosody_control
voice_cloning
```

A model may support only some of these capabilities.

Adapt the scene representation to the selected backend. Do not assume every TTS model supports every emotional or prosody parameter.

Generated audio must preserve the order defined by the scene representation.

The final output may combine:

```text
Narration
+
Character Dialogue
+
Sound Effects
+
Music
```

when those components are available.

---

## 7. User Interface

All functionality described above must be available through the UI.

### Characters

The UI should allow:

- Create characters
- Generate character appearance
- Edit character descriptions
- Configure voices
- Configure personality
- View character information

### Objects / Artifacts

The UI should allow:

- Create objects
- Generate object appearance
- Edit object descriptions
- Associate objects with characters
- Track recurring objects
- View object information

### Story

The UI should allow:

- Create chapters
- Create scenes
- Manage scenes
- Manage timeline
- Define characters present
- Define objects present
- Review generated content

### Scene Editor

The scene editor should expose:

- Narration
- Dialogue
- Speaker
- Voice
- Emotion
- Tone
- Intensity
- Delivery
- Scene order

The user must be able to manually modify these values.

### LLM Assistance

Provide an explicit mechanism for asking the LLM to modify or improve a scene.

Examples:

- Improve dialogue
- Change emotional tone
- Rewrite narration
- Add more tension
- Make dialogue more natural
- Adjust character behavior

The LLM should operate on the structured scene representation.

The user must be able to review the result before audio generation.

### Audio Preview

The UI should allow:

- Generate a preview of individual lines
- Preview a scene
- Regenerate a specific line
- Change voice
- Change emotion
- Change delivery
- Regenerate only the affected audio

The entire audiobook should not need to be regenerated when only one line changes.

---

## 8. Architecture

Inspect the existing Story Engine architecture before implementation.

Identify existing components for:

- Characters
- Image generation
- Voice generation
- LLM integration
- Tasks
- Task Manager
- Capabilities
- Storage
- UI
- Persistence

Reuse existing infrastructure wherever possible.

Extend the existing Story Engine instead of creating an unrelated audiobook subsystem.

Conceptually:

```text
                    STORY ENGINE
                         │
       ┌─────────────────┼─────────────────┬─────────────────┐
       │                 │                 │                 │
       ▼                 ▼                 ▼                 ▼
  Characters          Objects         Locations         Timeline
       │                 │                 │                 │
       └─────────────────┼─────────────────┼─────────────────┘
                         ▼
                    Story Engine
                         │
                         ▼
                  Scene Representation
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
          Narration   Dialogue    Metadata
              │          │
              └─────┬────┘
                    ▼
              User Review
                    │
              Optional LLM
                    │
                    ▼
             Voice Generation
                    │
                    ▼
             Audio Assembly
                    │
                    ▼
              Final Audiobook
```

---

## 9. Preserve Information

The system must NOT lose information during transformations.

Do not reduce a rich scene representation into plain text prematurely.

The intermediate representation should preserve:

- Character identity
- Object identity
- Scene context
- Timeline
- Dialogue
- Narration
- Emotion
- Tone
- Delivery
- Voice assignment
- Ordering
- Metadata

Use LLMs for semantic reasoning, interpretation, transformation and enrichment when appropriate.

Use deterministic application logic for:

- IDs
- Timeline state
- Relationships
- Persistence
- Resource/state tracking

Do not replace deterministic state with LLM guesses.

---

## 10. Planning Deliverable

Do NOT immediately implement the entire feature set.

First provide:

1. Analysis of the current architecture.
2. What can be reused.
3. What needs to change.
4. Proposed data model for Characters.
5. Proposed data model for Objects/Artifacts.
6. Proposed data model for Locations.
7. Proposed data model for Scenes.
8. Proposed data model for Timeline.
9. Proposed audio-scene intermediate representation.
10. Proposed voice-generation abstraction.
11. Proposed UI structure.
12. Proposed task/capability architecture.
13. Migration strategy from the current Story Engine.
14. MVP scope.
15. Features that should explicitly NOT be part of the MVP.

The implementation should be incremental.

Do not rewrite working parts of the Story Engine merely to make the architecture aesthetically cleaner.

The primary objective is to evolve the existing Story Engine into a robust **AI Story + Audiobook Engine**, preserving its existing strengths in character generation, image generation, voice generation and task orchestration.
