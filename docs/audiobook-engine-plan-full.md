# Story Engine — Audiobook & AI Story Engine: Full Implementation Plan

## Overview

This document provides the complete implementation plan for evolving the Story Engine into a robust AI Story + Audiobook Engine. It preserves existing character generation, image generation, voice generation, and task orchestration capabilities while adding objects, locations, timeline management, and audio scene generation.

---

## 1. Analysis of Current Architecture

### 1.1 Existing Components (Reusable)

| Area | Existing Implementation | Reuse Notes |
|------|------------------------|-------------|
| **Database (SQLite)** | `story_engine.db` with tables: `projects`, `characters`, `character_versions`, `character_attributes`, `scenes` | Extended with new tables for objects, locations, timeline |
| **Characters** | `CharacterService` — CRUD, attributes, voice path management | Pattern reused for Objects and Locations |
| **Scene Planning** | `core/scene_planner.py` — 4-stage LLM delegation (Stages A-D) | Adapted for object/location integration |
| **Voice Generation** | `core/voice_engine.py` — Qwen3-TTS CustomVoice + VoiceDesign | Extended with emotion/tone/intensity controls |
| **Image Generation** | `generators/image_engine.py` — SDXL, FLUX dev/klein, asset composition | Object/location visual identity generation added |
| **UI** | PyQt6 stacked widget: `ProjectListScreen` → `ProjectViewScreen` → `CharacterBuilderScreen`/`CharacterViewScreen` | New tabs/screens added incrementally |
| **Services** | `ProjectService`, `CharacterService`, `SceneService`, `DatabaseService` | CRUD patterns extended |
| **Model constants** | `models.py` — TTS models catalogued | Extended with capability metadata |
| **Token budget** | `utils/token_budget.py` | Reused for audio scene prompts |
| **LLM planning** | `core/scene_planner.py` — Stages A-D pipeline | Adapted for story engine scene construction |

### 1.2 Limitations to Address
- No objects/artifacts or locations as first-class entities
- No explicit timeline management
- No emotion/tone/intensity in dialogue beyond basic personality/expression
- No structured audio-scene intermediate representation
- No capability-based TTS abstraction
- UI lacks object/location management and scene editor features

---

## 2. What Can Be Reused

- **SQLite schema** — extend existing tables, add new ones for objects/locations/timeline
- **`CharacterService` pattern** — reuse CRUD for Objects and Locations
- **`SceneService`** — extend to track audio scene representations
- **`VoiceEngine`** — extend with emotion/tone/intensity controls
- **LLM scene planning pipeline** (Stages A-D) — adapt for story engine
- **PyQt6 UI framework** — reuse navigation, add new screens/tabs
- **Model constants** — TTS models already catalogued
- **Token budget management** — existing `utils/token_budget.py`
- **Image generation** — object/location visual identities generated similarly to characters

---

## 3. What Needs to Change / Add

### 3.1 New Data Models

#### 3.1.1 Characters (extend existing)
```json
{
  "id": "nikita",
  "name": "Nikita",
  "type": "character",
  "description": "A curious teenager...",
  "voice_path": "...",
  "voice_prompt": "...",
  "visual_identity": "...",
  "properties": {
    "age": "teenager",
    "ethnicity": "Mediterranean",
    "personality": "curious",
    "expression": "friendly"
  }
}
```

#### 3.1.2 Objects/Artifacts (NEW)
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

#### 3.1.3 Locations (NEW)
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
  },
  "state": "clean",
  "timeline_changes": [...],
  "objects_present": ["nikita_sword"],
  "characters_associated": ["nikita"]
}
```

#### 3.1.4 Scenes (extend existing)
```json
{
  "id": "scene_001",
  "scene_number": 1,
  "title": "The Morning",
  "timeline": {"day": 1, "time": "08:30"},
  "segments": [...],  // audio scene representation
  "characters_present": ["nikita", "roger"],
  "objects_present": ["nikita_sword"],
  "location": "central_square"
}
```

#### 3.1.5 Timeline (NEW)
```json
{
  "chapters": [
    {
      "id": "chapter_1",
      "number": 1,
      "title": "Chapter 1",
      "scenes": [
        {"id": "scene_001", "number": 1, "day": 1, "time": "08:00"},
        {"id": "scene_002", "number": 2, "day": 1, "time": "08:15"}
      ]
    }
  ],
  "anchors": [...],
  "events": [...]
}
```

### 3.2 New Infrastructure

- **Story Engine** — constructs scene content: who speaks, narration, dialogue, emotional state, tone
- **Capability-based TTS abstraction** — detect which emotion/prosody parameters a model supports; adapt scene representation
- **Object/Location image generation** — recognize objects/locations in image generation same way characters are recognized
- **Scene editor UI** — exposes narration, dialogue, speaker, voice, emotion, tone, intensity, delivery
- **LLM assistance mechanism** — operating on structured scene representation

---

## 4. Proposed Audio-Scene Intermediate Representation

```json
{
  "scene_id": "scene_001",
  "title": "The Morning",
  "timeline": {"day": 1, "time": "08:30"},
  "segments": [
    {
      "type": "narration",
      "speaker": "narrator",
      "text": "Nikita entered the kitchen and looked toward Roger.",
      "emotion": "neutral",
      "tone": "descriptive",
      "intensity": 0.3,
      "delivery": "calm",
      "voice": "nikita",
      "timing": {"start": 0.0, "end": 2.5}
    },
    {
      "type": "dialogue",
      "speaker": "nikita",
      "text": "Good morning, Roger.",
      "emotion": "warm",
      "tone": "friendly",
      "intensity": 0.3,
      "delivery": "calm",
      "voice": "nikita_voice",
      "timing": {"start": 2.5, "end": 4.0}
    }
  ]
}
```

User can inspect and modify this representation before audio generation. Supports LLM assistance commands like:
- "Make Nikita sound more irritated."
- "Rewrite this dialogue so Roger sounds afraid but tries to hide it."
- "Make the narration more cinematic."

---

## 5. Proposed Voice-Generation Abstraction

### 5.1 Capability-based Architecture (from plan §4, §6)

| Capability | Description | Supported Models |
|------------|-------------|-----------------|
| `voice_generation` | Basic TTS synthesis | all TTS models |
| `tts` | Text-to-speech output | all TTS models |
| `emotion_control` | Emotional expression in output | qwen3_tts (via instruct), qwen3_tts_voicedesign |
| `prosody_control` | Prosody/paralinguistic controls | limited |
| `voice_cloning` | Clone from reference audio | qwen3_tts_voicedesign |

### 5.2 Model Capability Mapping

- `qwen3_tts` (CustomVoice 1.7B): `voice_generation`, `tts`, basic `emotion_control` via `instruct`, limited `prosody_control`
- `qwen3_tts_voicedesign`: `voice_generation`, `tts`, `voice_cloning` (designs new voice from prompt)
- `qwen3_tts_base`: `voice_generation`, `tts` only (lighter weight)

### 5.3 Abstraction Layer

Adapts scene representation to selected backend. Does NOT assume every TTS model supports every emotional or prosody parameter. Maps scene-level emotion/tone/intensity/delivery to what the selected backend supports.

---

## 6. Proposed UI Structure

### 6.1 New Screens/Tabs

| Screen | Features |
|--------|----------|
| `ObjectManagerScreen` | Create objects, generate appearance, edit descriptions, associate with characters, track recurring objects, view object information |
| `LocationManagerScreen` | Create locations, edit descriptions, manage parent/child hierarchy, track state changes (clean/damaged, day/night), view objects/characters present |
| `SceneEditorScreen` | Expose: narration, dialogue, speaker, voice, emotion, tone, intensity, delivery, scene order; user can manually modify all values |
| `TimelineScreen` | Chapter/scene order management, time/duration editing, relative temporal relationships, timeline anchors |
| `NarratorConfigurationScreen` | Select dedicated narrator voice, generated narrator voice, or existing character as narrator (project-level) |

### 6.2 Updated ProjectViewScreen
- Add tabs for Objects and Locations alongside existing Characters tab

### 6.3 Scene Editor
Must expose all segment properties from the audio-scene representation for user review/editing before audio generation.

### 6.4 Audio Preview UI
- Generate preview of individual lines
- Preview a scene
- Regenerate a specific line
- Change voice, emotion, delivery
- Regenerate only the affected audio (entire audiobook not regenerated when one line changes)

---

## 7. Proposed Task/Capability Architecture

```
STORY ENGINE
│
├─ Characters (reused from existing)
├─ Objects/Artifacts (new)
├─ Locations (new)
└─ Timeline (new)

    │
Story Engine — constructs scene content
    │
Scene Representation (audio-scene JSON) — structured intermediate form
    │
│   ├─ Narration segments
│   ├─ Dialogue segments
│   └─ Metadata (emotion, tone, intensity, delivery, voice, timing)

    │
Voice Generation Abstraction — capability-based
    │   │
    │   ├─ Model capability detection
    │   ├─ Adapt scene representation to backend
    │   └─ Preserve order, combine components

    │
Audio Assembly — combine narration + dialogue + SFX + music
    │
    └─ Final audio output
```

**Task orchestration** follows existing pattern: `generate_scene_pipeline()` selects strategy, runs planning, executes pipeline with fallbacks.

---

## 8. Migration Strategy from Current Story Engine

1. **Database migration** — add new tables: `objects`, `locations`, `timeline`, `audio_scene_representations`; add columns to existing tables if needed
2. **Character backward compatibility** — existing characters continue to work unchanged
3. **Objects inherit character patterns** — reuse `CharacterService` CRUD pattern, store in `objects` table
4. **Locations inherit character patterns** — reuse CRUD pattern, store in `locations` table with parent_id self-referential foreign key
5. **Timeline data** — user defines explicitly; existing scene `scene_number` ordering can be used as base
6. **Audio scene representations** — stored alongside existing scene metadata; existing generated scenes remain unchanged
7. **Voice generation** — existing voice paths preserved; new emotion/tone controls optional
8. **Image generation** — object/location visual identities generated similarly to characters; existing character image generation unchanged

---

## 9. MVP Scope

### Must-have for MVP
- [x] Objects/Artifacts data model and CRUD (store/retrieve, associate with characters)
- [x] Locations data model and CRUD (store/retrieve, hierarchy, state tracking)
- [x] Timeline data model and basic management (chapter/scene order, time/duration)
- [x] Audio-scene intermediate representation (structured JSON with segments)
- [x] Extended voice generation with emotion/tone/intensity controls (within existing TTS capabilities)
- [x] Scene editor UI exposing: narration, dialogue, speaker, voice, emotion, tone, intensity, delivery
- [x] Narrator configuration (select from dedicated voice, generated voice, or existing character)
- [x] LLM assistance on structured scene representation (modify dialogue emotion, rewrite narration)
- [x] Audio preview UI (generate preview of individual lines, regenerate specific line)

### Nice-to-have (post-MVP)
- Full object/location image generation with consistent visual identity
- Advanced timeline inconsistency detection
- Multi-backend TTS capability adaptation
- Full audiobook assembly (narration + dialogue + SFX + music)
- Collaborative features, project sharing

---

## 10. Features Explicitly NOT Part of MVP

- ❌ Full video generation (local video generation hardware-limited per plan §1)
- ❌ Complete rewrite of existing character/image generation pipeline
- ❌ All possible emotion/prosody parameters for every TTS model (only support what's actually available)
- ❌ Real-time voice cloning from arbitrary audio (only designed voice from prompts)
- ❌ Automatic timeline consistency enforcement (user remains authority; system only warns per plan §5)
- ❌ Procedural generation of story content from scratch (LLM assistance on structured representation only per plan §4)
- ❌ Multi-language audio generation (start with English only)
- ❌ Real-time lip sync (existing i2v models archived due to hardware constraints per models.py)
- ❌ Music generation system (MusicGen exists but not integrated into MVP audio pipeline per plan §6)
- ❌ Full UI redesign (reuse existing framework, add new tabs/screens incrementally)
- ❌ Deterministic state replacing all LLM guesses everywhere (use LLMs for semantic reasoning, deterministic for IDs/timeline/relationships per plan §9)