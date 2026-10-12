# Audiobook Specification — Markup Tag Reference

This document describes every tag and attribute supported by the story markup language
used to produce immersive audiobooks. The markup compiles to the same internal structure
regardless of whether you use XML-like tags or bracket shorthand.

---

## `<scene>`

A scene boundary that groups segments into one audio scene. One or more `<scene>` elements
appear in the markup text; each becomes an `AudioSceneRepresentation` with its own timeline,
segments, and generated audio.

### Attributes

| Attribute | Required? | Description |
|---|---|---|
| `title` | No | Human-readable scene title. If omitted, defaults to `"Scene N"`. |

### Self-closing / nesting

- **Self-closing**: `<scene title="Opening"/>` — rare; normally `<scene>` is opened and closed
  explicitly or auto-closed at end of input.
- **Nested**: `<scene>` may wrap other tags (including other `<scene>` tags) for immersion,
  e.g. a scene wrapped inside a `<sound>` bed.

### Example

```xml
<scene title="The Maestro's Entrance">
  <character name="maestro" tone="confident">The orchestra takes the stage.</character>
  <sound preset="orchestra warm"/>
  <music prompt="soft strings"/>
</scene>
```

```text
[Scene: The Maestro's Entrance]
```

---

## `<character>`

A character tag defines a speaking voice with performance attributes. Every segment of text
between `<character>` tags (or bracket tokens targeting that speaker) inherits these
attributes unless overridden.

### Attributes

| Attribute | Required? | Description |
|---|---|---|
| `name` | Yes (for non-narrator) | The character's name. If the name matches a narrator alias (`narrator`,
  `narrador`, `narador`, `narator`), the segment is treated as narration. |
| `tone` | No | Voice quality/timbre cue (e.g. `"confident"`, `"mysterious"`, `"scary"`). |
| `emotion` / `feeling` | No | Emotional state. `emotion` is the preferred key; `feeling` is accepted as an alias. |
| `intensity` | No | Emotional intensity as a float 0.0–1.0 (0.0 = neutral, 1.0 = maximum). |
| `delivery` | No | Delivery style (e.g. `"fast"`, `"slow"`, `"dramatic"`). |
| `voice` | No | Explicit voice preset or designed voice override. If set, the voice generator
  uses this preset/prompt instead of auto-selecting from character attributes. |
| `speed` / `pace` | No | Speech speed multiplier: 0.5–2.0, where 1.0 = model's natural pace. |

### Self-closing behavior

`<character />` is **self-closing** and has no text; it is ignored (emits a parse issue).
Use `<character name="...">text</character>` for actual dialogue.

### Example

```xml
<character name="maestro" tone="confident" emotion="joyful" intensity="0.8" speed="1.1">
  The audience applauds.
</character>
```

```text
[maestro tone=confident emotion=joyful intensity=0.8 speed=1.1]The audience applauds.
```

---

## `<sound>` / `<sfx>`

A sound effect bed. Either a `preset` name or a `prompt` string must be provided.
The prompt/preset is passed to the dedicated sound-effects engine (Stable Audio Open) to generate a background bed
segment. `<sound>` can also contain nested `<character>` tags for dialogue within the bed.

### Attributes

| Attribute | Required? | Description |
|---|---|---|
| `preset` | No | A named music preset (e.g. `"adventure"`, `"orchestral"`). Mutually exclusive |
| | | with `prompt`; at least one must be provided. |
| `prompt` | No | A free-form text description used as the generation prompt. If no `preset` is |
| | | given, this string is used verbatim as the sound-effects prompt. |

### Example

```xml
<sound preset="adventure"/>
<sound prompt="a confusing swirling noise"/>
<sound prompt="rain on a roof">
  <character name="narrator" tone="mysterious">But then your hero shows up</character>
</sound>
```

```text
[sound preset=adventure]
[sound prompt=a confusing swirling noise]
```

---

### Local sound-effects model setup

Sound effects use [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0),
a text-to-audio model trained on field recordings and sound effects rather than
MusicGen's music-only training data. It generates up to 47 seconds of stereo
audio at 44.1 kHz through Diffusers, with `torchsde` for its cosine DPM scheduler.

Accept the model's gated license on Hugging Face and run `hf auth login` before
`make models` (or download only this checkpoint with `hf download
stabilityai/stable-audio-open-1.0 --local-dir models/sound_effects/stable_audio_open`).
Review the current Stability AI license for commercial-use eligibility; the
weights are not Apache/MIT licensed. Inference loads only the local checkpoint,
uses the available MPS/CUDA device or CPU, and uses float32 for numerical
stability. CPU inference is slow and accelerator inference requires several GB
of memory beyond the downloaded weights.
The installer downloads the full repository, including alternative checkpoint
formats, so disk usage is larger than the inference weights alone (the model
registry's download-size estimate is approximate).

Generation lives in `generators/sound_effects_engine.py`; cue routing and WAV
caching remain in `generators/sound_engine.py`. Cache keys include model identity
and cue type, so old MusicGen effects and music clips cannot be reused as new
sound effects. Missing effects weights skip effects without disabling music.

---

## `<music>`

A music bed segment. Like `<sound>`, either `preset` or `prompt` must be provided. The
generated music is mixed under the dialogue with the exporter's default ducking curve
(`MUSIC_VOLUME=0.30`), but can be adjusted per-scene.
Music continues to use MusicGen; it does not use the sound-effects model.

### Attributes

| Attribute | Required? | Description |
|---|---|---|
| `preset` | No | A named music preset (e.g. `"soft strings"`, `"jazz piano"`). |
| `prompt` | No | A free-form text description used as the generation prompt. |

### Example

```xml
<music prompt="soft strings"/>
<music preset="adventure"/>
```

```text
[music prompt=soft strings]
```

---

## Bracket shorthand

The bracket shorthand is a fast-to-type alternative to XML-like tags. It can be mixed with
XML tags; bracket tokens inside `<character>` text also work.

### Speaker switch

- `[Narrator]` / `[narrator]` — switch to narrator (or away from a character).
- `[Nikita]` / `[FeFe]` — switch to a named character. The character must be defined in
  the story; attributes carry forward until the next speaker switch.

### Performance attributes

- `[Feeling=Angry]` / `[Emotion=Angry]` — set emotion.
- `[Tone=cold]` — set tone.
- `[Intensity=0.8]` — set emotional intensity (0.0–1.0).
- `[Delivery=fast]` — set delivery style.
- `[Voice=serena]` — override the voice preset.
- `[Speed=1.2]` / `[Pace=1.2]` — set speech speed multiplier (0.5–2.0).

### Example (mixed)

```text
[Narrator] And so the story begins.
[maestro speed=1.1]The curtain rises.[Feeling=excited] Watch out!
[Speed=0.9]I tiptoed forward.
```

---

## Summary table

| Tag | Shorthand | Purpose |
|---|---|---|
| `<scene>` | `[Scene: title]` | Groups segments into one audio scene |
| `<character>` | `[Name]` | Defines a speaking voice with attributes |
| `<sound>` / `<sfx>` | `[sound prompt=...]` | Sound effect bed |
| `<music>` | `[music prompt=...]` | Music bed |
| Attributes | `[Key=value]` | Fine-tune performance per-segment |

---

## Migration note

The parser is deliberately tolerant: unknown tags/attributes are ignored rather than fatal,
unclosed tags are auto-closed at end of input, and plain text outside any tag becomes
narration. This allows authors to start with bracket shorthand and later add XML tags
for finer control without breaking existing content.