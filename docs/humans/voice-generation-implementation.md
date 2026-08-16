# Voice Generation Implementation Notes

> What worked and what didn't when wiring character voice generation into
> story-engine with the local Qwen3-TTS models. Companion to
> [character-voice-generation.md](character-voice-generation.md) (the
> reference-project investigation) — this is the story-engine implementation
> record. Updated Aug 2026.

## Status: WORKING (local, no cloud)

Characters can get a spoken introduction line — "Hi, my name is `<name>`, I'm
a `<character props>`" — generated fully locally and played in the UI.
**Prompt-based voice design works**: with a voice prompt, the VoiceDesign
model creates a brand-new timbre from the description.

- Models (both downloaded by `make install`):
  - `models/text_to_speech/qwen3_tts_voicedesign` = `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
    — **creates the voice from the prompt** (`generate_voice_design`)
  - `models/text_to_speech/qwen3_tts` = `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
    — preset timbres + delivery instruct (fallback for prompt-less generation)
- Engine: `core/voice_engine.py` (wrapper around the `qwen-tts` package)
- UI: mini player on the character detail screen (`QMediaPlayer` +
  `QAudioOutput`) with "Generate Voice" + "Regenerate with Prompt…"
- Storage: `outputs/<project>/characters/<name>/voice/voice.wav` + `voice_path`
  and `voice_prompt` columns on the `characters` table

## How the engine routes requests

`core/voice_engine.py::generate_character_voice()`:

```
generate_character_voice(name, attributes, instruct=None, force=False)
  ├─ instruct given AND VoiceDesign checkpoint present
  │    └─ VoiceDesign:  generate_voice_design(line, instruct, language)
  │                     → prompt CREATES the actual timbre
  ├─ instruct given but VoiceDesign missing
  │    └─ CustomVoice fallback: preset speaker + prompt as delivery instruct
  └─ no instruct (plain "Generate Voice")
       └─ CustomVoice: preset speaker (from attributes) + derived delivery
```

The WAV is cached (`voice/voice.wav`); pass `force=True` to regenerate.

## The working path (in order of operations)

1. **Install the package**
   `pip install qwen-tts` (v0.1.1). This **downgrades** pins:
   - `transformers` 5.14.1 → **4.57.3** (qwen-tts 0.1.1 requires `<5`)
   - `accelerate` 1.14.0 → **1.12.0** (requires `<1.13`)
   - `huggingface_hub` 1.26.0 → **0.36.2** (requires `<1.0`)
   `requirements.txt` was updated to the working pins with comments.
   **All existing tests pass** after the downgrades.

2. **Download the models with the project installer**
   `python scripts/install.py` — both TTS checkpoints are registered in
   `models.py` (`TEXT_TO_SPEECH_MODELS` + `MODEL_METADATA`), so
   `make install` fetches them into `models/text_to_speech/`.

3. **Load from the LOCAL directory** — do NOT pass the HF repo id (would
   re-download). The engine does this internally:

   ```python
   from qwen_tts import Qwen3TTSModel
   tts = Qwen3TTSModel.from_pretrained(
       "models/text_to_speech/qwen3_tts_voicedesign",  # or qwen3_tts
       device_map="mps", dtype="float16", attn_implementation="sdpa",
   )
   ```

   `Qwen3TTSTokenizer.from_pretrained` is **NOT** used — it crashes
   (`AutoFeatureExtractor` can't parse the `preprocessor_config.json`, which
   has no `feature_extractor_type`); the model wrapper's own `AutoProcessor`
   path works fine.

4. **Generate with a prompt** (VoiceDesign — the prompt creates the voice):

   ```python
   wavs, sr = tts.generate_voice_design(
       text="Hi, my name is Nikita, I'm a woman with an elegant presence.",
       instruct="A warm, elegant and feminine female voice with a calm, sophisticated presence. ...",
       language="English",
       non_streaming_mode=True,
   )
   # wavs[0]: np.ndarray float32, sr=24000
   ```

5. **Generate without a prompt** (CustomVoice — preset speaker + delivery):

   ```python
   wavs, sr = tts.generate_custom_voice(
       text="Hi, my name is Leila, I'm an elderly mediterranean woman.",
       speaker="serena",            # one of 9 presets in the checkpoint
       language="English",
       instruct="a warm, friendly voice, open and approachable",
       non_streaming_mode=True,
   )
   ```

6. **Write the WAV** with `soundfile` (`sf.write(path, wavs[0], sr)`).

Measured on this machine (Apple silicon, MPS): model load ~4 s, one sentence
~5–10 s, output 24 kHz mono WAV.

## Design decisions

| Decision | Why |
|---|---|
| **VoiceDesign when a prompt is given** | The whole point of a regeneration prompt is to *create* the voice. CustomVoice cannot do that — a prompt there only shapes delivery over a preset timbre. VoiceDesign (`generate_voice_design`) designs the timbre from the description. |
| CustomVoice fallback (no prompt / model missing) | Prompt-less "Generate Voice" has no description to design from, so a preset speaker + attribute-derived delivery is used. If VoiceDesign isn't downloaded, a prompt degrades gracefully to CustomVoice (prompt as instruct) with a log warning. |
| Preset speaker + `instruct`, not voice cloning | Cloning needs the Base checkpoint + a reference WAV (`create_voice_clone_prompt` + `generate_voice_clone`) — the reference project found Qwen3 cloning unstable and disabled it. Not needed for a greeting line. |
| Voice stored as a WAV + DB path + prompt | Matches the reference project's "persist one voice per character" pattern. The `voice_prompt` column records which prompt produced the voice. |
| Voice under `voice/` subfolder | Keeps the character folder organized: `reference.png` + `manifest.json` + `versions/` at the top, audio under `voice/voice.wav`. |
| `voice_path` / `voice_prompt` preserved on re-save | `save_character` uses `COALESCE(...)` so regenerating versions doesn't wipe the voice. |
| Attribute vocabulary tolerant | The builder screen stores a flat dict (`gender`, `age_range`, `mood`, ...) while the full attribute system uses `age`/`ethnicity`/`personality`. `voice_engine._attr()` reads either. |

## File-system layout

```
outputs/<project>/characters/<name>/
├── reference.png          # default version image
├── manifest.json
├── versions/              # v_1.png, v_2.png, ...
└── voice/
    └── voice.wav          # generated character voice
```

`outputs/` is gitignored; the DB records the absolute paths.

## Files changed

| File | Change |
|---|---|
| `core/voice_engine.py` | **new** — line building, speaker/instruct mapping, VoiceDesign + CustomVoice engines, lazy model singletons |
| `core/character_manager.py` | `generate_voice()` + `get_voice_path()`; `generate_versions(attributes=...)` |
| `services/database/character_service.py` | `voice_path` + `voice_prompt` columns (+ `ALTER TABLE` for existing DBs), `set_voice_path()`, save/get plumbing |
| `models.py` | `qwen3_tts_voicedesign` in `TEXT_TO_SPEECH_MODELS` + `MODEL_METADATA` |
| `ui/screens/character_view_screen.py` | mini player (Play/Stop), Generate Voice + Regenerate with Prompt… buttons, `_VoiceThread` |
| `ui/screens/character_builder_screen.py` | passes attributes into generation so the voice has props |
| `requirements.txt` | `qwen-tts` + downgraded pins |
| `tests/core/test_voice_engine.py` | **new** — 23 tests (line building, speaker selection, routing, persistence) |

## Known failures / what didn't work

| Attempt | Result | Notes |
|---|---|---|
| `Qwen3TTSTokenizer.from_pretrained(...)` | ❌ crashes | `AutoFeatureExtractor` fails on the repo's `preprocessor_config.json` (no `feature_extractor_type`). Unneeded — the model wrapper loads its own processor. |
| CustomVoice with a prompt only | ❌ wrong voice | The prompt shaped delivery but the **timbre stayed a preset** — this is why the user heard a male voice with a "female voice" prompt. Fixed by routing prompts to VoiceDesign. |
| `device_map="cpu"` / no `attn_implementation` | ❌ slow/unsupported path | `sdpa` + MPS is the fast path; the package warns if `flash-attn` is missing (fine, `sdpa` is used instead). |
| HF repo id as path | ❌ would re-download | Always point `from_pretrained` at the local `models/text_to_speech/<checkpoint>` dir. |
| Voice cloning from a reference WAV | ⏸ deferred | Reference project disabled Qwen3 cloning ("not giving good results and is not stable"). Revisit with the Base checkpoint if a character needs a locked, exact identity. |
| `soundfile` write to a non-existent dir | ❌ `LibsndfileError` | Trivial: ensure the output dir exists first (voice engine does). |
| SoX missing warning on import | ⚠️ cosmetic | `torchaudio`/`qwen_tts` prints "sox: command not found" on import. Benign — generation works fine without it. Can be silenced by installing `sox` (brew) if desired. |

## Where to go next

- **Consistent multi-line voice** (character says different lines across
  scenes with the same voice): design a reference WAV once with VoiceDesign,
  then clone it per line with the Base checkpoint
  (`create_voice_clone_prompt` → `generate_voice_clone`) — the "design → lock
  → consistent" pattern from `character-voice-generation.md`.
- **Prompt editing of an existing voice**: VoiceDesign generates a fresh
  voice per prompt; true identity-preserving edits (same voice, calmer) are
  still easier on cloud APIs.
- **Regenerate**: click "Regenerate with Prompt…" (a new prompt designs a new
  voice) or delete `voice/voice.wav` and click "Generate Voice" again.