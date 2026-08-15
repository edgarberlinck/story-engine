# Token Budget Implementation Summary

## Root Cause Identified

The Scene Generator was incorrectly reusing complete character generation prompts in scene contexts, causing:

1. **Token overflow**: Prompts exceeded CLIP's 77 token limit (reaching 166+ tokens)
2. **Contradictory instructions**: Character generation instructions like `standing upright` and `entire body visible` conflict with scene actions like `sitting behind drum kit`
3. **Irrelevant information**: Technical quality modifiers (`high detail`, `photorealistic`) consume token budget without benefit in scenes

## Solution Implemented

### 1. Token Budget Manager
**File**: `utils/token_budget.py`

New module provides:
- Token counting with word-based estimation (approximate for CLIP)
- Generation instruction stripping via regex patterns
- Priority-based prompt construction
- Logging of dropped items and token statistics

**Key patterns stripped**:
```python
CHARACTER_GEN_INSTRUCTIONS = [
    r"standing upright", r"entire body visible", r"full body",
    r"feet touching the ground", r"wide shot", r"not a portrait",
    r"high detail", r"detailed face", r"photorealistic"
]
```

### 2. Prompt Decomposer Enhancement
**File**: `core/prompt_decomposer.py`

Enhanced `extract_appearance_from_stored_prompt()` to:
- Call `strip_generation_instructions()` after decomposition
- Remove generation artifacts while preserving appearance attributes

### 3. Scene Generator Integration
**File**: `generators/image_engine.py`

Changes made:
1. **`_enrich_scene_prompt()`**: Now uses token-aware building for multi-character scenes
2. **`generate_scene()`**: Added token budget enforcement phase
3. **Result logging**: Token counts added to return values

## Prompt Examples

### Before Fix (166 tokens, truncated)
```
Roger is sitting behind a drum kit playing drums.

Roger appearance: full body photo of a muscular male character, age 30-40,
with short bald black hair, dark skin tone and brown eyes,
wearing formal clothing, serious expression, photorealistic,
detailed face, high detail, good lighting.
standing upright, entire body visible from head to toe including feet,
full length wide shot, camera far from subject,
feet touching the ground visible in frame,
not a portrait, not a close-up
```
**Warning**: `[transformers] Token indices sequence length is longer than specified maximum (166 > 77)`

### After Fix (~50 tokens, within limits)
```
Roger is sitting behind a drum kit playing drums. Roger, muscular male character, age 30-40, short bald black hair, dark skin tone and brown eyes, wearing formal clothing, serious expression
```
**Token count**: ~50 tokens (within 77 limit)

## Validation Tests

Created comprehensive tests:

1. **`tests/test_token_budget.py`** - Unit tests for token counting and instruction stripping
2. **`tests/test_scene_token_limits.py`** - Integration tests for scene generation scenarios

Test cases cover:
- Single character scenes
- Two character scenes with actions
- Long verbose descriptions with prioritization
- Generation instruction removal validation

All tests passing ✓

## Architecture Benefits

1. **Token awareness**: Prompts now respect CLIP's 77 token limit
2. **Information preservation**: Critical details (identity, actions, appearance) prioritized
3. **Clean separation**: Character generation vs scene description properly separated
4. **Debugging visibility**: Token stats logged for inspection
5. **Backward compatible**: Falls back to original behavior if needed

## Files Modified

1. `utils/token_budget.py` - NEW (token management)
2. `core/prompt_decomposer.py` - Enhanced extraction
3. `generators/image_engine.py` - Integrated token budgeting
4. `tests/test_token_budget.py` - NEW (unit tests)
5. `tests/test_scene_token_limits.py` - NEW (integration tests)

## Remaining Considerations

1. **FLUX vs CLIP**: FLUX model uses 512 sequence length, but warning mentions CLIP limit of 77. May need model-specific handling.

2. **Token estimation accuracy**: Current implementation uses word-based heuristics. Could integrate actual CLIP tokenizer for precise counting.

3. **Character type inference**: Currently defaults to "man". Could infer from stored attributes or database.

4. **Multi-step generation**: Token budget solves truncation but complex scenes may still benefit from multi-pass generation (layout → characters).

## Usage

The fix is automatic - no API changes required. Scene generation now:
- Automatically strips generation instructions
- Enforces token budgets
- Logs token statistics
- Preserves critical information

Example output:
```
Token analysis: Original=166, Token-aware=48
INFO: Using token-aware prompt (saved 118 tokens)
Token budget: 48/77 tokens
```

## Documentation

Full documentation in `README_TOKEN_BUDGET.md`
