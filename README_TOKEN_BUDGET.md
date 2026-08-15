# Token Budget Management for Scene Generation

## Problem

Scene generation prompts were exceeding CLIP's 77 token limit due to:
1. Full character generation prompts being reused in scenes
2. Character prompts containing generation-specific instructions (pose, framing, quality modifiers)
3. No token counting or budgeting before model invocation
4. Multi-character scenes accumulating prompt length rapidly

Example problematic prompt:
```
background roger : full body photo of a muscular male character , age 30-40 , 
with short bald black hair , dark skin tone and brown eyes , wearing formal clothing ,
serious expression , photorealistic , detailed face , high detail , good lighting .
standing upright , entire body visible from head to toe including feet , 
full length wide shot , camera far from subject , feet touching the ground visible in frame ,
not a portrait , not a close - up
```
**Token count: ~166 tokens (truncated by CLIP)**

## Solution

Implemented token-aware prompt construction with three key improvements:

### 1. Token Budget Manager (`utils/token_budget.py`)

- Estimates token counts using word-based heuristics
- Prioritizes information by category:
  - **Priority 1**: Scene description, character identity
  - **Priority 2**: Character actions, camera instructions
  - **Priority 3**: Environment details
  - **Priority 4+**: Secondary attributes
- Drops low-priority items when budget exceeded
- Logs token usage and dropped items

### 2. Generation Instruction Stripping

Removes character-generation-specific instructions that are irrelevant/contradictory in scenes:

**Stripped:**
- Pose: `standing upright`, `sitting`, `entire body visible`
- Framing: `full length wide shot`, `not a portrait`, `camera far`
- Quality: `high detail`, `photorealistic`, `good lighting`

**Preserved:**
- Physical attributes: age, build, hair, skin tone, eyes
- Clothing: wearing formal clothing
- Actions: playing drums, sitting on chair

### 3. Character Scene Description vs Generation Prompt

**Before (incorrect):**
```python
character['name'] appearance: {character['prompt']}  # Full generation prompt
```

**After (correct):**
```python
character['name'], [appearance attributes only]  # Stripped description
```

## Implementation Details

### Modified Files

1. **`utils/token_budget.py`** (new)
   - `TokenBudgetManager` class
   - Token counting and estimation
   - Generation instruction stripping
   - Priority-based prompt building

2. **`core/prompt_decomposer.py`**
   - Enhanced `extract_appearance_from_stored_prompt()` to strip generation instructions

3. **`generators/image_engine.py`**
   - Updated `_enrich_scene_prompt()` to use token-aware building
   - Added token budget enforcement in `generate_scene()`
   - Added token statistics to results

### Usage

```python
from utils.token_budget import build_token_aware_scene_prompt

prompt, stats = build_token_aware_scene_prompt(
    base_prompt="Roger is sitting behind a drum kit",
    characters=[{'name': 'Roger', 'prompt': '...'}]
)

print(f"Token count: {stats['total_tokens_estimated']}/{stats['max_tokens']}")
```

### Testing

Run tests:
```bash
PYTHONPATH=. python3 tests/test_token_budget.py
PYTHONPATH=. python3 tests/test_scene_token_limits.py
```

## Results

**Before:**
- Token count: 166+ tokens
- Warning: `[transformers] Token indices sequence length is longer than specified maximum (166 > 77)`
- Prompt truncated, losing character information

**After:**
- Token count: ~30-70 tokens
- All critical information preserved
- Generation instructions stripped
- Token stats logged for debugging

## Example Transformation

### Input Character Generation Prompt
```
full body photo of a muscular male character, age 30-40,
with short bald black hair, dark skin tone and brown eyes,
wearing formal clothing, serious expression, photorealistic,
detailed face, high detail, good lighting.
standing upright, entire body visible from head to toe including feet,
full length wide shot, camera far from subject,
feet touching the ground visible in frame,
not a portrait, not a close-up
```

### Output Scene Description
```
Roger, muscular male character, age 30-40, short bald black hair, dark skin tone and brown eyes, wearing formal clothing, serious expression
```

**Tokens reduced: 166 → ~28**

## Remaining Limitations

1. **Token estimation is approximate**: Uses word-based heuristics, not actual CLIP tokenization
2. **Attribute inference**: Character type (man/woman) is hardcoded to default
3. **No multi-step generation**: Still single-pass generation
4. **Model-specific**: Currently tuned for CLIP/77 token limit, may need adjustment for other models

## Future Improvements

1. **Actual token counting**: Use CLIP tokenizer directly for accurate counts
2. **Multi-step generation**: Generate scene layout first, then inpaint characters
3. **Dynamic character types**: Infer gender from stored attributes
4. **Scene-specific filtering**: Context-aware instruction removal
