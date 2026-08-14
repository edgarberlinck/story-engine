# Scene Generation Caveats

## Multi-Character Style Conflicts
### Overview
When generating scenes containing multiple characters with different visual styles, Story Engine currently exhibits limitations that can result in poor image quality and inconsistent character rendering.

The core issue stems from how scene prompts are constructed by concatenating individual character descriptions (including their style attributes) into a single prompt for diffusion model generation. When characters have conflicting visual styles, the model receives contradictory instructions.

### The Problem

#### Current Behavior
When a user requests a scene with multiple characters:

1. **Character Prompt Storage**: Each character's prompt is stored in the database with its complete style prefix and modifiers embedded:
   ```
   Example Character 1 (Ultra Realistic):
   "ultra realistic man, athletic build, short brown hair, ... highly detailed skin texture, natural facial features, realistic proportions..."
   
   Example Character 2 (Manga):
   "manga style woman, slender build, long black hair, ... black and white manga illustration, detailed ink lines, expressive facial features..."
   ```

2. **Prompt Enrichment**: The `_enrich_scene_prompt` function in `generators/image_engine.py` appends character appearance descriptions to the base scene prompt:
   ```python
   def _enrich_scene_prompt(prompt: str, project: str) -> str:
       enriched = prompt
       for character in character_service.find_characters_in_text(prompt, project):
           enriched += f"\n{character['name']} appearance: {character['prompt']}"
       return enriched
   ```

3. **Resulting Prompt**: The model receives a single prompt containing conflicting style instructions:
   ```
   Character 1 and Character 2 in a forest.
   
   Character 1 appearance: ultra realistic man, ... photographic realism, highly detailed skin texture...
   
   Character 2 appearance: manga style woman, ... black and white manga illustration, Japanese manga art style...
   ```

#### Why This Fails
Diffusion models process prompts holistically. When presented with contradictory style instructions:
- The model's attention mechanism becomes confused about which visual style to apply
- Style tokens from different characters compete for influence
- Results in hybrid/aberrant styling where neither character looks correct
- Characters may blend styles inconsistently across the image
- Overall composition suffers as the model attempts to satisfy conflicting constraints

### Root Cause Analysis

The fundamental architectural issue is **style leakage through prompt enrichment**:

1. **Style Entanglement**: Character prompts include style information (prefix + modifiers) as integral components of appearance
2. **Flat Concatenation**: Scene prompt enrichment treats character descriptions as independent text additions without style isolation
3. **Single Model Constraint**: The entire scene must use one diffusion model with one set of generation parameters
4. **No Style Mediation**: No logic exists to detect conflicting styles or resolve incompatibilities

### Architectural Limitations

**Current Data Flow**:
```
Character Creation → build_character_prompt() → Store prompt with style embedded
                                          ↓
Scene Generation → _enrich_scene_prompt() → Append character prompts → Single diffusion call
```

**Key Files Involved**:
- `core/character_attributes.py`: Style definitions and `build_character_prompt()` function
- `services/database/character_service.py`: Character storage and `find_characters_in_text()` lookup
- `generators/image_engine.py`: `_enrich_scene_prompt()` and `generate_scene()` functions
- `generators/video_engine.py`: Multi-character verification logic

### Limitations of Current Approach

1. **No Style Validation**: No check for style compatibility when adding characters to scenes
2. **Irreversible Prompt Construction**: Once character prompts are stored with embedded styles, they cannot be decoupled for scene use
3. **False Positives in Character Matching**: Simple substring matching in `find_characters_in_text()` can trigger unintended enrichments
4. **No Per-Character Style Control**: Scene generation applies one global style context to all characters

### Recommended Architectural Solutions

#### Option 1: Style-Aware Prompt Separation (Recommended)
**Separate appearance from style in storage**:
- Store character attributes as structured data (JSON) separate from style
- Rebuild prompts dynamically for scenes with scene-appropriate styling
- Extract only physical/appearance attributes for scene enrichment
- Apply scene-level style consistently to all characters

**Implementation**:
1. Extend character schema to store normalized attributes without embedded style
2. Create `decompose_character_prompt()` function to extract appearance vs style
3. Modify `_enrich_scene_prompt()` to rebuild character descriptions using only appearance attributes
4. Add style conflict detection and user warnings

#### Option 2: Style Harmonization
**Detect conflicts and harmonize**:
- Detect when scene characters have incompatible styles
- Automatically select a compromise style or enforce style consistency rules
- Generate scene with unified style, then apply per-character style adjustments via post-processing

#### Option 3: Multi-Pass Generation (Advanced)
**Generate characters separately then composite**:
- Generate each character individually with correct style
- Use inpainting or composition techniques to combine characters into scene
- Requires significant architectural changes and new generation pipeline

### Implementation Guidance

#### Phase 1: Immediate Mitigation
1. **Style Conflict Detection**: Add validation when user attempts to create scenes with conflicting character styles
2. **User Warnings**: Display warnings when mixing incompatible styles (e.g., Ultra Realistic + Anime/Manga)
3. **Documentation**: Update UI help text to inform users about style limitations

#### Phase 2: Architectural Refactor
1. **Schema Enhancement**: Add `character_attributes` table population during character creation
2. **Prompt Decomposition**: Create functions to extract appearance-only descriptions from stored prompts
3. **Scene Style Mediator**: Implement logic to either harmonize styles or use scene-level style override
4. **Attribute-Based Generation**: Move to attribute-based prompt generation rather than storing static prompts

#### Phase 3: Enhanced Features
1. **Per-Character Style Tokens**: Investigate advanced prompting techniques with negative prompts per character
2. **Style Blending**: Implement controlled style interpolation for gradual transitions
3. **Model Selection Logic**: Automatically select models optimized for specific style combinations

### Technical Considerations

**Prompt Structure**:
Current prompt building in `build_character_prompt()` includes:
- Style prefix: `"ultra realistic man"` or `"manga style woman"`
- Physical attributes: body type, hair, clothing, etc.
- Style modifiers: detailed descriptions of rendering technique

For multi-character scenes, ideal separation would be:
- **Scene-level**: Overall composition, lighting, environment style
- **Character-level**: Only physical appearance (gender, age, features) without style modifiers
- **Style mediation**: Either unify or explicitly handle per-character styling

**Database Schema Impact**:
The `characters` table currently stores static prompts. Moving to structured attributes would require:
- Migration path for existing character data
- Backward compatibility for current prompt-based workflows
- Population of `character_attributes` JSON table during character creation

### Testing Recommendations

1. **Create test suite for style conflicts**: Automated tests with character pairs having incompatible styles
2. **Visual validation**: Manual review of generation quality with mixed-style scenes
3. **Regression testing**: Ensure single-character scenes remain unaffected
4. **Performance testing**: Verify prompt decomposition doesn't impact generation speed

### Related Documentation
- `docs/ui_requirements.md`: Character builder specifications
- `core/character_attributes.py`: Style configuration and prompt building logic
- `generators/image_engine.py`: Scene generation and prompt enrichment

---

*Last updated: 2024*
*Issue identified during multi-character scene generation testing*
