"""
LLM-delegated scene planning.

Replaces heuristic rules with LLM reasoning for scene decomposition,
context resolution, and token budget fitting.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path

# Import with fallback if torch not available
try:
    from generators.text_generator import generate_text_with_llm
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False
    def generate_text_with_llm(*args, **kwargs):
        """Fallback when LLM unavailable."""
        return None
        
from utils.token_budget import count_tokens
try:
    from utils.token_budget import verify_token_budget
except ImportError:
    def verify_token_budget(prompt, budget):
        return count_tokens(prompt) <= budget, count_tokens(prompt)


@dataclass
class ResolvedCharacter:
    """LLM-resolved character with context-aware attributes."""
    name: str
    identity: List[str]
    default_presentation: List[str]
    presentation_decision: str  # KEEP | REPLACE | ADAPT
    scene_presentation: List[str]
    dropped: List[str]
    dropped_reason: str
    # Scene-specific fields used by the asset-composition path (design doc
    # §2.2). Optional; only populated when the scene text describes them.
    scene_pose: Optional[str] = None          # e.g. "sitting on a chair"
    scene_action: Optional[str] = None        # e.g. "playing a black Gibson Explorer guitar"
    scene_position_hint: Optional[str] = None  # e.g. "left side of the stage"


@dataclass
class SceneLayerPlan:
    """One layer in LLM-decomposed scene."""
    name: str
    prompt: str
    must_include: List[str] = field(default_factory=list)
    region_hint: Optional[str] = None
    depends_on: Optional[str] = None
    original_prompt: Optional[str] = None


@dataclass
class ScenePlan:
    """Complete scene decomposition from LLM."""
    camera: str
    layers: List[SceneLayerPlan]
    single_pass_feasible: bool
    rationale: str
    resolved_characters: List[ResolvedCharacter] = field(default_factory=list)
    strategy: str = "single_pass"  # "single_pass" | "progressive" | "asset_composition"
    canvas_layout: Optional[Dict[str, Any]] = None  # explicit placement hints (§2.3)


def _extract_json_blocks(text: str) -> str:
    """Best-effort extraction of JSON from a model's raw completion.

    Local models frequently wrap the requested JSON in code fences or add a
    short prose lead-in/out. This strips ```json fences and trims to the first
    `[` or `{` ... last `]` or `}` so `json.loads` can succeed. Returns the
    original text if nothing JSON-like is found.
    """
    import re
    if not text:
        return text
    # Strip markdown code fences (```json ... ```).
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1).strip() if fenced else text.strip()
    # Trim to the outermost JSON object/array.
    start = candidate.find("[")
    if start == -1 or (candidate.find("{") != -1 and candidate.find("{") < start):
        start = candidate.find("{")
    end = candidate.rfind("]")
    if end == -1 or (candidate.rfind("}") != -1 and candidate.rfind("}") > end):
        end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        return candidate[start:end + 1]
    return candidate


def _extract_character_snippets(
    scene_description: str,
    name: str,
    all_names: List[str],
) -> List[str]:
    """Deterministically pull the sentences describing one character out of
    the scene description (the sentence naming them + immediate pronoun
    continuations), so the fallback path never has to touch the raw stored
    character prompt (which contains generation-time artifacts)."""
    import re
    sentences = [
        s.strip().rstrip('.')
        for s in re.split(r"(?<=[.!?])\s+|\n+", scene_description)
        if s.strip()
    ]
    others = [n.lower() for n in all_names if n.lower() != name.lower()]
    snippets: List[str] = []
    capturing = False
    for sent in sentences:
        lower = sent.lower()
        if name.lower() in lower and not any(o in lower for o in others):
            # Trim positional lead-ins like "On the left side of the stage is Nikita,"
            m = re.search(rf"\b{re.escape(name)}\b[,:]?\s*(.+)", sent, re.IGNORECASE)
            snippet = m.group(1).strip(' ,') if m and m.group(1).strip(' ,') else sent
            snippets.append(snippet)
            capturing = True
            continue
        if capturing:
            if any(o in lower for o in others):
                capturing = False
                continue
            if re.match(r"^(she|he|they)\b", lower):
                snippets.append(sent)
            else:
                capturing = False
    return snippets


def stage_a_context_resolution(
    scene_description: str,
    characters: List[Dict[str, Any]]
) -> List[ResolvedCharacter]:
    """
    Stage A: Resolve character descriptions against scene context.
    
    Returns characters with identity preserved and presentation adapted.
    This prevents issues like Fantasy Clothing in bar scenes.
    """
    
    # Build prompt for LLM
    chars_json = json.dumps([
        {
            'name': c['name'],
            'stored_prompt': c.get('prompt', ''),
            'attributes': c.get('attributes', {})
        }
        for c in characters
    ], indent=2)
    
    llm_prompt = f"""You resolve character descriptions against a scene context.

SCENE: {scene_description}

CHARACTERS:
{chars_json}

For each character, classify its stored description into:
- identity: MUST survive verbatim (face, hair, build, age, species, scars). Copy verbatim.
- default_presentation: clothing/gear from generation-time style
- presentation_decision: KEEP | REPLACE | ADAPT (based on scene context)
- scene_presentation: what they should wear/appear as in THIS scene
- dropped: specific phrases removed with reason

Optionally, for the asset-composition path, also extract from the SCENE text:
- scene_pose: the character's pose/positioning in this scene (e.g. "sitting on a chair") or null
- scene_action: what the character is doing (e.g. "playing a black Gibson Explorer guitar") or null
- scene_position_hint: where in the frame they appear (e.g. "left side of the stage") or null

Rules:
- NEVER drop identity traits; copy them verbatim
- Clothing/gear: adapt to scene setting unless scene text explicitly describes it
- Everything removed must appear in "dropped" with a reason
- Return valid JSON only

Return JSON array:""" + """
[
  {
    "name": "CharacterName",
    "identity": ["trait1", "trait2"],
    "default_presentation": ["original clothing"],
    "presentation_decision": "REPLACE",
    "scene_presentation": ["appropriate clothes for scene"],
    "scene_pose": "sitting on a chair" or null,
    "scene_action": "playing guitar" or null,
    "scene_position_hint": "left side of the stage" or null,
    "dropped": ["phrase to drop"],
    "dropped_reason": "why dropped"
  }
]"""
    
    # Try LLM
    if not LLM_AVAILABLE:
        print("LLM unavailable, using fallback")
    else:
        try:
            result = generate_text_with_llm(llm_prompt, model_name="phi3_mini", max_new_tokens=1024)
            # Parse JSON from result (may need cleaning/fence stripping)
            if result:
                chars_data = json.loads(_extract_json_blocks(result))
                resolved = []
                for c in chars_data:
                    # Allow the LLM to omit the new optional fields.
                    for optional in ("scene_pose", "scene_action", "scene_position_hint"):
                        c.setdefault(optional, None)
                    resolved.append(ResolvedCharacter(**c))
                return resolved
        except Exception as e:
            print(f"LLM context resolution failed: {e}")
    
    # Fallback: derive identity + scene presentation from the SCENE TEXT
    # itself (never from the raw stored prompt, which carries generation
    # artifacts like "fantasy clothing" / "standing upright").
    all_names = [c['name'] for c in characters]
    fallback = []
    for c in characters:
        snippets = _extract_character_snippets(scene_description, c['name'], all_names)
        identity = snippets[:1]
        presentation = snippets[1:]
        fallback.append(ResolvedCharacter(
            name=c['name'],
            identity=identity,
            default_presentation=[],
            presentation_decision="ADAPT" if snippets else "KEEP",
            scene_presentation=presentation,
            dropped=["<stored character prompt>"] if snippets else [],
            dropped_reason=(
                "Fallback resolution: used scene-description snippet instead "
                "of raw stored prompt to avoid generation-time artifacts."
                if snippets else ""
            ),
        ))
    return fallback


def stage_b_decompose_scene(
    scene_description: str,
    resolved_characters: List[ResolvedCharacter],
    project_style: Optional[str] = None
) -> ScenePlan:
    """Stage B: Decompose scene into layers via LLM."""
    
    chars_desc = json.dumps([
        {
            'name': c.name,
            'identity': c.identity,
            'scene_presentation': c.scene_presentation
        }
        for c in resolved_characters
    ], indent=2)
    
    llm_prompt = f"""You are a scene director planning multi-step image generation.
The image model handles ~1-2 subjects well per step.

SCENE: {scene_description}
CHARACTERS: {chars_desc}
PROJECT STYLE: {project_style or 'photorealistic'}

Produce JSON with layers for incremental generation:
{{
  "camera": "description of camera angle",
  "layers": [
    {{
      "name": "base_environment",
      "prompt": "environment description without characters",
      "must_include": ["element1", "element2"],
      "region_hint": "full frame"
    }},
    {{
      "name": "character_Name",
      "prompt": "character with identity traits, action, and scene-appropriate appearance",
      "must_include": ["trait1"],
      "region_hint": "left/right/center area"
    }}
  ],
  "single_pass_feasible": true/false,
  "rationale": "why this decomposition",
  "canvas_layout": {{
    "width": 1024,
    "height": 1024,
    "placements": [
      {{"name": "CharacterName", "anchor": [0.3, 0.85], "scale": 0.5, "z": 1}}
    ]
  }}
}}

Rules:
- Order layers background → foreground
- Each layer prompt must be self-sufficient
- Include lighting/style continuity words in every layer
- If simple, set single_pass_feasible=true with one layer
- canvas_layout: provide explicit normalized anchor (x, y) in [0,1] where the
  character's feet/base should sit, a scale (fraction of canvas height the
  character occupies), and a z (layer order, lower = further back). Omit if
  not needed."""

    try:
        result = generate_text_with_llm(llm_prompt, model_name="phi3_mini", max_new_tokens=1024)
        # Parse JSON (strip fences/lead-in from the model's raw completion)
        data = json.loads(_extract_json_blocks(result))
        
        layers = []
        for l in data.get('layers', []):
            layers.append(SceneLayerPlan(
                name=l['name'],
                prompt=l['prompt'],
                must_include=l.get('must_include', []),
                region_hint=l.get('region_hint'),
                depends_on=l.get('depends_on')
            ))
        
        return ScenePlan(
            camera=data.get('camera', ''),
            layers=layers,
            single_pass_feasible=data.get('single_pass_feasible', False),
            rationale=data.get('rationale', ''),
            canvas_layout=data.get('canvas_layout'),
        )
    except Exception as e:
        print(f"LLM decomposition failed: {e}, using fallback")
        
        # Fallback to simple two-layer approach
        return ScenePlan(
            camera="wide shot",
            layers=[
                SceneLayerPlan(
                    name="base_environment",
                    prompt=scene_description,
                    must_include=[]
                )
            ],
            single_pass_feasible=True,
            rationale="LLM unavailable, using fallback"
        )


def stage_c_token_budget_fit(
    layer_plan: SceneLayerPlan,
    token_limit: int = 77
) -> SceneLayerPlan:
    """Stage C: Fit layer prompt to token budget via LLM rephrasing."""
    
    current_tokens = count_tokens(layer_plan.prompt)
    if current_tokens <= token_limit:
        return layer_plan
    
    # Try LLM compression
    compress_prompt = f"""Compress this image prompt to ≤ {token_limit} CLIP tokens (currently {current_tokens}).
Keep ALL identity traits verbatim.
Prioritize: 1) subject identity 2) action 3) setting

PROMPT: {layer_plan.prompt}

Return only the compressed prompt, no explanations."""
    
    try:
        result = generate_text_with_llm(compress_prompt, model_name="phi3_mini")
        
        new_tokens = count_tokens(result)
        if new_tokens <= token_limit:
            # Verify identity preserved
            layer_plan.original_prompt = layer_plan.prompt
            layer_plan.prompt = result
            print(f"LLM compressed: {current_tokens} → {new_tokens} tokens")
            return layer_plan
    except Exception as e:
        print(f"LLM compression failed: {e}")
    
    # Fallback to token budget truncation
    from utils.token_budget import TokenBudgetManager
    manager = TokenBudgetManager()
    layer_plan.original_prompt = layer_plan.prompt
    layer_plan.prompt = manager.truncate_to_tokens(layer_plan.prompt, token_limit)
    
    return layer_plan


# Strategy labels used by Stage D (design doc §2.1).
STRATEGY_SINGLE_PASS = "single_pass"
STRATEGY_PROGRESSIVE = "progressive"
STRATEGY_ASSET_COMPOSITION = "asset_composition"


@dataclass
class StrategyDecision:
    """Result of Stage D strategy selection."""
    strategy: str
    reason: str
    character_count: int = 0
    requires_spatial_precision: bool = False
    heuristic: bool = False  # True when decided by the pre-LLM guardrail gate.


def _pre_llm_strategy_gate(
    num_characters: int,
    over_clip_budget: bool = False,
) -> Optional[str]:
    """Cheap deterministic guardrail gate (design doc §2.1) that decides the
    trivial/cheap cases WITHOUT an LLM round-trip. Returns a strategy, or None
    if the case is non-trivial and the LLM should decide."""
    if num_characters == 0:
        return STRATEGY_SINGLE_PASS
    if num_characters == 1:
        # Single subject has no entity-binding problem; only escalate to
        # progressive if the prompt still overflows CLIP after Stage C.
        return STRATEGY_PROGRESSIVE if over_clip_budget else STRATEGY_SINGLE_PASS
    # >=2 characters: non-trivial, defer to LLM (default asset_composition).
    return None


def stage_d_select_strategy(
    scene_description: str,
    resolved_characters: List[ResolvedCharacter],
    plan: Optional[ScenePlan] = None,
    over_clip_budget: bool = False,
) -> StrategyDecision:
    """Stage D: select the generation strategy.

    Implements design doc §2.1. Runs a cheap deterministic guardrail gate
    first (0 chars -> single_pass, 1 char -> single_pass unless over budget).
    For >=2 characters it asks the LLM to choose between `progressive` and
    `asset_composition`, defaulting to `asset_composition` when the scene
    specifies explicit distinct positions for each character.
    """
    num_characters = len(resolved_characters)

    gated = _pre_llm_strategy_gate(num_characters, over_clip_budget)
    if gated is not None:
        reason = (
            "Deterministic guardrail gate "
            f"({num_characters} characters, over_clip_budget={over_clip_budget})."
        )
        return StrategyDecision(
            strategy=gated,
            reason=reason,
            character_count=num_characters,
            heuristic=True,
        )

    # >=2 characters: ask the LLM. Detect explicit spatial requirements as a
    # cheap pre-filter to seed the default when the LLM is unavailable.
    position_keywords = [
        "left", "right", "center", "behind", "in front", "beside",
        "on the stage", "next to", "foreground", "background",
    ]
    lower = scene_description.lower()
    has_position_hints = any(
        rc.scene_position_hint
        or any(k in lower for k in position_keywords)
        for rc in resolved_characters
    )

    llm_decision = None
    if LLM_AVAILABLE:
        try:
            chars_desc = json.dumps([
                {
                    "name": rc.name,
                    "scene_position_hint": rc.scene_position_hint,
                }
                for rc in resolved_characters
            ], indent=2)
            llm_prompt = f"""You select an image-generation strategy for a scene.

SCENE: {scene_description}
CHARACTERS: {chars_desc}

Choose one strategy:
- "single_pass": generate everything in one diffusion call.
- "progressive": generate background first, then each character via prompt
  continuity (no image editing).
- "asset_composition": generate each character asset separately against a
  plain background, then deterministically place them onto the background.

Prefer "asset_composition" when the scene requires strict, distinct spatial
positions per character or strong identity preservation. Prefer
"single_pass"/"progressive" for looser compositions.

Return valid JSON only: {{
  "strategy": "asset_composition",
  "reason": "2 characters with distinct spatial positions and identity requirements"
}}"""
            result = generate_text_with_llm(llm_prompt, model_name="phi3_mini", max_new_tokens=256)
            if result:
                data = json.loads(_extract_json_blocks(result))
                llm_decision = data.get("strategy")
                llm_reason = data.get("reason", "")
                if llm_decision in (
                    STRATEGY_SINGLE_PASS, STRATEGY_PROGRESSIVE, STRATEGY_ASSET_COMPOSITION
                ):
                    return StrategyDecision(
                        strategy=llm_decision,
                        reason=llm_reason or "LLM Stage D decision",
                        character_count=num_characters,
                        requires_spatial_precision=has_position_hints,
                    )
        except Exception as e:
            print(f"LLM strategy selection failed: {e}")

    # Default when LLM unavailable or returned something invalid: prefer
    # asset_composition when explicit spatial/identity requirements exist.
    strategy = STRATEGY_ASSET_COMPOSITION if has_position_hints else STRATEGY_PROGRESSIVE
    return StrategyDecision(
        strategy=strategy,
        reason=(
            "LLM unavailable/invalid; defaulting to asset_composition "
            "because the scene has explicit spatial requirements."
            if has_position_hints
            else "LLM unavailable/invalid; defaulting to progressive."
        ),
        character_count=num_characters,
        requires_spatial_precision=has_position_hints,
        heuristic=True,
    )


class LLMScenePlanner:
    """Orchestrates LLM-based scene planning."""
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
    
    def plan_scene(
        self,
        scene_description: str,
        characters: List[Dict[str, Any]],
        project_style: Optional[str] = None,
    ) -> ScenePlan:
        """Create complete scene plan via LLM delegation."""
        
        if not self.use_llm:
            # Return empty plan for fallback
            return ScenePlan(
                camera="",
                layers=[],
                single_pass_feasible=True,
                rationale="LLM disabled"
            )
        
        # Stage A: Context resolution
        print("Stage A: Resolving character context...")
        resolved = stage_a_context_resolution(scene_description, characters)
        
        # Stage B: Decomposition  
        print("Stage B: Decomposing scene...")
        plan = stage_b_decompose_scene(scene_description, resolved, project_style)
        plan.resolved_characters = resolved
        
        # Stage C: Token fitting
        print("Stage C: Fitting token budgets...")
        over_budget = False
        for i, layer in enumerate(plan.layers):
            fitted = stage_c_token_budget_fit(layer)
            if fitted.original_prompt is not None:
                over_budget = True
            plan.layers[i] = fitted
        
        # Stage D: Strategy selection
        print("Stage D: Selecting generation strategy...")
        decision = stage_d_select_strategy(
            scene_description,
            resolved,
            plan=plan,
            over_clip_budget=over_budget,
        )
        plan.strategy = decision.strategy
        
        return plan
    
    def save_plan(self, plan: ScenePlan, output_path: Path):
        """Save plan for audit trail."""
        data = {
            'camera': plan.camera,
            'single_pass_feasible': plan.single_pass_feasible,
            'rationale': plan.rationale,
            'strategy': plan.strategy,
            'canvas_layout': plan.canvas_layout,
            'layers': [
                {
                    'name': l.name,
                    'prompt': l.prompt,
                    'original_prompt': l.original_prompt,
                    'must_include': l.must_include,
                    'region_hint': l.region_hint
                }
                for l in plan.layers
            ],
            'resolved_characters': [
                {
                    'name': c.name,
                    'identity': c.identity,
                    'presentation_decision': c.presentation_decision,
                    'scene_presentation': c.scene_presentation,
                    'scene_pose': c.scene_pose,
                    'scene_action': c.scene_action,
                    'scene_position_hint': c.scene_position_hint,
                    'dropped': c.dropped
                }
                for c in plan.resolved_characters
            ]
        }
        
        output_path.write_text(json.dumps(data, indent=2))


def create_llm_scene_plan(
    scene_description: str,
    characters: List[Dict[str, Any]],
    project_name: str = "default"
) -> ScenePlan:
    """Convenience function to create LLM-based scene plan."""
    
    from utils.project_paths import scene_dir
    
    # Get project for style info (with fallback)
    try:
        from services.database.project_service import get_project
        project = get_project(project_name)
        project_style = project.get('style', 'photorealistic')
    except:
        project_style = 'photorealistic'
    
    planner = LLMScenePlanner(use_llm=True)
    plan = planner.plan_scene(scene_description, characters, project_style)
    
    # Save plan for audit
    try:
        next_num = len(list((scene_dir(1, project_name)).parent.glob('*')))
        plan_path = scene_dir(next_num or 1, project_name) / "plan.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        planner.save_plan(plan, plan_path)
        print(f"Plan saved to {plan_path}")
    except:
        pass
    
    return plan
