"""
Token budget management for CLIP text encoder limits.

CLIP has a maximum sequence length of 77 tokens.
This module provides token counting and prompt prioritization to stay within limits.
"""

import re
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Character generation instructions that should NOT be in scene prompts
# Only applied to CHARACTER PROMPTS, not scene descriptions
CHARACTER_GEN_INSTRUCTIONS = [
    # Pose/positioning - only strip from character prompts with word boundaries
    r"\bstanding upright\b",
    r"\bentire body visible\b",
    r"\bfull body\b",
    r"\bfull length\b",
    r"\bfeet touching the ground\b",
    r"\bfeet visible\b",
    r"\bposed\b",
    
    # Framing/composition - character generation artifacts
    r"\bwide shot\b",
    r"\bfull length wide shot\b",
    r"\bcamera far from subject\b",
    r"\bnot a portrait\b",
    r"\bnot a close-?up\b",
    r"\bclose-?up\b",
    r"\bmedium shot\b",
    r"\blong shot\b",
    
    # Technical quality modifiers (low priority for scenes)
    r"\bhigh detail\b",
    r"\bdetailed face\b",
    r"\bgood lighting\b",
    r"\bphotorealistic\b",
    r"\bultra realistic\b",
    r"\bhighly detailed\b",
    r"\bprofessional\b",
    
    # Generation artifacts
    r"\bfull body photo\b",
    r"\bphoto of a\b",
]

# Instructions to strip from character prompts but NOT scene descriptions
CHARACTER_PROMPT_STRIP = CHARACTER_GEN_INSTRUCTIONS

# Scene-safe stripping - only quality modifiers, never pose/composition words
SCENE_SAFE_STRIP = [
    r"\bhigh detail\b",
    r"\bdetailed face\b",
    r"\bgood lighting\b",
    r"\bphotorealistic\b",
    r"\bultra realistic\b",
    r"\bhighly detailed\b",
    r"\bprofessional\b",
]


@dataclass
class TokenBudgetItem:
    """Represents an item in the token budget with priority."""
    text: str
    priority: int  # 1=highest, 5=lowest
    category: str


class TokenBudgetManager:
    """Manages token budgeting for CLIP prompts."""
    
    # CLIP tokenizer limit
    MAX_TOKENS = 77
    
    # Rough tokens per word estimation (CLIP uses BPE, so this is approximate)
    TOKENS_PER_WORD = 1.3
    
    def __init__(self, max_tokens: int = None):
        self.max_tokens = max_tokens or self.MAX_TOKENS
        self.tokens_per_word = self.TOKENS_PER_WORD
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        if not text:
            return 0
        
        # Count words and apply rough estimate
        words = len(text.split())
        estimated_tokens = int(words * self.tokens_per_word) + 5  # Add overhead
        
        # Simple heuristic: commas, parentheses add tokens
        punctuation_count = len(re.findall(r'[,\.;:\(\)]', text))
        estimated_tokens += punctuation_count // 2
        
        return estimated_tokens
    
    def strip_generation_instructions(self, text: str, is_character_prompt: bool = True) -> str:
        """Remove character generation instructions from prompts.
        
        Args:
            text: Text to clean
            is_character_prompt: If True, strip pose/framing artifacts. 
                                 If False (scene text), only strip quality modifiers.
        """
        if not text:
            return text
        
        patterns = CHARACTER_PROMPT_STRIP if is_character_prompt else SCENE_SAFE_STRIP
        
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Clean up multiple spaces and commas
        cleaned = re.sub(r',\s*,+', ',', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip(' ,')
        
        return cleaned
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token budget by removing sentences."""
        if not text:
            return text
        
        current_tokens = self.count_tokens(text)
        if current_tokens <= max_tokens:
            return text
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        truncated = []
        current_count = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            if current_count + sentence_tokens <= max_tokens:
                truncated.append(sentence)
                current_count += sentence_tokens
            else:
                # Try adding partial sentence word by word
                words = sentence.split()
                for word in words:
                    test_text = ' '.join(truncated + [word])
                    if self.count_tokens(test_text) <= max_tokens:
                        truncated.append(word)
                    else:
                        break
                break
        
        result = ' '.join(truncated).strip()
        
        # If still too long, truncate by characters as last resort
        if self.count_tokens(result) > max_tokens:
            # Rough estimate: 1 token ≈ 0.77 words
            approx_chars = int(max_tokens * 4.5)
            result = result[:approx_chars].rsplit(' ', 1)[0]
        
        return result
    
    def truncate_to_tokens_smart(self, text: str, max_tokens: int) -> str:
        """Smart truncation that preserves sentences with actions/characters.

        Sentences are *selected* by importance score but always *emitted in
        their original order*, so the result stays coherent (previously the
        output was reordered by score, producing garbled prompts).
        """
        if not text:
            return text
        
        sentences = [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        
        # Score sentences by importance (contains actions, key scene nouns).
        scored = []
        for idx, sent in enumerate(sentences):
            score = 0
            sent_lower = sent.lower()
            # The opening sentence usually establishes the scene: keep it.
            if idx == 0:
                score += 6
            # Prefer sentences with actions
            if any(word in sent_lower for word in ['playing', 'sitting', 'standing', 'performing', 'watching']):
                score += 3
            # Prefer sentences with key scene nouns
            if any(word in sent_lower for word in ['stage', 'bar', 'guitar', 'drums', 'camera']):
                score += 2
            # Prefer shorter sentences
            score += max(0, 10 - len(sent.split()))
            scored.append((score, idx, sent))
        
        # Select best sentences by score...
        selected: List[Tuple[int, str]] = []
        current_tokens = 0
        for score, idx, sent in sorted(scored, key=lambda x: x[0], reverse=True):
            sent_tokens = self.count_tokens(sent)
            if current_tokens + sent_tokens <= max_tokens:
                selected.append((idx, sent))
                current_tokens += sent_tokens
        
        # ...but emit them in original order to keep the text coherent.
        selected.sort(key=lambda x: x[0])
        result = ' '.join(sent for _, sent in selected).strip()

        # Guarantee the input is NEVER fully dropped: if no complete sentence
        # fit within the budget (e.g. a single long sentence exceeding
        # max_tokens), fall back to word-level truncation so we always return
        # a non-empty, budget-fitted prefix. This preserves the "base prompt
        # is never dropped" contract relied on by build_scene_prompt.
        if not result and text:
            result = self.truncate_to_tokens(text, max_tokens)

        return result
    
    def prioritize_character_description(self, character_prompt: str, attributes: Dict = None) -> Tuple[str, int]:
        """Extract and prioritize character description for scenes."""
        # Strip generation instructions from CHARACTER prompts only
        cleaned = self.strip_generation_instructions(character_prompt, is_character_prompt=True)
        
        # If we have attributes, prefer building from them (more reliable)
        if attributes:
            from core.prompt_decomposer import build_appearance_prompt
            char_type = "man"  # Default, could be inferred from attributes
            built = build_appearance_prompt(char_type, attributes)
            # Compress to essential attributes only (8-12 words)
            words = built.split()
            if len(words) > 15:
                # Keep first 15 words - enough for distinguishing features
                built = ' '.join(words[:15])
            return built, 1
        
        # Compress character description to essential info
        if cleaned:
            words = cleaned.split()
            if len(words) > 20:
                # Keep only essential attributes (first 20 words usually has key features)
                cleaned = ' '.join(words[:20])
        
        return cleaned, 2
    
    def build_scene_prompt(
        self,
        base_prompt: str,
        characters: List[Dict],
        environment: str = "",
        camera_instructions: str = ""
    ) -> Tuple[str, Dict]:
        """Build token-aware scene prompt with priority budgeting."""
        
        budget = int(self.max_tokens * 0.9)
        dropped_items = []
        compression_log = []
        
        # Phase 1: Process base prompt - NEVER drop it, compress if needed
        if base_prompt:
            # Strip only scene-safe items (quality modifiers), never pose/composition
            base_clean = self.strip_generation_instructions(base_prompt, is_character_prompt=False)
            
            base_tokens = self.count_tokens(base_clean)
            
            # Reserve budget for characters but base gets priority
            # Base gets at least 50% of budget, characters share remaining (min 16 tokens for 2 chars)
            min_char_budget = len(characters) * 8
            max_base_budget = budget - min_char_budget
            if max_base_budget < int(budget * 0.40):
                max_base_budget = int(budget * 0.50)
            
            if base_tokens > max_base_budget:
                # Try smart truncation preserving key sentences with actions
                base_clean = self.truncate_to_tokens_smart(base_clean, max_base_budget)
                compression_log.append({
                    "category": "scene_description",
                    "original_tokens": base_tokens,
                    "compressed_tokens": self.count_tokens(base_clean),
                    "action": "truncated"
                })
            
            final_parts = [base_clean]
            current_tokens = self.count_tokens(base_clean)
        else:
            final_parts = []
            current_tokens = 0
        
        # Phase 2: Add characters (compressed descriptions)
        # For multi-character scenes, we need BOTH characters even if description is minimal
        char_items_included = 0
        for i, char in enumerate(characters):
            char_name = char.get("name", "")
            char_prompt = char.get("prompt", "")
            attributes = char.get("attributes")
            
            desc, _ = self.prioritize_character_description(char_prompt, attributes)
            
            if desc and char_name:
                # Ultra-minimal character reference - just name and 2-3 key attributes
                if attributes:
                    from core.prompt_decomposer import build_appearance_prompt
                    # Build very short description
                    built = build_appearance_prompt("man", attributes)
                    words = built.split()
                    # Take only 3-4 key words
                    key_desc = ' '.join(words[:4]) if len(words) > 4 else built
                else:
                    # Extract just 2-3 adjectives from description
                    words = desc.split(',')
                    key_desc = ','.join([w.strip() for w in words[:2]])
                
                char_section = f"{char_name}: {key_desc}"
                char_tokens = self.count_tokens(char_section)
                
                # Ultra-compressed: name + 4 words max
                if char_tokens > 10:
                    parts = char_section.split(':')
                    if len(parts) == 2:
                        name_part = parts[0].strip()
                        desc_words = parts[1].split()[:3]
                        char_section = f"{name_part}: {' '.join(desc_words)}"
                        char_tokens = self.count_tokens(char_section)
                
                # Try to fit, but force fit for first 2 characters
                if current_tokens + char_tokens <= budget:
                    final_parts.append(char_section)
                    current_tokens += char_tokens
                    char_items_included += 1
                elif i < 2 and current_tokens < budget - 5:
                    # Force minimal version
                    ultra_min = f"{char_name}"
                    if current_tokens + self.count_tokens(ultra_min) <= budget:
                        final_parts.append(ultra_min)
                        current_tokens += self.count_tokens(ultra_min)
                        char_items_included += 1
                    else:
                        dropped_items.append({
                            "category": "character",
                            "name": char_name,
                            "reason": "even minimal name exceeds remaining budget",
                        })
                else:
                    dropped_items.append({
                        "category": "character",
                        "name": char_name,
                        "reason": "budget exhausted after higher-priority items",
                    })
        
        # Phase 3: Add camera instructions if budget allows
        if camera_instructions and current_tokens < budget - 10:
            cam_clean = self.strip_generation_instructions(
                camera_instructions, is_character_prompt=False
            )
            cam_tokens = self.count_tokens(cam_clean)
            
            if current_tokens + cam_tokens <= budget:
                final_parts.append(f"Camera: {cam_clean}")
                current_tokens += cam_tokens
        
        # Phase 4: Add environment if budget allows
        if environment and current_tokens < budget - 10:
            env_tokens = self.count_tokens(environment)
            if current_tokens + env_tokens <= budget:
                final_parts.append(f"Environment: {environment}")
                current_tokens += env_tokens
        
        # Join without producing double periods ("..").
        final_prompt = ". ".join(p.strip().rstrip('.') for p in final_parts if p.strip())
        if final_prompt:
            final_prompt += "."
        
        stats = {
            "total_tokens_estimated": current_tokens,
            "max_tokens": self.max_tokens,
            "budget_used": current_tokens,
            "budget_total": budget,
            "items_included": len(final_parts),
            "items_dropped": len(dropped_items),
            "dropped_details": dropped_items[:5],
            "compression_log": compression_log,
            "base_compressed": len(compression_log) > 0,
            "characters_included": char_items_included
        }
        
        return final_prompt, stats


# Module-level convenience functions
_default_manager = TokenBudgetManager()


def count_tokens(text: str) -> int:
    """Estimate token count."""
    return _default_manager.count_tokens(text)


def strip_generation_instructions(text: str) -> str:
    """Strip character generation instructions."""
    return _default_manager.strip_generation_instructions(text)


def build_token_aware_scene_prompt(
    base_prompt: str,
    characters: List[Dict],
    environment: str = "",
    camera_instructions: str = ""
) -> Tuple[str, Dict]:
    """Build token-aware scene prompt."""
    return _default_manager.build_scene_prompt(
        base_prompt, characters, environment, camera_instructions
    )
