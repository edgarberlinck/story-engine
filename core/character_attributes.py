"""
Centralized character attribute & style configuration.

Structure:
    CHARACTER_STYLES: style_id -> {label, prefix, modifiers}
    CHARACTER_TYPES:  type_id  -> {label, categories: [{name, attributes: [...]}]}
    SHARED_CATEGORIES: categories appended to every type.

Each attribute:
    {key, label, values, template?, skip?}
    - template: phrase template, "{}" is the lowercase value (default "{}").
    - skip: values that produce no prompt phrase (e.g. "None").

The UI stores only identifiers/values; `build_character_prompt()` converts a
selection into the final generation prompt. The style is a first-class
attribute and must be reused for future references, scenes, poses and
full-body generations to keep visual consistency.
"""

from typing import Dict, Any

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

CHARACTER_STYLES: Dict[str, Dict[str, str]] = {
    "ultra_realistic": {
        "label": "Ultra Realistic",
        "prefix": "ultra realistic",
        "modifiers": "highly detailed skin texture, natural facial features, realistic proportions, detailed eyes, natural lighting, photographic realism",
    },
    "cinematic": {
        "label": "Cinematic",
        "prefix": "cinematic",
        "modifiers": "cinematic lighting, dramatic composition, realistic details, professional cinematography, depth of field",
    },
    "photorealistic": {
        "label": "Photorealistic",
        "prefix": "photorealistic",
        "modifiers": "realistic human proportions, natural skin texture, realistic lighting, high detail photography",
    },
    "realistic": {
        "label": "Realistic",
        "prefix": "realistic",
        "modifiers": "natural proportions, realistic textures, natural lighting, high detail",
    },
    "anime": {
        "label": "Anime",
        "prefix": "anime style",
        "modifiers": "detailed anime style, expressive eyes, clean line art, vibrant colors",
    },
    "manga": {
        "label": "Manga",
        "prefix": "manga style",
        "modifiers": "black and white manga illustration, detailed ink lines, expressive facial features, Japanese manga art style",
    },
    "comic_book": {
        "label": "Comic Book",
        "prefix": "comic book style",
        "modifiers": "bold ink outlines, dramatic shadows, detailed comic illustration, dynamic composition",
    },
    "cartoon": {
        "label": "Cartoon",
        "prefix": "cartoon",
        "modifiers": "stylized proportions, expressive facial features, clean shapes, colorful illustration",
    },
    "animation": {
        "label": "Animation",
        "prefix": "animated",
        "modifiers": "stylized design, expressive face, clean shapes, detailed animation art style",
    },
    "3d_animation": {
        "label": "3D Animation",
        "prefix": "3D animated",
        "modifiers": "high quality 3D animated character, stylized proportions, expressive face, detailed 3D render, animation movie quality",
    },
    "3d_render": {
        "label": "3D Render",
        "prefix": "3D rendered",
        "modifiers": "high quality 3D character render, detailed materials, realistic lighting, physically based rendering",
    },
    "pixar_like": {
        "label": "Pixar-like",
        "prefix": "Pixar-style",
        "modifiers": "high quality 3D animated character, expressive face, soft lighting, animation movie quality",
    },
    "disney_like": {
        "label": "Disney-like",
        "prefix": "Disney-style",
        "modifiers": "classic animation character design, expressive features, appealing shapes, animation movie quality",
    },
    "stylized": {
        "label": "Stylized",
        "prefix": "stylized",
        "modifiers": "artistic proportions, visually distinctive features, detailed character illustration",
    },
    "semi_realistic": {
        "label": "Semi-Realistic",
        "prefix": "semi-realistic",
        "modifiers": "realistic anatomy combined with stylized artistic features, detailed illustration",
    },
    "fantasy_art": {
        "label": "Fantasy Art",
        "prefix": "fantasy",
        "modifiers": "detailed fantasy illustration, dramatic lighting, magical atmosphere, epic character design",
    },
    "dark_fantasy": {
        "label": "Dark Fantasy",
        "prefix": "dark fantasy",
        "modifiers": "dramatic shadows, gothic atmosphere, detailed dark fantasy illustration",
    },
    "cyberpunk": {
        "label": "Cyberpunk",
        "prefix": "cyberpunk",
        "modifiers": "futuristic clothing, neon lights, futuristic environment, detailed cyberpunk aesthetic",
    },
    "sci_fi": {
        "label": "Sci-Fi",
        "prefix": "science fiction",
        "modifiers": "futuristic design, advanced technology, detailed sci-fi aesthetic",
    },
    "steampunk": {
        "label": "Steampunk",
        "prefix": "steampunk",
        "modifiers": "Victorian-inspired clothing, mechanical details, brass elements, retro-futuristic aesthetic",
    },
    "medieval_art": {
        "label": "Medieval Art",
        "prefix": "medieval-inspired",
        "modifiers": "historical clothing, detailed textures, medieval aesthetic",
    },
    "concept_art": {
        "label": "Concept Art",
        "prefix": "concept art of a",
        "modifiers": "professional character concept art, detailed character design, production quality, highly detailed illustration",
    },
    "digital_painting": {
        "label": "Digital Painting",
        "prefix": "digital painting of a",
        "modifiers": "detailed brushwork, professional concept art, rich lighting and textures",
    },
    "oil_painting": {
        "label": "Oil Painting",
        "prefix": "oil painting of a",
        "modifiers": "detailed brush strokes, classical composition, artistic lighting",
    },
    "watercolor": {
        "label": "Watercolor",
        "prefix": "watercolor illustration of a",
        "modifiers": "soft colors, visible watercolor texture, artistic brushwork",
    },
    "pencil_drawing": {
        "label": "Pencil Drawing",
        "prefix": "pencil drawing of a",
        "modifiers": "detailed graphite shading, hand-drawn character illustration",
    },
    "sketch": {
        "label": "Sketch",
        "prefix": "concept sketch of a",
        "modifiers": "rough expressive lines, hand-drawn illustration",
    },
    "ink_drawing": {
        "label": "Ink Drawing",
        "prefix": "ink drawing of a",
        "modifiers": "detailed ink illustration, strong line work, black and white drawing",
    },
    "pixel_art": {
        "label": "Pixel Art",
        "prefix": "pixel art",
        "modifiers": "detailed pixel art character, game sprite aesthetic, pixelated details, retro game art",
    },
    "low_poly": {
        "label": "Low Poly",
        "prefix": "low poly",
        "modifiers": "simplified geometric shapes, stylized 3D game asset",
    },
    "game_asset": {
        "label": "Game Asset",
        "prefix": "game-ready",
        "modifiers": "clean silhouette, readable design, detailed game character concept",
    },
    "clay_render": {
        "label": "Clay Render",
        "prefix": "clay render of a",
        "modifiers": "stylized clay character render, soft materials, studio lighting, sculpted appearance",
    },
    "minimalist": {
        "label": "Minimalist",
        "prefix": "minimalist",
        "modifiers": "simple shapes, clean composition, limited visual details",
    },
    "abstract": {
        "label": "Abstract",
        "prefix": "abstract",
        "modifiers": "artistic shapes, experimental composition, unconventional visual design",
    },
}

DEFAULT_STYLE = "ultra_realistic"

# ---------------------------------------------------------------------------
# Style compatibility families (Phase 1 conflict detection)
# ---------------------------------------------------------------------------
# Styles within the same family render coherently together; mixing across
# families in one scene is the failure mode described in
# docs/scene-generation-caveats.md.

STYLE_FAMILIES: Dict[str, str] = {
    # Photographic / realistic
    "ultra_realistic": "realistic",
    "cinematic": "realistic",
    "photorealistic": "realistic",
    "realistic": "realistic",
    "semi_realistic": "realistic",  # bridges realistic/stylized, see FAMILY_BRIDGES

    # Japanese-inspired 2D
    "anime": "anime_manga",
    "manga": "anime_manga",

    # Western 2D / comic
    "comic_book": "comic_cartoon",
    "cartoon": "comic_cartoon",
    "animation": "comic_cartoon",

    # 3D stylized
    "3d_animation": "3d_stylized",
    "3d_render": "3d_stylized",
    "pixar_like": "3d_stylized",
    "disney_like": "3d_stylized",
    "clay_render": "3d_stylized",
    "low_poly": "3d_stylized",
    "game_asset": "3d_stylized",

    # Painterly / traditional media
    "digital_painting": "painterly",
    "oil_painting": "painterly",
    "watercolor": "painterly",
    "concept_art": "painterly",
    "fantasy_art": "painterly",
    "dark_fantasy": "painterly",

    # Line art / sketch
    "pencil_drawing": "sketch",
    "sketch": "sketch",
    "ink_drawing": "sketch",

    # Genre-flavored but photographic-leaning (treated as realistic-compatible
    # unless explicitly combined with a 2D family)
    "cyberpunk": "realistic",
    "sci_fi": "realistic",
    "steampunk": "realistic",
    "medieval_art": "realistic",

    # Ambiguous / low-signal — excluded from conflict checks
    "stylized": "ambiguous",
    "pixel_art": "pixel",
    "minimalist": "ambiguous",
    "abstract": "ambiguous",
}

# Families considered mutually exclusive when they appear together in one
# scene (asymmetric/explicit pairs beyond simple "different family" checks).
INCOMPATIBLE_FAMILY_PAIRS = {
    frozenset({"realistic", "anime_manga"}),
    frozenset({"realistic", "comic_cartoon"}),
    frozenset({"realistic", "pixel"}),
    frozenset({"realistic", "3d_stylized"}),
    frozenset({"anime_manga", "3d_stylized"}),
    frozenset({"anime_manga", "painterly"}),
    frozenset({"comic_cartoon", "painterly"}),
}

# Families that are considered "close enough" not to warn even though they
# aren't identical (avoids false positives for near-neighbors).
FAMILY_BRIDGES = {
    frozenset({"realistic", "painterly"}),  # cinematic/concept art often coexist fine
    frozenset({"sketch", "painterly"}),
}

# ---------------------------------------------------------------------------
# Attribute helpers
# ---------------------------------------------------------------------------

_HUMAN_AGES = ["Child", "Teenager", "Young Adult", "Adult", "Middle-aged", "Elderly"]
_ETHNICITIES = [
    "European", "Scandinavian", "Mediterranean", "Latin American", "East Asian",
    "South Asian", "Southeast Asian", "Middle Eastern", "African", "Mixed", "Other",
]
_SKIN_TONES = ["Very Fair", "Fair", "Light", "Medium", "Olive", "Tan", "Brown", "Dark Brown", "Deep"]
_HEIGHTS = ["Very Short", "Short", "Average Height", "Tall", "Very Tall"]
_NOSES = ["Small", "Straight", "Wide", "Narrow", "Roman", "Button", "Aquiline"]
_EYE_SHAPES = ["Round", "Almond", "Narrow", "Hooded", "Deep-set", "Wide-set"]
_EYE_COLORS = ["Dark Brown", "Brown", "Hazel", "Green", "Blue", "Gray", "Amber", "Black"]
_EYEBROWS = ["Thin", "Natural", "Thick", "Bushy", "Straight", "Arched"]
_HAIR_COLORS = ["Black", "Dark Brown", "Brown", "Light Brown", "Blonde", "Platinum Blonde", "Red", "Gray", "White", "Dyed"]
_FRECKLES = ["None", "Light", "Visible", "Heavy"]
_SCARS = ["None", "Small Facial Scar", "Cheek Scar", "Eyebrow Scar", "Large Scar"]
_HAIR_TEXTURES = ["Straight", "Wavy", "Curly", "Coily"]
_HAIR_LENGTHS = ["Bald", "Very Short", "Short", "Medium", "Long", "Very Long"]
_HAIR_STYLES_MAN = ["Buzz Cut", "Short", "Messy", "Curly", "Wavy", "Straight", "Slicked Back", "Side Part", "Undercut", "Mohawk", "Dreadlocks", "Braids", "Ponytail", "Man Bun"]
_HAIR_STYLES_WOMAN = ["Loose", "Straight", "Curly", "Wavy", "Bob", "Pixie Cut", "Ponytail", "High Ponytail", "Low Ponytail", "Bun", "Braids", "Dreadlocks", "Side Part", "Bangs", "Afro"]


def _attr(key, label, values, template="{}", skip=("None",)):
    return {"key": key, "label": label, "values": list(values), "template": template, "skip": set(skip)}


# ---------------------------------------------------------------------------
# Character types
# ---------------------------------------------------------------------------

CHARACTER_TYPES: Dict[str, Dict[str, Any]] = {
    "man": {
        "label": "Man",
        "subject_noun": "man",
        "categories": [
            {"name": "Identity", "attributes": [
                _attr("age", "Age", _HUMAN_AGES),
                _attr("ethnicity", "Ethnicity / Appearance Origin", _ETHNICITIES, skip=("Other",)),
                _attr("skin_tone", "Skin Tone", _SKIN_TONES, "{} skin"),
            ]},
            {"name": "Body", "attributes": [
                _attr("body_type", "Body Type", ["Slim", "Lean", "Athletic", "Muscular", "Average", "Broad", "Stocky", "Overweight", "Heavyset"], "{} body"),
                _attr("height", "Height", _HEIGHTS),
                _attr("build", "Build", ["Delicate", "Narrow", "Average", "Broad Shoulders", "Powerful", "Bulky"], "{} build"),
            ]},
            {"name": "Face", "attributes": [
                _attr("face_shape", "Face Shape", ["Oval", "Round", "Square", "Rectangular", "Heart-shaped", "Diamond-shaped", "Long"], "{} face"),
                _attr("jaw", "Jaw", ["Soft", "Defined", "Strong", "Square", "Narrow"], "{} jaw"),
                _attr("nose", "Nose", _NOSES, "{} nose"),
                _attr("eye_shape", "Eye Shape", _EYE_SHAPES, "{} eye shape"),
                _attr("eye_color", "Eye Color", _EYE_COLORS, "{} eyes"),
                _attr("eyebrows", "Eyebrows", _EYEBROWS, "{} eyebrows"),
            ]},
            {"name": "Hair", "attributes": [
                _attr("hair_color", "Hair Color", _HAIR_COLORS, "{} hair"),
                _attr("hair_length", "Hair Length", _HAIR_LENGTHS, "{} hair"),
                _attr("hair_style", "Hair Style", _HAIR_STYLES_MAN, "{} hairstyle"),
                _attr("hair_texture", "Hair Texture", _HAIR_TEXTURES, "{} hair"),
            ]},
            {"name": "Facial Hair", "attributes": [
                _attr("beard", "Beard", ["Clean Shaven", "Stubble", "Short Beard", "Full Beard", "Long Beard", "Goatee", "Mustache", "Handlebar Mustache"], "{} beard"),
                _attr("beard_color", "Beard Color", ["Same as Hair", "Black", "Brown", "Blonde", "Gray", "White"], "{} beard", skip=("Same as Hair",)),
            ]},
            {"name": "Distinctive Features", "attributes": [
                _attr("freckles", "Freckles", _FRECKLES, "{} freckles"),
                _attr("scars", "Scars", _SCARS),
                _attr("tattoos", "Tattoos", ["None", "Small", "Arms", "Chest", "Neck", "Full Sleeve", "Multiple"], "{} tattoos"),
                _attr("piercings", "Piercings", ["None", "Ear", "Nose", "Eyebrow", "Multiple"], "{} piercings"),
                _attr("glasses", "Glasses", ["None", "Round", "Square", "Rectangular", "Aviator", "Sunglasses"], "{} glasses"),
            ]},
            {"name": "Style", "attributes": [
                _attr("style", "Style", [
                    "Ultra Realistic",
                    "Cinematic",
                    "Photorealistic",
                    "Realistic",
                    "Anime",
                    "Manga",
                    "Comic Book",
                    "Cartoon",
                    "Animation",
                    "3D Animation",
                    "3D Render",
                    "Pixar-like",
                    "Disney-like",
                    "Stylized",
                    "Semi-Realistic",
                    "Fantasy Art",
                    "Dark Fantasy",
                    "Cyberpunk",
                    "Sci-Fi",
                    "Steampunk",
                    "Medieval Art",
                    "Concept Art",
                    "Digital Painting",
                    "Oil Painting",
                    "Watercolor",
                    "Pencil Drawing",
                    "Sketch",
                    "Ink Drawing",
                    "Pixel Art",
                    "Low Poly",
                    "Game Asset",
                    "Clay Render",
                    "Minimalist",
                    "Abstract"
                ], skip=("None",)),
            ]},
        ],
    },
    "woman": {
        "label": "Woman",
        "subject_noun": "woman",
        "categories": [
            {"name": "Identity", "attributes": [
                _attr("age", "Age", _HUMAN_AGES),
                _attr("ethnicity", "Ethnicity / Appearance Origin", _ETHNICITIES, skip=("Other",)),
                _attr("skin_tone", "Skin Tone", _SKIN_TONES, "{} skin"),
            ]},
            {"name": "Body", "attributes": [
                _attr("body_type", "Body Type", ["Slim", "Lean", "Athletic", "Curvy", "Average", "Hourglass", "Petite", "Plus Size", "Muscular"], "{} body"),
                _attr("height", "Height", _HEIGHTS),
                _attr("build", "Build", ["Delicate", "Petite", "Average", "Broad", "Athletic", "Strong"], "{} build"),
            ]},
            {"name": "Face", "attributes": [
                _attr("face_shape", "Face Shape", ["Oval", "Round", "Square", "Heart-shaped", "Diamond-shaped", "Long"], "{} face"),
                _attr("jaw", "Jaw", ["Soft", "Defined", "Strong", "Narrow"], "{} jaw"),
                _attr("nose", "Nose", _NOSES, "{} nose"),
                _attr("eye_shape", "Eye Shape", _EYE_SHAPES, "{} eye shape"),
                _attr("eye_color", "Eye Color", _EYE_COLORS, "{} eyes"),
                _attr("eyebrows", "Eyebrows", _EYEBROWS, "{} eyebrows"),
            ]},
            {"name": "Hair", "attributes": [
                _attr("hair_color", "Hair Color", _HAIR_COLORS, "{} hair"),
                _attr("hair_length", "Hair Length", _HAIR_LENGTHS, "{} hair"),
                _attr("hair_texture", "Hair Texture", _HAIR_TEXTURES, "{} hair"),
                _attr("hair_style", "Hair Style", _HAIR_STYLES_WOMAN, "{} hairstyle"),
            ]},
            {"name": "Distinctive Features", "attributes": [
                _attr("freckles", "Freckles", _FRECKLES, "{} freckles"),
                _attr("scars", "Scars", _SCARS),
                _attr("tattoos", "Tattoos", ["None", "Small", "Arms", "Back", "Legs", "Neck", "Full Sleeve", "Multiple"], "{} tattoos"),
                _attr("piercings", "Piercings", ["None", "Ear", "Nose", "Eyebrow", "Lip", "Multiple"], "{} piercings"),
                _attr("glasses", "Glasses", ["None", "Round", "Square", "Rectangular", "Cat Eye", "Aviator", "Sunglasses"], "{} glasses"),
            ]},
            {"name": "Makeup", "attributes": [
                _attr("makeup_style", "Makeup Style", ["None", "Natural", "Light", "Glamorous", "Dramatic", "Smokey Eyes", "Gothic", "Colorful"], "{} makeup"),
                _attr("lip_color", "Lip Color", ["Natural", "Nude", "Pink", "Red", "Dark Red", "Purple", "Black"], "{} lips", skip=("Natural",)),
            ]},
            {"name": "Style", "attributes": [
                _attr("style", "Style", [
                    "Ultra Realistic",
                    "Cinematic",
                    "Photorealistic",
                    "Realistic",
                    "Anime",
                    "Manga",
                    "Comic Book",
                    "Cartoon",
                    "Animation",
                    "3D Animation",
                    "3D Render",
                    "Pixar-like",
                    "Disney-like",
                    "Stylized",
                    "Semi-Realistic",
                    "Fantasy Art",
                    "Dark Fantasy",
                    "Cyberpunk",
                    "Sci-Fi",
                    "Steampunk",
                    "Medieval Art",
                    "Concept Art",
                    "Digital Painting",
                    "Oil Painting",
                    "Watercolor",
                    "Pencil Drawing",
                    "Sketch",
                    "Ink Drawing",
                    "Pixel Art",
                    "Low Poly",
                    "Game Asset",
                    "Clay Render",
                    "Minimalist",
                    "Abstract"
                ], skip=("None",)),
            ]},
        ],
    },
    "animal": {
        "label": "Animal",
        "subject_noun": None,  # subject is built from species
        "categories": [
            {"name": "Identity", "attributes": [
                _attr("species", "Species", ["Dog", "Cat", "Horse", "Wolf", "Fox", "Bear", "Lion", "Tiger", "Bird", "Rabbit", "Deer", "Dragon", "Fantasy Creature", "Other"]),
                _attr("gender", "Gender", ["Male", "Female", "Unknown"], skip=("Unknown",)),
                _attr("age", "Age", ["Baby", "Young", "Adult", "Elderly"]),
            ]},
            {"name": "Appearance", "attributes": [
                _attr("size", "Size", ["Tiny", "Small", "Medium", "Large", "Very Large", "Giant"], "{} size"),
                _attr("body_type", "Body Type", ["Slim", "Lean", "Athletic", "Strong", "Heavy", "Large", "Muscular"], "{} body"),
                _attr("surface", "Fur / Skin / Surface", ["Short Fur", "Long Fur", "Smooth", "Scaly", "Feathered", "Hairless", "Rough Skin"]),
                _attr("primary_color", "Primary Color", ["Black", "White", "Gray", "Brown", "Dark Brown", "Light Brown", "Golden", "Red", "Orange", "Cream", "Mixed"], "{} colored"),
                _attr("pattern", "Pattern", ["Solid", "Spotted", "Striped", "Patches", "Gradient", "Mixed"], "{} pattern", skip=("Solid",)),
            ]},
            {"name": "Head", "attributes": [
                _attr("eye_color", "Eye Color", ["Brown", "Dark Brown", "Blue", "Green", "Amber", "Yellow", "Gray", "Black"], "{} eyes"),
                _attr("ear_type", "Ear Type", ["Small", "Large", "Pointed", "Rounded", "Floppy", "Long"], "{} ears"),
                _attr("tail", "Tail", ["None", "Short", "Long", "Bushy", "Curled", "Thin"], "{} tail"),
            ]},
            {"name": "Distinctive Features", "attributes": [
                _attr("scars", "Scars", ["None", "Small Scar", "Facial Scar", "Body Scar", "Multiple Scars"]),
                _attr("accessories", "Accessories", ["None", "Collar", "Tag", "Harness", "Armor", "Clothing", "Jewelry", "Fantasy Accessories"], "wearing a {}"),
                _attr("special_features", "Special Features", ["None", "Horns", "Wings", "Multiple Tails", "Glowing Eyes", "Glowing Fur", "Unusual Color", "Magical Aura", "Mechanical Parts"]),
            ]},
            {"name": "Style", "attributes": [
                _attr("style", "Style", [
                    "Ultra Realistic",
                    "Cinematic",
                    "Photorealistic",
                    "Realistic",
                    "Anime",
                    "Manga",
                    "Comic Book",
                    "Cartoon",
                    "Animation",
                    "3D Animation",
                    "3D Render",
                    "Pixar-like",
                    "Disney-like",
                    "Stylized",
                    "Semi-Realistic",
                    "Fantasy Art",
                    "Dark Fantasy",
                    "Cyberpunk",
                    "Sci-Fi",
                    "Steampunk",
                    "Medieval Art",
                    "Concept Art",
                    "Digital Painting",
                    "Oil Painting",
                    "Watercolor",
                    "Pencil Drawing",
                    "Sketch",
                    "Ink Drawing",
                    "Pixel Art",
                    "Low Poly",
                    "Game Asset",
                    "Clay Render",
                    "Minimalist",
                    "Abstract"
                ], skip=("None",)),
            ]},
        ],
    },
}

# Shared categories appended to all types (Style is handled separately in the
# UI/prompt builder because it maps to CHARACTER_STYLES).
SHARED_CATEGORIES = [
    {"name": "Clothing", "attributes": [
        _attr("clothing_style", "Clothing Style", ["Casual", "Formal", "Business", "Streetwear", "Sportswear", "Military", "Medieval", "Fantasy", "Sci-Fi", "Cyberpunk", "Historical", "Traditional", "Luxury", "Minimalist"], "wearing {} clothing"),
        _attr("clothing_color", "Primary Clothing Color", ["Black", "White", "Gray", "Brown", "Blue", "Red", "Green", "Yellow", "Purple", "Orange", "Pink", "Mixed"], "{} colored clothing"),
    ]},
    {"name": "Personality / Expression", "attributes": [
        _attr("expression", "General Expression", ["Neutral", "Happy", "Serious", "Angry", "Sad", "Confident", "Friendly", "Mysterious", "Aggressive", "Calm"], "{} expression"),
        _attr("personality", "Personality", ["Brave", "Intelligent", "Curious", "Serious", "Funny", "Mysterious", "Friendly", "Aggressive", "Shy", "Confident", "Elegant", "Chaotic"], "{} personality"),
    ]},
]

# Attribute keys consumed by the subject line (excluded from generic phrases).
_SUBJECT_KEYS = {"age", "ethnicity", "species", "gender"}


def get_categories(char_type: str):
    """All categories (type-specific + shared) for a character type."""
    return CHARACTER_TYPES[char_type]["categories"] + SHARED_CATEGORIES


def _build_subject(char_type: str, attributes: Dict[str, str]) -> str:
    age = attributes.get("age", "").lower()
    if char_type == "animal":
        species = attributes.get("species", "animal")
        species = "animal" if species == "Other" else species.lower()
        gender = attributes.get("gender", "")
        gender = "" if gender in ("", "Unknown") else gender.lower()
        return " ".join(p for p in (age, gender, species) if p)
    ethnicity = attributes.get("ethnicity", "")
    ethnicity = "" if ethnicity in ("", "Other") else ethnicity.lower()
    noun = CHARACTER_TYPES[char_type]["subject_noun"]
    return " ".join(p for p in (age, ethnicity, noun) if p)


def build_character_prompt(
    char_type: str,
    style_id: str,
    attributes: Dict[str, str],
    custom_description: str = "",
) -> str:
    """Build the final generation prompt.

    Order: [style prefix + subject] + [physical attributes] +
    [distinctive features] + [clothing] + [custom description] +
    [style modifiers].
    """
    style = CHARACTER_STYLES.get(style_id, CHARACTER_STYLES[DEFAULT_STYLE])
    parts = [f"{style['prefix']} {_build_subject(char_type, attributes)}"]

    for category in get_categories(char_type):
        for attr in category["attributes"]:
            if attr["key"] in _SUBJECT_KEYS:
                continue
            value = attributes.get(attr["key"], "")
            if not value or value in attr["skip"]:
                continue
            # Special handling for style attribute to prevent duplication
            if attr["key"] == "style":
                continue
            parts.append(attr["template"].format(value.lower()))

    custom_description = custom_description.strip()
    if custom_description:
        parts.append(custom_description)

    parts.append(style["modifiers"])
    return ", ".join(parts)
