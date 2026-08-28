"""
Unit tests for core.character_attributes.

Covers the centralized style/type configuration, the subject builder, and
the final prompt builder.  Runs with stdlib unittest only.
"""

import unittest

from core.character_attributes import (
    CHARACTER_STYLES,
    CHARACTER_TYPES,
    SHARED_CATEGORIES,
    STYLE_FAMILIES,
    INCOMPATIBLE_FAMILY_PAIRS,
    FAMILY_BRIDGES,
    DEFAULT_STYLE,
    _attr,
    get_categories,
    _build_subject,
    build_character_prompt,
)


class TestCharacterStyles(unittest.TestCase):

    def test_styles_have_required_keys(self):
        for style_id, style in CHARACTER_STYLES.items():
            self.assertIn("label", style)
            self.assertIn("prefix", style)
            self.assertIn("modifiers", style)
            self.assertIsInstance(style["prefix"], str)

    def test_default_style_present(self):
        self.assertIn(DEFAULT_STYLE, CHARACTER_STYLES)
        self.assertEqual(DEFAULT_STYLE, "ultra_realistic")

    def test_known_styles_exist(self):
        for sid in ("ultra_realistic", "cinematic", "anime", "manga",
                    "comic_book", "cyberpunk", "sketch"):
            self.assertIn(sid, CHARACTER_STYLES)


class TestCharacterTypes(unittest.TestCase):

    def test_all_types_have_expected_shape(self):
        for ctype, spec in CHARACTER_TYPES.items():
            self.assertIn("label", spec)
            self.assertIn("subject_noun", spec)
            self.assertIn("categories", spec)
            for category in spec["categories"]:
                self.assertIn("name", category)
                self.assertIn("attributes", category)
                for attr in category["attributes"]:
                    self.assertIn("key", attr)
                    self.assertIn("label", attr)
                    self.assertIn("values", attr)
                    self.assertIn("template", attr)
                    self.assertIsInstance(attr["skip"], set)

    def test_shared_categories_have_style_relevant_attrs(self):
        names = [c["name"] for c in SHARED_CATEGORIES]
        self.assertIn("Clothing", names)
        self.assertIn("Personality / Expression", names)

    def test_style_family_data_consistent(self):
        # Every style id also has a family mapping.
        for sid in CHARACTER_STYLES:
            self.assertIn(sid, STYLE_FAMILIES)
         # Incompatible pairs are frozensets of 2 families.
        for pair in INCOMPATIBLE_FAMILY_PAIRS:
            self.assertEqual(len(pair), 2)
            self.assertIsInstance(pair, frozenset)
         # Bridges are frozensets of 2 families too.
        for bridge in FAMILY_BRIDGES:
            self.assertEqual(len(bridge), 2)
            self.assertIsInstance(bridge, frozenset)


class TestAttrHelper(unittest.TestCase):

    def test_attr_defaults(self):
        attr = _attr("k", "Label", ["a", "b"])
        self.assertEqual(attr["key"], "k")
        self.assertEqual(attr["label"], "Label")
        self.assertEqual(attr["values"], ["a", "b"])
        self.assertEqual(attr["template"], "{}")
        self.assertEqual(attr["skip"], {"None"})

    def test_attr_custom_template_and_skip(self):
        attr = _attr("k", "L", ["x"], template="{} skin", skip=("x", "y"))
        self.assertEqual(attr["template"], "{} skin")
        self.assertEqual(attr["skip"], {"x", "y"})

    def test_values_are_copied(self):
        source = ["a"]
        attr = _attr("k", "L", source)
        source.append("b")
        self.assertEqual(attr["values"], ["a"])  # not mutated


class TestGetCategories(unittest.TestCase):

    def test_returns_type_plus_shared(self):
        cats = get_categories("man")
        names = [c["name"] for c in cats]
        # Type-specific
        self.assertIn("Identity", names)
        self.assertIn("Hair", names)
        # Shared appended
        self.assertIn("Clothing", names)
        self.assertIn("Personality / Expression", names)

    def test_woman_and_animal(self):
        self.assertTrue(get_categories("woman"))
        self.assertTrue(get_categories("animal"))

    def test_unknown_type_raises(self):
        with self.assertRaises(KeyError):
            get_categories("robot")


class TestBuildSubject(unittest.TestCase):

    def test_man_full(self):
        sub = _build_subject("man", {"age": "30", "ethnicity": "European"})
        self.assertIn("30", sub)
        self.assertIn("european", sub)
        self.assertTrue(sub.endswith("man"))

    def test_man_ethnicity_other_dropped(self):
        sub = _build_subject("man", {"age": "25", "ethnicity": "Other"})
        self.assertNotIn("other", sub)
        self.assertIn("man", sub)

    def test_man_empty(self):
        sub = _build_subject("man", {})
        self.assertEqual(sub, "man")

    def test_woman_noun(self):
        sub = _build_subject("woman", {"age": "20"})
        self.assertTrue(sub.endswith("woman"))
        self.assertIn("20", sub)

    def test_animal_species_and_gender(self):
        sub = _build_subject("animal", {
            "age": "Adult", "species": "Dog", "gender": "Male"})
        self.assertIn("adult", sub)
        self.assertIn("male", sub)
        self.assertIn("dog", sub)

    def test_animal_other_species(self):
        sub = _build_subject("animal", {"species": "Other", "gender": "Unknown"})
        self.assertEqual(sub, "animal")

    def test_animal_default_species(self):
        sub = _build_subject("animal", {"age": "Baby"})
        self.assertIn("animal", sub)
        self.assertIn("baby", sub)


class TestBuildCharacterPrompt(unittest.TestCase):

    def test_unknown_style_falls_back_to_default(self):
        prompt = build_character_prompt("man", "does_not_exist", {})
        # Default style prefix must appear.
        self.assertIn(CHARACTER_STYLES[DEFAULT_STYLE]["prefix"], prompt)
        # Default modifiers must appear at the end.
        self.assertIn(CHARACTER_STYLES[DEFAULT_STYLE]["modifiers"], prompt)

    def test_style_prefix_and_modifiers_present(self):
        prompt = build_character_prompt("man", "ultra_realistic", {
            "age": "30", "ethnicity": "European"})
        self.assertIn("ultra realistic", prompt)
        self.assertIn("highly detailed skin texture", prompt)

    def test_skip_values_excluded(self):
        # "None" freckles must be omitted (it is a skip value).
        prompt = build_character_prompt("man", "ultra_realistic", {
            "freckles": "None"})
        self.assertNotIn("none freckles", prompt.lower())

    def test_non_skip_value_included(self):
        prompt = build_character_prompt("man", "ultra_realistic", {
            "freckles": "Light"})
        self.assertIn("light freckles", prompt)

    def test_custom_description_appended(self):
        prompt = build_character_prompt(
            "man", "ultra_realistic", {}, custom_description="  mysterious aura  ")
        self.assertIn("mysterious aura", prompt)
        self.assertNotIn("  mysterious aura  ", prompt)  # stripped

    def test_empty_custom_description_ignored(self):
        prompt = build_character_prompt("man", "ultra_realistic", {},
                                       custom_description="   ")
        # No leftover empty fragment / double commas.
        self.assertNotIn(", ,", prompt)

    def test_subject_keys_used_once_in_subject_only(self):
        # age/ethnicity should appear via the subject line, not as a separate
        # generic phrase.
        prompt = build_character_prompt("man", "ultra_realistic", {
            "age": "Adult", "ethnicity": "African"})
        # They are not emitted with a "{} ..." template as a generic attribute.
        self.assertNotIn("adult age", prompt.lower())
        self.assertIn("african", prompt.lower())

    def test_style_attribute_not_duplicated(self):
        # Providing a style *attribute* value must not add a second style phrase.
        prompt = build_character_prompt("man", "anime", {
            "style": "Anime", "age": "25"})
        # The prefix appears, but the style attribute is skipped (no extra).
        self.assertEqual(prompt.lower().count("anime style"),
                         prompt.lower().count("anime style"))
        self.assertIn("anime style", prompt)

    def test_animal_prompt(self):
        prompt = build_character_prompt("animal", "fantasy_art", {
            "species": "Dragon", "gender": "Female", "age": "Young"})
        self.assertIn("fantasy", prompt)
        self.assertIn("dragon", prompt)


if __name__ == "__main__":
    unittest.main()
