"""
Tests for the story markup parser (core/story_markup.py).

Covers both notations (XML-like markup and bracket shorthand), mixing them,
and the tolerant-parsing guarantees (unclosed tags, typos, unknown tags).
"""

import unittest

from core.story_markup import (
    parse_story_markup,
    compile_story_markup,
    scenes_to_representations,
)


class TestBracketShorthand(unittest.TestCase):
    def test_narrator_and_speaker_switch(self):
        text = "[Narrator]\nSome narration. [Nikita] Don't worry about it..."
        scenes, issues = parse_story_markup(text)
        self.assertEqual(len(scenes), 1)
        segs = scenes[0].segments
        self.assertEqual(segs[0].segment_type, "narration")
        self.assertEqual(segs[0].speaker, "narrator")
        self.assertEqual(segs[1].segment_type, "dialogue")
        self.assertEqual(segs[1].speaker, "Nikita")
        self.assertEqual(issues, [])

    def test_feeling_changes_split_segments(self):
        text = "[Narrator] calm start [Feeling=Angry] angry middle [Feeling=calm] calm end"
        scenes, _ = parse_story_markup(text)
        segs = scenes[0].segments
        self.assertEqual([s.emotion for s in segs], [None, "Angry", "calm"])
        self.assertEqual(segs[1].text, "angry middle")

    def test_speaker_switch_resets_performance_state(self):
        text = "[Nikita] hello [Feeling=Angry] grr [Roger] hi there"
        scenes, _ = parse_story_markup(text)
        segs = scenes[0].segments
        self.assertEqual(segs[1].emotion, "Angry")
        self.assertIsNone(segs[2].emotion)
        self.assertEqual(segs[2].speaker, "Roger")

    def test_all_bracket_attributes(self):
        text = ("[Nikita] [Tone=cold] [Delivery=slow] [Intensity=0.8] "
                "[Voice=roger] line here")
        scenes, issues = parse_story_markup(text)
        seg = scenes[0].segments[0]
        self.assertEqual(seg.tone, "cold")
        self.assertEqual(seg.delivery, "slow")
        self.assertEqual(seg.intensity, 0.8)
        self.assertEqual(seg.voice, "roger")
        self.assertEqual(issues, [])

    def test_sound_and_music_brackets(self):
        text = "[Sound=door slamming] [Narrator] then... [Music=soft strings]"
        scenes, _ = parse_story_markup(text)
        types = [s.segment_type for s in scenes[0].segments]
        self.assertEqual(types, ["sound_effect", "narration", "music"])
        self.assertEqual(scenes[0].segments[0].sound_effects, ["door slamming"])
        self.assertTrue(scenes[0].segments[2].music)

    def test_invalid_intensity_warns(self):
        scenes, issues = parse_story_markup("[Nikita] [Intensity=loud] hi")
        self.assertIsNone(scenes[0].segments[0].intensity)
        self.assertTrue(any("intensity" in i.message.lower() for i in issues))

    def test_unknown_bracket_key_warns_but_continues(self):
        scenes, issues = parse_story_markup("[Nikita] [Wibble=x] hi")
        self.assertEqual(scenes[0].segments[0].text, "hi")
        self.assertTrue(any("Wibble" in i.message for i in issues))


class TestXmlMarkup(unittest.TestCase):
    def test_full_example_from_the_plan(self):
        text = """<scene>
          <sound preset='adventure' />
          <character name='narrator' tone='neutral'>
            And your story begins with Fable receiving strange instructions
          </character>
          <character name='claude fable' tone='scary'>
            No, it's too dificult to me
          </character>
          <sound prompt="confusing sound">
            <scene>
               <character name="Narator" tone="misterious">
                   But then your hero shows up
               </character>
            </scene>
          </sound>
        <scene>"""
        scenes, issues = parse_story_markup(text)
        self.assertEqual(len(scenes), 1)
        segs = scenes[0].segments
        self.assertEqual(
            [s.segment_type for s in segs],
            ["sound_effect", "narration", "dialogue", "sound_effect", "narration"],
        )
        # Misspelled "Narator" is still the narrator.
        self.assertEqual(segs[4].speaker, "narrator")
        self.assertEqual(segs[4].tone, "misterious")
        self.assertEqual(segs[2].speaker, "claude fable")
        # Unclosed tags produce a warning, not an error.
        self.assertTrue(any("auto-closed" in i.message for i in issues))

    def test_multiple_scenes(self):
        text = ("<scene title='One'><character name='a'>x</character></scene>"
                "<scene title='Two'><character name='b'>y</character></scene>")
        scenes, _ = parse_story_markup(text)
        self.assertEqual([s.title for s in scenes], ["One", "Two"])
        self.assertEqual(scenes[0].segments[0].speaker, "a")
        self.assertEqual(scenes[1].segments[0].speaker, "b")

    def test_character_attributes(self):
        text = ("<character name='nikita' emotion='warm' tone='friendly' "
                "intensity='0.3' delivery='calm' voice='custom'>Good morning.</character>")
        scenes, _ = parse_story_markup(text)
        seg = scenes[0].segments[0]
        self.assertEqual(
            (seg.emotion, seg.tone, seg.intensity, seg.delivery, seg.voice),
            ("warm", "friendly", 0.3, "calm", "custom"),
        )

    def test_state_restored_after_character_block(self):
        text = ("[Narrator] before "
                "<character name='nikita' emotion='angry'>line</character>"
                " after")
        scenes, _ = parse_story_markup(text)
        segs = scenes[0].segments
        self.assertEqual(segs[0].speaker, "narrator")
        self.assertEqual(segs[1].speaker, "nikita")
        # After the block the narrator state is restored.
        self.assertEqual(segs[2].speaker, "narrator")
        self.assertIsNone(segs[2].emotion)

    def test_unknown_tag_ignored_with_warning(self):
        scenes, issues = parse_story_markup("<blink>hey</blink> [Narrator] hi")
        self.assertTrue(any("blink" in i.message for i in issues))
        self.assertEqual(scenes[0].segments[-1].text, "hi")

    def test_bracket_tokens_inside_character_text(self):
        text = ("<character name='nikita'>calm... [Feeling=Angry] furious!"
                "</character>")
        scenes, _ = parse_story_markup(text)
        segs = scenes[0].segments
        self.assertIsNone(segs[0].emotion)
        self.assertEqual(segs[1].emotion, "Angry")
        self.assertEqual(segs[1].speaker, "nikita")

    def test_empty_input(self):
        scenes, issues = parse_story_markup("")
        self.assertEqual(scenes, [])

    def test_plain_text_becomes_narration(self):
        scenes, _ = parse_story_markup("Just a plain paragraph.")
        self.assertEqual(scenes[0].segments[0].segment_type, "narration")


class TestConversion(unittest.TestCase):
    def test_scenes_to_representations(self):
        text = ("<scene title='Morning'>"
                "<character name='nikita' emotion='warm'>Good morning.</character>"
                "<character name='roger'>Morning!</character>"
                "</scene>")
        scenes, _ = parse_story_markup(text)
        reps = scenes_to_representations(scenes, scene_id_prefix="ch1_en")
        self.assertEqual(len(reps), 1)
        rep = reps[0]
        self.assertEqual(rep.scene_id, "ch1_en_001")
        self.assertEqual(rep.title, "Morning")
        self.assertEqual(rep.characters_present, ["nikita", "roger"])
        self.assertEqual(rep.segments[0].emotion, "warm")

    def test_compile_story_markup_roundtrip(self):
        reps, issues = compile_story_markup("[Narrator] once upon a time")
        self.assertEqual(len(reps), 1)
        self.assertEqual(reps[0].segments[0].segment_type, "narration")


if __name__ == "__main__":
    unittest.main()

class TestSceneStateReset(unittest.TestCase):
    def test_performance_state_resets_at_scene_boundary(self):
        text = ("<scene>[Narrator] one [Feeling=worried] uneasy</scene>"
                "<scene>[Narrator] fresh start</scene>")
        scenes, _ = parse_story_markup(text)
        self.assertEqual(scenes[0].segments[1].emotion, "worried")
        self.assertIsNone(scenes[1].segments[0].emotion)
