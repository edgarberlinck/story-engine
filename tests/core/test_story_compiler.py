"""
Tests for the story compiler: written chapters -> persisted audio scenes.
"""

import unittest
import tempfile
import os
from unittest.mock import patch

from core import story_compiler
from services.audio_scene_service import AudioSceneService
from services.database.manuscript_service import ManuscriptService


class TestStoryCompiler(unittest.TestCase):
    def setUp(self):
        self.tmpdb = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.tmpdb.name
        self.tmpdb.close()
        self.manuscripts = ManuscriptService(self.db_path)
        self.audio = AudioSceneService(self.db_path)
        self.p1 = patch.object(story_compiler, "manuscript_service", self.manuscripts)
        self.p2 = patch.object(story_compiler, "audio_scene_service", self.audio)
        self.p1.start()
        self.p2.start()

    def tearDown(self):
        self.p1.stop()
        self.p2.stop()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def _chapter(self, content, locale="en", title="C1"):
        ch = self.manuscripts.create_chapter("p", title)
        self.manuscripts.save_content(ch, locale, content)
        return ch

    def test_compile_chapter_persists_scenes(self):
        ch = self._chapter(
            "<scene title='Morning'>"
            "<character name='nikita' emotion='warm'>Good morning.</character>"
            "</scene>"
            "<scene title='Evening'>[Narrator] The sun sets.</scene>"
        )
        result = story_compiler.compile_chapter("p", ch)
        self.assertIsNone(result.error)
        self.assertEqual(result.scene_numbers, [1, 2])
        rep = self.audio.get_representation("p", 1)
        self.assertEqual(rep.title, "Morning")
        self.assertEqual(rep.segments[0].speaker, "nikita")
        rep2 = self.audio.get_representation("p", 2)
        self.assertEqual(rep2.segments[0].segment_type, "narration")

    def test_recompile_updates_in_place(self):
        ch = self._chapter("[Narrator] version one")
        first = story_compiler.compile_chapter("p", ch)
        self.manuscripts.save_content(ch, "en", "[Narrator] version two")
        second = story_compiler.compile_chapter("p", ch)
        self.assertEqual(first.scene_numbers, second.scene_numbers)
        rep = self.audio.get_representation("p", second.scene_numbers[0])
        self.assertIn("version two", rep.segments[0].text)

    def test_growing_chapter_appends_new_scene_numbers(self):
        ch = self._chapter("<scene>[Narrator] one</scene>")
        first = story_compiler.compile_chapter("p", ch)
        self.manuscripts.save_content(
            ch,
            "en",
            "<scene>[Narrator] one</scene><scene>[Narrator] two</scene>",
        )
        second = story_compiler.compile_chapter("p", ch)
        self.assertEqual(second.scene_numbers[0], first.scene_numbers[0])
        self.assertEqual(len(second.scene_numbers), 2)

    def test_two_chapters_get_distinct_scene_numbers(self):
        ch1 = self._chapter("[Narrator] chapter one")
        ch2 = self._chapter("[Narrator] chapter two", title="C2")
        r1 = story_compiler.compile_chapter("p", ch1)
        r2 = story_compiler.compile_chapter("p", ch2)
        self.assertEqual(set(r1.scene_numbers) & set(r2.scene_numbers), set())

    def test_locales_compile_independently(self):
        ch = self._chapter("[Narrator] hello")
        self.manuscripts.save_content(ch, "pt-BR", "[Narrator] ol\u00e1")
        r_en = story_compiler.compile_chapter("p", ch, "en")
        r_pt = story_compiler.compile_chapter("p", ch, "pt-BR")
        self.assertNotEqual(r_en.scene_numbers, r_pt.scene_numbers)
        chapter = self.manuscripts.get_chapter(ch)
        self.assertEqual(set(chapter["compiled_scenes"]), {"en", "pt-BR"})

    def test_empty_content_errors(self):
        ch = self.manuscripts.create_chapter("p", "Empty")
        result = story_compiler.compile_chapter("p", ch)
        self.assertIsNotNone(result.error)

    def test_missing_chapter_errors(self):
        result = story_compiler.compile_chapter("p", 999)
        self.assertIsNotNone(result.error)

    def test_issues_are_surfaced(self):
        ch = self._chapter("<scene><character name='a'>unclosed")
        result = story_compiler.compile_chapter("p", ch)
        self.assertIsNone(result.error)
        self.assertTrue(any("auto-closed" in i.message for i in result.issues))

    def test_compile_project_compiles_all_chapters(self):
        self._chapter("[Narrator] one")
        self._chapter("[Narrator] two", title="C2")
        results = story_compiler.compile_project("p")
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.error is None for r in results))


if __name__ == "__main__":
    unittest.main()
