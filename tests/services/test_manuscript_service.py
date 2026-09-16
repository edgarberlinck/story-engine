"""
Tests for the manuscript service (writing layer ported from writing-tools).
"""

import unittest
import tempfile
import os

from services.database.manuscript_service import ManuscriptService


class TestManuscriptService(unittest.TestCase):
    def setUp(self):
        self.tmpdb = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.tmpdb.name
        self.tmpdb.close()
        self.svc = ManuscriptService(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_create_and_list_chapters_ordered(self):
        a = self.svc.create_chapter("p", "Intro", "introduction")
        b = self.svc.create_chapter("p", "Chapter 1")
        chapters = self.svc.list_chapters("p")
        self.assertEqual([c["id"] for c in chapters], [a, b])
        self.assertEqual(chapters[0]["chapter_type"], "introduction")
        self.assertEqual([c["position"] for c in chapters], [1, 2])

    def test_invalid_chapter_type_rejected(self):
        with self.assertRaises(ValueError):
            self.svc.create_chapter("p", "X", "moodboard-of-doom")
        ch = self.svc.create_chapter("p", "X")
        with self.assertRaises(ValueError):
            self.svc.set_chapter_type(ch, "nope")

    def test_rename_and_retype(self):
        ch = self.svc.create_chapter("p", "Old")
        self.assertTrue(self.svc.rename_chapter(ch, "New"))
        self.assertTrue(self.svc.set_chapter_type(ch, "epigraph"))
        got = self.svc.get_chapter(ch)
        self.assertEqual((got["title"], got["chapter_type"]), ("New", "epigraph"))

    def test_move_chapter_swaps_neighbors(self):
        a = self.svc.create_chapter("p", "A")
        b = self.svc.create_chapter("p", "B")
        c = self.svc.create_chapter("p", "C")
        self.assertTrue(self.svc.move_chapter("p", c, -1))
        self.assertEqual([x["id"] for x in self.svc.list_chapters("p")], [a, c, b])
        # Can't move first up or last down.
        self.assertFalse(self.svc.move_chapter("p", a, -1))
        self.assertFalse(self.svc.move_chapter("p", b, 1))

    def test_multilanguage_contents_are_independent(self):
        ch = self.svc.create_chapter("p", "C1")
        self.svc.save_content(ch, "en", "[Narrator] Hello")
        self.svc.save_content(ch, "pt-BR", "[Narrator] Ol\u00e1")
        self.assertEqual(self.svc.get_content(ch, "en"), "[Narrator] Hello")
        self.assertEqual(self.svc.get_content(ch, "pt-BR"), "[Narrator] Ol\u00e1")
        self.assertEqual(self.svc.get_content(ch, "fr"), "")
        contents = self.svc.get_contents(ch)
        self.assertEqual(set(contents), {"en", "pt-BR"})

    def test_save_content_upserts(self):
        ch = self.svc.create_chapter("p", "C1")
        self.svc.save_content(ch, "en", "v1")
        self.svc.save_content(ch, "en", "v2")
        self.assertEqual(self.svc.get_content(ch, "en"), "v2")

    def test_delete_chapter_removes_contents(self):
        ch = self.svc.create_chapter("p", "C1")
        self.svc.save_content(ch, "en", "text")
        self.assertTrue(self.svc.delete_chapter(ch))
        self.assertIsNone(self.svc.get_chapter(ch))
        self.assertEqual(self.svc.get_content(ch, "en"), "")

    def test_compiled_scenes_tracking(self):
        ch = self.svc.create_chapter("p", "C1")
        self.svc.set_compiled_scenes(ch, "en", [3, 4])
        self.svc.set_compiled_scenes(ch, "pt-BR", [5])
        got = self.svc.get_chapter(ch)
        self.assertEqual(got["compiled_scenes"], {"en": [3, 4], "pt-BR": [5]})

    def test_delete_project_chapters(self):
        ch = self.svc.create_chapter("p", "C1")
        self.svc.save_content(ch, "en", "x")
        self.svc.create_chapter("other", "Keep")
        self.svc.delete_project_chapters("p")
        self.assertEqual(self.svc.list_chapters("p"), [])
        self.assertEqual(len(self.svc.list_chapters("other")), 1)


if __name__ == "__main__":
    unittest.main()
