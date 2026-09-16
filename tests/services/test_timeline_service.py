"""
Tests for the timeline service (plan §4: user-directed timeline).
"""

import unittest
import tempfile
import os
from services.database.timeline_service import TimelineService


class TestTimelineService(unittest.TestCase):
    def setUp(self):
        self.tmpdb = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.tmpdb.name
        self.tmpdb.close()
        self.service = TimelineService(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_save_and_list_entries_in_user_order(self):
        self.service.save_entry("p", 2, 1, 1, "19:30")
        self.service.save_entry("p", 1, 2, 1, "08:15", duration_minutes=10)
        self.service.save_entry("p", 1, 1, 1, "08:00", events=["breakfast"])
        entries = self.service.list_entries("p")
        self.assertEqual(len(entries), 3)
        self.assertEqual(
            [(e["chapter_number"], e["scene_number"]) for e in entries],
            [(1, 1), (1, 2), (2, 1)],
        )
        self.assertEqual(entries[0]["events"], ["breakfast"])
        self.assertEqual(entries[1]["duration_minutes"], 10)

    def test_upsert_updates_existing_entry(self):
        self.service.save_entry("p", 1, 1, 1, "08:00")
        self.service.save_entry("p", 1, 1, 3, "22:00")
        entries = self.service.list_entries("p")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["day"], 3)
        self.assertEqual(entries[0]["time_of_day"], "22:00")

    def test_delete_entry(self):
        self.service.save_entry("p", 1, 1, 1, "08:00")
        self.assertTrue(self.service.delete_entry("p", 1, 1))
        self.assertFalse(self.service.delete_entry("p", 1, 1))
        self.assertEqual(self.service.list_entries("p"), [])

    def test_projects_are_isolated(self):
        self.service.save_entry("a", 1, 1, 1, "08:00")
        self.assertEqual(self.service.list_entries("b"), [])

    def test_detect_inconsistencies_warns_but_never_fixes(self):
        self.service.save_entry("p", 1, 1, 5, "08:00")
        self.service.save_entry("p", 1, 2, 2, "08:00")   # earlier day
        self.service.save_entry("p", 2, 1, 2, "06:00")   # earlier time same day
        warnings = self.service.detect_inconsistencies("p")
        self.assertEqual(len(warnings), 2)
        # Entries remain exactly as the user wrote them.
        entries = self.service.list_entries("p")
        self.assertEqual([e["day"] for e in entries], [5, 2, 2])

    def test_consistent_timeline_has_no_warnings(self):
        self.service.save_entry("p", 1, 1, 1, "08:00")
        self.service.save_entry("p", 1, 2, 1, "08:15")
        self.service.save_entry("p", 2, 1, 1, "19:30")
        self.assertEqual(self.service.detect_inconsistencies("p"), [])


if __name__ == "__main__":
    unittest.main()
