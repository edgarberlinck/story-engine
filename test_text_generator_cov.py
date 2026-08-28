import unittest

from generators import text_generator as tg


class TextFilenameCoverageTest(unittest.TestCase):
    def test_filename_generation_edges(self):
        self.assertEqual(tg.generate_filename_from_prompt(None), "generated_image")
        self.assertEqual(tg.generate_filename_from_prompt("A!"), "a")
        self.assertEqual(tg.generate_filename_from_prompt("@@@"), "generated_image")
        self.assertEqual(tg.generate_filename_from_prompt("red dragon flying"), "red_dragon")
        self.assertLessEqual(len(tg.generate_filename_from_prompt("x" * 50)), 20)


if __name__ == "__main__":
    unittest.main()
