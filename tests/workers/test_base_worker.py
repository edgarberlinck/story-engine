"""
Tests for base worker infrastructure.
"""

import unittest
from workers.base_worker import GenerationWorker


class TestBaseWorker(unittest.TestCase):
    def test_submit_task(self):
        worker = GenerationWorker(max_workers=1)
        future = worker.submit(lambda x: x + 1, 5)
        result = future.result(timeout=1)
        self.assertEqual(result, 6)
        worker.shutdown()


if __name__ == "__main__":
    unittest.main()
