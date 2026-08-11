"""
Tests for CLI commands.
"""

import unittest
from cli.main import app
from typer.testing import CliRunner


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_list_projects_command(self):
        result = self.runner.invoke(app, ["list-projects"])
        self.assertEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
