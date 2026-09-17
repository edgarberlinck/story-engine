"""
Extra coverage tests for cli/main.py.

Complements tests/cli/test_cli.py by exercising every CLI command through
typer's CliRunner while the core managers are mocked, so no real database or
model is touched.
"""

import unittest
from unittest.mock import patch
from typer.testing import CliRunner
from cli.main import app


class TestCLICoverage(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    # -- list-projects ----------------------------------------------------

    def test_list_projects_with_data(self):
        with patch(
            "cli.main.project_manager.list_projects",
            return_value=[{"id": "1", "name": "Alpha", "description": "d"}],
        ):
            result = self.runner.invoke(app, ["list-projects"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Alpha", result.output)

    def test_list_projects_description_default(self):
        # A project dict without 'description' exercises the .get default.
        with patch(
            "cli.main.project_manager.list_projects",
            return_value=[{"id": "2", "name": "Beta"}],
        ):
            result = self.runner.invoke(app, ["list-projects"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Beta", result.output)

    # -- new-project -----------------------------------------------------

    def test_new_project(self):
        with patch(
            "cli.main.project_manager.create_project", return_value="project_123"
        ) as mock_create:
            result = self.runner.invoke(app, ["new-project", "Awesome"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("project_123", result.output)
        mock_create.assert_called_once_with("Awesome", "")

    def test_new_project_with_description(self):
        with patch(
            "cli.main.project_manager.create_project", return_value="project_456"
        ) as mock_create:
            result = self.runner.invoke(
                app, ["new-project", "Great", "--description", "A great project"]
            )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("project_456", result.output)
        mock_create.assert_called_once_with("Great", "A great project")

    # -- delete-project --------------------------------------------------

    def test_delete_project_success(self):
        with patch(
            "cli.main.project_manager.delete_project", return_value=True
        ) as mock_delete:
            result = self.runner.invoke(app, ["delete-project", "pid_1"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Deleted project pid_1", result.output)
        mock_delete.assert_called_once_with("pid_1")

    def test_delete_project_not_found(self):
        with patch("cli.main.project_manager.delete_project", return_value=False):
            result = self.runner.invoke(app, ["delete-project", "pid_x"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Project not found pid_x", result.output)

    # -- list-characters -------------------------------------------------

    def test_list_characters_default_project(self):
        with patch(
            "cli.main.character_manager.list_characters",
            return_value=[{"name": "Bob", "prompt": "a tall man standing tall"}],
        ) as mock:
            result = self.runner.invoke(app, ["list-characters"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Bob", result.output)
        # Default project is "test_project".
        self.assertEqual(mock.call_args.args[0], "test_project")

    def test_list_characters_with_option(self):
        with patch(
            "cli.main.character_manager.list_characters",
            return_value=[{"name": "Alice", "prompt": "a woman"}],
        ) as mock:
            result = self.runner.invoke(app, ["list-characters", "-p", "story1"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Alice", result.output)
        self.assertEqual(mock.call_args.args[0], "story1")

    # -- generate-character ------------------------------------------------

    def test_generate_character_default_variants(self):
        with patch(
            "cli.main.character_manager.generate_versions",
            return_value=[{"version": 1}, {"version": 2}, {"version": 3}],
        ) as mock:
            result = self.runner.invoke(
                app, ["generate-character", "p1", "Hero", "a hero"]
            )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Generated 3 versions for Hero", result.output)
        # Variants default to 3.
        self.assertEqual(mock.call_args.kwargs["num_versions"], 3)

    def test_generate_character_custom_variants(self):
        with patch(
            "cli.main.character_manager.generate_versions",
            return_value=[{"version": 1}, {"version": 2}],
        ) as mock:
            result = self.runner.invoke(
                app,
                ["generate-character", "p2", "Villain", "a villain", "--variants", "2"],
            )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Generated 2 versions for Villain", result.output)
        self.assertEqual(mock.call_args.kwargs["num_versions"], 2)

    # -- help / invocation -------------------------------------------------

    def test_help_shows_commands(self):
        result = self.runner.invoke(app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("list-projects", result.output)
        self.assertIn("generate-character", result.output)

    def test_invoke_app_calls_all_registered_commands(self):
        # Invoking the app without subcommands exercises the __main__ path's
        # command registration; ensure it returns a clean exit code on error.
        result = self.runner.invoke(app, [])
        # No subcommand -> typer prints usage; that is a non-zero exit.
        self.assertNotEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
