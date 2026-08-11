"""
Main application entry point with stack-based navigation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication, QStackedWidget

from ui.screens.project_list_screen import ProjectListScreen

APP_STYLE = """
QWidget { font-family: -apple-system, "Segoe UI", sans-serif; font-size: 13px; }
QPushButton {
    background-color: #4CAF50; color: white; border: none;
    border-radius: 4px; padding: 6px 14px; font-weight: 600;
}
QPushButton:hover { background-color: #43a047; }
QPushButton:disabled { background-color: #bdbdbd; }
QPushButton[flat="true"] {
    background: transparent; color: #4CAF50; padding: 4px 8px;
}
QPushButton[danger="true"] { background-color: #e53935; }
QPushButton[danger="true"]:hover { background-color: #d32f2f; }
QLineEdit, QTextEdit, QComboBox, QSpinBox {
    border: 1px solid #ccc; border-radius: 4px; padding: 5px;
    background: white; color: #222;
}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus {
    border: 1px solid #4CAF50;
}
QComboBox {
    min-width: 180px; min-height: 22px; padding: 5px 10px;
}
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView {
    background: white; color: #222;
    border: 1px solid #ccc;
    selection-background-color: #4CAF50;
    selection-color: white;
    padding: 4px;
}
QComboBox QAbstractItemView::item {
    min-height: 26px; padding: 4px 8px;
}
QScrollArea { border: none; }
"""


class App(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.setStyleSheet(APP_STYLE)

        self.stack = QStackedWidget()
        self.stack.setWindowTitle("Story Engine")
        self.stack.resize(1200, 800)

        self.project_list = ProjectListScreen()
        self.project_list.project_selected.connect(self.show_project_view)
        self.stack.addWidget(self.project_list)

        self.stack.show()

    # -- navigation helpers -------------------------------------------------

    def _push(self, widget):
        self.stack.addWidget(widget)
        self.stack.setCurrentWidget(widget)

    def _pop_to_root(self):
        """Remove every screen except the project list and show it."""
        while self.stack.count() > 1:
            w = self.stack.widget(self.stack.count() - 1)
            self.stack.removeWidget(w)
            w.deleteLater()
        self.project_list.load_projects()
        self.stack.setCurrentWidget(self.project_list)

    def _pop(self):
        """Remove the top screen and show the one below it."""
        if self.stack.count() <= 1:
            return
        w = self.stack.widget(self.stack.count() - 1)
        self.stack.removeWidget(w)
        w.deleteLater()
        current = self.stack.widget(self.stack.count() - 1)
        if hasattr(current, "refresh"):
            current.refresh()
        self.stack.setCurrentWidget(current)

    # -- screens -------------------------------------------------------------

    def show_project_view(self, project_id):
        from ui.screens.project_view_screen import ProjectViewScreen
        view = ProjectViewScreen(
            project_id,
            on_back=self._pop_to_root,
            on_character_selected=lambda char, pid=project_id: self.show_character_view(pid, char["name"]),
            on_new_character=lambda pid=project_id: self.show_character_builder(pid),
        )
        self._push(view)

    def show_character_view(self, project_id, character_name):
        from ui.screens.character_view_screen import CharacterViewScreen
        view = CharacterViewScreen(project_id, character_name, on_back=self._pop)
        self._push(view)

    def show_character_builder(self, project_id):
        from ui.screens.character_builder_screen import CharacterBuilderScreen
        view = CharacterBuilderScreen(
            project_id,
            on_back=self._pop,
            on_created=self._pop,
        )
        self._push(view)


if __name__ == "__main__":
    app = App(sys.argv)
    sys.exit(app.exec())
