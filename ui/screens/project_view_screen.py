"""
Project view screen with breadcrumb and tabs for Characters and Scenes.
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGridLayout, QScrollArea, QMessageBox, QTabWidget, QDialog,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap

from core.project_manager import project_manager
from core.character_manager import character_manager
from core.scene_manager import scene_manager
from ui.components.character_card import CharacterCard
from ui.components.scene_card import SceneCard


class _SceneGenerateThread(QThread):
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, project, prompt):
        super().__init__()
        self.project = project
        self.prompt = prompt

    def run(self):
        try:
            scene_manager.create_scene(self.project, self.prompt)
            self.finished_ok.emit()
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class ProjectViewScreen(QWidget):
    def __init__(self, project_id, on_back, on_character_selected, on_new_character):
        super().__init__()
        self.project_id = project_id
        self.on_back = on_back
        self.on_character_selected = on_character_selected
        self.on_new_character = on_new_character
        self._scene_thread = None
        self._pending_scene_ghost = False

        self.project = project_manager.get_project(project_id) or {"name": "Unknown", "description": ""}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # Breadcrumb / back
        crumb_row = QHBoxLayout()
        back_btn = QPushButton("\u2190 Home")
        back_btn.setProperty("flat", True)
        back_btn.clicked.connect(self.on_back)
        breadcrumb = QLabel(f"Home  \u203a  {self.project['name']}")
        breadcrumb.setStyleSheet("font-size: 12px; color: #666;")
        crumb_row.addWidget(back_btn)
        crumb_row.addWidget(breadcrumb)
        crumb_row.addStretch()
        layout.addLayout(crumb_row)

        # Header
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel(self.project["name"])
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        title_box.addWidget(title)
        desc = self.project.get("description") or ""
        if desc:
            desc_label = QLabel(desc)
            desc_label.setStyleSheet("color: #666;")
            desc_label.setWordWrap(True)
            title_box.addWidget(desc_label)
        header.addLayout(title_box)
        header.addStretch()
        layout.addLayout(header)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                padding: 8px 20px; font-weight: 600;
                border: 1px solid #ddd; border-bottom: none;
                border-top-left-radius: 6px; border-top-right-radius: 6px;
                background: #f0f0f0;
            }
            QTabBar::tab:selected { background: white; color: #4CAF50; }
            QTabWidget::pane { border: 1px solid #ddd; border-radius: 0 6px 6px 6px; background: white; }
        """)
        self.tabs.addTab(self._build_characters_tab(), "Characters")
        self.tabs.addTab(self._build_scenes_tab(), "Scenes")
        layout.addWidget(self.tabs)

        self.load_characters()
        self.load_scenes()

    @property
    def project_slug(self):
        return self.project["name"].replace(" ", "_")

    def refresh(self):
        self.load_characters()
        self.load_scenes()

    # -- Characters tab -------------------------------------------------------

    def _build_characters_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)

        toolbar = QHBoxLayout()
        toolbar.addStretch()
        btn_new_char = QPushButton("+ New Character")
        btn_new_char.clicked.connect(lambda: self.on_new_character())
        toolbar.addWidget(btn_new_char)
        layout.addLayout(toolbar)

        self.char_empty_label = QLabel("No characters yet. Click \u201c+ New Character\u201d to create one.")
        self.char_empty_label.setStyleSheet("color: #999; padding: 20px;")
        self.char_empty_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.char_empty_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        grid_host = QWidget()
        self.char_grid = QGridLayout(grid_host)
        self.char_grid.setSpacing(15)
        self.char_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll.setWidget(grid_host)
        layout.addWidget(scroll)
        return tab

    def load_characters(self):
        while self.char_grid.count():
            item = self.char_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        chars = character_manager.list_characters(self.project_slug)
        self.char_empty_label.setVisible(not chars)

        for idx, char in enumerate(chars):
            card = CharacterCard(char)
            card.clicked.connect(self.on_character_selected)
            card.delete_requested.connect(self.delete_character)
            self.char_grid.addWidget(card, idx // 5, idx % 5)

    def delete_character(self, character):
        reply = QMessageBox.question(
            self,
            "Delete Character",
            f"Delete character \u201c{character['name']}\u201d?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            character_manager.delete_character(self.project_slug, character["name"])
            self.load_characters()

    # -- Scenes tab -------------------------------------------------------------

    def _build_scenes_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)

        toolbar = QHBoxLayout()
        self.scene_status_label = QLabel("")
        self.scene_status_label.setStyleSheet("color: #4CAF50; font-style: italic;")
        toolbar.addWidget(self.scene_status_label)
        toolbar.addStretch()
        self.btn_new_scene = QPushButton("+ New Scene")
        self.btn_new_scene.clicked.connect(self.create_scene)
        toolbar.addWidget(self.btn_new_scene)
        layout.addLayout(toolbar)

        self.scene_empty_label = QLabel("No scenes yet. Click \u201c+ New Scene\u201d to generate one.")
        self.scene_empty_label.setStyleSheet("color: #999; padding: 20px;")
        self.scene_empty_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.scene_empty_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        grid_host = QWidget()
        self.scene_grid = QGridLayout(grid_host)
        self.scene_grid.setSpacing(15)
        self.scene_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll.setWidget(grid_host)
        layout.addWidget(scroll)
        return tab

    def load_scenes(self):
        while self.scene_grid.count():
            item = self.scene_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        scenes = scene_manager.list_scenes(self.project_slug)
        self.scene_empty_label.setVisible(not scenes and not self._pending_scene_ghost)

        cells = []
        if self._pending_scene_ghost:
            from ui.components.ghost_card import GhostCard
            cells.append(GhostCard(240, 220, "Generating scene\u2026"))
        for scene in scenes:
            card = SceneCard(scene)
            card.clicked.connect(self.show_scene)
            cells.append(card)
        for idx, widget in enumerate(cells):
            self.scene_grid.addWidget(widget, idx // 4, idx % 4)

    def show_scene(self, scene):
        """Show the full scene image and prompt in a simple viewer dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Scene {scene.get('scene_number', '?')}")
        layout = QVBoxLayout(dialog)

        img = QLabel()
        img.setAlignment(Qt.AlignCenter)
        path = scene.get("image_path")
        pix = QPixmap(path) if path and Path(path).is_file() else QPixmap()
        if not pix.isNull():
            img.setPixmap(pix.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img.setText("No image")
            img.setStyleSheet("background: #eee; color: #999; padding: 60px;")
        layout.addWidget(img)

        prompt_label = QLabel(f"<b>Prompt:</b> {scene.get('prompt', '')}")
        prompt_label.setWordWrap(True)
        layout.addWidget(prompt_label)

        meta = QLabel(f"Seed: {scene.get('seed', '\u2014')}   Model: {scene.get('model', '\u2014')}   "
                      f"Created: {str(scene.get('created_at', ''))[:19]}")
        meta.setStyleSheet("color: #999; font-size: 11px;")
        layout.addWidget(meta)

        btn_row = QHBoxLayout()
        btn_export = QPushButton("Export Image\u2026")
        btn_export.clicked.connect(
            lambda: self._export_scene(dialog, scene)
        )
        btn_close = QPushButton("Close")
        btn_close.setProperty("flat", True)
        btn_close.clicked.connect(dialog.accept)
        btn_row.addStretch()
        btn_row.addWidget(btn_export)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        dialog.exec()

    def _export_scene(self, parent, scene):
        from ui.helpers import export_image
        export_image(parent, scene.get("image_path"),
                     f"{self.project_slug}_scene_{scene.get('scene_number', 'x')}")

    def create_scene(self):
        from ui.dialogs.scene_dialog import SceneDialog
        dialog = SceneDialog(self, project=self.project_slug)
        if not dialog.exec():
            return
        prompt = dialog.get_data()["prompt"].strip()
        if not prompt:
            QMessageBox.warning(self, "Missing prompt", "Please enter a scene prompt.")
            return

        self.btn_new_scene.setEnabled(False)
        self.scene_status_label.setText("Generating scene\u2026 this may take a while.")
        self._pending_scene_ghost = True
        self.load_scenes()

        self._scene_thread = _SceneGenerateThread(self.project_slug, prompt)
        self._scene_thread.finished_ok.connect(self._on_scene_generated)
        self._scene_thread.failed.connect(self._on_scene_failed)
        self._scene_thread.start()

    def _on_scene_generated(self):
        self.btn_new_scene.setEnabled(True)
        self.scene_status_label.setText("")
        self._pending_scene_ghost = False
        self.load_scenes()

    def _on_scene_failed(self, error):
        self.btn_new_scene.setEnabled(True)
        self.scene_status_label.setText("")
        self._pending_scene_ghost = False
        self.load_scenes()
        QMessageBox.critical(self, "Scene generation failed", error)
