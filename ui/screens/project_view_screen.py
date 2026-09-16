"""
Project view screen with breadcrumb and tabs for Characters, Objects, Locations and Scenes.
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
from core.object_manager import object_manager
from core.location_manager import location_manager
from ui.components.character_card import CharacterCard
from ui.components.scene_card import SceneCard
from ui.components.object_card import ObjectCard
from ui.components.location_card import LocationCard


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
        btn_narrator = QPushButton("Narrator\u2026")
        btn_narrator.clicked.connect(self.configure_narrator)
        header.addWidget(btn_narrator)
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
        self.tabs.addTab(self._build_writing_tab(), "Writing")
        self.tabs.addTab(self._build_characters_tab(), "Characters")
        self.tabs.addTab(self._build_objects_tab(), "Objects")
        self.tabs.addTab(self._build_locations_tab(), "Locations")
        self.tabs.addTab(self._build_scenes_tab(), "Scenes")
        layout.addWidget(self.tabs)

        self.load_characters()
        self.load_objects()
        self.load_locations()
        self.load_scenes()
        self.load_audio_scenes()

    @property
    def project_slug(self):
        return self.project["name"].replace(" ", "_")

    def refresh(self):
        self.load_characters()
        self.load_objects()
        self.load_locations()
        self.load_scenes()
        self.load_audio_scenes()

    def configure_narrator(self):
        from ui.dialogs.narrator_config_dialog import NarratorConfigDialog
        chars = character_manager.list_characters(self.project_slug)
        dialog = NarratorConfigDialog(self, project=self.project_slug, characters=chars)
        dialog.exec()

    # -- Writing tab -----------------------------------------------------------

    def _build_writing_tab(self):
        from ui.components.writing_tab import WritingTab
        self.writing_tab = WritingTab(
            self.project_slug, on_compiled=self._on_chapter_compiled
        )
        return self.writing_tab

    def _on_chapter_compiled(self, scene_numbers):
        """After compiling a chapter, refresh and reveal the audio scenes."""
        self.load_audio_scenes()
        self.tabs.setCurrentIndex(self.tabs.count() - 1)  # jump to Scenes tab
        # Highlight the first compiled scene.
        if scene_numbers:
            for row in range(self.audio_scene_list.count()):
                entry = self.audio_scene_list.item(row).data(Qt.UserRole)
                if entry["scene_number"] == scene_numbers[0]:
                    self.audio_scene_list.setCurrentRow(row)
                    break

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

        self.char_empty_label = QLabel("No characters yet. Click \"+ New Character\" to create one.")
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
            f"Delete character \"{character['name']}\"?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            character_manager.delete_character(self.project_slug, character["name"])
            self.load_characters()

    # -- Objects tab --------------------------------------------------------

    def _build_objects_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)

        toolbar = QHBoxLayout()
        toolbar.addStretch()
        btn_new_obj = QPushButton("+ New Object")
        btn_new_obj.clicked.connect(self.create_object)
        toolbar.addWidget(btn_new_obj)
        layout.addLayout(toolbar)

        self.obj_empty_label = QLabel("No objects yet. Click \"+ New Object\" to create one.")
        self.obj_empty_label.setStyleSheet("color: #999; padding: 20px;")
        self.obj_empty_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.obj_empty_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        grid_host = QWidget()
        self.obj_grid = QGridLayout(grid_host)
        self.obj_grid.setSpacing(15)
        self.obj_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll.setWidget(grid_host)
        layout.addWidget(scroll)
        return tab

    def load_objects(self):
        while self.obj_grid.count():
            item = self.obj_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        objs = object_manager.list_objects(self.project_slug)
        self.obj_empty_label.setVisible(not objs)

        for idx, obj in enumerate(objs):
            card = ObjectCard(obj)
            card.clicked.connect(lambda checked, o=obj: self.on_object_selected(o))
            card.delete_requested.connect(self.delete_object)
            self.obj_grid.addWidget(card, idx // 5, idx % 5)

    def on_object_selected(self, obj):
        QMessageBox.information(self, "Object Selected", f"Selected: {obj['name']}")

    def create_object(self):
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "New Object", "Object name:")
        if not ok or not name.strip():
            return
        obj_type, ok = QInputDialog.getText(self, "New Object", "Object type (e.g. weapon, artifact):", text="artifact")
        if not ok:
            return
        description, ok = QInputDialog.getMultiLineText(self, "New Object", "Description:")
        if not ok:
            return
        object_manager.create_object(
            self.project_slug, name.strip(),
            obj_type.strip() or "artifact", description.strip(),
        )
        self.load_objects()

    def delete_object(self, obj):
        reply = QMessageBox.question(
            self,
            "Delete Object",
            f"Delete object \"{obj['name']}\"?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            object_manager.delete_object(self.project_slug, obj["name"])
            self.load_objects()

    # -- Locations tab ------------------------------------------------------

    def _build_locations_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)

        toolbar = QHBoxLayout()
        toolbar.addStretch()
        btn_new_loc = QPushButton("+ New Location")
        btn_new_loc.clicked.connect(self.create_location)
        toolbar.addWidget(btn_new_loc)
        layout.addLayout(toolbar)

        self.loc_empty_label = QLabel("No locations yet. Click \"+ New Location\" to create one.")
        self.loc_empty_label.setStyleSheet("color: #999; padding: 20px;")
        self.loc_empty_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.loc_empty_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        grid_host = QWidget()
        self.loc_grid = QGridLayout(grid_host)
        self.loc_grid.setSpacing(15)
        self.loc_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll.setWidget(grid_host)
        layout.addWidget(scroll)
        return tab

    def load_locations(self):
        while self.loc_grid.count():
            item = self.loc_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        locs = location_manager.list_locations(self.project_slug)
        self.loc_empty_label.setVisible(not locs)

        for idx, loc in enumerate(locs):
            card = LocationCard(loc)
            card.clicked.connect(lambda checked, l=loc: self.on_location_selected(l))
            card.delete_requested.connect(self.delete_location)
            self.loc_grid.addWidget(card, idx // 5, idx % 5)

    def on_location_selected(self, loc):
        QMessageBox.information(self, "Location Selected", f"Selected: {loc['name']}")

    def create_location(self):
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "New Location", "Location name:")
        if not ok or not name.strip():
            return
        loc_type, ok = QInputDialog.getText(self, "New Location", "Location type (e.g. city, room):", text="place")
        if not ok:
            return
        description, ok = QInputDialog.getMultiLineText(self, "New Location", "Description:")
        if not ok:
            return
        existing = [l["name"] for l in location_manager.list_locations(self.project_slug)]
        parent = None
        if existing:
            from PySide6.QtWidgets import QInputDialog as _QID
            parent_choice, ok = _QID.getItem(
                self, "New Location", "Parent location (optional):",
                ["<none>"] + existing, 0, False,
            )
            if ok and parent_choice != "<none>":
                parent = parent_choice
        location_manager.create_location(
            self.project_slug, name.strip(),
            loc_type.strip() or "place", description.strip(),
            parent_location=parent,
        )
        self.load_locations()

    def delete_location(self, loc):
        reply = QMessageBox.question(
            self,
            "Delete Location",
            f"Delete location \"{loc['name']}\"?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            location_manager.delete_location(self.project_slug, loc["name"])
            self.load_locations()

    # -- Scenes tab ---------------------------------------------------------

    def _build_scenes_tab(self):
        from PySide6.QtWidgets import QListWidget, QSplitter, QGroupBox
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)

        splitter = QSplitter(Qt.Vertical)

        # -- Audio scenes (compiled from the Writing tab) ----------------------
        audio_group = QGroupBox("Audio Scenes (compiled from Writing)")
        audio_layout = QVBoxLayout(audio_group)
        audio_toolbar = QHBoxLayout()
        self.audio_scene_hint = QLabel(
            "Double-click a scene to edit segments, voices and preview audio."
        )
        self.audio_scene_hint.setStyleSheet("color: #666;")
        audio_toolbar.addWidget(self.audio_scene_hint, 1)
        btn_edit_audio = QPushButton("Edit Selected\u2026")
        btn_edit_audio.clicked.connect(self._edit_selected_audio_scene)
        audio_toolbar.addWidget(btn_edit_audio)
        audio_layout.addLayout(audio_toolbar)

        self.audio_scene_list = QListWidget()
        self.audio_scene_list.itemDoubleClicked.connect(
            lambda item: self._edit_audio_scene_row(item)
        )
        audio_layout.addWidget(self.audio_scene_list)
        splitter.addWidget(audio_group)

        # -- Image scenes ------------------------------------------------------
        image_group = QGroupBox("Image Scenes")
        image_layout = QVBoxLayout(image_group)
        toolbar = QHBoxLayout()
        self.scene_status_label = QLabel("")
        self.scene_status_label.setStyleSheet("color: #4CAF50; font-style: italic;")
        toolbar.addWidget(self.scene_status_label)
        toolbar.addStretch()
        self.btn_new_scene = QPushButton("+ New Scene")
        self.btn_new_scene.clicked.connect(self.create_scene)
        toolbar.addWidget(self.btn_new_scene)
        image_layout.addLayout(toolbar)

        self.scene_empty_label = QLabel("No image scenes yet. Click \"+ New Scene\" to generate one.")
        self.scene_empty_label.setStyleSheet("color: #999; padding: 20px;")
        self.scene_empty_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.scene_empty_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        grid_host = QWidget()
        self.scene_grid = QGridLayout(grid_host)
        self.scene_grid.setSpacing(15)
        self.scene_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll.setWidget(grid_host)
        image_layout.addWidget(scroll)
        splitter.addWidget(image_group)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        return tab

    def load_audio_scenes(self):
        """List the compiled audio scene representations."""
        import json
        from PySide6.QtWidgets import QListWidgetItem
        from services.audio_scene_service import audio_scene_service

        self.audio_scene_list.clear()
        entries = audio_scene_service.list_representations(self.project_slug)
        entries.sort(key=lambda e: e["scene_number"])
        for e in entries:
            try:
                data = json.loads(e["representation_json"])
                title = data.get("title") or f"Scene {e['scene_number']}"
                n_segments = len(data.get("segments", []))
                speakers = ", ".join(data.get("characters_present", [])[:4])
            except (ValueError, KeyError):
                title, n_segments, speakers = f"Scene {e['scene_number']}", 0, ""
            label = f"Scene {e['scene_number']:>3} \u2014 {title}  ({n_segments} segments)"
            if speakers:
                label += f"  [{speakers}]"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, e)
            self.audio_scene_list.addItem(item)
        self.audio_scene_hint.setText(
            f"{len(entries)} audio scene(s). Double-click to edit segments, "
            "voices and preview audio."
            if entries else
            "No audio scenes yet. Write a chapter in the Writing tab and "
            "click \u266a Compile to Audio Scenes."
        )

    def _edit_selected_audio_scene(self):
        item = self.audio_scene_list.currentItem()
        if item:
            self._edit_audio_scene_row(item)

    def _edit_audio_scene_row(self, item):
        entry = item.data(Qt.UserRole)
        from ui.dialogs.scene_editor_dialog import SceneEditorDialog
        dialog = SceneEditorDialog(
            self,
            project=self.project_slug,
            scene_id=str(entry.get("scene_id", "")),
            scene_number=entry["scene_number"],
        )
        dialog.exec()
        self.load_audio_scenes()

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
        btn_audio = QPushButton("Edit Audio Scene\u2026")
        btn_audio.clicked.connect(
            lambda: self._edit_audio_scene(scene)
        )
        btn_export = QPushButton("Export Image\u2026")
        btn_export.clicked.connect(
            lambda: self._export_scene(dialog, scene)
        )
        btn_close = QPushButton("Close")
        btn_close.setProperty("flat", True)
        btn_close.clicked.connect(dialog.accept)
        btn_row.addStretch()
        btn_row.addWidget(btn_audio)
        btn_row.addWidget(btn_export)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        dialog.exec()

    def _edit_audio_scene(self, scene):
        """Open the audio-scene representation editor for this scene."""
        from ui.dialogs.scene_editor_dialog import SceneEditorDialog
        dialog = SceneEditorDialog(
            self,
            project=self.project_slug,
            scene_id=str(scene.get("id", "")),
            scene_number=scene.get("scene_number"),
        )
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