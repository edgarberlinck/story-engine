"""
Character detail view with versions gallery.
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QGridLayout, QFrame, QMessageBox, QInputDialog,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap

from core.project_manager import project_manager
from core.character_manager import character_manager


class _GenerateThread(QThread):
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, project, name, prompt, num_versions):
        super().__init__()
        self.project = project
        self.name = name
        self.prompt = prompt
        self.num_versions = num_versions

    def run(self):
        try:
            character_manager.generate_versions(
                self.project, self.name, self.prompt,
                num_versions=self.num_versions,
            )
            self.finished_ok.emit()
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class VersionThumb(QFrame):
    clicked = Signal(dict)
    export_requested = Signal(dict)

    def __init__(self, version):
        super().__init__()
        self.version = version
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("versionThumb")
        is_default = bool(version.get("is_default"))
        border = "#4CAF50" if is_default else "#ddd"
        self.setStyleSheet(
            f"QFrame#versionThumb {{ border: 2px solid {border}; "
            f"border-radius: 6px; background: white; }} "
            f"QLabel {{ border: none; background: transparent; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        img = QLabel()
        img.setFixedSize(140, 140)
        img.setAlignment(Qt.AlignCenter)
        path = version.get("image_path")
        pix = QPixmap(path) if path and Path(path).is_file() else QPixmap()
        if not pix.isNull():
            img.setPixmap(pix.scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img.setText("No image")
            img.setStyleSheet("background: #eee; color: #999;")

        caption = QLabel(f"v{version['version']}" + ("  \u2605 default" if is_default else ""))
        caption.setAlignment(Qt.AlignCenter)
        caption.setStyleSheet("font-size: 11px;" + (" color: #4CAF50; font-weight: bold;" if is_default else ""))

        layout.addWidget(img)
        layout.addWidget(caption)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.version)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        export_action = menu.addAction("Export Image\u2026")
        if menu.exec(event.globalPos()) == export_action:
            self.export_requested.emit(self.version)


class CharacterViewScreen(QWidget):
    def __init__(self, project_id, character_name, on_back):
        super().__init__()
        self.project_id = project_id
        self.character_name = character_name
        self.on_back = on_back
        self._gen_thread = None

        self.project = project_manager.get_project(project_id) or {"name": "Unknown"}
        self.project_slug = self.project["name"].replace(" ", "_")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # Breadcrumb / back
        crumb_row = QHBoxLayout()
        back_btn = QPushButton("\u2190 Back")
        back_btn.setProperty("flat", True)
        back_btn.clicked.connect(self.on_back)
        breadcrumb = QLabel(f"Home  \u203a  {self.project['name']}  \u203a  {character_name}")
        breadcrumb.setStyleSheet("font-size: 12px; color: #666;")
        crumb_row.addWidget(back_btn)
        crumb_row.addWidget(breadcrumb)
        crumb_row.addStretch()
        layout.addLayout(crumb_row)

        title = QLabel(character_name)
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(title)

        # Main content
        content = QHBoxLayout()

        # Left: reference image and details
        left = QVBoxLayout()
        self.img_label = QLabel()
        self.img_label.setFixedSize(360, 360)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("background-color: #eee; border: 1px solid #ddd; border-radius: 6px;")
        left.addWidget(self.img_label)

        self.details_label = QLabel()
        self.details_label.setWordWrap(True)
        self.details_label.setStyleSheet("color: #444;")
        left.addWidget(self.details_label)

        self.btn_generate = QPushButton("Generate New Versions")
        self.btn_generate.clicked.connect(self.generate_versions)
        left.addWidget(self.btn_generate)

        self.btn_export = QPushButton("Export Image\u2026")
        self.btn_export.clicked.connect(self.export_reference)
        left.addWidget(self.btn_export)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #4CAF50; font-style: italic;")
        left.addWidget(self.status_label)
        left.addStretch()

        # Right: versions gallery
        right = QVBoxLayout()
        versions_label = QLabel("Versions  (click to set default \u00b7 right-click to export)")
        versions_label.setStyleSheet("font-weight: bold;")
        right.addWidget(versions_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        gallery_host = QWidget()
        self.gallery = QGridLayout(gallery_host)
        self.gallery.setSpacing(10)
        self.gallery.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll.setWidget(gallery_host)
        right.addWidget(scroll)

        content.addLayout(left, 1)
        content.addLayout(right, 2)
        layout.addLayout(content)

        self.refresh()

    def refresh(self):
        character = character_manager.get_character(self.project_slug, self.character_name) or {}

        # Reference image
        ref = character.get("reference_image")
        pix = QPixmap(ref) if ref and Path(ref).is_file() else QPixmap()
        if not pix.isNull():
            self.img_label.setPixmap(pix.scaled(360, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.img_label.setText("No reference image")

        prompt = character.get("prompt", "\u2014")
        self.details_label.setText(
            f"<b>Prompt:</b> {prompt}<br>"
            f"<b>Seed:</b> {character.get('seed', '\u2014')}<br>"
            f"<b>Model:</b> {character.get('model', '\u2014')}<br>"
            f"<b>Created:</b> {str(character.get('created_at', ''))[:19]}"
        )
        self._prompt = prompt

        # Versions gallery
        while self.gallery.count():
            item = self.gallery.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        versions = character_manager.list_versions(self.project_slug, self.character_name)
        if not versions:
            empty = QLabel("No versions yet.")
            empty.setStyleSheet("color: #999; padding: 20px;")
            self.gallery.addWidget(empty, 0, 0)
        for idx, version in enumerate(versions):
            thumb = VersionThumb(version)
            thumb.clicked.connect(self.set_default_version)
            thumb.export_requested.connect(self.export_version)
            self.gallery.addWidget(thumb, idx // 3, idx % 3)

    def set_default_version(self, version):
        character_manager.set_default_version(self.project_slug, self.character_name, version["version"])
        self.refresh()

    def export_reference(self):
        from ui.helpers import export_image
        from utils.project_paths import slugify
        character = character_manager.get_character(self.project_slug, self.character_name) or {}
        export_image(self, character.get("reference_image"), slugify(self.character_name))

    def export_version(self, version):
        from ui.helpers import export_image
        from utils.project_paths import slugify
        export_image(self, version.get("image_path"),
                     f"{slugify(self.character_name)}_v{version['version']}")

    def generate_versions(self):
        num, ok = QInputDialog.getInt(self, "Generate Versions", "Number of new versions:", 3, 1, 10)
        if not ok:
            return
        self.btn_generate.setEnabled(False)
        self.status_label.setText("Generating\u2026 this may take a while.")
        self._gen_thread = _GenerateThread(self.project_slug, self.character_name, self._prompt, num)
        self._gen_thread.finished_ok.connect(self._on_generated)
        self._gen_thread.failed.connect(self._on_generation_failed)
        self._gen_thread.start()

    def _on_generated(self):
        self.btn_generate.setEnabled(True)
        self.status_label.setText("")
        self.refresh()

    def _on_generation_failed(self, error):
        self.btn_generate.setEnabled(True)
        self.status_label.setText("")
        QMessageBox.critical(self, "Generation failed", error)
