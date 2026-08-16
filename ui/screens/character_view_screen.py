"""
Character detail view with versions gallery and voice player.
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QGridLayout, QFrame, QMessageBox, QInputDialog,
)
from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtGui import QPixmap
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

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


class _VoiceThread(QThread):
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, project, name, attributes, instruct=None, force=False):
        super().__init__()
        self.project = project
        self.name = name
        self.attributes = attributes or {}
        self.instruct = instruct
        self.force = force

    def run(self):
        try:
            path = character_manager.generate_voice(
                self.project, self.name,
                attributes=self.attributes,
                instruct=self.instruct,
                force=self.force,
            )
            self.finished_ok.emit(path)
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
        self._pending_ghosts = 0

        self.project = project_manager.get_project(project_id) or {"name": "Unknown"}
        self.project_slug = self.project["name"].replace(" ", "_")

        # Voice playback
        self._voice_thread = None
        self._voice_path = None
        self.audio_output = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.mediaStatusChanged.connect(self._on_media_status)

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

        # -- Voice section (mini player) --------------------------------------
        voice_box = QFrame()
        voice_box.setStyleSheet(
            "QFrame { border: 1px solid #ddd; border-radius: 6px; "
            "background: #fafafa; } QLabel { border: none; background: transparent; }"
        )
        voice_layout = QVBoxLayout(voice_box)
        voice_layout.setContentsMargins(10, 10, 10, 10)

        voice_title = QLabel("\U0001F3A4 Character Voice")
        voice_title.setStyleSheet("font-weight: bold;")
        voice_layout.addWidget(voice_title)

        self.voice_line_label = QLabel("")
        self.voice_line_label.setWordWrap(True)
        self.voice_line_label.setStyleSheet("color: #666; font-size: 11px;")
        voice_layout.addWidget(self.voice_line_label)

        voice_controls = QHBoxLayout()
        self.btn_generate_voice = QPushButton("Generate Voice")
        self.btn_generate_voice.clicked.connect(self.generate_voice)
        voice_controls.addWidget(self.btn_generate_voice)

        self.btn_regenerate_voice = QPushButton("Regenerate with Prompt\u2026")
        self.btn_regenerate_voice.clicked.connect(self.regenerate_voice)
        voice_controls.addWidget(self.btn_regenerate_voice)

        self.btn_play_voice = QPushButton("\u25B6 Play")
        self.btn_play_voice.setEnabled(False)
        self.btn_play_voice.clicked.connect(self.play_voice)
        voice_controls.addWidget(self.btn_play_voice)

        self.btn_stop_voice = QPushButton("\u25A0 Stop")
        self.btn_stop_voice.setEnabled(False)
        self.btn_stop_voice.clicked.connect(self.stop_voice)
        voice_controls.addWidget(self.btn_stop_voice)
        voice_controls.addStretch()
        voice_layout.addLayout(voice_controls)

        self.voice_status_label = QLabel("")
        self.voice_status_label.setStyleSheet("color: #4CAF50; font-style: italic; font-size: 11px;")
        voice_layout.addWidget(self.voice_status_label)

        self.voice_prompt_label = QLabel("")
        self.voice_prompt_label.setWordWrap(True)
        self.voice_prompt_label.setStyleSheet("color: #888; font-size: 11px;")
        voice_layout.addWidget(self.voice_prompt_label)

        left.addWidget(voice_box)

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
        self._character_attributes = character.get("attributes") or {}

        # Voice state
        self._voice_path = character.get("voice_path")
        voice_prompt = character.get("voice_prompt") or ""
        if self._voice_path and Path(self._voice_path).is_file():
            self.btn_play_voice.setEnabled(True)
            self.voice_status_label.setText("Voice ready \u2014 click \u25B6 Play.")
        else:
            self.btn_play_voice.setEnabled(False)
            self.voice_status_label.setText("No voice yet. Click \u201cGenerate Voice\u201d.")
        if voice_prompt:
            self.voice_prompt_label.setText(f"Voice prompt: {voice_prompt}")
        else:
            self.voice_prompt_label.setText("")
        self._update_voice_line()

        # Versions gallery
        while self.gallery.count():
            item = self.gallery.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        versions = character_manager.list_versions(self.project_slug, self.character_name)
        if not versions and not self._pending_ghosts:
            empty = QLabel("No versions yet.")
            empty.setStyleSheet("color: #999; padding: 20px;")
            self.gallery.addWidget(empty, 0, 0)

        cells = []
        if self._pending_ghosts:
            from ui.components.ghost_card import GhostCard
            cells.extend(GhostCard(160, 180) for _ in range(self._pending_ghosts))
        for version in versions:
            thumb = VersionThumb(version)
            thumb.clicked.connect(self.set_default_version)
            thumb.export_requested.connect(self.export_version)
            cells.append(thumb)
        for idx, widget in enumerate(cells):
            self.gallery.addWidget(widget, idx // 3, idx % 3)

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

    # -- Voice -----------------------------------------------------------------

    def _update_voice_line(self):
        from core.voice_engine import build_voice_line
        line = build_voice_line(self.character_name, attributes=self._character_attributes)
        self.voice_line_label.setText(f"Line: \u201c{line}\u201d")

    def generate_voice(self):
        self._start_voice_thread(instruct=None, force=False)

    def regenerate_voice(self):
        """Ask for a natural-language voice prompt, then re-synthesize.

        The prompt shapes style/emotion/pace (the model's `instruct`); the
        timbre stays on the character's preset speaker — see
        docs/humans/voice-generation-implementation.md for the caveat.
        """
        if self._voice_thread and self._voice_thread.isRunning():
            return
        from PySide6.QtWidgets import QInputDialog

        text, ok = QInputDialog.getMultiLineText(
            self,
            "Regenerate Voice",
            "Describe the voice you want (timbre, style, emotion, pace).\n"
            "The local VoiceDesign model creates the voice from this\n"
            "description. Example:\n"
            "\"a calm, warm grandmotherly voice, speaking slowly\"",
            "",
        )
        if not ok or not text.strip():
            return
        self._start_voice_thread(instruct=text.strip(), force=True)

    def _start_voice_thread(self, instruct=None, force=False):
        if self._voice_thread and self._voice_thread.isRunning():
            return
        self.btn_generate_voice.setEnabled(False)
        self.btn_regenerate_voice.setEnabled(False)
        self.voice_status_label.setText(
            "Regenerating voice with prompt\u2026" if force
            else "Generating voice\u2026 this may take a while."
        )
        self._voice_thread = _VoiceThread(
            self.project_slug, self.character_name, self._character_attributes,
            instruct=instruct, force=force,
        )
        self._voice_thread.finished_ok.connect(self._on_voice_generated)
        self._voice_thread.failed.connect(self._on_voice_failed)
        self._voice_thread.start()

    def _on_voice_generated(self, path):
        self.btn_generate_voice.setEnabled(True)
        self.btn_regenerate_voice.setEnabled(True)
        self.voice_status_label.setText("Voice ready \u2014 click \u25B6 Play.")
        self._voice_path = path
        self.btn_play_voice.setEnabled(True)
        self.refresh()

    def _on_voice_failed(self, error):
        self.btn_generate_voice.setEnabled(True)
        self.btn_regenerate_voice.setEnabled(True)
        self.voice_status_label.setText("")
        QMessageBox.critical(self, "Voice generation failed", error)

    def play_voice(self):
        if not self._voice_path or not Path(self._voice_path).is_file():
            return
        self.player.setSource(QUrl.fromLocalFile(str(Path(self._voice_path).resolve())))
        self.player.play()
        self.btn_play_voice.setEnabled(False)
        self.btn_stop_voice.setEnabled(True)
        self.voice_status_label.setText("Playing\u2026")

    def stop_voice(self):
        self.player.stop()
        self.btn_play_voice.setEnabled(True)
        self.btn_stop_voice.setEnabled(False)
        self.voice_status_label.setText("Voice ready \u2014 click \u25B6 Play.")

    def _on_media_status(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.stop_voice()

    def closeEvent(self, event):
        self.player.stop()
        super().closeEvent(event)

    def generate_versions(self):
        num, ok = QInputDialog.getInt(self, "Generate Versions", "Number of new versions:", 3, 1, 10)
        if not ok:
            return
        self.btn_generate.setEnabled(False)
        self.status_label.setText("Generating\u2026 this may take a while.")
        self._pending_ghosts = num
        self.refresh()
        self._gen_thread = _GenerateThread(self.project_slug, self.character_name, self._prompt, num)
        self._gen_thread.finished_ok.connect(self._on_generated)
        self._gen_thread.failed.connect(self._on_generation_failed)
        self._gen_thread.start()

    def _on_generated(self):
        self.btn_generate.setEnabled(True)
        self.status_label.setText("")
        self._pending_ghosts = 0
        self.refresh()

    def _on_generation_failed(self, error):
        self.btn_generate.setEnabled(True)
        self.status_label.setText("")
        self._pending_ghosts = 0
        self.refresh()
        QMessageBox.critical(self, "Generation failed", error)
