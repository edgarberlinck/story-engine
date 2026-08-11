"""
Dialog for creating scenes, with a character reference panel.
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton,
    QScrollArea, QWidget, QFrame, QApplication,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap

from core.character_manager import character_manager


class _CharacterChip(QFrame):
    """Small clickable character entry: thumbnail + name. Click copies the name."""

    def __init__(self, character, on_copied):
        super().__init__()
        self.character = character
        self.on_copied = on_copied
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("charChip")
        self.setToolTip(f"Click to copy \u201c{character['name']}\u201d to the clipboard")
        self.setStyleSheet("""
            QFrame#charChip {
                border: 1px solid #ddd; border-radius: 6px; background: white;
            }
            QFrame#charChip:hover { border: 2px solid #4CAF50; background: #f0f8f0; }
            QLabel { border: none; background: transparent; }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 8, 6)
        layout.setSpacing(8)

        thumb = QLabel()
        thumb.setFixedSize(48, 48)
        thumb.setAlignment(Qt.AlignCenter)
        path = character.get("reference_image")
        pix = QPixmap(path) if path and Path(path).is_file() else QPixmap()
        if not pix.isNull():
            thumb.setPixmap(pix.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            thumb.setText("?")
            thumb.setStyleSheet("background: #eee; color: #999; border-radius: 4px;")

        name = QLabel(character["name"])
        name.setStyleSheet("font-weight: 600; font-size: 12px;")
        name.setWordWrap(True)

        layout.addWidget(thumb)
        layout.addWidget(name, 1)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            QApplication.clipboard().setText(self.character["name"])
            self.on_copied(self.character["name"])
        super().mousePressEvent(event)


class SceneDialog(QDialog):
    def __init__(self, parent=None, project=None):
        super().__init__(parent)
        self.setWindowTitle("New Scene")
        self.setMinimumSize(700, 420)

        outer = QVBoxLayout(self)
        content = QHBoxLayout()
        outer.addLayout(content)

        # Left: prompt
        left = QVBoxLayout()
        left.addWidget(QLabel("Prompt:"))
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText(
            "Describe the scene\u2026 Mention character names to include their looks automatically."
        )
        left.addWidget(self.prompt_edit)
        content.addLayout(left, 2)

        # Right: character reference panel
        characters = character_manager.list_characters(project) if project else []
        if characters:
            right = QVBoxLayout()
            header = QLabel("Characters")
            header.setStyleSheet("font-weight: bold;")
            right.addWidget(header)

            hint = QLabel("Click a character to copy its name")
            hint.setStyleSheet("color: #999; font-size: 10px;")
            right.addWidget(hint)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            host = QWidget()
            chip_layout = QVBoxLayout(host)
            chip_layout.setSpacing(6)
            chip_layout.setAlignment(Qt.AlignTop)
            for char in characters:
                chip_layout.addWidget(_CharacterChip(char, self._on_name_copied))
            scroll.setWidget(host)
            right.addWidget(scroll)
            content.addLayout(right, 1)

        # Footer: status + buttons
        footer = QHBoxLayout()
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #4CAF50; font-size: 11px; font-style: italic;")
        footer.addWidget(self.status_label)
        footer.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setProperty("flat", True)
        btn_ok = QPushButton("Generate")
        btn_ok.setDefault(True)
        footer.addWidget(btn_cancel)
        footer.addWidget(btn_ok)
        outer.addLayout(footer)

        btn_cancel.clicked.connect(self.reject)
        btn_ok.clicked.connect(self.accept)

    def _on_name_copied(self, name):
        self.status_label.setText(f"Copied \u201c{name}\u201d to clipboard")
        QTimer.singleShot(2000, lambda: self.status_label.setText(""))

    def get_data(self):
        return {
            "prompt": self.prompt_edit.toPlainText()
        }
