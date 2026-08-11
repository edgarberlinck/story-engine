"""
Scene card component with thumbnail.
"""

from pathlib import Path

from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap


class SceneCard(QFrame):
    clicked = Signal(dict)  # scene dict

    def __init__(self, scene):
        super().__init__()
        self.scene = scene
        self.setFixedSize(240, 220)
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("sceneCard")
        self.setStyleSheet("""
            QFrame#sceneCard {
                border: 1px solid #ddd;
                border-radius: 6px;
                background-color: white;
            }
            QFrame#sceneCard:hover { border: 2px solid #4CAF50; }
            QLabel { border: none; background: transparent; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        thumb = QLabel()
        thumb.setFixedSize(220, 130)
        thumb.setAlignment(Qt.AlignCenter)
        image_path = scene.get("image_path")
        pix = QPixmap(image_path) if image_path and Path(image_path).is_file() else QPixmap()
        if not pix.isNull():
            thumb.setPixmap(pix.scaled(220, 130, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            thumb.setText("No image")
            thumb.setStyleSheet("background-color: #eee; color: #999; border-radius: 4px;")

        title = QLabel(f"Scene {scene.get('scene_number', '?')}")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")

        prompt = scene.get("prompt") or ""
        if len(prompt) > 80:
            prompt = prompt[:80] + "…"
        prompt_label = QLabel(prompt)
        prompt_label.setWordWrap(True)
        prompt_label.setStyleSheet("font-size: 10px; color: #666;")

        layout.addWidget(thumb, alignment=Qt.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(prompt_label)
        layout.addStretch()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.scene)
        super().mousePressEvent(event)
