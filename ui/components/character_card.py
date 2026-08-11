"""
Character card component with thumbnail.
"""

from pathlib import Path

from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap


class CharacterCard(QFrame):
    clicked = Signal(dict)          # character dict
    delete_requested = Signal(dict)

    def __init__(self, character):
        super().__init__()
        self.character = character
        self.setFixedSize(170, 220)
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("characterCard")
        self.setStyleSheet("""
            QFrame#characterCard {
                border: 1px solid #ddd;
                border-radius: 6px;
                background-color: white;
            }
            QFrame#characterCard:hover { border: 2px solid #4CAF50; }
            QLabel { border: none; background: transparent; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        thumb = QLabel()
        thumb.setFixedSize(150, 150)
        thumb.setAlignment(Qt.AlignCenter)
        image_path = character.get("reference_image")
        if image_path and Path(image_path).is_file():
            pix = QPixmap(image_path)
            if not pix.isNull():
                thumb.setPixmap(pix.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                thumb.setText("No image")
        else:
            thumb.setText("No image")
            thumb.setStyleSheet("background-color: #eee; color: #999; border-radius: 4px;")

        name_label = QLabel(character["name"])
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        name_label.setWordWrap(True)

        layout.addWidget(thumb, alignment=Qt.AlignCenter)
        layout.addWidget(name_label)
        layout.addStretch()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.character)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        delete_action = menu.addAction("Delete Character")
        if menu.exec(event.globalPos()) == delete_action:
            self.delete_requested.emit(self.character)
