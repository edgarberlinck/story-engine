"""
Location card component with thumbnail.
"""

from pathlib import Path

from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap


class LocationCard(QFrame):
    clicked = Signal(dict)          # location dict
    delete_requested = Signal(dict)

    def __init__(self, loc):
        super().__init__()
        self.location = loc
        self.setFixedSize(170, 200)
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("locationCard")
        self.setStyleSheet("""
            QFrame#locationCard {
                border: 1px solid #ddd;
                border-radius: 6px;
                background-color: white;
            }
            QFrame#locationCard:hover { border: 2px solid #4CAF50; }
            QLabel { border: none; background: transparent; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        thumb = QLabel()
        thumb.setFixedSize(150, 120)
        thumb.setAlignment(Qt.AlignCenter)
        image_path = self.location.get("visual_identity")
        if image_path and Path(image_path).is_file():
            pix = QPixmap(image_path)
            if not pix.isNull():
                thumb.setPixmap(pix.scaled(150, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                thumb.setText("No image")
        else:
            thumb.setText("No image")
            thumb.setStyleSheet("background-color: #eee; color: #999; border-radius: 4px;")

        name_label = QLabel(self.location["name"])
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        name_label.setWordWrap(True)

        type_label = QLabel(f"Type: {self.location.get('type', '')}")
        type_label.setAlignment(Qt.AlignCenter)
        type_label.setStyleSheet("color: #666; font-size: 10px;")
        type_label.setWordWrap(True)

        state_label = QLabel(f"State: {self.location.get('state', 'initial')}")
        state_label.setAlignment(Qt.AlignCenter)
        state_label.setStyleSheet("color: #666; font-size: 10px;")
        state_label.setWordWrap(True)

        layout.addWidget(thumb, alignment=Qt.AlignCenter)
        layout.addWidget(name_label)
        layout.addWidget(type_label)
        layout.addWidget(state_label)
        layout.addStretch()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.location)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        delete_action = menu.addAction("Delete Location")
        if menu.exec(event.globalPos()) == delete_action:
            self.delete_requested.emit(self.location)