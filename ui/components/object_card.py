"""
Object/Artifact card component with thumbnail.
"""

from pathlib import Path

from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap


class ObjectCard(QFrame):
    clicked = Signal(dict)  # object dict
    delete_requested = Signal(dict)

    def __init__(self, obj):
        super().__init__()
        self.object = obj
        self.setFixedSize(170, 200)
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("objectCard")
        self.setStyleSheet("""
            QFrame#objectCard {
                border: 1px solid #ddd;
                border-radius: 6px;
                background-color: white;
            }
            QFrame#objectCard:hover { border: 2px solid #4CAF50; }
            QLabel { border: none; background: transparent; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        thumb = QLabel()
        thumb.setFixedSize(150, 120)
        thumb.setAlignment(Qt.AlignCenter)
        image_path = self.object.get("visual_identity")
        if image_path and Path(image_path).is_file():
            pix = QPixmap(image_path)
            if not pix.isNull():
                thumb.setPixmap(
                    pix.scaled(150, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
            else:
                thumb.setText("No image")
        else:
            thumb.setText("No image")
            thumb.setStyleSheet(
                "background-color: #eee; color: #999; border-radius: 4px;"
            )

        name_label = QLabel(self.object["name"])
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        name_label.setWordWrap(True)

        type_label = QLabel(f"Type: {self.object.get('type', '')}")
        type_label.setAlignment(Qt.AlignCenter)
        type_label.setStyleSheet("color: #666; font-size: 10px;")
        type_label.setWordWrap(True)

        layout.addWidget(thumb, alignment=Qt.AlignCenter)
        layout.addWidget(name_label)
        layout.addWidget(type_label)
        layout.addStretch()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.object)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        delete_action = menu.addAction("Delete Object")
        if menu.exec(event.globalPos()) == delete_action:
            self.delete_requested.emit(self.object)
