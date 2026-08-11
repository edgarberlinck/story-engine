"""
Ghost placeholder card shown while images are being generated.
"""

from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QGraphicsOpacityEffect
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve


class GhostCard(QFrame):
    """A pulsing placeholder tile."""

    def __init__(self, width=150, height=180, label="Generating\u2026"):
        super().__init__()
        self.setFixedSize(width, height)
        self.setObjectName("ghostCard")
        self.setStyleSheet("""
            QFrame#ghostCard {
                border: 2px dashed #bbb;
                border-radius: 6px;
                background-color: #f2f2f2;
            }
            QLabel { border: none; background: transparent; color: #999; }
        """)

        layout = QVBoxLayout(self)
        text = QLabel(label)
        text.setAlignment(Qt.AlignCenter)
        text.setWordWrap(True)
        layout.addWidget(text)

        # Pulse animation
        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)
        self._anim = QPropertyAnimation(effect, b"opacity", self)
        self._anim.setDuration(1200)
        self._anim.setStartValue(1.0)
        self._anim.setKeyValueAt(0.5, 0.35)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.InOutSine)
        self._anim.setLoopCount(-1)
        self._anim.start()
