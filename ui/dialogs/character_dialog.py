"""
Dialog for creating characters with multiple variants.
"""

from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout, QSpinBox


class CharacterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Character")
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        layout.addWidget(self.name_edit)

        layout.addWidget(QLabel("Prompt:"))
        self.prompt_edit = QLineEdit()
        layout.addWidget(self.prompt_edit)

        layout.addWidget(QLabel("Number of variants:"))
        self.variant_spin = QSpinBox()
        self.variant_spin.setRange(1, 10)
        self.variant_spin.setValue(3)
        layout.addWidget(self.variant_spin)

        buttons = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_ok = QPushButton("Generate")
        buttons.addWidget(btn_cancel)
        buttons.addWidget(btn_ok)
        layout.addLayout(buttons)

        btn_cancel.clicked.connect(self.reject)
        btn_ok.clicked.connect(self.accept)

    def get_data(self):
        return {
            "name": self.name_edit.text(),
            "prompt": self.prompt_edit.text(),
            "variants": self.variant_spin.value()
        }
