"""
Project-level narrator configuration dialog (plan §2 "Narrator configuration").

The narrator can be:
- A dedicated (designed) narrator voice described by a prompt, or
- One of the existing characters.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QRadioButton, QButtonGroup, QComboBox, QTextEdit, QGroupBox,
    QMessageBox,
)

from services.database.project_settings_service import (
    project_settings_service,
    NARRATOR_MODE_DEDICATED,
    NARRATOR_MODE_CHARACTER,
)


class NarratorConfigDialog(QDialog):
    def __init__(self, parent=None, project=None, characters=None):
        super().__init__(parent)
        self.project = project
        self.characters = characters or []
        self.setWindowTitle("Narrator Configuration")
        self.setMinimumWidth(480)

        current = project_settings_service.get_narrator(project)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Choose who narrates this project. Narration segments use this "
            "voice unless a segment overrides it."
        ))

        # Mode selection
        self.mode_group = QButtonGroup(self)
        self.radio_dedicated = QRadioButton("Dedicated narrator voice (designed from a prompt)")
        self.radio_character = QRadioButton("An existing character narrates")
        self.mode_group.addButton(self.radio_dedicated)
        self.mode_group.addButton(self.radio_character)
        layout.addWidget(self.radio_dedicated)

        # Dedicated voice prompt
        self.prompt_box = QGroupBox("Narrator voice description")
        prompt_layout = QVBoxLayout(self.prompt_box)
        self.voice_prompt = QTextEdit()
        self.voice_prompt.setPlaceholderText(
            "e.g. A deep, calm male voice with a slight rasp, "
            "unhurried and cinematic."
        )
        self.voice_prompt.setPlainText(current.get("voice_prompt") or "")
        self.voice_prompt.setMaximumHeight(80)
        prompt_layout.addWidget(self.voice_prompt)
        layout.addWidget(self.prompt_box)

        layout.addWidget(self.radio_character)

        # Character picker
        self.char_combo = QComboBox()
        self.char_combo.addItems([c["name"] for c in self.characters])
        layout.addWidget(self.char_combo)

        # Initialize from current config
        if current.get("mode") == NARRATOR_MODE_CHARACTER and self.characters:
            self.radio_character.setChecked(True)
            idx = self.char_combo.findText(current.get("character") or "")
            if idx >= 0:
                self.char_combo.setCurrentIndex(idx)
        else:
            self.radio_dedicated.setChecked(True)

        if not self.characters:
            self.radio_character.setEnabled(False)
            self.char_combo.setEnabled(False)

        self.radio_dedicated.toggled.connect(self._sync_enabled)
        self._sync_enabled()

        # Buttons
        btn_row = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_save = QPushButton("Save")
        btn_save.clicked.connect(self._save)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_save)
        layout.addLayout(btn_row)

    def _sync_enabled(self):
        dedicated = self.radio_dedicated.isChecked()
        self.prompt_box.setEnabled(dedicated)
        self.char_combo.setEnabled(not dedicated and bool(self.characters))

    def _save(self):
        try:
            if self.radio_dedicated.isChecked():
                project_settings_service.set_narrator(
                    self.project,
                    NARRATOR_MODE_DEDICATED,
                    voice_prompt=self.voice_prompt.toPlainText().strip(),
                )
            else:
                project_settings_service.set_narrator(
                    self.project,
                    NARRATOR_MODE_CHARACTER,
                    voice_prompt=self.voice_prompt.toPlainText().strip(),
                    character=self.char_combo.currentText(),
                )
        except ValueError as e:
            QMessageBox.warning(self, "Narrator", str(e))
            return
        self.accept()
