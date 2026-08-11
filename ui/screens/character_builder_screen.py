"""
Character creation builder with attributes.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QSpinBox, QLineEdit, QFormLayout, QTextEdit, QMessageBox, QScrollArea,
)
from PySide6.QtCore import Qt, QThread, Signal

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


class CharacterBuilderScreen(QWidget):
    def __init__(self, project_id, on_back, on_created):
        super().__init__()
        self.project_id = project_id
        self.on_back = on_back
        self.on_created = on_created
        self._gen_thread = None

        project = project_manager.get_project(project_id) or {"name": "Unknown"}
        self.project_slug = project["name"].replace(" ", "_")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)

        # Breadcrumb / back
        crumb_row = QHBoxLayout()
        back_btn = QPushButton("\u2190 Back")
        back_btn.setProperty("flat", True)
        back_btn.clicked.connect(self.on_back)
        breadcrumb = QLabel(f"Home  \u203a  {project['name']}  \u203a  New Character")
        breadcrumb.setStyleSheet("font-size: 12px; color: #666;")
        crumb_row.addWidget(back_btn)
        crumb_row.addWidget(breadcrumb)
        crumb_row.addStretch()
        outer.addLayout(crumb_row)

        title = QLabel("Create New Character")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        outer.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_host = QWidget()
        form = QFormLayout(form_host)
        form.setSpacing(10)
        scroll.setWidget(form_host)
        outer.addWidget(scroll)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Character name (required)")
        form.addRow("Name:", self.name_input)

        def combo(items):
            c = QComboBox()
            c.addItems(items)
            return c

        self.gender_combo = combo(["Female", "Male", "Non-binary", "Other"])
        form.addRow("Gender:", self.gender_combo)

        self.age_combo = combo(["Child", "Teen", "20-30", "30-40", "40-50", "50+"])
        form.addRow("Age Range:", self.age_combo)

        self.body_combo = combo(["Slender", "Athletic", "Curvy", "Muscular", "Plus Size"])
        form.addRow("Body Type:", self.body_combo)

        self.hair_type_combo = combo(["Straight", "Wavy", "Curly", "Coily", "Bald"])
        form.addRow("Hair Type:", self.hair_type_combo)

        self.hair_color_combo = combo(["Black", "Brown", "Blonde", "Red", "Gray", "White"])
        form.addRow("Hair Color:", self.hair_color_combo)

        self.hair_length_combo = combo(["Short", "Medium", "Long"])
        form.addRow("Hair Length:", self.hair_length_combo)

        self.skin_combo = combo(["Very Light", "Light", "Medium", "Tan", "Dark", "Very Dark"])
        form.addRow("Skin Tone:", self.skin_combo)

        self.eye_combo = combo(["Blue", "Green", "Brown", "Hazel", "Gray"])
        form.addRow("Eye Color:", self.eye_combo)

        self.clothing_combo = combo(["Casual", "Formal", "Fantasy", "Sci-fi", "Historical", "Sporty"])
        form.addRow("Clothing Style:", self.clothing_combo)

        self.mood_combo = combo(["Neutral", "Happy", "Serious", "Mysterious", "Confident", "Melancholic"])
        form.addRow("Mood/Expression:", self.mood_combo)

        self.variant_spin = QSpinBox()
        self.variant_spin.setRange(1, 10)
        self.variant_spin.setValue(3)
        form.addRow("Variants to generate:", self.variant_spin)

        self.prompt_preview = QTextEdit()
        self.prompt_preview.setPlaceholderText("Prompt preview \u2014 click \u201cPreview Prompt\u201d to generate, then edit freely.")
        self.prompt_preview.setMaximumHeight(120)
        form.addRow("Prompt:", self.prompt_preview)

        # Status
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #4CAF50; font-style: italic;")
        outer.addWidget(self.status_label)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setProperty("flat", True)
        btn_cancel.clicked.connect(self.on_back)
        btn_preview = QPushButton("Preview Prompt")
        btn_preview.clicked.connect(self.preview_prompt)
        self.btn_generate = QPushButton("Generate Character")
        self.btn_generate.clicked.connect(self.generate_character)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_preview)
        btn_layout.addWidget(self.btn_generate)
        outer.addLayout(btn_layout)

    # -- prompt building ------------------------------------------------------

    def get_attributes(self):
        return {
            "gender": self.gender_combo.currentText(),
            "age_range": self.age_combo.currentText(),
            "body_type": self.body_combo.currentText(),
            "hair_type": self.hair_type_combo.currentText(),
            "hair_color": self.hair_color_combo.currentText(),
            "hair_length": self.hair_length_combo.currentText(),
            "skin_tone": self.skin_combo.currentText(),
            "eye_color": self.eye_combo.currentText(),
            "clothing": self.clothing_combo.currentText(),
            "mood": self.mood_combo.currentText(),
        }

    def build_prompt(self):
        a = {k: v.lower() for k, v in self.get_attributes().items()}
        return (
            f"Photorealistic portrait of a {a['body_type']} {a['gender']} character, "
            f"age {a['age_range']}, with {a['hair_length']} {a['hair_type']} {a['hair_color']} hair, "
            f"{a['skin_tone']} skin tone and {a['eye_color']} eyes, wearing {a['clothing']} clothing, "
            f"{a['mood']} expression, high detail, studio lighting, neutral background"
        )

    def preview_prompt(self):
        self.prompt_preview.setPlainText(self.build_prompt())

    # -- generation -----------------------------------------------------------

    def generate_character(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing name", "Please enter a character name.")
            return

        if character_manager.get_character(self.project_slug, name):
            QMessageBox.warning(self, "Duplicate", f"Character \u201c{name}\u201d already exists in this project.")
            return

        prompt = self.prompt_preview.toPlainText().strip() or self.build_prompt()

        self.btn_generate.setEnabled(False)
        self.status_label.setText("Generating\u2026 this may take a while.")

        self._gen_thread = _GenerateThread(self.project_slug, name, prompt, self.variant_spin.value())
        self._gen_thread.finished_ok.connect(self._on_generated)
        self._gen_thread.failed.connect(self._on_generation_failed)
        self._gen_thread.start()

    def _on_generated(self):
        self.status_label.setText("")
        self.on_created()

    def _on_generation_failed(self, error):
        self.btn_generate.setEnabled(True)
        self.status_label.setText("")
        QMessageBox.critical(self, "Generation failed", error)
