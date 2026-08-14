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
from core.character_attributes import CHARACTER_TYPES, build_character_prompt
from models import MODEL_METADATA


def get_diffusion_models():
    """Dynamically load diffusion models from MODEL_METADATA"""
    diffusion_models = []
    for model_id, metadata in MODEL_METADATA.items():
        if metadata.get("type") == "diffusion":
            # Prefer the display name if available, otherwise use the model ID
            display_name = metadata.get("name", model_id)
            diffusion_models.append((display_name, model_id))
    
    # Sort by display name for consistent ordering
    diffusion_models.sort(key=lambda x: x[0])
    return diffusion_models


class _GenerateThread(QThread):
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, project, name, prompt, num_versions, model_id):
        super().__init__()
        self.project = project
        self.name = name
        self.prompt = prompt
        self.num_versions = num_versions
        self.model_id = model_id

    def run(self):
        try:
            character_manager.generate_versions(
                self.project, self.name, self.prompt,
                num_versions=self.num_versions,
                model=self.model_id
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
        
        # Add Style attribute
        self.style_combo = combo([
            "Ultra Realistic", "Cinematic", "Photorealistic", "Realistic",
            "Anime", "Manga", "Comic Book", "Cartoon",
            "Animation", "3D Animation", "3D Render", "Pixar-like",
            "Disney-like", "Stylized", "Semi-Realistic", "Fantasy Art",
            "Dark Fantasy", "Cyberpunk", "Sci-Fi", "Steampunk",
            "Medieval Art", "Concept Art", "Digital Painting", "Oil Painting",
            "Watercolor", "Pencil Drawing", "Sketch", "Ink Drawing",
            "Pixel Art", "Low Poly", "Game Asset", "Clay Render",
            "Minimalist", "Abstract"
        ])
        form.addRow("Visual Style:", self.style_combo)

        # Add Model selector - dynamically loaded from models.py
        diffusion_models = get_diffusion_models()
        model_display_names = [model[0] for model in diffusion_models]
        self.model_combo = combo(model_display_names)
        form.addRow("Generation Model:", self.model_combo)
        
        # Store mapping from display names to actual model IDs
        self.model_id_mapping = {display_name: model_id for display_name, model_id in diffusion_models}

        self.variant_spin = QSpinBox()
        self.variant_spin.setRange(1, 10)
        self.variant_spin.setValue(3)
        form.addRow("Variants to generate:", self.variant_spin)

        self.prompt_preview = QTextEdit()
        self.prompt_preview.setPlaceholderText("Prompt preview \u2014 click \u201cPreview Prompt\u201d to generate, then edit freely.")
        self.prompt_preview.setMaximumHeight(120)
        form.addRow("Prompt:", self.prompt_preview)

        # Status + ghost previews
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #4CAF50; font-style: italic;")
        outer.addWidget(self.status_label)

        self.ghost_row = QHBoxLayout()
        self.ghost_row.setSpacing(10)
        outer.addLayout(self.ghost_row)

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
            "style": self.style_combo.currentText()
        }

    def build_prompt(self):
        a = {k: v.lower() for k, v in self.get_attributes().items()}
        
        # Build a character using the character attributes system
        # but maintain compatibility with the existing simple format
        if a["gender"] == "male":
            char_type = "man"
        elif a["gender"] == "female":
            char_type = "woman"
        else:
            char_type = "animal"  # Fallback for non-binary or other
        
        # For now we'll use the simpler approach, but the system supports style
        base_prompt = (
            f"Full body photo of a {a['body_type']} {a['gender']} character, "
            f"age {a['age_range']}, with {a['hair_length']} {a['hair_type']} {a['hair_color']} hair, "
            f"{a['skin_tone']} skin tone and {a['eye_color']} eyes, wearing {a['clothing']} clothing, "
            f"{a['mood']} expression, photorealistic, detailed face, high detail, good lighting"
        )
        
        # Append style if selected (but since this is a simpler UI, just indicate it's included)
        if a["style"] and a["style"] != "Default":
            base_prompt = f"{a['style'].lower()} {base_prompt}"
            
        return base_prompt

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

        # Get selected model ID
        selected_model_display = self.model_combo.currentText()
        model_id = self.model_id_mapping.get(selected_model_display, "flux_dev")  # fallback to flux_dev if not found

        self.btn_generate.setEnabled(False)
        self.status_label.setText("Generating\u2026 this may take a while.")
        self._show_ghosts(self.variant_spin.value())
        
        # For a more complete integration we could build full attribute dict
        full_attributes = self.get_attributes()
        if "style" in full_attributes:
            style_id = full_attributes["style"].lower().replace(" ", "_")
            if style_id in ["ultra_realistic", "cinematic", "photorealistic", "realistic"]:
                # This would be passed to our more complete prompt building function...
                pass

        self._gen_thread = _GenerateThread(self.project_slug, name, prompt, self.variant_spin.value(), model_id)
        self._gen_thread.finished_ok.connect(self._on_generated)
        self._gen_thread.failed.connect(self._on_generation_failed)
        self._gen_thread.start()

    def _show_ghosts(self, count):
        from ui.components.ghost_card import GhostCard
        self._clear_ghosts()
        for _ in range(count):
            self.ghost_row.addWidget(GhostCard(110, 130))
        self.ghost_row.addStretch()

    def _clear_ghosts(self):
        while self.ghost_row.count():
            item = self.ghost_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _on_generated(self):
        self.status_label.setText("")
        self._clear_ghosts()
        self.on_created()

    def _on_generation_failed(self, error):
        self.btn_generate.setEnabled(True)
        self.status_label.setText("")
        self._clear_ghosts()
        QMessageBox.critical(self, "Generation failed", error)
