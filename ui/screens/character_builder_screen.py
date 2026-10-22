"""Character creation with an editable prompt and optional image reference."""

import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QComboBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.character_manager import character_manager, ensure_full_body
from core.project_manager import project_manager
from models import MODEL_METADATA


def get_diffusion_models():
    """Return available diffusion models as display-name/model-id pairs."""
    diffusion_models = [
        (metadata.get("name", model_id), model_id)
        for model_id, metadata in MODEL_METADATA.items()
        if metadata.get("type") == "diffusion"
    ]
    return sorted(diffusion_models, key=lambda model: model[0])


class _PromptThread(QThread):
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        try:
            from generators.image_prompt_generator import generate_prompt_from_image

            self.finished_ok.emit(generate_prompt_from_image(self.image_path))
        except Exception as error:  # noqa: BLE001
            self.failed.emit(str(error))


class _PreviewThread(QThread):
    finished_ok = Signal(str, str, int)
    failed = Signal(str)

    def __init__(self, prompt, model_id, reference_image, output_path):
        super().__init__()
        self.prompt = prompt
        self.model_id = model_id
        self.reference_image = reference_image
        self.output_path = output_path
        self.seed = uuid4().int % (2**31)

    def run(self):
        try:
            if self.reference_image:
                from generators.img2img_engine import generate_image_from_reference

                generate_image_from_reference(
                    self.reference_image,
                    self.prompt,
                    model_name=self.model_id,
                    seed=self.seed,
                    output_path=self.output_path,
                )
                prompt_used = self.prompt
            else:
                from generators.image_generator import generate_images

                prompt_used = ensure_full_body(self.prompt)
                files = generate_images(
                    prompt=prompt_used,
                    model_name=self.model_id,
                    seed=self.seed,
                    task_name=f"character_preview_{uuid4().hex}",
                )
                if not files:
                    raise RuntimeError("Image generation did not return an image.")
                shutil.copy2(files[0], self.output_path)

            self.finished_ok.emit(self.output_path, prompt_used, self.seed)
        except Exception as error:  # noqa: BLE001
            self.failed.emit(str(error))


class CharacterBuilderScreen(QWidget):
    def __init__(self, project_id, on_back, on_created):
        super().__init__()
        self.project_id = project_id
        self.on_back = on_back
        self.on_created = on_created
        self.project = project_manager.get_project(project_id) or {"name": "Unknown"}
        self.project_slug = self.project["name"].replace(" ", "_")
        self.reference_image = None
        self._preview_path = None
        self._preview_prompt = None
        self._busy = False
        self._preview_directory = tempfile.TemporaryDirectory(
            prefix="story_engine_character_"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)

        crumb_row = QHBoxLayout()
        self.back_button = QPushButton("\u2190 Back")
        self.back_button.setProperty("flat", True)
        self.back_button.clicked.connect(self.cancel)
        breadcrumb = QLabel(
            f"Home  \u203a  {self.project['name']}  \u203a  New Character"
        )
        breadcrumb.setStyleSheet("font-size: 12px; color: #666;")
        crumb_row.addWidget(self.back_button)
        crumb_row.addWidget(breadcrumb)
        crumb_row.addStretch()
        outer.addLayout(crumb_row)

        title = QLabel("Create New Character")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        outer.addWidget(title)

        form = QFormLayout()
        form.setSpacing(10)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Character name (required)")
        form.addRow("Name:", self.name_input)

        self.model_combo = QComboBox()
        for display_name, model_id in get_diffusion_models():
            self.model_combo.addItem(display_name, model_id)
        sdxl_index = self.model_combo.findData("sdxl")
        if sdxl_index >= 0:
            self.model_combo.setCurrentIndex(sdxl_index)
        self.model_combo.currentIndexChanged.connect(self._invalidate_preview)
        form.addRow("Generation model:", self.model_combo)
        outer.addLayout(form)

        image_row = QHBoxLayout()
        source_column = QVBoxLayout()
        source_title = QLabel("Reference image (optional)")
        self.source_preview = self._image_preview_label("No reference image")
        self.choose_image_button = QPushButton("Choose Image\u2026")
        self.choose_image_button.clicked.connect(self.choose_reference_image)
        self.describe_button = QPushButton("Generate Prompt from Image")
        self.describe_button.setEnabled(False)
        self.describe_button.clicked.connect(self.generate_prompt_from_image)
        source_column.addWidget(source_title)
        source_column.addWidget(self.source_preview)
        source_column.addWidget(self.choose_image_button)
        source_column.addWidget(self.describe_button)
        image_row.addLayout(source_column)

        generated_column = QVBoxLayout()
        generated_column.addWidget(QLabel("Generated preview"))
        self.generated_preview = self._image_preview_label(
            "Generate a preview to review it here"
        )
        generated_column.addWidget(self.generated_preview)
        image_row.addLayout(generated_column)
        image_row.addStretch()
        outer.addLayout(image_row)

        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText(
            "Describe the character, or upload an image and generate a prompt."
        )
        self.prompt_input.setMinimumHeight(110)
        self.prompt_input.textChanged.connect(self._invalidate_preview)
        outer.addWidget(QLabel("Prompt:"))
        outer.addWidget(self.prompt_input)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        outer.addWidget(self.status_label)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setProperty("flat", True)
        self.cancel_button.clicked.connect(self.cancel)
        buttons.addWidget(self.cancel_button)
        self.preview_button = QPushButton("Generate Preview")
        self.preview_button.clicked.connect(self.generate_preview)
        buttons.addWidget(self.preview_button)
        self.save_button = QPushButton("Save Character")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_character)
        buttons.addWidget(self.save_button)
        outer.addLayout(buttons)

    @staticmethod
    def _image_preview_label(placeholder):
        label = QLabel(placeholder)
        label.setFixedSize(220, 220)
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        label.setStyleSheet(
            "background-color: #eee; border: 1px solid #ddd; border-radius: 6px;"
        )
        return label

    @staticmethod
    def _set_image(label, image_path, placeholder):
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            label.setPixmap(QPixmap())
            label.setText(placeholder)
            return False
        label.setText("")
        label.setPixmap(
            pixmap.scaled(
                label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        return True

    def _invalidate_preview(self):
        if not self._preview_path:
            return
        Path(self._preview_path).unlink(missing_ok=True)
        self._preview_path = None
        self._preview_prompt = None
        self._preview_seed = None
        self.save_button.setEnabled(False)
        self.generated_preview.setPixmap(QPixmap())
        self.generated_preview.setText("Prompt changed; generate a new preview")

    def choose_reference_image(self):
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose character reference image",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if not image_path:
            return
        self._invalidate_preview()
        self.reference_image = None
        self.describe_button.setEnabled(False)
        if not self._set_image(self.source_preview, image_path, "Could not load image"):
            QMessageBox.warning(
                self, "Invalid image", "The selected image could not be opened."
            )
            return
        self.reference_image = image_path
        self.describe_button.setEnabled(True)

    def _set_busy(self, busy, message=""):
        self._busy = busy
        self.status_label.setText(message)
        for button in (
            self.back_button,
            self.cancel_button,
            self.choose_image_button,
            self.describe_button,
            self.preview_button,
            self.save_button,
        ):
            button.setEnabled(not busy)
        self.name_input.setEnabled(not busy)
        self.model_combo.setEnabled(not busy)
        self.prompt_input.setEnabled(not busy)
        if not busy:
            self.describe_button.setEnabled(bool(self.reference_image))
            self.save_button.setEnabled(bool(self._preview_path))

    def generate_prompt_from_image(self):
        if not self.reference_image:
            return
        self._set_busy(True, "Analyzing reference image\u2026")
        self._prompt_thread = _PromptThread(self.reference_image)
        self._prompt_thread.finished_ok.connect(self._on_prompt_generated)
        self._prompt_thread.failed.connect(self._on_task_failed)
        self._prompt_thread.start()

    def _on_prompt_generated(self, prompt):
        self._set_busy(False)
        self.prompt_input.setPlainText(prompt)

    def generate_preview(self):
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Missing prompt", "Enter a character prompt.")
            return
        model_id = self.model_combo.currentData()
        if self.reference_image and model_id not in {"sdxl", "flux_dev"}:
            QMessageBox.warning(
                self,
                "Unsupported model",
                "Reference-image generation supports SDXL and FLUX.1 Dev. "
                "Choose one of those models to preserve the pose.",
            )
            return

        self._invalidate_preview()
        output_path = str(Path(self._preview_directory.name) / "candidate.png")
        self._set_busy(True, "Generating preview\u2026")
        self._preview_thread = _PreviewThread(
            prompt, model_id, self.reference_image, output_path
        )
        self._preview_thread.finished_ok.connect(self._on_preview_generated)
        self._preview_thread.failed.connect(self._on_task_failed)
        self._preview_thread.start()

    def _on_preview_generated(self, image_path, prompt_used, seed):
        self._set_busy(False)
        if not self._set_image(
            self.generated_preview, image_path, "Generated image could not be loaded"
        ):
            self._on_task_failed("Generated image could not be loaded.")
            return
        self._preview_path = image_path
        self._preview_prompt = prompt_used
        self._preview_seed = seed
        self.save_button.setEnabled(True)
        self.status_label.setText("Review the preview, edit the prompt, or save it.")

    def _on_task_failed(self, error):
        self._set_busy(False)
        self.status_label.setText("")
        QMessageBox.critical(self, "Character image failed", error)

    def save_character(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing name", "Please enter a character name.")
            return
        if character_manager.get_character(self.project_slug, name):
            QMessageBox.warning(
                self,
                "Duplicate",
                f"Character \u201c{name}\u201d already exists in this project.",
            )
            return
        if not self._preview_path or not self._preview_prompt:
            QMessageBox.warning(
                self, "Missing preview", "Generate and review a preview before saving."
            )
            return

        try:
            character_manager.save_preview(
                self.project_slug,
                name,
                self._preview_prompt,
                self.model_combo.currentData(),
                self._preview_path,
                seed=self._preview_seed,
            )
        except Exception as error:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(error))
            return

        self._preview_directory.cleanup()
        self.on_created()

    def cancel(self):
        if self._busy:
            return
        self._preview_directory.cleanup()
        self.on_back()
