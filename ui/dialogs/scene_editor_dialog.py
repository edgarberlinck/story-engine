"""
Dialog for creating/editing scenes with full audio-scene representation.

Allows users to inspect and modify:
- Narration and dialogue segments
- Speaker, voice, emotion, tone, intensity, delivery
- Timing/order of segments
- Sound effects and music
- Characters and objects present
- Location

The user can also request LLM assistance to modify the representation
(e.g. "Make Nikita sound more irritated", "Rewrite this dialogue so Roger
sounds afraid but tries to hide it").
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton,
    QScrollArea, QWidget, QFrame, QApplication, QComboBox, QDoubleSpinBox,
    QSpinBox, QGroupBox, QFormLayout, QListWidget, QListWidgetItem,
    QMessageBox, QLineEdit, QCheckBox
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QPixmap

from services.audio_scene_service import (
    audio_scene_service, AudioSceneRepresentation, AudioSceneSegment,
)


class _LLMAssistThread(QThread):
    """Run LLM scene assistance off the UI thread."""

    finished_ok = Signal(object)  # proposed AudioSceneRepresentation
    failed = Signal(str)

    def __init__(self, representation, user_request):
        super().__init__()
        self.representation = representation
        self.user_request = user_request

    def run(self):
        try:
            from core.scene_assist import assist_scene
            proposal = assist_scene(self.representation, self.user_request)
            if proposal is None:
                self.failed.emit(
                    "The LLM did not return a usable modified representation. "
                    "Is a local text-generation model installed (make install)?"
                )
            else:
                self.finished_ok.emit(proposal)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class _PreviewThread(QThread):
    """Generate (and optionally play) segment audio off the UI thread."""

    finished_ok = Signal(list)  # list of generated wav paths
    failed = Signal(str)

    def __init__(self, project, scene_number, jobs, force=False):
        """jobs: list of (segment_index, AudioSceneSegment)."""
        super().__init__()
        self.project = project
        self.scene_number = scene_number
        self.jobs = jobs
        self.force = force

    def run(self):
        try:
            from core.audio_preview import generate_segment_audio
            paths = []
            for idx, segment in self.jobs:
                path = generate_segment_audio(
                    self.project, self.scene_number, idx, segment, force=self.force
                )
                if path:
                    paths.append(str(path))
            self.finished_ok.emit(paths)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class _EntityChip(QFrame):
    """Small clickable entity entry: thumbnail + name."""

    def __init__(self, entity, entity_type, on_toggled, entity_id=None):
        super().__init__()
        self.entity = entity
        self.entity_type = entity_type  # "character", "object", "location"
        self.on_toggled = on_toggled
        self.entity_id = entity_id
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("entityChip")
        self.setStyleSheet("""
            QFrame#entityChip {
                border: 1px solid #ddd; border-radius: 6px; background: white;
            }
            QFrame#entityChip:hover { border: 2px solid #4CAF50; background: #f0f8f0; }
            QLabel { border: none; background: transparent; }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 8, 6)
        layout.setSpacing(8)

        thumb = QLabel()
        thumb.setFixedSize(48, 48)
        thumb.setAlignment(Qt.AlignCenter)
        img_path = entity.get("visual_identity") or entity.get("reference_image")
        pix = QPixmap(img_path) if img_path and Path(img_path).is_file() else QPixmap()
        if not pix.isNull():
            thumb.setPixmap(pix.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            thumb.setText("?")
            thumb.setStyleSheet("background: #eee; color: #999; border-radius: 4px;")

        name = QLabel(entity["name"])
        name.setStyleSheet("font-weight: 600; font-size: 12px;")
        name.setWordWrap(True)

        layout.addWidget(thumb)
        layout.addWidget(name, 1)

        # Check state/toggle
        if entity_type == "location":
            state = entity.get("state", "initial")
            self.setProperty("state", state)
            self.setStyleSheet(self.styleSheet() + f'QFrame#entityChip[state="{state}"] {{ border-color: #4CAF50; }}')

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.on_toggled(self.entity, self.entity_id)
        super().mousePressEvent(event)


class _SegmentEditor(QWidget):
    """Widget for editing one audio scene segment."""

    changed = Signal(dict)  # segment dict changes

    def __init__(self, segment, all_characters, all_objects, all_locations, parent=None):
        super().__init__(parent)
        self.segment = segment
        self.all_characters = all_characters
        self.all_objects = all_objects
        self.all_locations = all_locations

        layout = QFormLayout(self)

        # Type
        self.type_combo = QComboBox()
        self.type_combo.addItems(["dialogue", "narration", "sound_effect", "music"])
        self.type_combo.setCurrentText(segment.segment_type)
        self.type_combo.currentTextChanged.connect(self._on_changed)
        layout.addRow("Type:", self.type_combo)

        # Speaker
        self.speaker_combo = QComboBox()
        self.speaker_combo.addItem("")  # empty for narration
        char_names = [c["name"] for c in all_characters]
        obj_names = [o["name"] for o in all_objects]
        loc_names = [l["name"] for l in all_locations]
        all_speakers = [""] + char_names + obj_names + loc_names
        self.speaker_combo.addItems(all_speakers)
        # Find and set current speaker
        idx = self.speaker_combo.findText(segment.speaker)
        if idx >= 0:
            self.speaker_combo.setCurrentIndex(idx)
        else:
            self.speaker_combo.setCurrentText(segment.speaker)
        self.speaker_combo.currentTextChanged.connect(self._on_changed)
        layout.addRow("Speaker:", self.speaker_combo)

        # Text
        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(segment.text)
        self.text_edit.textChanged.connect(self._on_changed)
        layout.addRow("Text:", self.text_edit)

        # Emotion
        self.emotion_combo = QComboBox()
        emotion_values = [""] + ["happy", "serious", "calm", "friendly", "confident",
                                  "mysterious", "sad", "angry", "fearful", "stressed"]
        self.emotion_combo.addItems(emotion_values)
        idx = self.emotion_combo.findText(segment.emotion or "")
        if idx >= 0:
            self.emotion_combo.setCurrentIndex(idx)
        self.emotion_combo.currentTextChanged.connect(self._on_changed)
        layout.addRow("Emotion:", self.emotion_combo)

        # Tone
        self.tone_combo = QComboBox()
        tone_values = [""] + ["confrontational", "descriptive", "friendly", "narrative",
                               "cinematic", "natural"]
        self.tone_combo.addItems(tone_values)
        idx = self.tone_combo.findText(segment.tone or "")
        if idx >= 0:
            self.tone_combo.setCurrentIndex(idx)
        self.tone_combo.currentTextChanged.connect(self._on_changed)
        layout.addRow("Tone:", self.tone_combo)

        # Intensity
        self.intensity_spin = QDoubleSpinBox()
        self.intensity_spin.setRange(0.0, 1.0)
        self.intensity_spin.setSingleStep(0.1)
        self.intensity_spin.setValue(segment.intensity if segment.intensity is not None else 0.5)
        self.intensity_spin.valueChanged.connect(self._on_changed)
        layout.addRow("Intensity:", self.intensity_spin)

        # Delivery
        self.delivery_combo = QComboBox()
        delivery_values = [""] + ["calm", "fast and forceful", "breathless", "tense", "quiet",
                                   "shaky", "relaxed", "fast and unpredictable"]
        self.delivery_combo.addItems(delivery_values)
        idx = self.delivery_combo.findText(segment.delivery or "")
        if idx >= 0:
            self.delivery_combo.setCurrentIndex(idx)
        self.delivery_combo.currentTextChanged.connect(self._on_changed)
        layout.addRow("Delivery:", self.delivery_combo)

        # Voice
        self.voice_combo = QComboBox()
        # Characters have voice paths; we'll just list character names
        self.voice_combo.addItems([""] + char_names)
        idx = self.voice_combo.findText(segment.voice or "")
        if idx >= 0:
            self.voice_combo.setCurrentIndex(idx)
        self.voice_combo.currentTextChanged.connect(self._on_changed)
        layout.addRow("Voice:", self.voice_combo)

        # Timing
        self.timing_layout = QHBoxLayout()
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0.0, 60.0)
        self.start_spin.setSingleStep(0.5)
        self.start_spin.setValue(segment.timing.get("start", 0.0))
        self.start_spin.valueChanged.connect(self._on_changed)
        self.timing_layout.addWidget(QLabel("Start(s):"))
        self.timing_layout.addWidget(self.start_spin)

        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(0.0, 120.0)
        self.end_spin.setSingleStep(0.5)
        self.end_spin.setValue(segment.timing.get("end", 5.0))
        self.end_spin.valueChanged.connect(self._on_changed)
        self.timing_layout.addWidget(QLabel("End(s):"))
        self.timing_layout.addWidget(self.end_spin)
        layout.addRow("Timing:", self.timing_layout)

        # Sound effects
        self.sfx_list = QListWidget()
        sfx = segment.sound_effects or []
        for s in sfx:
            self.sfx_list.addItem(s)
        self.sfx_list.itemChanged.connect(self._on_changed)
        layout.addRow("Sound Effects:", self.sfx_list)

        # Music
        self.music_check = QCheckBox("Music")
        self.music_check.setChecked(segment.music if segment.music is not None else False)
        self.music_check.toggled.connect(self._on_changed)
        layout.addRow("", self.music_check)

    def _on_changed(self):
        if getattr(self, "_loading", False):
            return
        # Update segment from UI
        self.segment.segment_type = self.type_combo.currentText()
        self.segment.speaker = self.speaker_combo.currentText() or None
        self.segment.text = self.text_edit.toPlainText()
        self.segment.emotion = self.emotion_combo.currentText() or None
        self.segment.tone = self.tone_combo.currentText() or None
        self.segment.intensity = self.intensity_spin.value()
        self.segment.delivery = self.delivery_combo.currentText() or None
        self.segment.voice = self.voice_combo.currentText() or None
        self.segment.timing = {
            "start": self.start_spin.value(),
            "end": self.end_spin.value(),
        }
        self.segment.sound_effects = [self.sfx_list.item(i).text()
                                      for i in range(self.sfx_list.count())]
        self.segment.music = self.music_check.isChecked()

        self.changed.emit(self.segment.to_dict())

    def set_segment(self, segment):
        """Populate UI from a segment."""
        self._loading = True
        try:
            self.segment = segment
            self.type_combo.setCurrentText(segment.segment_type)
            idx = self.speaker_combo.findText(segment.speaker or "")
            if idx >= 0:
                self.speaker_combo.setCurrentIndex(idx)
            self.text_edit.setPlainText(segment.text)
            idx = self.emotion_combo.findText(segment.emotion or "")
            if idx >= 0:
                self.emotion_combo.setCurrentIndex(idx)
            idx = self.tone_combo.findText(segment.tone or "")
            if idx >= 0:
                self.tone_combo.setCurrentIndex(idx)
            self.intensity_spin.setValue(segment.intensity if segment.intensity is not None else 0.5)
            idx = self.delivery_combo.findText(segment.delivery or "")
            if idx >= 0:
                self.delivery_combo.setCurrentIndex(idx)
            idx = self.voice_combo.findText(segment.voice or "")
            if idx >= 0:
                self.voice_combo.setCurrentIndex(idx)
            self.start_spin.setValue(segment.timing.get("start", 0.0))
            self.end_spin.setValue(segment.timing.get("end", 5.0))
            self.sfx_list.clear()
            for s in segment.sound_effects or []:
                self.sfx_list.addItem(s)
            self.music_check.setChecked(bool(segment.music))
        finally:
            self._loading = False


class SceneEditorDialog(QDialog):
    """Dialog for creating and editing scenes with full audio-scene representation."""

    # Signals
    segment_edited = Signal(dict)  # when a segment is edited
    representation_edited = Signal(dict)  # when the full representation is edited
    segment_added = Signal(dict)  # when a new segment is added
    segment_removed = Signal(str)  # when a segment is removed (by scene_id)

    def __init__(self, parent=None, project=None, scene_id=None, scene_number=None):
        super().__init__(parent)
        self.project = project
        self.scene_id = scene_id
        self.scene_number = scene_number
        self._llm_thread = None
        self._preview_thread = None
        self._play_queue = []
        self.setWindowTitle(f"Scene {scene_number} Editor" if scene_number else "New Scene Editor")
        self.setMinimumSize(900, 700)

        # Load existing representation or create new
        if scene_id and project:
            existing = audio_scene_service.get_representation(project, scene_number)
            if existing:
                self.representation = existing
            else:
                self.representation = AudioSceneRepresentation(
                    scene_id=scene_id,
                    title=f"Scene {scene_number}",
                    timeline={"day": 1, "time": "08:00"},
                )
        else:
            self.representation = AudioSceneRepresentation(
                scene_id=scene_id or "",
                title=f"Scene {scene_number or ''}",
                timeline={"day": 1, "time": "08:00"},
            )

        # Get entities from database
        from core.character_manager import character_manager
        from core.object_manager import object_manager
        from core.location_manager import location_manager

        all_characters = character_manager.list_characters(project) if project else []
        all_objects = object_manager.list_objects(project) if project else []
        all_locations = location_manager.list_locations(project) if project else []

        self._setup_ui(all_characters, all_objects, all_locations)

    def _setup_ui(self, all_characters, all_objects, all_locations):
        layout = QVBoxLayout(self)

        # Toolbar
        toolbar = QHBoxLayout()
        btn_add_segment = QPushButton("+ Add Segment")
        btn_add_segment.clicked.connect(self._add_segment)
        btn_remove_segment = QPushButton("- Remove Selected Segment")
        btn_remove_segment.clicked.connect(self._remove_segment)
        self.btn_preview_line = QPushButton("\u25b6 Preview Line")
        self.btn_preview_line.clicked.connect(self._preview_line)
        self.btn_regen_line = QPushButton("\u21bb Regenerate Line")
        self.btn_regen_line.clicked.connect(self._regenerate_line)
        self.btn_preview_scene = QPushButton("\u25b6\u25b6 Preview Scene")
        self.btn_preview_scene.clicked.connect(self._preview_scene)
        self.btn_help = QPushButton("LLM Assistance")
        self.btn_help.clicked.connect(self._llm_assistance)
        toolbar.addWidget(btn_add_segment)
        toolbar.addWidget(btn_remove_segment)
        toolbar.addWidget(self.btn_preview_line)
        toolbar.addWidget(self.btn_regen_line)
        toolbar.addWidget(self.btn_preview_scene)
        toolbar.addStretch()
        toolbar.addWidget(self.btn_help)
        layout.addLayout(toolbar)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #4CAF50; font-style: italic;")
        layout.addWidget(self.status_label)

        # Scene info
        info_group = QGroupBox("Scene Information")
        info_layout = QFormLayout(info_group)

        self.title_edit = QLineEdit(self.representation.title)
        self.title_edit.textChanged.connect(self._on_representation_changed)
        info_layout.addRow("Title:", self.title_edit)

        # Timeline
        self.day_spin = QSpinBox()
        self.day_spin.setRange(1, 365)
        self.day_spin.setValue(self.representation.timeline.get("day", 1))
        self.day_spin.valueChanged.connect(self._on_representation_changed)
        info_layout.addRow("Day:", self.day_spin)

        self.time_edit = QLineEdit(self.representation.timeline.get("time", "08:00"))
        self.time_edit.textChanged.connect(self._on_representation_changed)
        info_layout.addRow("Time:", self.time_edit)

        # Characters present
        self.char_list = QListWidget()
        for c in all_characters:
            item = QListWidgetItem(c["name"])
            item.setData(0, c)  # store full char dict
            self.char_list.addItem(item)
        layout.addWidget(info_group)
        layout.addWidget(QLabel("Characters Present:"))
        layout.addWidget(self.char_list)

        # Objects present
        self.obj_list = QListWidget()
        for o in all_objects:
            item = QListWidgetItem(o["name"])
            item.setData(0, o)
            self.obj_list.addItem(item)
        layout.addWidget(QLabel("Objects Present:"))
        layout.addWidget(self.obj_list)

        # Locations present
        self.loc_list = QListWidget()
        for l in all_locations:
            item = QListWidgetItem(l["name"])
            item.setData(0, l)
            self.loc_list.addItem(item)
        layout.addWidget(QLabel("Location:"))
        layout.addWidget(self.loc_list)

        # Segments list and editor area
        split_layout = QHBoxLayout()

        # Left: segments list
        left_group = QGroupBox("Segments")
        left_layout = QVBoxLayout(left_group)
        self.segment_list = QListWidget()
        self.segment_list.addItems([(s.speaker or "narrator") + ": " + (s.text[:30] if s.text else "") for s in self.representation.segments])
        self.segment_list.currentItemChanged.connect(self._segment_selected)
        left_layout.addWidget(self.segment_list)
        left_layout.addStretch()
        split_layout.addWidget(left_group, 1)

        # Right: segment editor
        right_group = QGroupBox("Segment Editor")
        right_layout = QVBoxLayout(right_group)
        self._segment_editor = _SegmentEditor(
            self.representation.segments[0] if self.representation.segments
            else AudioSceneSegment(segment_type="narration", speaker="narrator", text=""),
            all_characters, all_objects, all_locations
        )
        # Connect segment editor changes to update the list
        self._segment_editor.changed.connect(self._on_segment_editor_changed)
        right_layout.addWidget(self._segment_editor)
        right_layout.addStretch()
        split_layout.addWidget(right_group, 2)

        layout.addLayout(split_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_export = QPushButton("Export Representation")
        btn_export.clicked.connect(self._export_representation)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self._save_and_accept)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_export)
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_ok)
        layout.addLayout(btn_layout)

        # Initialize with first segment
        self._segment_selected(self.segment_list.currentItem())

    def _on_representation_changed(self, *args):
        """Sync scene-level fields (title, timeline) into the representation."""
        self.representation.title = self.title_edit.text()
        self.representation.timeline = {
            "day": self.day_spin.value(),
            "time": self.time_edit.text(),
        }
        self.representation_edited.emit(self.representation.to_dict())

    def _segment_selected(self, item):
        """Load the selected segment into the editor."""
        if not item:
            self._segment_editor.set_segment(AudioSceneSegment(
                segment_type="narration",
                speaker="narrator",
                text=""
            ))
            return

        # The list rows map 1:1 to representation.segments.
        row = self.segment_list.currentRow()
        if 0 <= row < len(self.representation.segments):
            self._segment_editor.set_segment(self.representation.segments[row])
            return

        # Fallback to first segment
        if self.representation.segments:
            self._segment_editor.set_segment(self.representation.segments[0])

    def _on_segment_editor_changed(self, segment_dict):
        """Update the list when the editor changes a segment.

        The editor mutates the AudioSceneSegment in place (it holds the same
        object stored in representation.segments), so only the list labels
        need refreshing here.
        """
        row = self.segment_list.currentRow()
        self._refresh_segment_list()
        if 0 <= row < self.segment_list.count():
            self.segment_list.blockSignals(True)
            self.segment_list.setCurrentRow(row)
            self.segment_list.blockSignals(False)

        # Emit signal
        self.representation_edited.emit(self.representation.to_dict())

    def _add_segment(self):
        """Add a new empty segment."""
        new_seg = AudioSceneSegment(
            segment_type="dialogue",
            speaker="",
            text="",
            emotion="",
            tone="",
            intensity=0.5,
            delivery="",
            voice=""
        )
        self.representation.segments.append(new_seg)
        self.segment_list.addItem("narrator: ")
        # Select the new item
        row = self.segment_list.count() - 1
        self.segment_list.setCurrentRow(row)
        self._segment_selected(self.segment_list.currentItem())

    def _remove_segment(self):
        """Remove the selected segment."""
        current_row = self.segment_list.currentRow()
        if current_row < 0:
            return

        # Don't remove the last segment
        if self.segment_list.count() <= 1:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Cannot Remove", "Must keep at least one segment.")
            return

        # Emit signal with scene_id for tracking
        from PySide6.QtWidgets import QMessageBox
        self.segment_removed.emit(self.scene_id or "unsaved")
        del self.representation.segments[current_row]

        # Refresh list
        self.segment_list.clear()
        self.segment_list.addItems([(s.speaker or "narrator") + ": " + (s.text[:30] if s.text else "") for s in self.representation.segments])

        # Select the next or first segment
        if self.segment_list.count() > 0:
            self.segment_list.setCurrentRow(min(current_row, self.segment_list.count() - 1))
            self._segment_selected(self.segment_list.currentItem())

    def _llm_assistance(self):
        """Ask the LLM to modify the structured scene representation.

        The LLM operates on the representation (never on audio) and the user
        reviews the proposal before it is applied (plan §5).
        """
        from PySide6.QtWidgets import QInputDialog
        request, ok = QInputDialog.getMultiLineText(
            self, "LLM Assistance",
            "Describe the change you want. Examples:\n"
            "- Make Nikita sound more irritated.\n"
            "- Rewrite this dialogue so Roger sounds afraid but tries to hide it.\n"
            "- Make the narration more cinematic.",
        )
        if not ok or not request.strip():
            return

        self.btn_help.setEnabled(False)
        self.status_label.setText("Asking the LLM\u2026 this may take a while.")
        self._llm_thread = _LLMAssistThread(self.representation, request.strip())
        self._llm_thread.finished_ok.connect(self._on_llm_proposal)
        self._llm_thread.failed.connect(self._on_llm_failed)
        self._llm_thread.start()

    def _on_llm_proposal(self, proposal):
        self.btn_help.setEnabled(True)
        self.status_label.setText("")

        # User review before applying (plan §5): show a readable summary.
        import json
        preview = json.dumps(proposal.to_dict(), indent=2)
        if len(preview) > 4000:
            preview = preview[:4000] + "\n\u2026 (truncated)"
        reply = QMessageBox.question(
            self, "Review LLM Proposal",
            "The LLM proposes this modified representation. Apply it?\n\n" + preview,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.representation = proposal
            self._refresh_segment_list()
            if self.representation.segments:
                self.segment_list.setCurrentRow(0)
                self._segment_editor.set_segment(self.representation.segments[0])
            self.representation_edited.emit(self.representation.to_dict())

    def _on_llm_failed(self, error):
        self.btn_help.setEnabled(True)
        self.status_label.setText("")
        QMessageBox.warning(self, "LLM Assistance", error)

    # -- Audio preview -------------------------------------------------------

    def _refresh_segment_list(self):
        self.segment_list.clear()
        self.segment_list.addItems(
            [(s.speaker or "narrator") + ": " + (s.text[:30] if s.text else "")
             for s in self.representation.segments]
        )

    def _current_segment_job(self):
        """(index, segment) for the selected list row, or None."""
        row = self.segment_list.currentRow()
        if row < 0 or row >= len(self.representation.segments):
            return None
        return row, self.representation.segments[row]

    def _preview_line(self):
        job = self._current_segment_job()
        if job is None:
            QMessageBox.information(self, "Preview", "Select a segment first.")
            return
        self._start_preview([job], force=False)

    def _regenerate_line(self):
        """Regenerate ONLY the selected line (plan §7: never the whole book)."""
        job = self._current_segment_job()
        if job is None:
            QMessageBox.information(self, "Regenerate", "Select a segment first.")
            return
        self._start_preview([job], force=True)

    def _preview_scene(self):
        jobs = list(enumerate(self.representation.segments))
        if not jobs:
            QMessageBox.information(self, "Preview", "No segments to preview.")
            return
        self._start_preview(jobs, force=False)

    def _set_preview_enabled(self, enabled):
        self.btn_preview_line.setEnabled(enabled)
        self.btn_regen_line.setEnabled(enabled)
        self.btn_preview_scene.setEnabled(enabled)

    def _start_preview(self, jobs, force):
        if self.scene_number is None or not self.project:
            QMessageBox.information(
                self, "Preview", "Save the scene first so audio can be cached."
            )
            return
        self._set_preview_enabled(False)
        self.status_label.setText("Generating audio\u2026 (cached lines are reused)")
        self._preview_thread = _PreviewThread(
            self.project, self.scene_number, jobs, force=force
        )
        self._preview_thread.finished_ok.connect(self._on_preview_ready)
        self._preview_thread.failed.connect(self._on_preview_failed)
        self._preview_thread.start()

    def _on_preview_ready(self, paths):
        self._set_preview_enabled(True)
        self.status_label.setText("")
        if not paths:
            QMessageBox.information(
                self, "Preview", "Nothing to play (empty or non-speech segments)."
            )
            return
        self._play_queue = list(paths)
        self._play_next()

    def _on_preview_failed(self, error):
        self._set_preview_enabled(True)
        self.status_label.setText("")
        QMessageBox.warning(self, "Audio Preview", error)

    def _play_next(self):
        """Play queued WAVs sequentially with QProcess (afplay on macOS)."""
        if not self._play_queue:
            return
        import sys
        from PySide6.QtCore import QProcess
        path = self._play_queue.pop(0)
        player = "afplay" if sys.platform == "darwin" else "aplay"
        self._player = QProcess(self)
        self._player.finished.connect(lambda *_: self._play_next())
        self._player.start(player, [path])

    def _export_representation(self):
        """Export the current representation as JSON."""
        import json
        repr_dict = self.representation.to_dict()
        json_str = json.dumps(repr_dict, indent=2)
        # Copy to clipboard
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(json_str)
        QMessageBox.information(self, "Exported", "Scene representation JSON copied to clipboard.")

    def _save_and_accept(self):
        """Persist the representation (when bound to a scene) and close."""
        self.representation.title = self.title_edit.text()
        self.representation.timeline = {
            "day": self.day_spin.value(),
            "time": self.time_edit.text(),
        }
        if self.project and self.scene_number is not None:
            audio_scene_service.save_representation(
                self.project,
                self.scene_id or "",
                self.scene_number,
                self.representation,
            )
        self.accept()

    def get_data(self):
        """Return the full scene representation data."""
        return {
            "representation": self.representation.to_dict(),
            "title": self.title_edit.text(),
            "day": self.day_spin.value(),
            "time": self.time_edit.text(),
            "characters_present": [self.char_list.item(i).text() for i in range(self.char_list.count())],
            "objects_present": [self.obj_list.item(i).text() for i in range(self.obj_list.count())],
            "locations_present": [self.loc_list.item(i).text() for i in range(self.loc_list.count())],
        }