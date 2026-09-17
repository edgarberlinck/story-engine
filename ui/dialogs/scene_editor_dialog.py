"""Timeline-first scene audio editor.

Redesigned around the review workflow: the multitrack timeline IS the
editor. On first open of a freshly written scene, a banner prompts the
author to click "\u2699 Generate Scene", which renders every ungenerated
fragment in one pass (core.scene_generation). After that the author can:

- listen to any fragment (click a clip, "\u25b6 Listen"),
- edit the essentials in a compact inspector (text/prompt, voice, speed;
  the exhaustive per-segment fields live behind a collapsed "Advanced\u2026"
  expander),
- accept a fragment when satisfied ("\u2714 Accept" — green clip),
- finalize the scene and play the final mix.

Edits invalidate acceptance (cache-key hash compare) and are persisted via
audio_scene_service as they happen.
"""

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from services.audio_scene_service import (
    AudioSceneRepresentation,
    audio_scene_service,
)

EMOTIONS = [
    "",
    "happy",
    "serious",
    "calm",
    "friendly",
    "confident",
    "mysterious",
    "sad",
    "angry",
    "fearful",
    "stressed",
]
TONES = [
    "",
    "confrontational",
    "descriptive",
    "friendly",
    "narrative",
    "cinematic",
    "natural",
]
DELIVERIES = [
    "",
    "calm",
    "fast and forceful",
    "breathless",
    "tense",
    "quiet",
    "shaky",
    "relaxed",
    "fast and unpredictable",
]


class _PreviewThread(QThread):
    """Generate (and optionally play) fragment audio off the UI thread."""

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
                    self.project,
                    self.scene_number,
                    idx,
                    segment,
                    force=self.force,
                )
                if path:
                    paths.append(str(path))
            self.finished_ok.emit(paths)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class _GenerateSceneThread(QThread):
    """Bulk-generate every missing fragment of the scene off the UI thread."""

    progress = Signal(int, int)  # done, total
    finished_ok = Signal(int)  # number of fragments generated
    failed = Signal(str)

    def __init__(self, project, scene_number, representation):
        super().__init__()
        self.project = project
        self.scene_number = scene_number
        self.representation = representation

    def run(self):
        try:
            from core.scene_generation import generate_missing_fragments

            count = generate_missing_fragments(
                self.project,
                self.scene_number,
                self.representation,
                progress_cb=lambda done, total, _label: self.progress.emit(done, total),
            )
            self.finished_ok.emit(count)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class _FinalizeThread(QThread):
    """Run scene finalization (mix speech + beds) off the UI thread."""

    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, project, scene_number, representation):
        super().__init__()
        self.project = project
        self.scene_number = scene_number
        self.representation = representation

    def run(self):
        try:
            from core.scene_finalizer import finalize_scene

            path = finalize_scene(self.project, self.scene_number, self.representation)
            self.finished_ok.emit(str(path))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class _FragmentInspector(QWidget):
    """Compact editor for one fragment: prompt, voice, speed + actions.

    Only the essentials are visible; the full per-segment fields (emotion,
    tone, intensity, delivery) are tucked behind "Advanced\u2026".
    """

    edited = Signal()  # any field changed (segment already updated)
    listen = Signal()
    regenerate = Signal()
    accept = Signal()

    def __init__(self, char_names, parent=None):
        super().__init__(parent)
        self.segment = None
        self._loading = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        self.header = QLabel("")
        self.header.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.header)

        form = QFormLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setFixedHeight(64)
        self.text_edit.setPlaceholderText(
            "Dialogue text, or the sound/music prompt for beds"
        )
        self.text_edit.textChanged.connect(self._on_changed)
        form.addRow("Prompt:", self.text_edit)

        self.voice_combo = QComboBox()
        self.voice_combo.addItems([""] + list(char_names))
        self.voice_combo.currentTextChanged.connect(self._on_changed)
        form.addRow("Voice:", self.voice_combo)

        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.5, 2.0)
        self.speed_spin.setSingleStep(0.05)
        self.speed_spin.setSpecialValueText("auto")
        self.speed_spin.setValue(self.speed_spin.minimum())  # "auto"
        self.speed_spin.valueChanged.connect(self._on_changed)
        form.addRow("Speed:", self.speed_spin)
        layout.addLayout(form)

        # Advanced (collapsed by default).
        self.btn_advanced = QPushButton("Advanced\u2026")
        self.btn_advanced.setCheckable(True)
        self.btn_advanced.setFlat(True)
        self.btn_advanced.toggled.connect(self._toggle_advanced)
        layout.addWidget(self.btn_advanced, 0, Qt.AlignLeft)

        self.advanced = QWidget()
        adv_form = QFormLayout(self.advanced)
        self.emotion_combo = QComboBox()
        self.emotion_combo.addItems(EMOTIONS)
        self.emotion_combo.currentTextChanged.connect(self._on_changed)
        adv_form.addRow("Emotion:", self.emotion_combo)
        self.tone_combo = QComboBox()
        self.tone_combo.addItems(TONES)
        self.tone_combo.currentTextChanged.connect(self._on_changed)
        adv_form.addRow("Tone:", self.tone_combo)
        self.intensity_spin = QDoubleSpinBox()
        self.intensity_spin.setRange(0.0, 1.0)
        self.intensity_spin.setSingleStep(0.1)
        self.intensity_spin.valueChanged.connect(self._on_changed)
        adv_form.addRow("Intensity:", self.intensity_spin)
        self.delivery_combo = QComboBox()
        self.delivery_combo.addItems(DELIVERIES)
        self.delivery_combo.currentTextChanged.connect(self._on_changed)
        adv_form.addRow("Delivery:", self.delivery_combo)
        self.advanced.setVisible(False)
        layout.addWidget(self.advanced)

        # Actions.
        actions = QHBoxLayout()
        self.btn_listen = QPushButton("\u25b6 Listen")
        self.btn_listen.clicked.connect(self.listen)
        self.btn_regen = QPushButton("\u21bb Regenerate")
        self.btn_regen.clicked.connect(self.regenerate)
        self.btn_accept = QPushButton("\u2714 Accept")
        self.btn_accept.setStyleSheet("font-weight: 600; color: #2e7d32;")
        self.btn_accept.clicked.connect(self.accept)
        actions.addWidget(self.btn_listen)
        actions.addWidget(self.btn_regen)
        actions.addWidget(self.btn_accept)
        actions.addStretch()
        layout.addLayout(actions)

    def _toggle_advanced(self, checked):
        self.advanced.setVisible(checked)

    def set_segment(self, segment):
        """Populate from a segment (speech or bed)."""
        self._loading = True
        try:
            self.segment = segment
            is_bed = segment.segment_type in ("sound_effect", "music")
            who = segment.speaker or (
                "Music"
                if segment.segment_type == "music"
                else "Sound FX" if is_bed else "narrator"
            )
            state = "\u2714 accepted" if segment.accepted else "pending"
            self.header.setText(f"{who} \u2014 {segment.segment_type} ({state})")
            self.text_edit.setPlainText(segment.text or "")
            self.voice_combo.setVisible(not is_bed)
            self.speed_spin.setVisible(not is_bed)
            self.btn_advanced.setVisible(not is_bed)
            if self.btn_advanced.isChecked():
                self.advanced.setVisible(not is_bed)
            idx = self.voice_combo.findText(segment.voice or "")
            self.voice_combo.setCurrentIndex(max(idx, 0))
            self.speed_spin.setValue(
                segment.speed if segment.speed else self.speed_spin.minimum()
            )
            idx = self.emotion_combo.findText(segment.emotion or "")
            self.emotion_combo.setCurrentIndex(max(idx, 0))
            idx = self.tone_combo.findText(segment.tone or "")
            self.tone_combo.setCurrentIndex(max(idx, 0))
            self.intensity_spin.setValue(
                segment.intensity if segment.intensity is not None else 0.5
            )
            idx = self.delivery_combo.findText(segment.delivery or "")
            self.delivery_combo.setCurrentIndex(max(idx, 0))
        finally:
            self._loading = False

    def _on_changed(self, *args):
        if self._loading or self.segment is None:
            return
        seg = self.segment
        seg.text = self.text_edit.toPlainText()
        if self.voice_combo.isVisible():
            seg.voice = self.voice_combo.currentText() or None
        speed = self.speed_spin.value()
        seg.speed = None if speed <= self.speed_spin.minimum() else speed
        seg.emotion = self.emotion_combo.currentText() or None
        seg.tone = self.tone_combo.currentText() or None
        seg.intensity = self.intensity_spin.value()
        seg.delivery = self.delivery_combo.currentText() or None
        self.edited.emit()


class SceneEditorDialog(QDialog):
    """Timeline-first scene audio editor (listen / edit / accept / finalize)."""

    def __init__(self, parent=None, project=None, scene_id=None, scene_number=None):
        super().__init__(parent)
        self.project = project
        self.scene_id = scene_id
        self.scene_number = scene_number
        self._preview_thread = None
        self._finalize_thread = None
        self._generate_thread = None
        self._play_queue = []
        self._player = None
        self._selected_index = -1
        self.setWindowTitle(
            f"Scene {scene_number} Audio" if scene_number else "Scene Audio"
        )
        self.setMinimumSize(1100, 640)
        self.resize(1250, 720)

        # Load existing representation or create an empty one.
        existing = None
        if project and scene_number is not None:
            existing = audio_scene_service.get_representation(project, scene_number)
        self.representation = existing or AudioSceneRepresentation(
            scene_id=scene_id or "",
            title=f"Scene {scene_number or ''}".strip(),
            timeline={"day": 1, "time": "08:00"},
        )

        from core.character_manager import character_manager

        char_names = [
            c["name"]
            for c in (character_manager.list_characters(project) if project else [])
        ]

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(700)
        self._save_timer.timeout.connect(self._persist_and_refresh)

        self._build_ui(char_names)
        self._update_summary()

    # -- UI ------------------------------------------------------------------

    def _build_ui(self, char_names):
        layout = QVBoxLayout(self)

        # Header: title + review status.
        header = QHBoxLayout()
        self.title_label = QLabel(self.representation.title or "Scene")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: 700;")
        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("color: #607d8b;")
        header.addWidget(self.title_label)
        header.addStretch()
        header.addWidget(self.summary_label)
        layout.addLayout(header)

        # Generation banner: a freshly written scene has no audio yet — the
        # author must click "⚙ Generate Scene" once to render every fragment.
        self.generate_banner = QWidget()
        self.generate_banner.setStyleSheet(
            "background-color: #fff8e1; border: 1px solid #ffe082;"
            " border-radius: 4px;"
        )
        banner_layout = QHBoxLayout(self.generate_banner)
        banner_layout.setContentsMargins(10, 6, 10, 6)
        self.generate_banner_label = QLabel("")
        self.generate_banner_label.setStyleSheet(
            "color: #795548; font-weight: 600; border: none;"
        )
        self.btn_generate = QPushButton("\u2699 Generate Scene")
        self.btn_generate.setStyleSheet("font-weight: 700;")
        self.btn_generate.clicked.connect(self._generate_scene)
        banner_layout.addWidget(self.generate_banner_label, 1)
        banner_layout.addWidget(self.btn_generate)
        layout.addWidget(self.generate_banner)
        self.generate_banner.setVisible(False)

        # Center: the timeline IS the editor.
        from ui.components.scene_timeline import SceneTimelinePanel

        self.timeline = SceneTimelinePanel(
            self.project, self.scene_number, self.representation
        )
        self.timeline.clip_selected.connect(self._on_clip_selected)
        self.timeline.representation_changed.connect(self._update_summary)
        layout.addWidget(self.timeline, 1)

        # Inspector (hidden until a clip is selected).
        self.inspector = _FragmentInspector(char_names)
        self.inspector.setVisible(False)
        self.inspector.edited.connect(self._on_fragment_edited)
        self.inspector.listen.connect(self._listen_fragment)
        self.inspector.regenerate.connect(self._regenerate_fragment)
        self.inspector.accept.connect(self._accept_fragment)
        layout.addWidget(self.inspector)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #4CAF50; font-style: italic;")
        layout.addWidget(self.status_label)

        # Footer.
        footer = QHBoxLayout()
        self.btn_finalize = QPushButton("Finalize Scene")
        self.btn_finalize.clicked.connect(self._finalize_scene)
        self.btn_play_final = QPushButton("\u25b6 Play Final")
        self.btn_play_final.clicked.connect(self._play_final)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self._save_and_accept)
        footer.addWidget(self.btn_finalize)
        footer.addWidget(self.btn_play_final)
        footer.addStretch()
        footer.addWidget(btn_close)
        layout.addLayout(footer)
        self._update_final_buttons()
        self._update_generate_banner()

    # -- bulk generation ---------------------------------------------------------

    def _missing_count(self):
        if not self.project or self.scene_number is None:
            return 0
        from core.scene_generation import count_missing_fragments

        try:
            return count_missing_fragments(
                self.project, self.scene_number, self.representation
            )
        except Exception:  # noqa: BLE001
            return 0

    def _update_generate_banner(self):
        missing = self._missing_count()
        if missing <= 0:
            self.generate_banner.setVisible(False)
            return
        self.generate_banner_label.setText(
            f"This scene has {missing} ungenerated fragment"
            f"{'s' if missing != 1 else ''} \u2014 Generate Scene to render audio."
        )
        self.btn_generate.setText("\u2699 Generate Scene")
        self.btn_generate.setEnabled(True)
        self.generate_banner.setVisible(True)

    def _generate_scene(self):
        if not self.project or self.scene_number is None:
            QMessageBox.information(
                self, "Generate Scene", "Save the scene first so audio can be cached."
            )
            return
        self._save_timer.stop()
        self._save_representation()
        self.btn_generate.setEnabled(False)
        self.btn_finalize.setEnabled(False)
        self.inspector.setEnabled(False)
        self.status_label.setText("Generating scene audio\u2026")
        self._generate_thread = _GenerateSceneThread(
            self.project, self.scene_number, self.representation
        )
        self._generate_thread.progress.connect(self._on_generate_progress)
        self._generate_thread.finished_ok.connect(self._on_generate_done)
        self._generate_thread.failed.connect(self._on_generate_failed)
        self._generate_thread.start()

    def _on_generate_progress(self, done, total):
        self.btn_generate.setText(f"Generating {min(done + 1, total)}/{total}\u2026")

    def _on_generate_done(self, count):
        self.btn_finalize.setEnabled(True)
        self.inspector.setEnabled(True)
        self.timeline.reload(self.representation)
        self._update_generate_banner()
        self._update_summary()
        self.status_label.setText(
            f"Generated {count} fragment{'s' if count != 1 else ''} \u2714"
            if count
            else "All fragments already generated."
        )

    def _on_generate_failed(self, error):
        self.btn_finalize.setEnabled(True)
        self.inspector.setEnabled(True)
        self._update_generate_banner()
        self.status_label.setText("")
        QMessageBox.warning(self, "Generate Scene", error)

    # -- selection / editing ---------------------------------------------------

    def _current_segment(self):
        if 0 <= self._selected_index < len(self.representation.segments):
            return self.representation.segments[self._selected_index]
        return None

    def _on_clip_selected(self, segment_index):
        self._selected_index = segment_index
        segment = self._current_segment()
        if segment is None:
            self.inspector.setVisible(False)
            return
        self.inspector.set_segment(segment)
        self.inspector.setVisible(True)

    def _on_fragment_edited(self):
        """Prompt/voice/speed edited: invalidate acceptance, save (debounced)."""
        segment = self._current_segment()
        if segment is None:
            return
        if segment.accepted and self.project:
            from core.audio_preview import segment_cache_key

            if segment_cache_key(self.project, segment) != segment.accepted_hash:
                segment.accepted = False
                segment.accepted_hash = ""
        self._save_timer.start()

    def _persist_and_refresh(self):
        self._save_representation()
        self.timeline.reload(self.representation)
        self._update_summary()
        self._update_generate_banner()

    def _save_representation(self):
        if self.project and self.scene_number is not None:
            audio_scene_service.save_representation(
                self.project,
                self.scene_id or "",
                self.scene_number,
                self.representation,
            )

    def _update_summary(self):
        if not self.project or self.scene_number is None:
            self.summary_label.setText("")
            return
        from core.scene_status import scene_status

        status = scene_status(self.project, self.scene_number, self.representation)
        self.summary_label.setText(
            f"{status.accepted}/{status.total} fragments accepted"
            + ("  \u2022  \U0001f7e2 finalized" if status.finalized else "")
        )

    # -- listen / regenerate / accept ------------------------------------------

    def _start_generation(self, force, on_done):
        segment = self._current_segment()
        if segment is None:
            return
        if not self.project or self.scene_number is None:
            QMessageBox.information(
                self, "Audio", "Save the scene first so audio can be cached."
            )
            return
        self.inspector.setEnabled(False)
        self.status_label.setText("Generating audio\u2026")
        self._preview_thread = _PreviewThread(
            self.project,
            self.scene_number,
            [(self._selected_index, segment)],
            force=force,
        )
        self._preview_thread.finished_ok.connect(on_done)
        self._preview_thread.failed.connect(self._on_generation_failed)
        self._preview_thread.start()

    def _on_generation_failed(self, error):
        self.inspector.setEnabled(True)
        self.status_label.setText("")
        QMessageBox.warning(self, "Audio", error)

    def _listen_fragment(self):
        self._save_timer.stop()
        self._save_representation()
        self._start_generation(force=False, on_done=self._on_listen_ready)

    def _regenerate_fragment(self):
        self._save_timer.stop()
        self._save_representation()
        self._start_generation(force=True, on_done=self._on_listen_ready)

    def _on_listen_ready(self, paths):
        self.inspector.setEnabled(True)
        self.status_label.setText("")
        self.timeline.reload(self.representation)
        self._update_generate_banner()
        if not paths:
            self.status_label.setText("Nothing to play (empty fragment).")
            return
        self._play_queue = list(paths)
        self._play_next()

    def _accept_fragment(self):
        self._save_timer.stop()
        self._start_generation(force=False, on_done=self._on_accept_ready)

    def _on_accept_ready(self, paths):
        self.inspector.setEnabled(True)
        self.status_label.setText("")
        segment = self._current_segment()
        if segment is None:
            return
        if not paths:
            self.status_label.setText("Nothing was generated (empty fragment).")
            return
        from core.audio_preview import segment_cache_key

        segment.accepted = True
        segment.accepted_hash = segment_cache_key(self.project, segment)
        self._save_representation()
        self.timeline.reload(self.representation)
        self.inspector.set_segment(segment)
        self._update_summary()
        self.status_label.setText("Fragment accepted \u2714")

    # -- finalize / play final ---------------------------------------------------

    def _final_wav_path(self):
        if self.scene_number is None or not self.project:
            return None
        from core.scene_finalizer import final_scene_path

        return final_scene_path(self.project, self.scene_number)

    def _update_final_buttons(self):
        path = self._final_wav_path()
        self.btn_play_final.setEnabled(bool(path and path.exists()))

    def _finalize_scene(self):
        if self.scene_number is None or not self.project:
            QMessageBox.information(
                self, "Finalize", "Save the scene first so audio can be cached."
            )
            return
        from core.scene_status import scene_status

        status = scene_status(self.project, self.scene_number, self.representation)
        if status.accepted < status.total:
            reply = QMessageBox.question(
                self,
                "Finalize Scene",
                f"{status.total - status.accepted} fragment(s) are not "
                "accepted yet. Finalize anyway (missing audio will be "
                "generated)?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        self._save_timer.stop()
        self._save_representation()
        self.btn_finalize.setEnabled(False)
        self.status_label.setText("Finalizing scene\u2026 (mixing speech + beds)")
        self._finalize_thread = _FinalizeThread(
            self.project, self.scene_number, self.representation
        )
        self._finalize_thread.finished_ok.connect(self._on_finalize_done)
        self._finalize_thread.failed.connect(self._on_finalize_failed)
        self._finalize_thread.start()

    def _on_finalize_done(self, path):
        self.btn_finalize.setEnabled(True)
        self.status_label.setText(f"Finalized: {path}")
        self._update_final_buttons()
        self._update_summary()

    def _on_finalize_failed(self, error):
        self.btn_finalize.setEnabled(True)
        self.status_label.setText("")
        QMessageBox.warning(self, "Finalize Scene", error)

    def _play_final(self):
        path = self._final_wav_path()
        if not path or not path.exists():
            QMessageBox.information(self, "Play Final", "No final mix yet.")
            return
        self._play_queue = [str(path)]
        self._play_next()

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

    # -- close ---------------------------------------------------------------

    def _save_and_accept(self):
        self._save_timer.stop()
        self._save_representation()
        self.accept()
