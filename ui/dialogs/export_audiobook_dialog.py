"""
Export Audiobook dialog: select chapters, render every segment (voices +
sound/music cues) and join them into one final audio file with ffmpeg.
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QComboBox,
    QCheckBox,
    QFileDialog,
    QMessageBox,
    QPlainTextEdit,
)
from PySide6.QtCore import Qt, QThread, Signal

from services.database.manuscript_service import manuscript_service, SUPPORTED_LOCALES


class _ExportThread(QThread):
    progressed = Signal(str)
    finished_ok = Signal(object)  # ExportResult
    failed = Signal(str)

    def __init__(self, project, chapter_ids, output_path, locale, include_cues):
        super().__init__()
        self.project = project
        self.chapter_ids = chapter_ids
        self.output_path = output_path
        self.locale = locale
        self.include_cues = include_cues

    def run(self):
        try:
            from core.audiobook_exporter import export_chapters

            result = export_chapters(
                self.project,
                self.chapter_ids,
                self.output_path,
                locale=self.locale,
                include_cues=self.include_cues,
                progress=self.progressed.emit,
            )
            if result.error:
                self.failed.emit(result.error)
            else:
                self.finished_ok.emit(result)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class ExportAudiobookDialog(QDialog):
    def __init__(self, parent=None, project=None, default_locale="en"):
        super().__init__(parent)
        self.project = project
        self._thread = None
        self.setWindowTitle("Export Audiobook")
        self.setMinimumSize(560, 520)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Select the chapters to export. Voices, sound effects and music "
                "cues are generated (cached lines are reused) and joined into one "
                "audio file."
            )
        )

        # Chapter checklist
        self.chapter_list = QListWidget()
        chapters = manuscript_service.list_chapters(project)
        for ch in chapters:
            compiled = ch["compiled_scenes"]
            locales = ", ".join(sorted(compiled)) if compiled else "not compiled"
            item = QListWidgetItem(f"{ch['position']}. {ch['title']}  [{locales}]")
            item.setData(Qt.UserRole, ch)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if compiled else Qt.Unchecked)
            self.chapter_list.addItem(item)
        layout.addWidget(self.chapter_list)

        row = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.clicked.connect(lambda: self._set_all(Qt.Checked))
        btn_none = QPushButton("Select None")
        btn_none.clicked.connect(lambda: self._set_all(Qt.Unchecked))
        row.addWidget(btn_all)
        row.addWidget(btn_none)
        row.addStretch()

        row.addWidget(QLabel("Language:"))
        self.locale_combo = QComboBox()
        for code, label in SUPPORTED_LOCALES:
            self.locale_combo.addItem(label, code)
        idx = self.locale_combo.findData(default_locale)
        if idx >= 0:
            self.locale_combo.setCurrentIndex(idx)
        row.addWidget(self.locale_combo)
        layout.addLayout(row)

        self.cues_check = QCheckBox("Generate sound effects and music cues (MusicGen)")
        self.cues_check.setChecked(True)
        layout.addWidget(self.cues_check)

        # Progress log
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Export progress will appear here…")
        layout.addWidget(self.log)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.reject)
        self.btn_export = QPushButton("Export\u2026")
        self.btn_export.clicked.connect(self._export)
        btn_row.addWidget(self.btn_close)
        btn_row.addWidget(self.btn_export)
        layout.addLayout(btn_row)

    def _set_all(self, state):
        for i in range(self.chapter_list.count()):
            self.chapter_list.item(i).setCheckState(state)

    def _checked_chapter_ids(self):
        ids = []
        for i in range(self.chapter_list.count()):
            item = self.chapter_list.item(i)
            if item.checkState() == Qt.Checked:
                ids.append(item.data(Qt.UserRole)["id"])
        return ids

    def _export(self):
        chapter_ids = self._checked_chapter_ids()
        if not chapter_ids:
            QMessageBox.information(self, "Export", "Select at least one chapter.")
            return

        locale = self.locale_combo.currentData()
        default_name = f"{self.project}_{locale}.m4a"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Audiobook As",
            str(Path.home() / default_name),
            "Audio files (*.m4a *.mp3 *.wav)",
        )
        if not output_path:
            return

        self.btn_export.setEnabled(False)
        self.log.clear()
        self.log.appendPlainText(
            "Starting export\u2026 (voices load on first use, be patient)"
        )
        self._thread = _ExportThread(
            self.project,
            chapter_ids,
            output_path,
            locale,
            self.cues_check.isChecked(),
        )
        self._thread.progressed.connect(self.log.appendPlainText)
        self._thread.finished_ok.connect(self._on_done)
        self._thread.failed.connect(self._on_failed)
        self._thread.start()

    def _on_done(self, result):
        self.btn_export.setEnabled(True)
        msg = (
            f"Done!\n\nFile: {result.output_path}\n"
            f"Voice segments: {result.segments_rendered}\n"
            f"Sound/music cues: {result.cues_rendered}"
        )
        if result.skipped:
            msg += "\n\nSkipped:\n" + "\n".join(f"- {s}" for s in result.skipped[:10])
        self.log.appendPlainText(msg)
        QMessageBox.information(self, "Export Audiobook", msg)

    def _on_failed(self, error):
        self.btn_export.setEnabled(True)
        self.log.appendPlainText(f"FAILED: {error}")
        QMessageBox.critical(self, "Export Audiobook", error)
