"""
Writing tab: the book-writing layer ported from the writing-tools app.

Left: ordered chapter list (typed, movable). Right: per-language editor tabs
(EN, PT-BR, ES, FR, DE) with instant autosave, writing in story markup that
compiles into audio scenes.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QListWidget,
    QListWidgetItem, QPlainTextEdit, QTabWidget, QComboBox, QSplitter,
    QMessageBox, QInputDialog,
)
from PySide6.QtCore import Qt

from services.database.manuscript_service import (
    manuscript_service, CHAPTER_TYPES, SUPPORTED_LOCALES,
)

MARKUP_HELP = """\
Story markup quick reference — both styles work, even mixed:

Bracket shorthand:
  [Narrator] Some narration text.
  [Nikita] Hello! [Feeling=Angry] Now I'm angry. [Feeling=calm]
  [Sound=door slamming]  [Music=soft strings]
  Attributes: [Feeling=..] [Tone=..] [Delivery=..] [Intensity=0.0-1.0] [Voice=..]

Markup tags (nestable, more control):
  <scene title="The Morning">
    <sound preset="adventure" />
    <character name="narrator" tone="neutral">Your story begins...</character>
    <character name="nikita" emotion="warm" intensity="0.3">Good morning.</character>
    <music prompt="soft strings" />
  </scene>

Each <scene> (or the whole text, if you use no scene tags) becomes one audio
scene you can refine in the Scenes tab and preview line by line."""


class WritingTab(QWidget):
    def __init__(self, project_slug, parent=None, on_compiled=None):
        super().__init__(parent)
        self.project_slug = project_slug
        self.on_compiled = on_compiled
        self.current_chapter_id = None
        self._loading = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        splitter = QSplitter(Qt.Horizontal)

        # -- Left: chapters ---------------------------------------------------
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(QLabel("Chapters"))
        self.chapter_list = QListWidget()
        self.chapter_list.currentItemChanged.connect(self._chapter_selected)
        left_layout.addWidget(self.chapter_list)

        self.type_combo = QComboBox()
        self.type_combo.addItems(CHAPTER_TYPES)
        self.type_combo.currentTextChanged.connect(self._type_changed)
        left_layout.addWidget(self.type_combo)

        row1 = QHBoxLayout()
        btn_add = QPushButton("+ Chapter")
        btn_add.clicked.connect(self._add_chapter)
        btn_rename = QPushButton("Rename")
        btn_rename.clicked.connect(self._rename_chapter)
        btn_delete = QPushButton("Delete")
        btn_delete.clicked.connect(self._delete_chapter)
        row1.addWidget(btn_add)
        row1.addWidget(btn_rename)
        row1.addWidget(btn_delete)
        left_layout.addLayout(row1)

        row2 = QHBoxLayout()
        btn_up = QPushButton("\u2191 Up")
        btn_up.clicked.connect(lambda: self._move_chapter(-1))
        btn_down = QPushButton("\u2193 Down")
        btn_down.clicked.connect(lambda: self._move_chapter(1))
        row2.addWidget(btn_up)
        row2.addWidget(btn_down)
        left_layout.addLayout(row2)

        splitter.addWidget(left)

        # -- Right: language editors -------------------------------------------
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QHBoxLayout()
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #4CAF50; font-style: italic;")
        toolbar.addWidget(self.status_label, 1)
        btn_help = QPushButton("Markup Help")
        btn_help.clicked.connect(self._show_help)
        self.btn_compile = QPushButton("\u266a Compile to Audio Scenes")
        self.btn_compile.clicked.connect(self._compile_chapter)
        self.btn_export = QPushButton("\U0001f3a7 Export Audiobook\u2026")
        self.btn_export.clicked.connect(self._export_audiobook)
        toolbar.addWidget(btn_help)
        toolbar.addWidget(self.btn_compile)
        toolbar.addWidget(self.btn_export)
        right_layout.addLayout(toolbar)

        self.lang_tabs = QTabWidget()
        self.editors = {}
        for code, label in SUPPORTED_LOCALES:
            editor = QPlainTextEdit()
            editor.setPlaceholderText(
                "Write your story here.\n\n"
                "[Narrator] It was a quiet morning...\n"
                "[Nikita] Good morning! [Feeling=warm]\n\n"
                "Or use markup: <scene><character name=\"nikita\">...</character></scene>"
            )
            editor.textChanged.connect(
                lambda c=code: self._content_changed(c)
            )
            self.editors[code] = editor
            self.lang_tabs.addTab(editor, label)
        right_layout.addWidget(self.lang_tabs)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)

        self.load_chapters()

    # -- Chapter list ---------------------------------------------------------

    def load_chapters(self):
        self._loading = True
        try:
            self.chapter_list.clear()
            for ch in manuscript_service.list_chapters(self.project_slug):
                item = QListWidgetItem(f"{ch['position']}. {ch['title']}")
                item.setData(Qt.UserRole, ch)
                self.chapter_list.addItem(item)
        finally:
            self._loading = False
        if self.chapter_list.count() and self.current_chapter_id is None:
            self.chapter_list.setCurrentRow(0)

    def _current_chapter(self):
        item = self.chapter_list.currentItem()
        return item.data(Qt.UserRole) if item else None

    def _chapter_selected(self, item, _prev=None):
        if self._loading or not item:
            return
        ch = item.data(Qt.UserRole)
        self.current_chapter_id = ch["id"]
        self._loading = True
        try:
            contents = manuscript_service.get_contents(ch["id"])
            for code, editor in self.editors.items():
                editor.setPlainText(contents.get(code, ""))
            idx = self.type_combo.findText(ch["chapter_type"])
            if idx >= 0:
                self.type_combo.setCurrentIndex(idx)
        finally:
            self._loading = False

    def _add_chapter(self):
        title, ok = QInputDialog.getText(self, "New Chapter", "Chapter title:")
        if not ok or not title.strip():
            return
        chapter_id = manuscript_service.create_chapter(self.project_slug, title.strip())
        self.current_chapter_id = None
        self.load_chapters()
        # Select the new chapter (last row).
        self.chapter_list.setCurrentRow(self.chapter_list.count() - 1)
        self.current_chapter_id = chapter_id

    def _rename_chapter(self):
        ch = self._current_chapter()
        if not ch:
            return
        title, ok = QInputDialog.getText(
            self, "Rename Chapter", "Chapter title:", text=ch["title"]
        )
        if not ok or not title.strip():
            return
        manuscript_service.rename_chapter(ch["id"], title.strip())
        row = self.chapter_list.currentRow()
        self.load_chapters()
        self.chapter_list.setCurrentRow(row)

    def _delete_chapter(self):
        ch = self._current_chapter()
        if not ch:
            return
        reply = QMessageBox.question(
            self, "Delete Chapter",
            f"Delete chapter \"{ch['title']}\" and all its language contents?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        manuscript_service.delete_chapter(ch["id"])
        self.current_chapter_id = None
        for editor in self.editors.values():
            editor.clear()
        self.load_chapters()

    def _move_chapter(self, direction):
        ch = self._current_chapter()
        if not ch:
            return
        if manuscript_service.move_chapter(self.project_slug, ch["id"], direction):
            row = self.chapter_list.currentRow() + direction
            self.load_chapters()
            self.chapter_list.setCurrentRow(row)

    def _type_changed(self, chapter_type):
        if self._loading:
            return
        ch = self._current_chapter()
        if ch and chapter_type:
            manuscript_service.set_chapter_type(ch["id"], chapter_type)

    # -- Editing ---------------------------------------------------------------

    def _content_changed(self, locale):
        """Instant autosave, like writing-tools: every keystroke persists."""
        if self._loading or self.current_chapter_id is None:
            return
        manuscript_service.save_content(
            self.current_chapter_id, locale, self.editors[locale].toPlainText()
        )

    def _current_locale(self):
        return SUPPORTED_LOCALES[self.lang_tabs.currentIndex()][0]

    # -- Compilation -------------------------------------------------------------

    def _compile_chapter(self):
        ch = self._current_chapter()
        if not ch:
            QMessageBox.information(self, "Compile", "Select a chapter first.")
            return
        locale = self._current_locale()

        from core.story_compiler import compile_chapter
        result = compile_chapter(self.project_slug, ch["id"], locale)

        if result.error:
            QMessageBox.warning(self, "Compile", result.error)
            return

        msg = (
            f"Compiled into {len(result.scene_numbers)} audio scene(s): "
            f"{', '.join(str(n) for n in result.scene_numbers)}.\n"
            "Open the Scenes tab to refine and preview them."
        )
        if result.issues:
            msg += "\n\nWarnings:\n" + "\n".join(
                f"- {i.message}" for i in result.issues[:8]
            )
        QMessageBox.information(self, "Compile", msg)
        self.status_label.setText(
            f"Chapter compiled \u2192 scenes {', '.join(str(n) for n in result.scene_numbers)}"
        )
        if self.on_compiled:
            self.on_compiled(result.scene_numbers)

    def _export_audiobook(self):
        from ui.dialogs.export_audiobook_dialog import ExportAudiobookDialog
        dialog = ExportAudiobookDialog(
            self, project=self.project_slug, default_locale=self._current_locale()
        )
        dialog.exec()

    def _show_help(self):
        QMessageBox.information(self, "Story Markup", MARKUP_HELP)
