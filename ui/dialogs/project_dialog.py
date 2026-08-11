"""
Dialog for creating/editing projects.
"""

from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit


class ProjectDialog(QDialog):
    def __init__(self, parent=None, project=None):
        super().__init__(parent)
        self.setWindowTitle("New Project" if not project else "Edit Project")
        self.setMinimumWidth(400)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        layout.addWidget(self.name_edit)

        layout.addWidget(QLabel("Description:"))
        self.desc_edit = QTextEdit()
        layout.addWidget(self.desc_edit)

        if project:
            self.name_edit.setText(project.get("name", ""))
            self.desc_edit.setPlainText(project.get("description", ""))

        buttons = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_ok = QPushButton("Save")
        btn_ok.setDefault(True)
        buttons.addStretch()
        buttons.addWidget(btn_cancel)
        buttons.addWidget(btn_ok)
        layout.addLayout(buttons)

        btn_cancel.clicked.connect(self.reject)
        btn_ok.clicked.connect(self.accept)

    def get_data(self):
        return {
            "name": self.name_edit.text(),
            "description": self.desc_edit.toPlainText()
        }
