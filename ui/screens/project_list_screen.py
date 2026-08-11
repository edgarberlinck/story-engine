"""
Project list screen - initial screen with clickable project panels.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QGridLayout,
    QLabel, QScrollArea, QMessageBox,
)
from PySide6.QtCore import Qt, Signal

from ui.components.project_card import ProjectCard
from core.project_manager import project_manager


class ProjectListScreen(QWidget):
    project_selected = Signal(str)  # project_id

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header with New Project button
        header = QHBoxLayout()
        title = QLabel("Projects")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        btn_new = QPushButton("+ New Project")
        btn_new.clicked.connect(self.create_project)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(btn_new)
        layout.addLayout(header)

        self.empty_label = QLabel("No projects yet. Click \u201c+ New Project\u201d to get started.")
        self.empty_label.setStyleSheet("color: #999; font-size: 14px; padding: 30px;")
        self.empty_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.empty_label)

        # Scrollable grid for project cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        grid_host = QWidget()
        self.grid = QGridLayout(grid_host)
        self.grid.setSpacing(15)
        self.grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll.setWidget(grid_host)
        layout.addWidget(scroll)

        self.load_projects()

    def load_projects(self):
        # Clear existing cards
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        projects = project_manager.list_projects()
        self.empty_label.setVisible(not projects)

        for idx, proj in enumerate(projects):
            card = ProjectCard(proj)
            card.clicked.connect(self.project_selected.emit)
            card.edit_requested.connect(self.edit_project)
            card.delete_requested.connect(self.delete_project)
            self.grid.addWidget(card, idx // 3, idx % 3)

    def refresh(self):
        self.load_projects()

    def create_project(self):
        from ui.dialogs.project_dialog import ProjectDialog
        dialog = ProjectDialog(self)
        if dialog.exec():
            data = dialog.get_data()
            if not data["name"].strip():
                QMessageBox.warning(self, "Invalid name", "Project name is required.")
                return
            project_manager.create_project(data["name"].strip(), data["description"])
            self.load_projects()

    def edit_project(self, project):
        from ui.dialogs.project_dialog import ProjectDialog
        dialog = ProjectDialog(self, project=project)
        if dialog.exec():
            data = dialog.get_data()
            if not data["name"].strip():
                QMessageBox.warning(self, "Invalid name", "Project name is required.")
                return
            project_manager.update_project(project["id"], data["name"].strip(), data["description"])
            self.load_projects()

    def delete_project(self, project):
        reply = QMessageBox.question(
            self,
            "Delete Project",
            f"Delete project \u201c{project['name']}\u201d and all of its data?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            project_manager.delete_project(project["id"])
            self.load_projects()
