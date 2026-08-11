"""
Clickable project card component.
"""

from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel
from PySide6.QtCore import Signal, Qt


class ProjectCard(QFrame):
    clicked = Signal(str)          # project_id
    edit_requested = Signal(dict)  # project dict
    delete_requested = Signal(dict)

    def __init__(self, project):
        super().__init__()
        self.project = project
        self.setFixedSize(260, 150)
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("projectCard")
        self.setStyleSheet("""
            QFrame#projectCard {
                border: 2px solid #ddd;
                border-radius: 8px;
                background-color: #fafafa;
            }
            QFrame#projectCard:hover {
                border: 2px solid #4CAF50;
                background-color: #f0f8f0;
            }
            QLabel { border: none; background: transparent; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)

        name_label = QLabel(project["name"])
        name_label.setStyleSheet("font-weight: bold; font-size: 15px;")
        name_label.setWordWrap(True)

        desc = project.get("description") or ""
        if len(desc) > 100:
            desc = desc[:100] + "…"
        desc_label = QLabel(desc)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("font-size: 11px; color: #666;")

        created = str(project.get("created_at", ""))[:10]
        date_label = QLabel(f"Created: {created}" if created else "")
        date_label.setStyleSheet("font-size: 10px; color: #999;")

        layout.addWidget(name_label)
        layout.addWidget(desc_label)
        layout.addStretch()
        layout.addWidget(date_label)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.project["id"])
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        edit_action = menu.addAction("Edit Project")
        delete_action = menu.addAction("Delete Project")
        action = menu.exec(event.globalPos())
        if action == edit_action:
            self.edit_requested.emit(self.project)
        elif action == delete_action:
            self.delete_requested.emit(self.project)
