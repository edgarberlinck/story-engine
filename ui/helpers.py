"""
Shared UI helpers.
"""

import shutil
from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMessageBox


def export_image(parent, source_path, suggested_name):
    """Export a full-resolution image to a user-chosen location."""
    if not source_path or not Path(source_path).is_file():
        QMessageBox.warning(parent, "Export", "No image available to export.")
        return

    ext = Path(source_path).suffix or ".png"
    dest, _ = QFileDialog.getSaveFileName(
        parent,
        "Export Image",
        str(Path.home() / f"{suggested_name}{ext}"),
        f"Images (*{ext});;All files (*)",
    )
    if not dest:
        return
    if not Path(dest).suffix:
        dest += ext
    try:
        shutil.copy2(source_path, dest)
        QMessageBox.information(parent, "Export", f"Exported to:\n{dest}")
    except OSError as e:
        QMessageBox.critical(parent, "Export failed", str(e))
