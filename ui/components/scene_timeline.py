"""Audacity-style multitrack timeline for a scene's audio.

One lane per voice (each character + narrator), plus Sound FX and Music
lanes. Clips are positioned by ``core.scene_timeline_model.compute_timeline``
(the same layout math as the exporter), rendered with per-pixel waveform
peaks, and can be dragged horizontally to override their start offset
(persisted on the segment as ``start_offset``).

Interactive: per-track mute/solo/volume, zoom (slider or Ctrl+wheel),
playback of the audible mix with an animated playhead, and ruler-click
seeking.
"""

import sys
import tempfile
import time
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core.audio_peaks import compute_peaks
from core.scene_timeline_model import TRACK_MUSIC, TRACK_SFX, compute_timeline
from core.timeline_mixer import TrackState, mix_timeline

RULER_H = 26
TRACK_H = 72
CLIP_PAD = 6
HEADER_W = 170
MIN_PX_PER_S = 8
MAX_PX_PER_S = 300
DEFAULT_PX_PER_S = 40

_TRACK_COLORS = {
    TRACK_SFX: QColor("#8d6e63"),
    TRACK_MUSIC: QColor("#7e57c2"),
}
_VOICE_COLOR = QColor("#1976d2")


def _clip_color(kind):
    return _TRACK_COLORS.get(kind, _VOICE_COLOR)


class _MixThread(QThread):
    """Run the ffmpeg timeline mix off the UI thread."""

    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, model, track_states, out_path, start_at):
        super().__init__()
        self.model = model
        self.track_states = track_states
        self.out_path = out_path
        self.start_at = start_at

    def run(self):
        try:
            path = mix_timeline(
                self.model,
                self.track_states,
                self.out_path,
                start_at=self.start_at,
            )
            self.finished_ok.emit(str(path))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class _TrackHeader(QWidget):
    """Fixed left column cell: track name + Mute/Solo + volume slider."""

    changed = Signal()

    def __init__(self, track_name, parent=None):
        super().__init__(parent)
        self.track_name = track_name
        self.setFixedSize(HEADER_W, TRACK_H)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        name = QLabel(track_name)
        name.setStyleSheet("font-weight: 600;")
        layout.addWidget(name)

        row = QHBoxLayout()
        self.btn_mute = QPushButton("M")
        self.btn_mute.setCheckable(True)
        self.btn_mute.setFixedWidth(26)
        self.btn_mute.setToolTip("Mute")
        self.btn_mute.toggled.connect(lambda *_: self.changed.emit())
        self.btn_solo = QPushButton("S")
        self.btn_solo.setCheckable(True)
        self.btn_solo.setFixedWidth(26)
        self.btn_solo.setToolTip("Solo")
        self.btn_solo.toggled.connect(lambda *_: self.changed.emit())
        self.volume = QSlider(Qt.Horizontal)
        self.volume.setRange(0, 150)
        self.volume.setValue(100)
        self.volume.setToolTip("Track volume (%)")
        self.volume.valueChanged.connect(lambda *_: self.changed.emit())
        row.addWidget(self.btn_mute)
        row.addWidget(self.btn_solo)
        row.addWidget(self.volume, 1)
        layout.addLayout(row)
        layout.addStretch()

    def state(self) -> TrackState:
        return TrackState(
            mute=self.btn_mute.isChecked(),
            solo=self.btn_solo.isChecked(),
            volume=self.volume.value() / 100.0,
        )


class _TimelineCanvas(QWidget):
    """The scrollable ruler + lanes + clips + playhead surface."""

    seek_requested = Signal(float)  # ruler click (seconds)
    clip_moved = Signal(int, float)  # (segment_index, new_start_s)
    clip_clicked = Signal(int)  # segment_index (selection)
    zoom_requested = Signal(int)  # wheel delta steps

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.px_per_s = DEFAULT_PX_PER_S
        self.playhead_s = 0.0
        self.selected_index = -1
        self._drag = None  # (track_i, clip, grab_dx_px)
        self.setMouseTracking(False)

    # -- geometry ---------------------------------------------------------

    def set_model(self, model):
        self.model = model
        self._resize_to_content()
        self.update()

    def set_zoom(self, px_per_s):
        self.px_per_s = max(MIN_PX_PER_S, min(MAX_PX_PER_S, px_per_s))
        self._resize_to_content()
        self.update()

    def set_playhead(self, seconds):
        self.playhead_s = max(0.0, seconds)
        self.update()

    def _resize_to_content(self):
        duration = self.model.duration if self.model else 0.0
        tracks = len(self.model.tracks) if self.model else 0
        width = int((duration + 5.0) * self.px_per_s) + 40
        height = RULER_H + max(tracks, 1) * TRACK_H
        self.setMinimumSize(max(width, 400), height)
        self.resize(max(width, 400), height)

    def _x(self, seconds):
        return int(seconds * self.px_per_s)

    def _clip_rect(self, track_i, clip):
        y = RULER_H + track_i * TRACK_H + CLIP_PAD
        h = TRACK_H - 2 * CLIP_PAD
        x = self._x(clip.start)
        w = max(12, self._x(clip.duration))
        return x, y, w, h

    # -- painting ---------------------------------------------------------

    def paintEvent(self, event):  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#fafafa"))
        self._paint_ruler(painter)
        if self.model:
            for track_i, track in enumerate(self.model.tracks):
                self._paint_track(painter, track_i, track)
        self._paint_playhead(painter)
        painter.end()

    def _paint_ruler(self, painter):
        painter.fillRect(0, 0, self.width(), RULER_H, QColor("#eceff1"))
        painter.setPen(QPen(QColor("#607d8b")))
        step = 1 if self.px_per_s >= 30 else 5
        seconds = 0
        while self._x(seconds) < self.width():
            x = self._x(seconds)
            painter.drawLine(x, RULER_H - 8, x, RULER_H)
            painter.drawText(x + 3, RULER_H - 10, f"{seconds}s")
            seconds += step

    def _paint_track(self, painter, track_i, track):
        y = RULER_H + track_i * TRACK_H
        painter.setPen(QPen(QColor("#e0e0e0")))
        painter.drawLine(0, y + TRACK_H, self.width(), y + TRACK_H)
        for clip in track.clips:
            self._paint_clip(painter, track_i, clip)

    def _paint_clip(self, painter, track_i, clip):
        x, y, w, h = self._clip_rect(track_i, clip)
        color = _clip_color(clip.kind)
        selected = clip.segment_index == self.selected_index
        if clip.missing:
            # Ghost: outlined, no waveform.
            pen = QPen(color)
            pen.setStyle(Qt.DashLine)
            if selected:
                pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(QColor(255, 255, 255, 0))
            painter.drawRoundedRect(x, y, w, h, 4, 4)
            painter.drawText(x + 6, y + h // 2 + 4, clip.label)
            return
        if clip.accepted:
            # Reviewed and locked in: green tint + border.
            border = QColor("#2e7d32")
            fill = QColor("#2e7d32")
            fill.setAlpha(45)
        else:
            border = QColor(color)
            fill = QColor(color)
            fill.setAlpha(50)
        pen = QPen(border)
        if selected:
            pen.setWidth(3)
        painter.setPen(pen)
        painter.setBrush(fill)
        painter.drawRoundedRect(x, y, w, h, 4, 4)
        self._paint_waveform(painter, clip, x, y, w, h, border)
        painter.setPen(QPen(QColor("#37474f")))
        label = ("\u2714 " if clip.accepted else "") + clip.label
        painter.drawText(x + 6, y + 14, label)

    def _paint_waveform(self, painter, clip, x, y, w, h, color):
        peaks = compute_peaks(clip.path, max(1, w // 2))
        if not peaks:
            return
        mid = y + h // 2
        half = (h // 2) - 4
        painter.setPen(QPen(color))
        px_per_peak = w / len(peaks)
        for i, (lo, hi) in enumerate(peaks):
            px = x + int(i * px_per_peak)
            painter.drawLine(px, mid - int(hi * half), px, mid - int(lo * half))

    def _paint_playhead(self, painter):
        x = self._x(self.playhead_s)
        painter.setPen(QPen(QColor("#e53935"), 2))
        painter.drawLine(x, 0, x, self.height())

    # -- interaction ------------------------------------------------------

    def _clip_at(self, pos):
        if not self.model:
            return None
        for track_i, track in enumerate(self.model.tracks):
            for clip in track.clips:
                x, y, w, h = self._clip_rect(track_i, clip)
                if x <= pos.x() <= x + w and y <= pos.y() <= y + h:
                    return track_i, clip
        return None

    def mousePressEvent(self, event):  # noqa: N802
        if event.position().y() <= RULER_H:
            self.seek_requested.emit(max(0.0, event.position().x() / self.px_per_s))
            return
        hit = self._clip_at(event.position())
        if hit:
            track_i, clip = hit
            self.selected_index = clip.segment_index
            self.clip_clicked.emit(clip.segment_index)
            grab_dx = event.position().x() - self._x(clip.start)
            self._drag = (track_i, clip, grab_dx)
            self.update()

    def mouseMoveEvent(self, event):  # noqa: N802
        if not self._drag:
            return
        _track_i, clip, grab_dx = self._drag
        clip.start = max(0.0, (event.position().x() - grab_dx) / self.px_per_s)
        self.update()

    def mouseReleaseEvent(self, event):  # noqa: N802
        if not self._drag:
            return
        _track_i, clip, _grab_dx = self._drag
        self._drag = None
        self.clip_moved.emit(clip.segment_index, clip.start)

    def wheelEvent(self, event):  # noqa: N802
        if event.modifiers() & Qt.ControlModifier:
            self.zoom_requested.emit(1 if event.angleDelta().y() > 0 else -1)
            event.accept()
        else:
            event.ignore()


class SceneTimelinePanel(QWidget):
    """Multitrack timeline panel for one scene."""

    representation_changed = Signal()  # start_offset persisted
    clip_selected = Signal(int)  # segment index clicked on the timeline

    def __init__(self, project, scene_number, representation, parent=None):
        super().__init__(parent)
        self.project = project
        self.scene_number = scene_number
        self.representation = representation
        self.model = None
        self._headers = []
        self._mix_thread = None
        self._player = None
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(33)
        self._play_timer.timeout.connect(self._advance_playhead)
        self._play_started_at = 0.0
        self._play_from_s = 0.0
        self._tmp_mix = None

        self._build_ui()
        self.reload()

    # -- UI ----------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        toolbar = QHBoxLayout()
        self.btn_play = QPushButton("\u25b6 Play")
        self.btn_play.clicked.connect(self._play)
        self.btn_stop = QPushButton("\u25a0 Stop")
        self.btn_stop.clicked.connect(self._stop)
        self.btn_stop.setEnabled(False)
        toolbar.addWidget(self.btn_play)
        toolbar.addWidget(self.btn_stop)
        toolbar.addSpacing(20)
        toolbar.addWidget(QLabel("Zoom:"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(MIN_PX_PER_S, MAX_PX_PER_S)
        self.zoom_slider.setValue(DEFAULT_PX_PER_S)
        self.zoom_slider.setFixedWidth(160)
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider)
        toolbar.addWidget(self.zoom_slider)
        self.status = QLabel("")
        self.status.setStyleSheet("color: #4CAF50; font-style: italic;")
        toolbar.addWidget(self.status, 1)
        layout.addLayout(toolbar)

        body = QHBoxLayout()
        body.setSpacing(0)

        header_col = QWidget()
        self._header_layout = QVBoxLayout(header_col)
        self._header_layout.setContentsMargins(0, RULER_H, 0, 0)
        self._header_layout.setSpacing(0)
        header_col.setFixedWidth(HEADER_W)
        body.addWidget(header_col, 0, Qt.AlignTop)

        self.canvas = _TimelineCanvas()
        self.canvas.seek_requested.connect(self._on_seek)
        self.canvas.clip_moved.connect(self._on_clip_moved)
        self.canvas.clip_clicked.connect(self.clip_selected)
        self.canvas.zoom_requested.connect(self._on_zoom_wheel)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.canvas)
        self.scroll.setWidgetResizable(False)
        body.addWidget(self.scroll, 1)

        layout.addLayout(body, 1)

    def reload(self, representation=None):
        """Recompute the layout model and rebuild track headers."""
        if representation is not None:
            self.representation = representation
        self.model = compute_timeline(
            self.project, self.scene_number, self.representation
        )
        # Rebuild headers.
        while self._header_layout.count():
            item = self._header_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._headers = []
        for track in self.model.tracks:
            header = _TrackHeader(track.name)
            self._headers.append(header)
            self._header_layout.addWidget(header)
        self._header_layout.addStretch()
        self.canvas.set_model(self.model)

    def track_states(self):
        return {h.track_name: h.state() for h in self._headers}

    # -- zoom / seek / drag --------------------------------------------------

    def _on_zoom_slider(self, value):
        self.canvas.set_zoom(value)

    def _on_zoom_wheel(self, steps):
        self.zoom_slider.setValue(self.zoom_slider.value() + steps * 8)

    def _on_seek(self, seconds):
        self._play_from_s = seconds
        self.canvas.set_playhead(seconds)

    def _on_clip_moved(self, segment_index, new_start):
        """Persist a manual start offset on drop."""
        if not (0 <= segment_index < len(self.representation.segments)):
            return
        segment = self.representation.segments[segment_index]
        segment.start_offset = round(float(new_start), 3)
        try:
            from services.audio_scene_service import audio_scene_service

            audio_scene_service.save_representation(
                self.project,
                self.representation.scene_id or "",
                self.scene_number,
                self.representation,
            )
        except Exception:  # noqa: BLE001
            self.status.setText("Warning: could not save clip offset")
        self.reload()
        self.representation_changed.emit()

    # -- playback -------------------------------------------------------------

    def _play(self):
        if self.model is None or not self.model.all_clips():
            self.status.setText("Nothing to play.")
            return
        self.btn_play.setEnabled(False)
        self.status.setText("Mixing\u2026")
        tmp = tempfile.NamedTemporaryFile(
            prefix="timeline_mix_", suffix=".wav", delete=False
        )
        tmp.close()
        self._tmp_mix = Path(tmp.name)
        self._mix_thread = _MixThread(
            self.model, self.track_states(), self._tmp_mix, self._play_from_s
        )
        self._mix_thread.finished_ok.connect(self._on_mix_ready)
        self._mix_thread.failed.connect(self._on_mix_failed)
        self._mix_thread.start()

    def _on_mix_ready(self, path):
        self.status.setText("Playing\u2026")
        self.btn_stop.setEnabled(True)
        player = "afplay" if sys.platform == "darwin" else "aplay"
        self._player = QProcess(self)
        self._player.finished.connect(lambda *_: self._on_play_finished())
        self._player.start(player, [path])
        self._play_started_at = time.monotonic()
        self._play_timer.start()

    def _on_mix_failed(self, error):
        self.btn_play.setEnabled(True)
        self.status.setText(f"Mix failed: {error}")

    def _advance_playhead(self):
        elapsed = time.monotonic() - self._play_started_at
        self.canvas.set_playhead(self._play_from_s + elapsed)

    def _on_play_finished(self):
        self._play_timer.stop()
        self.btn_play.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status.setText("")
        self.canvas.set_playhead(self._play_from_s)
        if self._tmp_mix:
            self._tmp_mix.unlink(missing_ok=True)
            self._tmp_mix = None

    def _stop(self):
        if self._player is not None:
            self._player.kill()

    def closeEvent(self, event):  # noqa: N802
        self._stop()
        super().closeEvent(event)
