from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QScrollArea, QVBoxLayout, QWidget

from betternte.models import ROI


def ndarray_to_qpixmap(image: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    qimage = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


class RectangleCanvas(QWidget):
    selection_changed = Signal(QRect)

    def __init__(self, frame: np.ndarray, initial_roi: ROI | None = None) -> None:
        super().__init__()
        self._pixmap = ndarray_to_qpixmap(frame)
        self._start: QPoint | None = None
        self._current_rect = QRect()
        if initial_roi and initial_roi.valid():
            self._current_rect = QRect(initial_roi.x, initial_roi.y, initial_roi.w, initial_roi.h)
        self.setMinimumSize(self._pixmap.size())
        self.setMaximumSize(self._pixmap.size())

    def current_rect(self) -> QRect:
        return self._current_rect.normalized()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._start = event.position().toPoint()
        self._current_rect = QRect(self._start, self._start)
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._start is None:
            return
        self._current_rect = QRect(self._start, event.position().toPoint()).normalized()
        self.selection_changed.emit(self._current_rect)
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._start is None:
            return
        self._current_rect = QRect(self._start, event.position().toPoint()).normalized()
        self._start = None
        self.selection_changed.emit(self._current_rect)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._pixmap)
        if self._current_rect.isValid():
            painter.setPen(QPen(QColor("#00d084"), 2))
            painter.drawRect(self._current_rect)


class PointCanvas(QWidget):
    point_selected = Signal(int, int)

    def __init__(self, frame: np.ndarray) -> None:
        super().__init__()
        self._pixmap = ndarray_to_qpixmap(frame)
        self._point: QPoint | None = None
        self.setMinimumSize(self._pixmap.size())
        self.setMaximumSize(self._pixmap.size())

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._point = event.position().toPoint()
        self.point_selected.emit(self._point.x(), self._point.y())
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._pixmap)
        if self._point is not None:
            painter.setPen(QPen(QColor("#ffd400"), 2))
            painter.drawEllipse(self._point, 3, 3)


class ROISelectionDialog(QDialog):
    def __init__(self, frame: np.ndarray, title: str, initial_roi: ROI | None = None) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1280, 900)
        self.canvas = RectangleCanvas(frame, initial_roi)
        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(False)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("拖拽鼠标框选区域，然后点击确定。"))
        layout.addWidget(scroll)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_roi(self, name: str) -> ROI:
        rect = self.canvas.current_rect()
        return ROI(name=name, x=rect.x(), y=rect.y(), w=rect.width(), h=rect.height())


class PointSampleDialog(QDialog):
    def __init__(self, frame: np.ndarray, title: str) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1280, 900)
        self._point: tuple[int, int] | None = None
        self.canvas = PointCanvas(frame)
        self.canvas.point_selected.connect(self._set_point)
        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(False)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("点击截图中的颜色样本点，然后点击确定。"))
        layout.addWidget(scroll)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _set_point(self, x: int, y: int) -> None:
        self._point = (x, y)

    def selected_point(self) -> tuple[int, int] | None:
        return self._point