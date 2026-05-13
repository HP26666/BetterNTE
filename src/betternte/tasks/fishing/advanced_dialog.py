from __future__ import annotations

import cv2
import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from betternte.core.models import HSVThreshold, ROI
from betternte.core.vision.hsv import crop_global, threshold_mask
from betternte.gui.components import Card
from betternte.tasks.fishing.models import (
    AppConfig,
    DEFAULT_BAR_THRESHOLD,
    DEFAULT_BLUE_CIRCLE_THRESHOLD,
    DEFAULT_CAPTURE_FPS,
    DEFAULT_DOT_THRESHOLD,
)
from betternte.tasks.fishing.vision import analyze_frame, detect_bar, detect_dot


class _PreviewLabel(QLabel):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(280, 180)
        self.setWordWrap(True)
        self.setText(title)
        self.setStyleSheet(
            "background:#f8fafc; border:1px dashed #cbd5e1;"
            "border-radius:8px; color:#94a3b8; font-size:12px;"
        )


def _to_pixmap(image: np.ndarray, target_size: tuple[int, int]) -> QPixmap:
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    qimage = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimage).scaled(
        target_size[0],
        target_size[1],
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class _ThresholdPreviewPanel(QWidget):
    def __init__(
        self,
        config: AppConfig,
        preview_frame: np.ndarray | None,
        preview_source_roi: ROI | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config.clone()
        self._frame = preview_frame.copy() if preview_frame is not None else None
        self._source_roi = preview_source_roi or ROI(name="screen", x=0, y=0, w=0, h=0)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        card = Card("实时阈值预览")
        body = card.layout()

        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        lbl = QLabel("预览目标")
        lbl.setStyleSheet("color:#334155; font-size:12px;")
        self._target_combo = QComboBox()
        self._target_combo.addItem("绿条", "bar")
        self._target_combo.addItem("黄线", "dot")
        self._target_combo.addItem("蓝圈", "blue_circle")
        self._target_combo.currentIndexChanged.connect(self._update_preview)
        top_row.addWidget(lbl)
        top_row.addWidget(self._target_combo, stretch=1)
        body.addLayout(top_row)

        self._hint_label = QLabel(
            "预览使用当前钓鱼截图或上传图片，修改 HSV 后会即时刷新黄线、绿条和蓝圈结果。"
        )
        self._hint_label.setWordWrap(True)
        self._hint_label.setStyleSheet("color:#64748b; font-size:11px;")
        body.addWidget(self._hint_label)

        image_row = QHBoxLayout()
        image_row.setSpacing(8)

        left_col = QVBoxLayout()
        left_title = QLabel("ROI 原图")
        left_title.setStyleSheet("color:#334155; font-size:11px; font-weight:700;")
        self._source_label = _PreviewLabel("等待预览")
        left_col.addWidget(left_title)
        left_col.addWidget(self._source_label, stretch=1)
        image_row.addLayout(left_col, stretch=1)

        right_col = QVBoxLayout()
        right_title = QLabel("阈值掩码")
        right_title.setStyleSheet("color:#334155; font-size:11px; font-weight:700;")
        self._mask_label = _PreviewLabel("等待预览")
        right_col.addWidget(right_title)
        right_col.addWidget(self._mask_label, stretch=1)
        image_row.addLayout(right_col, stretch=1)

        body.addLayout(image_row)

        self._stats_label = QLabel("-")
        self._stats_label.setWordWrap(True)
        self._stats_label.setStyleSheet(
            "background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;"
            "padding:8px; color:#475569; font-size:11px;"
        )
        body.addWidget(self._stats_label)

        root.addWidget(card)
        self._update_preview()

    def update_config(self, config: AppConfig) -> None:
        self._config = config.clone()
        self._update_preview()

    def focus_target(self, target_key: str) -> None:
        for index in range(self._target_combo.count()):
            if self._target_combo.itemData(index) == target_key:
                self._target_combo.setCurrentIndex(index)
                return

    def _show_placeholder(self, text: str) -> None:
        self._source_label.setText(text)
        self._source_label.setPixmap(QPixmap())
        self._mask_label.setText(text)
        self._mask_label.setPixmap(QPixmap())
        self._stats_label.setText(text)

    def _preview_target(self) -> tuple[str, ROI, HSVThreshold, bool]:
        key = str(self._target_combo.currentData())
        if key == "blue_circle":
            return key, self._config.rois.blue_circle_area, self._config.hsv_thresholds.blue_circle, True
        if key == "dot":
            return key, self._config.rois.bar_area, self._config.hsv_thresholds.dot, False
        return key, self._config.rois.bar_area, self._config.hsv_thresholds.bar, True

    def _update_preview(self) -> None:
        target_key, search_roi, threshold, morph = self._preview_target()
        if self._frame is None:
            self._show_placeholder("暂无截图可预览，请先上传截图或截取屏幕。")
            return
        if not search_roi.valid():
            missing_text = "请先框选对应 ROI，实时预览才会有意义。"
            self._show_placeholder(missing_text)
            return

        view, relative = crop_global(self._frame, self._source_roi, search_roi)
        if view is None or relative is None or view.size == 0:
            self._show_placeholder("当前 ROI 超出截图范围，无法生成预览。")
            return

        mask = threshold_mask(view, threshold, morph=morph)
        if target_key == "bar":
            bridge_kernel = np.ones((3, 9), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, bridge_kernel)

        observation = analyze_frame(self._frame, self._source_roi, self._config)
        raw_dot = None
        filtered_dot = None
        if target_key == "dot":
            bar = detect_bar(
                self._frame,
                self._source_roi,
                self._config.rois.bar_area,
                self._config.hsv_thresholds.bar,
                self._config.control.min_area,
            )
            raw_dot = detect_dot(
                self._frame,
                self._source_roi,
                self._config.rois.bar_area,
                self._config.hsv_thresholds.dot,
                self._config.control.min_area,
                bar=None,
            )
            filtered_dot = detect_dot(
                self._frame,
                self._source_roi,
                self._config.rois.bar_area,
                self._config.hsv_thresholds.dot,
                self._config.control.min_area,
                bar=bar,
            )
        overlay = view.copy()
        view_origin_x = self._source_roi.x + relative.x
        view_origin_y = self._source_roi.y + relative.y

        if observation.bar is not None and target_key in ("bar", "dot"):
            rel_bar = observation.bar.bbox.relative_to(view_origin_x, view_origin_y)
            cv2.rectangle(
                overlay,
                (rel_bar.x, rel_bar.y),
                (rel_bar.x + rel_bar.w, rel_bar.y + rel_bar.h),
                (34, 197, 94),
                2,
            )
            center_x = observation.bar.center - view_origin_x
            cv2.line(
                overlay,
                (center_x, rel_bar.y),
                (center_x, rel_bar.y + rel_bar.h),
                (34, 197, 94),
                2,
            )

        if observation.dot is not None and target_key in ("bar", "dot"):
            rel_dot = observation.dot.bbox.relative_to(view_origin_x, view_origin_y)
            cv2.rectangle(
                overlay,
                (rel_dot.x, rel_dot.y),
                (rel_dot.x + rel_dot.w, rel_dot.y + rel_dot.h),
                (234, 179, 8),
                2,
            )
            dot_center_x = observation.dot.x - view_origin_x
            cv2.line(
                overlay,
                (dot_center_x, rel_dot.y),
                (dot_center_x, rel_dot.y + rel_dot.h),
                (234, 179, 8),
                2,
            )
        elif target_key == "dot" and raw_dot is not None:
            rel_dot = raw_dot.bbox.relative_to(view_origin_x, view_origin_y)
            cv2.rectangle(
                overlay,
                (rel_dot.x, rel_dot.y),
                (rel_dot.x + rel_dot.w, rel_dot.y + rel_dot.h),
                (250, 204, 21),
                1,
            )
            dot_center_x = raw_dot.x - view_origin_x
            cv2.line(
                overlay,
                (dot_center_x, rel_dot.y),
                (dot_center_x, rel_dot.y + rel_dot.h),
                (250, 204, 21),
                1,
            )

        if observation.blue_circle is not None and observation.blue_circle.found and observation.blue_circle.bbox is not None and target_key == "blue_circle":
            rel_blue = observation.blue_circle.bbox.relative_to(view_origin_x, view_origin_y)
            cv2.rectangle(
                overlay,
                (rel_blue.x, rel_blue.y),
                (rel_blue.x + rel_blue.w, rel_blue.y + rel_blue.h),
                (37, 99, 235),
                2,
            )

        self._source_label.setText("")
        self._mask_label.setText("")
        self._source_label.setPixmap(_to_pixmap(overlay, (self._source_label.width(), self._source_label.height())))
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        self._mask_label.setPixmap(_to_pixmap(mask_bgr, (self._mask_label.width(), self._mask_label.height())))

        contour_count = len(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        pixels = int(cv2.countNonZero(mask))
        detected_text = "未命中"
        if target_key == "bar" and observation.bar is not None:
            detected_text = f"center={observation.bar.center} left={observation.bar.left} right={observation.bar.right}"
        elif target_key == "dot" and observation.dot is not None:
            detected_text = f"黄线 x={observation.dot.x} y={observation.dot.y} area={observation.dot.area:.0f}"
        elif target_key == "dot" and raw_dot is not None:
            filtered_text = "通过" if filtered_dot is not None else "被约束过滤"
            detected_text = (
                f"原始黄线候选 x={raw_dot.x} y={raw_dot.y} area={raw_dot.area:.0f}"
                f" · 最终={filtered_text}"
            )
        elif target_key == "blue_circle" and observation.blue_circle is not None and observation.blue_circle.found:
            detected_text = f"x={observation.blue_circle.x} y={observation.blue_circle.y}"

        self._stats_label.setText(
            f"ROI: {search_roi.name} ({search_roi.w}x{search_roi.h})\n"
            f"阈值: lower={threshold.lower} upper={threshold.upper}\n"
            f"像素: {pixels} · 轮廓: {contour_count} · 识别: {detected_text}"
        )


class _HSVThresholdEditor(QWidget):
    threshold_changed = Signal()

    def __init__(self, title: str, threshold: HSVThreshold, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._spins: dict[str, QSpinBox] = {}

        card = Card(title, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(card)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        grid.addWidget(QLabel(""), 0, 0)
        for column, label in enumerate(("H", "S", "V"), start=1):
            head = QLabel(label)
            head.setStyleSheet("color:#64748b; font-size:11px; font-weight:700;")
            grid.addWidget(head, 0, column)

        labels = (("lower", "下限"), ("upper", "上限"))
        values = {"lower": threshold.lower, "upper": threshold.upper}
        limits = [179, 255, 255]
        body = card.layout()
        for row, (key, text) in enumerate(labels, start=1):
            row_label = QLabel(text)
            row_label.setStyleSheet("color:#334155; font-size:11px;")
            grid.addWidget(row_label, row, 0)
            for column, (value, upper) in enumerate(zip(values[key], limits), start=1):
                spin = QSpinBox()
                spin.setRange(0, upper)
                spin.setValue(int(value))
                spin.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
                spin.valueChanged.connect(lambda _value, signal=self.threshold_changed: signal.emit())
                self._spins[f"{key}_{column - 1}"] = spin
                grid.addWidget(spin, row, column)
        body.addLayout(grid)

    def threshold(self) -> HSVThreshold:
        return HSVThreshold(
            lower=[self._spins[f"lower_{index}"].value() for index in range(3)],
            upper=[self._spins[f"upper_{index}"].value() for index in range(3)],
        )

    def set_threshold(self, threshold: HSVThreshold) -> None:
        for index, value in enumerate(threshold.lower):
            self._spins[f"lower_{index}"].setValue(int(value))
        for index, value in enumerate(threshold.upper):
            self._spins[f"upper_{index}"].setValue(int(value))


class FishingAdvancedSettingsDialog(QDialog):
    def __init__(
        self,
        config: AppConfig,
        preview_frame: np.ndarray | None = None,
        preview_source_roi: ROI | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config.clone()
        self.setWindowTitle("钓鱼高级设置")
        self.resize(1080, 760)

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        left_scroll = QScrollArea(self)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        left_scroll.setMinimumWidth(420)

        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(10)
        left_scroll.setWidget(left_content)

        intro = QLabel(
            "这里的设置只影响当前钓鱼配置。实时预览使用你当前的截图/上传图片和已框好的 ROI，"
            "方便直接观察阈值是否把目标筛出来。"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#475569; font-size:12px;")
        left_layout.addWidget(intro)

        capture_card = Card("截图频率")
        capture_layout = capture_card.layout()
        fps_row = QHBoxLayout()
        fps_row.setSpacing(8)
        fps_label = QLabel("截图 FPS")
        fps_label.setStyleSheet("color:#334155; font-size:12px;")
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(5, 240)
        self._fps_spin.setSuffix(" FPS")
        self._fps_spin.setValue(self._config.control.capture_fps())
        self._fps_spin.valueChanged.connect(self._update_fps_hint)
        fps_row.addWidget(fps_label)
        fps_row.addWidget(self._fps_spin, stretch=1)
        capture_layout.addLayout(fps_row)
        self._fps_hint = QLabel("")
        self._fps_hint.setStyleSheet("color:#64748b; font-size:11px;")
        capture_layout.addWidget(self._fps_hint)
        left_layout.addWidget(capture_card)

        self._bar_editor = _HSVThresholdEditor("绿条 HSV", self._config.hsv_thresholds.bar)
        self._dot_editor = _HSVThresholdEditor("黄线 HSV", self._config.hsv_thresholds.dot)
        self._blue_editor = _HSVThresholdEditor("蓝圈 HSV", self._config.hsv_thresholds.blue_circle)
        left_layout.addWidget(self._bar_editor)
        left_layout.addWidget(self._dot_editor)
        left_layout.addWidget(self._blue_editor)

        self._preview_panel = _ThresholdPreviewPanel(self._config, preview_frame, preview_source_roi, self)

        self._bar_editor.threshold_changed.connect(lambda: self._on_threshold_changed("bar"))
        self._dot_editor.threshold_changed.connect(lambda: self._on_threshold_changed("dot"))
        self._blue_editor.threshold_changed.connect(lambda: self._on_threshold_changed("blue_circle"))

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        reset_btn = QPushButton("恢复默认阈值")
        reset_btn.setProperty("cssClass", "accent")
        reset_btn.clicked.connect(self._reset_thresholds)
        reset_fps_btn = QPushButton("恢复默认 FPS")
        reset_fps_btn.clicked.connect(lambda: self._fps_spin.setValue(DEFAULT_CAPTURE_FPS))
        action_row.addWidget(reset_btn)
        action_row.addWidget(reset_fps_btn)
        action_row.addStretch()

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        action_row.addWidget(button_box)
        left_layout.addLayout(action_row)
        left_layout.addStretch()

        root.addWidget(left_scroll, stretch=0)
        root.addWidget(self._preview_panel, stretch=1)

        self._update_fps_hint()
        self._preview_panel.update_config(self.updated_config())

    def _update_fps_hint(self) -> None:
        fps = self._fps_spin.value()
        interval_ms = max(1, int(round(1000.0 / max(1, fps))))
        actual_fps = max(1, int(round(1000.0 / interval_ms)))
        self._fps_hint.setText(
            f"当前目标为 {fps} FPS，执行时约 {interval_ms} ms/帧，实际极限约 {actual_fps} FPS。"
        )

    def _reset_thresholds(self) -> None:
        self._bar_editor.set_threshold(HSVThreshold(lower=list(DEFAULT_BAR_THRESHOLD[0]), upper=list(DEFAULT_BAR_THRESHOLD[1])))
        self._dot_editor.set_threshold(HSVThreshold(lower=list(DEFAULT_DOT_THRESHOLD[0]), upper=list(DEFAULT_DOT_THRESHOLD[1])))
        self._blue_editor.set_threshold(
            HSVThreshold(lower=list(DEFAULT_BLUE_CIRCLE_THRESHOLD[0]), upper=list(DEFAULT_BLUE_CIRCLE_THRESHOLD[1]))
        )

    def _on_threshold_changed(self, target_key: str) -> None:
        self._preview_panel.focus_target(target_key)
        self._preview_panel.update_config(self.updated_config())

    def updated_config(self) -> AppConfig:
        config = self._config.clone()
        config.hsv_thresholds.bar = self._bar_editor.threshold()
        config.hsv_thresholds.dot = self._dot_editor.threshold()
        config.hsv_thresholds.blue_circle = self._blue_editor.threshold()
        config.control.set_capture_fps(self._fps_spin.value())
        return config