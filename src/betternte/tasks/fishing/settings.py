from __future__ import annotations

import json
import queue
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPalette, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from betternte.core.capture import ScreenCapture, find_game_window
from betternte.core.config_manager import ConfigManager
from betternte.core.image_io import load_image_bgr
from betternte.core.input import InputController, PostMessageController, is_admin
from betternte.core.models import CaptureConfig, CaptureMode, InputMode
from betternte.core.task_executor import TaskExecutor
from betternte.gui.components import Badge, Card
from betternte.tasks.fishing.models import STATE_COLORS, SUGGESTION_COLORS
from betternte.tasks.fishing.advanced_dialog import FishingAdvancedSettingsDialog
from betternte.tasks.fishing.controller import compute_ad_pulse
from betternte.tasks.fishing.models import AppConfig, FishingState, ResultPacket, ROI, Suggestion
from betternte.tasks.fishing.vision import analyze_frame, debug_detect, dot_debug_lines, draw_debug_panel
from betternte.tasks.fishing.worker import FishingTask
from betternte.gui.annotation import ROISelectionDialog

CONFIGS_DIR = Path("configs")


def _ensure_configs_dir(base_path: Path) -> Path:
    configs = base_path / CONFIGS_DIR
    configs.mkdir(parents=True, exist_ok=True)
    return configs


def _list_configs(base_path: Path) -> list[Path]:
    return sorted(_ensure_configs_dir(base_path).glob("*.json"))


def _load_config(config_path: Path, screen_size: tuple[int, int]) -> AppConfig:
    if not config_path.exists():
        return AppConfig.from_dict({}, screen_size)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return AppConfig.from_dict(payload, screen_size)


def _save_config(config: AppConfig, config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


class FishingSettingsPanel(QWidget):
    """自动钓鱼功能的完整设置面板（自包含）。"""

    status_changed = Signal(str, str)  # text, color — 通知主窗口更新顶部状态徽章

    def __init__(self, base_path: Path, monitors: list[dict], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.base_path = base_path
        self._monitors = monitors

        self.capture = ScreenCapture()
        self.screen_size = self._detect_screen_size()
        self.config: AppConfig | None = None
        self._config_files: list[Path] = []

        self.result_queue: queue.Queue[ResultPacket] = queue.Queue(maxsize=2)
        self.executor: TaskExecutor | None = None
        self.fishing_task: FishingTask | None = None

        self.last_capture_frame: np.ndarray | None = None
        self.last_capture_roi = ROI(name="screen", x=0, y=0, w=self.screen_size[0], h=self.screen_size[1])
        self.current_packet: ResultPacket | None = None
        self._setup_images: list[np.ndarray] = []
        self._setup_index = 0

        self._build_ui()
        self.sync_capture_from_global()
        self._sync_ui_from_config()

        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(20)
        self.preview_timer.timeout.connect(self._drain_results)
        self.preview_timer.start()

    # ─────────────────────────────────────────────────────────
    def _detect_screen_size(self) -> tuple[int, int]:
        monitor = self._monitor_for_index(ConfigManager.instance().get_monitor_index())
        if monitor is not None:
            return int(monitor["width"]), int(monitor["height"])
        return 1920, 1080

    def _monitor_for_index(self, monitor_index: int) -> dict | None:
        for index, monitor in enumerate(self._monitors):
            if int(monitor.get("_mss_index", index + 1)) == int(monitor_index):
                return monitor
        if self._monitors:
            return self._monitors[0]
        return None

    def _screen_size_for_monitor(self, monitor_index: int) -> tuple[int, int]:
        monitor = self._monitor_for_index(monitor_index)
        if monitor is None:
            return self.screen_size
        return int(monitor["width"]), int(monitor["height"])

    def _describe_capture_source(self, capture_config: CaptureConfig) -> str:
        if capture_config.mode == CaptureMode.WINDOW_BITBLT:
            title = capture_config.window_title or "*"
            window_class = capture_config.window_class or "*"
            return f"窗口截图[{title} / {window_class}]"
        return f"全屏截图[显示器 {capture_config.monitor_index}]"

    def _apply_capture_config(self, capture_config: CaptureConfig, *, log_change: bool = False) -> CaptureConfig:
        config = self._ensure_config()
        config.capture = CaptureConfig.from_dict(capture_config.to_dict())
        if config.capture.mode == CaptureMode.FULLSCREEN:
            monitor = self._monitor_for_index(config.capture.monitor_index)
            width, height = self._screen_size_for_monitor(config.capture.monitor_index)
            self.screen_size = (width, height)
            config.screen.width = width
            config.screen.height = height
            left = int(monitor["left"]) if monitor is not None else 0
            top = int(monitor["top"]) if monitor is not None else 0
            self.last_capture_roi = ROI(name="screen", x=left, y=top, w=width, h=height)
        if self.fishing_task is not None:
            self.fishing_task.update_config(config)
        if self.executor is not None:
            self.executor.set_capture_config(config.capture)
            self.executor.frame_interval_ms = max(1, config.control.loop_interval_ms)
        if log_change:
            self._log(f"截图源已同步: {self._describe_capture_source(config.capture)}")
        return config.capture

    def sync_capture_from_global(self, log_change: bool = False) -> CaptureConfig:
        capture_config = ConfigManager.instance().get_capture_config()
        return self._apply_capture_config(capture_config, log_change=log_change)

    def _capture_current_source(self, *, append_to_setup_images: bool = False) -> tuple[np.ndarray, ROI]:
        capture_config = self.sync_capture_from_global()
        frame, source_roi = self.capture.grab_source(capture_config)
        self.last_capture_frame = frame
        self.last_capture_roi = source_roi
        if append_to_setup_images:
            self._setup_images.append(frame)
            self.image_combo.addItem(f"截图 #{len(self._setup_images)}")
            self.image_combo.setCurrentIndex(len(self._setup_images) - 1)
        self._set_preview(frame, source_roi)
        return frame, source_roi

    def on_monitor_changed(self, index: int, monitor: dict) -> None:
        """由主窗口调用，通知显示器切换。"""
        capture_config = self.sync_capture_from_global()
        self._log(
            f"已切换截图源: {self._describe_capture_source(capture_config)}"
            f" · {monitor['width']}x{monitor['height']}"
        )

    def stop(self, reason: str = "停止") -> None:
        self.stop_worker(reason)

    def cleanup(self) -> None:
        self.preview_timer.stop()
        self.stop_worker("关闭", silent=True)

    # ─────────────────────────────────────────────────────────
    #  UI 构建
    # ─────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setContentsMargins(8, 8, 8, 0)
        root.addWidget(splitter, stretch=1)

        # ── 左侧：预览 + 状态条 + 日志 ──
        left = QWidget()
        left.setStyleSheet("background:transparent;")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(6)

        preview_card = Card("预览")
        prev_lay = preview_card.layout()
        self.preview_label = QLabel("等待配置...")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(320, 240)
        self.preview_label.setStyleSheet(
            "background:#f8fafc; border:1px dashed #cbd5e1;"
            "border-radius:6px; color:#94a3b8; font-size:13px;"
        )
        prev_lay.addWidget(self.preview_label, stretch=1)

        strip = QHBoxLayout()
        strip.setSpacing(8)
        self._state_badge = Badge("IDLE", "#64748b")
        self._suggest_badge = Badge("-", "#94a3b8")
        self._bar_label = QLabel("绿条: -")
        self._bar_label.setStyleSheet("color:#16a34a; font-size:11px;")
        self._dot_label = QLabel("黄线: -")
        self._dot_label.setStyleSheet("color:#d97706; font-size:11px;")
        self._blue_label = QLabel("蓝圈: -")
        self._blue_label.setStyleSheet("color:#2563eb; font-size:11px;")
        for t in ("状态", "建议"):
            lbl = QLabel(t)
            lbl.setStyleSheet("color:#94a3b8; font-size:11px;")
            strip.addWidget(lbl)
        strip.addWidget(self._state_badge)
        strip.addWidget(self._suggest_badge)
        strip.addStretch()
        strip.addWidget(self._bar_label)
        strip.addWidget(self._dot_label)
        strip.addWidget(self._blue_label)
        prev_lay.addLayout(strip)
        left_layout.addWidget(preview_card, stretch=1)

        log_card = Card("日志")
        log_lay = log_card.layout()
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(80)
        self.log_output.setMaximumHeight(200)
        log_lay.addWidget(self.log_output)
        left_layout.addWidget(log_card)

        splitter.addWidget(left)

        # ── 右侧：配置面板（可滚动）──
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFixedWidth(270)
        right_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        right_panel = QWidget()
        right_panel.setStyleSheet("background: transparent;")
        right_scroll.setWidget(right_panel)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(6)
        right_layout.setContentsMargins(0, 0, 4, 4)
        splitter.addWidget(right_scroll)

        splitter.setSizes([740, 270])

        # ── 配置管理 ──
        cfg = Card("配置管理")
        cfg_lay = cfg.layout()
        self.config_combo = QComboBox()
        self.config_combo.wheelEvent = lambda e: e.ignore()
        self._refresh_config_list()
        cfg_row = QHBoxLayout()
        cfg_row.setSpacing(4)
        cfg_row.addWidget(self.config_combo, stretch=1)
        btn_load = QPushButton("加载")
        btn_load.clicked.connect(self._on_load_config)
        btn_save = QPushButton("保存")
        btn_save.clicked.connect(self._on_save_config)
        cfg_row.addWidget(btn_load)
        cfg_row.addWidget(btn_save)
        cfg_lay.addLayout(cfg_row)
        btn_new = QPushButton("新建配置")
        btn_new.setProperty("cssClass", "accent")
        btn_new.clicked.connect(self._on_new_config)
        cfg_lay.addWidget(btn_new)
        right_layout.addWidget(cfg)

        # ── 截图 ──
        img = Card("截图")
        img_lay = img.layout()
        img_row = QHBoxLayout()
        img_row.setSpacing(4)
        btn_upload = QPushButton("上传")
        btn_upload.clicked.connect(self._on_upload_images)
        btn_capture = QPushButton("截屏")
        btn_capture.clicked.connect(self._on_capture_screen)
        img_row.addWidget(btn_upload)
        img_row.addWidget(btn_capture)
        img_lay.addLayout(img_row)
        self.image_combo = QComboBox()
        self.image_combo.setPlaceholderText("请上传截图")
        self.image_combo.wheelEvent = lambda e: e.ignore()
        self.image_combo.currentIndexChanged.connect(self._on_image_switched)
        self.image_combo.setEditable(True)
        self.image_combo.lineEdit().setReadOnly(True)
        self.image_combo.lineEdit().setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_combo.setStyleSheet(
            "QComboBox QLineEdit { background:transparent; border:none; padding:0; }"
        )
        pal = self.image_combo.lineEdit().palette()
        pal.setColor(QPalette.ColorRole.PlaceholderText, QColor("#94a3b8"))
        self.image_combo.lineEdit().setPalette(pal)
        img_lay.addWidget(self.image_combo)
        right_layout.addWidget(img)

        # ── 区域框选 ──
        roi = Card("区域框选")
        roi_lay = roi.layout()
        btn_bar = QPushButton("绿条 + 黄线区域")
        btn_bar.clicked.connect(lambda: self._select_roi("bar_area", "选择绿条和黄线搜索区域"))
        btn_blue = QPushButton("蓝圈区域")
        btn_blue.clicked.connect(lambda: self._select_roi("blue_circle_area", "选择蓝圈搜索区域"))
        roi_lay.addWidget(btn_bar)
        roi_lay.addWidget(btn_blue)
        right_layout.addWidget(roi)

        # ── 运行控制 ──
        run = Card("运行控制")
        run_lay = run.layout()
        run_row = QHBoxLayout()
        run_row.setSpacing(6)
        self.start_button = QPushButton("开始运行")
        self.start_button.setProperty("cssClass", "primary")
        self.start_button.setMinimumHeight(34)
        self.start_button.clicked.connect(self.start_worker)
        self.stop_button = QPushButton("停止")
        self.stop_button.setProperty("cssClass", "danger")
        self.stop_button.setMinimumHeight(34)
        self.stop_button.clicked.connect(lambda: self.stop_worker("GUI 停止按钮"))
        run_row.addWidget(self.start_button, stretch=1)
        run_row.addWidget(self.stop_button, stretch=1)
        run_lay.addLayout(run_row)

        btn_test = QPushButton("测试识别")
        btn_test.setProperty("cssClass", "accent")
        btn_test.clicked.connect(self._test_recognition)
        btn_advanced = QPushButton("高级设置")
        btn_advanced.setProperty("cssClass", "accent")
        btn_advanced.clicked.connect(self._open_advanced_settings)
        act_row = QHBoxLayout()
        act_row.setSpacing(6)
        act_row.addWidget(btn_test, stretch=1)
        act_row.addWidget(btn_advanced, stretch=1)
        run_lay.addLayout(act_row)

        s_row = QHBoxLayout()
        s_row.setSpacing(6)
        s_lbl = QLabel("力度")
        s_lbl.setStyleSheet("color:#64748b; font-size:11px;")
        s_row.addWidget(s_lbl)
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(50)
        self.strength_slider.valueChanged.connect(self._on_strength_changed)
        self.strength_label = QLabel("50%")
        self.strength_label.setStyleSheet("color:#2563eb; font-weight:700; font-size:11px; min-width:32px;")
        s_row.addWidget(self.strength_slider, stretch=1)
        s_row.addWidget(self.strength_label)
        run_lay.addLayout(s_row)

        self.capture_rate_label = QLabel("截图频率: 50 FPS")
        self.capture_rate_label.setStyleSheet("color:#64748b; font-size:11px;")
        run_lay.addWidget(self.capture_rate_label)
        right_layout.addWidget(run)

        # ── 使用教程 ──
        tut = Card("使用教程")
        tut_lay = tut.layout()
        btn_tut = QPushButton("打开教程")
        btn_tut.clicked.connect(self._show_tutorial)
        tut_lay.addWidget(btn_tut)
        right_layout.addWidget(tut)

        right_layout.addStretch()

    # ─────────────────────────────────────────────────────────
    #  配置管理
    # ─────────────────────────────────────────────────────────
    def _refresh_config_list(self) -> None:
        self._config_files = _list_configs(self.base_path)
        self.config_combo.clear()
        for p in self._config_files:
            self.config_combo.addItem(p.stem, str(p))

    def _on_load_config(self) -> None:
        path_str = self.config_combo.currentData()
        if not path_str:
            QMessageBox.information(self, "提示", "没有可加载的配置")
            return
        path = Path(path_str)
        self.config = _load_config(path, self.screen_size)
        self.sync_capture_from_global()
        self._sync_ui_from_config()
        self._log(
            f"已加载: {path.name} · {self._describe_capture_source(self.config.capture)}"
            f" · {self.config.screen.width}x{self.config.screen.height}"
        )

    def _sync_ui_from_config(self) -> None:
        config = self._ensure_config()
        strength_pct = int(config.control.strength * 100)
        self.strength_slider.blockSignals(True)
        self.strength_slider.setValue(strength_pct)
        self.strength_slider.blockSignals(False)
        self.strength_label.setText(f"{strength_pct}%")
        self._update_capture_rate_label()

    def _apply_runtime_config(self) -> None:
        if self.config is None:
            return
        if self.fishing_task is not None:
            self.fishing_task.update_config(self.config)
        if self.executor is not None:
            self.executor.set_capture_config(self.config.capture)
            self.executor.frame_interval_ms = max(1, self.config.control.loop_interval_ms)
        if hasattr(self, "preview_timer"):
            self.preview_timer.setInterval(max(1, self.config.control.loop_interval_ms))
        self._update_capture_rate_label()

    def _update_capture_rate_label(self) -> None:
        if self.config is None:
            self.capture_rate_label.setText("截图频率: -")
            return
        target_fps = self.config.control.capture_fps()
        actual_fps = self.config.control.actual_capture_fps()
        self.capture_rate_label.setText(
            f"截图频率: {target_fps} FPS ({self.config.control.loop_interval_ms} ms/帧, 实际约 {actual_fps} FPS)"
        )

    def _on_save_config(self) -> None:
        if self.config is None:
            self.config = AppConfig.from_dict({}, self.screen_size)
        name, ok = QFileDialog.getSaveFileName(
            self, "保存配置", str(self.base_path / CONFIGS_DIR / "config.json"), "JSON (*.json)"
        )
        if not ok or not name:
            return
        _save_config(self.config, Path(name))
        self._refresh_config_list()
        self._log(f"已保存: {Path(name).name}")

    def _on_strength_changed(self, value: int) -> None:
        self.strength_label.setText(f"{value}%")
        if self.config is not None:
            self.config.control.strength = value / 100.0
            self._apply_runtime_config()

    def _on_new_config(self) -> None:
        self.config = AppConfig.from_dict({}, self.screen_size)
        self.sync_capture_from_global()
        self._setup_images.clear()
        self._setup_index = 0
        self.image_combo.clear()
        self._sync_ui_from_config()
        self._log("已新建配置，请上传截图并框选区域")

    def _ensure_config(self) -> AppConfig:
        if self.config is None:
            self.config = AppConfig.from_dict({}, self.screen_size)
        return self.config

    # ─────────────────────────────────────────────────────────
    #  截图
    # ─────────────────────────────────────────────────────────
    def _on_upload_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "上传截图", str(self.base_path / "image"), "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not files:
            return
        loaded_count = 0
        failed_files: list[str] = []
        for f in files:
            img = load_image_bgr(f)
            if img is not None:
                self._setup_images.append(img)
                self.image_combo.addItem(Path(f).name)
                loaded_count += 1
            else:
                failed_files.append(Path(f).name)
        if self._setup_images:
            self.image_combo.setCurrentIndex(len(self._setup_images) - 1)
            self._log(f"已上传 {loaded_count} 张截图")
        if failed_files:
            self._log(f"加载失败: {', '.join(failed_files)}")

    def _open_advanced_settings(self) -> None:
        config = self._ensure_config()
        dialog = FishingAdvancedSettingsDialog(
            config,
            preview_frame=self._get_current_frame(),
            preview_source_roi=self.last_capture_roi,
            parent=self,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        self.config = dialog.updated_config()
        self.sync_capture_from_global()
        self._sync_ui_from_config()
        self._apply_runtime_config()
        self._refresh_preview_analysis()
        self._log(
            "已应用钓鱼高级设置"
            f" · 目标FPS={self.config.control.capture_fps()}"
            f" · 实际约={self.config.control.actual_capture_fps()}"
        )

    def _on_capture_screen(self) -> None:
        capture_config = self.sync_capture_from_global()
        if capture_config.mode == CaptureMode.WINDOW_BITBLT:
            try:
                self._log(f"正在截取{self._describe_capture_source(capture_config)}...")
                self._capture_current_source(append_to_setup_images=True)
                self._log(f"已截取{self._describe_capture_source(capture_config)}")
            except Exception as exc:
                QMessageBox.warning(self, "截图失败", str(exc))
                self._log(f"截图失败: {exc}")
            return
        ret = QMessageBox.information(
            self, "延时截图",
            "点击「确定」后，窗口将最小化并在 5 秒后自动截取当前屏幕。\n\n"
            "请确保 5 秒内游戏画面已准备好。",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        )
        if ret != QMessageBox.StandardButton.Ok:
            return
        window = self.window()
        if window:
            window.showMinimized()
        QTimer.singleShot(5000, self._do_capture)

    def _do_capture(self) -> None:
        capture_config = self.sync_capture_from_global()
        self._log(f"正在截取{self._describe_capture_source(capture_config)}...")
        capture_ok = False
        try:
            self._capture_current_source(append_to_setup_images=True)
            capture_ok = True
        except Exception as exc:
            QMessageBox.warning(self, "截图失败", str(exc))
            self._log(f"截图失败: {exc}")
        window = self.window()
        if window:
            window.showNormal()
            window.activateWindow()
        if capture_ok:
            self._log(f"已截取{self._describe_capture_source(capture_config)}")

    def _on_image_switched(self, index: int) -> None:
        if 0 <= index < len(self._setup_images):
            self._setup_index = index
            frame = self._setup_images[index]
            roi = ROI(name="screen", x=0, y=0, w=frame.shape[1], h=frame.shape[0])
            self.last_capture_frame = frame
            self.last_capture_roi = roi
            self._set_preview(frame, roi, None)

    # ─────────────────────────────────────────────────────────
    #  ROI 框选
    # ─────────────────────────────────────────────────────────
    def _get_current_frame(self) -> np.ndarray | None:
        if self.last_capture_frame is not None:
            return self.last_capture_frame
        if self._setup_images:
            return self._setup_images[self._setup_index]
        return None

    def _select_roi(self, key: str, title: str) -> None:
        frame = self._get_current_frame()
        if frame is None:
            QMessageBox.information(self, "提示", "请先上传截图或截取屏幕")
            return
        config = self._ensure_config()
        current_roi = config.rois.get(key)
        initial_roi = None
        if current_roi.valid():
            candidate = current_roi.relative_to(self.last_capture_roi.x, self.last_capture_roi.y)
            if candidate.valid():
                initial_roi = candidate
        dialog = ROISelectionDialog(frame, title, initial_roi)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        roi = dialog.selected_roi(key).offset(self.last_capture_roi.x, self.last_capture_roi.y)
        if not roi.valid():
            self._log(f"未保存 {key}，框选区域无效")
            return
        config.rois.set(key, roi)
        self._log(f"已保存区域 {key}: x={roi.x} y={roi.y} w={roi.w} h={roi.h}")
        self._set_preview(frame, self.last_capture_roi)

    # ─────────────────────────────────────────────────────────
    #  测试识别
    # ─────────────────────────────────────────────────────────
    def _test_recognition(self) -> None:
        config = self.config
        if config is None:
            QMessageBox.information(self, "提示", "请先加载或新建配置")
            return
        try:
            frame, source_roi = self._capture_current_source()
        except Exception as exc:
            QMessageBox.warning(self, "测试识别失败", str(exc))
            self._log(f"测试识别失败: {exc}")
            return

        observation = analyze_frame(frame, source_roi, config)
        suggestion, _ = compute_ad_pulse(observation.bar, observation.dot, config.control)

        for name, roi_key, thresh_key in [
            ("绿条", "bar_area", "bar"),
            ("黄线", "bar_area", "dot"),
            ("蓝圈", "blue_circle_area", "blue_circle"),
        ]:
            roi = config.rois.get(roi_key)
            thresh = getattr(config.hsv_thresholds, thresh_key)
            debug_mode = "dot" if thresh_key == "dot" else "generic"
            info = debug_detect(
                frame,
                source_roi,
                roi,
                thresh,
                config.control.min_area,
                mode=debug_mode,
                bar=observation.bar if thresh_key == "dot" else None,
            )
            if not info["roi_valid"]:
                self._log(f"  [{name}] ROI 未设置或无效")
            elif thresh_key == "dot":
                edge_touches = info.get("edge_touches", {})
                edge_text = "".join(
                    label
                    for key, label in [
                        ("touches_top", "T"),
                        ("touches_bottom", "B"),
                        ("touches_left", "L"),
                        ("touches_right", "R"),
                    ]
                    if edge_touches.get(key)
                ) or "-"
                self._log(
                    f"  [{name}] 裁剪={info['view_shape']} raw像素={info.get('raw_mask_pixels', 0)} "
                    f"prep像素={info.get('prepared_mask_pixels', 0)} raw轮廓={info.get('raw_contours', 0)} "
                    f"prep轮廓={info.get('prepared_contours', 0)} 面积={info['contour_areas']} "
                    f"投影后备={'Y' if info.get('projection_found') else 'N'} 触边={edge_text} "
                    f"min_area={info['min_area']}"
                )
            else:
                self._log(
                    f"  [{name}] 裁剪={info['view_shape']} 匹配像素={info['mask_pixels']} "
                    f"轮廓={info['contours']} 面积={info['contour_areas']} "
                    f"min_area={info['min_area']}"
                )

        packet = ResultPacket(
            frame=frame,
            source_roi=source_roi,
            observation=observation,
            state=FishingState.CONTROLLING if observation.bar_visible else FishingState.IDLE,
            suggestion=suggestion,
            message=(
                f"测试: bar={'YES' if observation.bar_visible else 'no'}"
                f" · line={'YES' if observation.dot_visible else 'no'}"
                f" · blue={'YES' if observation.blue_circle and observation.blue_circle.found else 'no'}"
            ),
        )
        self.current_packet = packet
        self._update_preview_from_packet(packet)
        self._log(packet.message)

    def _refresh_preview_analysis(self) -> None:
        frame = self._get_current_frame()
        config = self.config
        if frame is None or config is None:
            return
        observation = analyze_frame(frame, self.last_capture_roi, config)
        suggestion, _ = compute_ad_pulse(observation.bar, observation.dot, config.control)
        packet = ResultPacket(
            frame=frame,
            source_roi=self.last_capture_roi,
            observation=observation,
            state=FishingState.CONTROLLING if observation.bar_visible else FishingState.IDLE,
            suggestion=suggestion,
            message="advanced-settings-preview",
        )
        self.current_packet = packet
        self._update_preview_from_packet(packet)

    # ─────────────────────────────────────────────────────────
    #  Worker 生命周期
    # ─────────────────────────────────────────────────────────
    def start_worker(self) -> None:
        if self.config is None:
            QMessageBox.information(self, "提示", "请先加载或新建配置")
            return

        cfg_mgr = ConfigManager.instance()
        capture_config = self.sync_capture_from_global()
        input_mode = cfg_mgr.get_input_mode()

        if capture_config.mode == CaptureMode.WINDOW_BITBLT:
            hwnd = find_game_window(
                title_pattern=capture_config.window_title,
                class_name=capture_config.window_class,
            )
            if hwnd is None:
                QMessageBox.warning(
                    self, "找不到窗口",
                    "窗口截图模式需要找到目标窗口。\n"
                    "请先在「全局设置」中确认窗口标题/类名，并保持游戏窗口可见且未最小化。"
                )
                return

        if input_mode == InputMode.POST_MESSAGE:
            # PostMessage 模式：查找游戏窗口 HWND
            hwnd = find_game_window(
                title_pattern=capture_config.window_title,
                class_name=capture_config.window_class,
            )
            if hwnd is None:
                QMessageBox.warning(
                    self, "找不到窗口",
                    "PostMessage 模式需要找到游戏窗口。\n"
                    "请先在「全局设置」中配置窗口标题/类名，并确认游戏已运行。"
                )
                return
            input_ctrl = PostMessageController(hwnd)
            self._log(f"[PostMessage] HWND={hwnd} · 已连接游戏窗口")
        else:
            # DirectInput 模式：需要管理员权限
            available, reason = InputController.backend_supported()
            if not available:
                QMessageBox.critical(self, "不可用", f"无法加载 pydirectinput: {reason}")
                return
            if not is_admin():
                QMessageBox.warning(self, "非管理员", "请用管理员身份运行 run_admin.bat")
                return
            input_ctrl = InputController()

        self.stop_worker("重新启动", silent=True)
        self.fishing_task = FishingTask(self.config, self.result_queue)
        self.executor = TaskExecutor(
            input_controller=input_ctrl,
            capture_config=self.config.capture,
            frame_interval_ms=self.config.control.loop_interval_ms,
            parent=self,
        )
        self.executor.register_trigger_task(self.fishing_task)
        self.executor.log_message.connect(self._log)
        self.executor.task_state_changed.connect(self._on_task_state_changed)
        self.executor.task_failed.connect(self._on_task_failed)
        self.executor.start()
        self.executor.start_task(self.fishing_task.name)
        self._log(
            f"识别线程已启动 · 输入模式={input_mode.value}"
            f" · 截图源={self._describe_capture_source(self.config.capture)}"
            f" · 目标FPS={self.config.control.capture_fps()}"
            f" · 实际约={self.config.control.actual_capture_fps()}"
        )

    def stop_worker(self, reason: str = "用户停止", silent: bool = False) -> None:
        if self.executor is None or self.fishing_task is None:
            if not silent:
                self._log(reason)
            return
        executor = self.executor
        task = self.fishing_task
        self.executor = None
        self.fishing_task = None
        if silent:
            task.suppress_next_stop_log()
        executor.stop_task(task.name, reason)
        executor.stop()
        executor.wait(1500)

    def _on_task_state_changed(self, name: str, running: bool) -> None:
        if name != FishingTask.name:
            return
        if running:
            self.status_changed.emit("RUN", "#16a34a")
        else:
            self.status_changed.emit("OFF", "#94a3b8")

    def _on_task_failed(self, name: str, error_message: str) -> None:
        if name != FishingTask.name:
            return
        self._log(f"识别线程异常: {error_message}")
        self.stop_worker("异常停止", silent=True)

    # ─────────────────────────────────────────────────────────
    #  结果处理 & 预览
    # ─────────────────────────────────────────────────────────
    def _drain_results(self) -> None:
        latest = None
        while True:
            try:
                latest = self.result_queue.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self.current_packet = latest
            self._update_preview_from_packet(latest)

    def _update_preview_from_packet(self, packet: ResultPacket) -> None:
        self._set_preview(packet.frame, packet.source_roi, packet)
        sc, st = STATE_COLORS.get(packet.state, ("#64748b", packet.state.value))
        uc, ut = SUGGESTION_COLORS.get(packet.suggestion, ("#94a3b8", packet.suggestion.value))
        self._state_badge.set_status(st, sc)
        self._suggest_badge.set_status(ut, uc)
        self._bar_label.setText(f"绿条: {packet.observation.bar.center if packet.observation.bar else '-'}")
        self._dot_label.setText(f"黄线: {packet.observation.dot.x if packet.observation.dot else '-'}")
        blue_found = packet.observation.blue_circle is not None and packet.observation.blue_circle.found
        self._blue_label.setText(f"蓝圈: {'YES' if blue_found else '-'}")

    def _set_preview(self, frame: np.ndarray, source_roi: ROI, packet: ResultPacket | None = None) -> None:
        overlay = frame.copy()
        if self.config is not None:
            for roi in [self.config.rois.bar_area, self.config.rois.blue_circle_area]:
                if not roi.valid():
                    continue
                rel = roi.relative_to(source_roi.x, source_roi.y)
                if not rel.valid():
                    continue
                cv2.rectangle(overlay, (rel.x, rel.y), (rel.x + rel.w, rel.y + rel.h), (37, 99, 235), 2)
                cv2.putText(overlay, roi.name, (rel.x, max(16, rel.y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (37, 99, 235), 1)

        if packet is not None:
            obs = packet.observation
            dot_info = None
            if self.config is not None and self.config.rois.bar_area.valid():
                dot_info = debug_detect(
                    frame,
                    source_roi,
                    self.config.rois.bar_area,
                    self.config.hsv_thresholds.dot,
                    self.config.control.min_area,
                    mode="dot",
                    bar=obs.bar,
                )
                dot_info = dict(dot_info)
                dot_info["visible"] = obs.dot is not None
            if obs.bar is not None:
                rel = obs.bar.bbox.relative_to(source_roi.x, source_roi.y)
                cv2.rectangle(overlay, (rel.x, rel.y), (rel.x + rel.w, rel.y + rel.h), (34, 197, 94), 2)
                cx = obs.bar.center - source_roi.x
                cv2.line(overlay, (cx, rel.y), (cx, rel.y + rel.h), (34, 197, 94), 2)
                left_x = obs.bar.left - source_roi.x
                right_x = obs.bar.right - source_roi.x
                cv2.line(overlay, (left_x, rel.y), (left_x, rel.y + rel.h), (0, 0, 255), 2)
                cv2.line(overlay, (right_x, rel.y), (right_x, rel.y + rel.h), (0, 0, 255), 2)
            if obs.dot is not None:
                rel = obs.dot.bbox.relative_to(source_roi.x, source_roi.y)
                cv2.rectangle(overlay, (rel.x, rel.y), (rel.x + rel.w, rel.y + rel.h), (234, 179, 8), 2)
                cx = obs.dot.x - source_roi.x
                cv2.line(overlay, (cx, rel.y), (cx, rel.y + rel.h), (234, 179, 8), 2)
            if obs.blue_circle is not None and obs.blue_circle.found and obs.blue_circle.bbox is not None:
                rel = obs.blue_circle.bbox.relative_to(source_roi.x, source_roi.y)
                cv2.rectangle(overlay, (rel.x, rel.y), (rel.x + rel.w, rel.y + rel.h), (37, 99, 235), 2)
            cv2.putText(overlay, f"State: {packet.state.value}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(overlay, f"Suggest: {packet.suggestion.value}", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            draw_debug_panel(
                overlay,
                dot_debug_lines(dot_info, smoothed_x=packet.dot_x_smoothed),
                origin=(10, 58),
                accent_color=(245, 158, 11),
                font_scale=0.44,
            )

        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        qi = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qi).scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(pix)

    # ─────────────────────────────────────────────────────────
    #  教程
    # ─────────────────────────────────────────────────────────
    def _show_tutorial(self) -> None:
        from PySide6.QtWidgets import QDialog, QTextBrowser, QDialogButtonBox

        html_path = self.base_path / "tutorial.html"
        html_content = ""
        if html_path.exists():
            html_content = html_path.read_text(encoding="utf-8")
        else:
            html_content = "<p>教程文件丢失: tutorial.html</p>"

        dlg = QDialog(self)
        dlg.setWindowTitle("BetterNTE 使用教程")
        dlg.resize(620, 560)
        dlg.setStyleSheet("QDialog { background: #ffffff; }")

        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        from PySide6.QtWidgets import QTextBrowser
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setStyleSheet(
            "QTextBrowser { border: none; padding: 20px 24px; font-size: 13px; "
            "color: #1e293b; background: #ffffff; }"
        )
        browser.setHtml(html_content)
        lay.addWidget(browser, stretch=1)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btn_box.button(QDialogButtonBox.StandardButton.Ok).setText("知道了")
        btn_box.setStyleSheet(
            "QPushButton { background:#dbeafe; color:#1e40af; border:1px solid #93c5fd; "
            "border-radius:6px; padding:6px 24px; font-weight:600; min-height:28px; }"
            "QPushButton:hover { background:#bfdbfe; }"
        )
        btn_box.accepted.connect(dlg.accept)
        lay.addWidget(btn_box)
        dlg.exec()

    # ─────────────────────────────────────────────────────────
    def _log(self, message: str) -> None:
        self.log_output.appendPlainText(message)
