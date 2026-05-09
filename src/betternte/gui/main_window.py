from __future__ import annotations

import queue
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from betternte.capture import ScreenCapture
from betternte.config import CONFIGS_DIR, list_configs, load_config, save_config
from betternte.control import InputController, compute_ad_pulse, is_admin, admin_status_text
from betternte.models import AppConfig, FishingState, ResultPacket, ROI, Suggestion
from betternte.vision import analyze_frame, debug_detect
from betternte.worker import VisionWorker
from betternte.gui.annotation import ROISelectionDialog


# ══════════════════════════════════════════════════════════
STYLE = r"""
* {
    font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
    font-size: 12px;
}
QMainWindow {
    background: #f1f5f9;
}

/* ── 卡片 ── */
QFrame[cssClass="card"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
}
QLabel[cssClass="card-title"] {
    color: #64748b;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 4px 0;
    background: transparent;
}

/* ── 按钮 ── */
QPushButton {
    background: #ffffff;
    color: #334155;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 5px 12px;
    min-height: 28px;
    font-size: 12px;
}
QPushButton:hover {
    background: #f1f5f9;
    color: #0f172a;
    border-color: #94a3b8;
}
QPushButton:pressed {
    background: #e2e8f0;
    border-color: #64748b;
}
QPushButton[cssClass="primary"] {
    background: #dcfce7;
    color: #166534;
    border: 1px solid #86efac;
    font-weight: 600;
}
QPushButton[cssClass="primary"]:hover {
    background: #bbf7d0;
    border-color: #4ade80;
    color: #14532d;
}
QPushButton[cssClass="danger"] {
    background: #fee2e2;
    color: #991b1b;
    border: 1px solid #fca5a5;
    font-weight: 600;
}
QPushButton[cssClass="danger"]:hover {
    background: #fecaca;
    border-color: #f87171;
    color: #7f1d1d;
}
QPushButton[cssClass="accent"] {
    background: #dbeafe;
    color: #1e40af;
    border: 1px solid #93c5fd;
    font-weight: 600;
}
QPushButton[cssClass="accent"]:hover {
    background: #bfdbfe;
    border-color: #60a5fa;
    color: #1e3a8a;
}

/* ── 下拉框 ── */
QComboBox {
    background: #ffffff;
    color: #1e293b;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 26px;
}
QComboBox:hover { border-color: #94a3b8; }
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox::down-arrow {
    image: none;
    border: none;
}
QComboBox QAbstractItemView {
    background: #ffffff;
    color: #1e293b;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    outline: none;
    selection-background-color: #dbeafe;
    selection-color: #1e293b;
}
QComboBox QAbstractItemView::item {
    padding: 4px 8px;
    min-height: 22px;
}
QComboBox QAbstractItemView::item:hover {
    background: #eff6ff;
    color: #1e293b;
}

/* ── 滑块 ── */
QSlider::groove:horizontal {
    background: #e2e8f0;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #2563eb;
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #1d4ed8;
}
QSlider::sub-page:horizontal {
    background: #2563eb;
    border-radius: 3px;
}

/* ── 日志 ── */
QPlainTextEdit {
    background: #f8fafc;
    color: #334155;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    font-family: "Cascadia Code", "Consolas", "Courier New", monospace;
    font-size: 11px;
    padding: 6px;
}

/* ── 滚动条 ── */
QScrollBar:vertical {
    background: #f1f5f9;
    width: 6px;
    border: none;
}
QScrollBar::handle:vertical {
    background: #cbd5e1;
    border-radius: 3px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover { background: #94a3b8; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QScrollArea {
    background: transparent;
    border: none;
}

/* ── 分割线 ── */
QSplitter::handle:vertical {
    background: #e2e8f0;
    height: 1px;
}
QSplitter::handle:horizontal {
    background: #e2e8f0;
    width: 1px;
}

/* ── 状态栏 ── */
QStatusBar {
    background: #ffffff;
    color: #64748b;
    border-top: 1px solid #e2e8f0;
    font-size: 10px;
    padding: 2px 10px;
}
"""

_STATE_COLORS: dict[FishingState, tuple[str, str]] = {
    FishingState.IDLE:          ("#64748b", "IDLE"),
    FishingState.CASTING:       ("#d97706", "CAST"),
    FishingState.WAITING_BITE:  ("#2563eb", "WAIT"),
    FishingState.HOOKING:       ("#dc2626", "HOOK"),
    FishingState.CONTROLLING:   ("#16a34a", "CTRL"),
    FishingState.FINISHED:      ("#7c3aed", "DONE"),
}
_SUGGESTION_COLORS: dict[Suggestion, tuple[str, str]] = {
    Suggestion.NONE:        ("#94a3b8", "-"),
    Suggestion.A:           ("#2563eb", "A"),
    Suggestion.D:           ("#2563eb", "D"),
    Suggestion.PRESS_F:     ("#d97706", "F"),
    Suggestion.CLICK_SCREEN:("#dc2626", "CLK"),
    Suggestion.STOP:        ("#ef4444", "STOP"),
}


# ══════════════════════════════════════════════════════════
class Card(QFrame):
    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("cssClass", "card")
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(0)
        if title:
            lbl = QLabel(title.upper())
            lbl.setProperty("cssClass", "card-title")
            lbl.setContentsMargins(14, 8, 14, 2)
            self._outer.addWidget(lbl)
        self.body = QWidget(self)
        self.body.setStyleSheet("background: transparent;")
        self._outer.addWidget(self.body)

    def layout(self) -> QVBoxLayout:
        lay = QVBoxLayout(self.body)
        lay.setContentsMargins(14, 4, 14, 10)
        lay.setSpacing(5)
        return lay


# ══════════════════════════════════════════════════════════
class Badge(QLabel):
    def __init__(self, text: str = "", color: str = "#94a3b8") -> None:
        super().__init__(text)
        self._base = color
        self.setFixedHeight(22)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._repaint()

    def set_status(self, text: str, color: str) -> None:
        self.setText(text)
        self._base = color
        self._repaint()

    def _repaint(self) -> None:
        c = self._base
        self.setStyleSheet(
            f"color:{c}; background:{c}18; border:1px solid {c}40;"
            f"border-radius:4px; padding:2px 10px; font-weight:700;"
            f"font-size:10px; letter-spacing:1px;"
        )


# ══════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    stop_requested = Signal(str)

    def __init__(self, base_path: Path) -> None:
        super().__init__()
        self.base_path = base_path
        self.setWindowTitle("BetterNTE")
        self.resize(1060, 700)
        self.setMinimumSize(860, 540)

        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(STYLE)

        self.capture = ScreenCapture()
        self._monitors = self.capture.list_monitors()

        self._config_files = list_configs(self.base_path)
        self.config: AppConfig | None = None
        self.screen_size = self._detect_screen_size()

        self.result_queue: queue.Queue[ResultPacket] = queue.Queue(maxsize=2)
        self.worker: VisionWorker | None = None
        self.hotkey_listener = None
        self.last_capture_frame: np.ndarray | None = None
        self.last_capture_roi = ROI(name="screen", x=0, y=0, w=self.screen_size[0], h=self.screen_size[1])
        self.current_packet: ResultPacket | None = None
        self._setup_images: list[np.ndarray] = []
        self._setup_index = 0

        self._build_ui()
        self._bind_hotkey()

        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(33)
        self.preview_timer.timeout.connect(self._drain_results)
        self.preview_timer.start()

        self.log(f"BetterNTE 已启动 · {admin_status_text()}")

    def closeEvent(self, event) -> None:
        self.stop_worker("关闭窗口")
        if self.hotkey_listener is not None:
            self.hotkey_listener.stop()
        return super().closeEvent(event)

    # ════════════════════════════════════════════════════
    def _detect_screen_size(self) -> tuple[int, int]:
        if self._monitors:
            m = self._monitors[0]
            return int(m["width"]), int(m["height"])
        return 1920, 1080

    # ════════════════════════════════════════════════════
    #  Header
    # ════════════════════════════════════════════════════
    def _build_header(self) -> QWidget:
        hdr = QWidget()
        hdr.setFixedHeight(44)
        hdr.setStyleSheet(
            "background:#ffffff; border-bottom:1px solid #e2e8f0;"
        )
        row = QHBoxLayout(hdr)
        row.setContentsMargins(16, 0, 12, 0)
        row.setSpacing(10)

        logo = QLabel("fish<b>PP</b>")
        logo.setTextFormat(Qt.TextFormat.RichText)
        logo.setStyleSheet("color:#1e293b; font-size:16px; font-weight:800; letter-spacing:2px; background:transparent;")

        self.monitor_combo = QComboBox()
        self.monitor_combo.setFixedWidth(210)
        self.monitor_combo.wheelEvent = lambda e: e.ignore()
        for i, m in enumerate(self._monitors):
            name = m.get("name", "")
            label = f"显示器 {i + 1}: {m['width']}x{m['height']}"
            if name:
                label += f" ({name})"
            self.monitor_combo.addItem(label, int(m["_mss_index"]))
        self.monitor_combo.currentIndexChanged.connect(self._on_monitor_changed)

        admin_ok = is_admin()
        admin_text = "管理员" if admin_ok else "非管理员"
        admin_color = "#16a34a" if admin_ok else "#dc2626"
        admin_bg = "#dcfce7" if admin_ok else "#fee2e2"
        admin_border = "#86efac" if admin_ok else "#fca5a5"
        self._admin_badge = QLabel(admin_text)
        self._admin_badge.setStyleSheet(
            f"color:{admin_color}; background:{admin_bg}; border:1px solid {admin_border}; border-radius:3px;"
            f"padding:1px 8px; font-size:10px; font-weight:600;"
        )

        f8 = QLabel("F8 停止")
        f8.setStyleSheet(
            "color:#dc2626; background:#fee2e2; border:1px solid #fca5a5; border-radius:3px;"
            "padding:1px 8px; font-size:10px; font-weight:600;"
        )

        self._header_status = Badge("OFF", "#94a3b8")

        row.addWidget(logo)
        row.addStretch()
        row.addWidget(self.monitor_combo)
        row.addWidget(self._admin_badge)
        row.addWidget(f8)
        row.addWidget(self._header_status)
        return hdr

    # ════════════════════════════════════════════════════
    #  Main UI
    # ════════════════════════════════════════════════════
    def _build_ui(self) -> None:
        container = QWidget()
        self.setCentralWidget(container)
        root = QVBoxLayout(container)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        root.addWidget(self._build_header())

        # ── 水平分割：预览 | 右侧面板 ──
        hsplit = QSplitter(Qt.Orientation.Horizontal)
        hsplit.setContentsMargins(8, 8, 8, 0)
        root.addWidget(hsplit, stretch=1)

        # ── 左侧预览区 ──
        left = QWidget()
        left.setStyleSheet("background:transparent;")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(6)

        preview_card = Card("预览")
        prev_outer = preview_card.layout()
        self.preview_label = QLabel("等待配置...")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(320, 240)
        self.preview_label.setStyleSheet(
            "background:#f8fafc; border:1px dashed #cbd5e1;"
            "border-radius:6px; color:#94a3b8; font-size:13px;"
        )
        prev_outer.addWidget(self.preview_label, stretch=1)

        # 状态指示条
        strip = QHBoxLayout()
        strip.setSpacing(8)
        self._state_badge = Badge("IDLE", "#64748b")
        self._suggest_badge = Badge("-", "#94a3b8")
        self._bar_label = QLabel("绿条: -")
        self._bar_label.setStyleSheet("color:#16a34a; font-size:11px;")
        self._dot_label = QLabel("黄点: -")
        self._dot_label.setStyleSheet("color:#d97706; font-size:11px;")
        self._blue_label = QLabel("蓝圈: -")
        self._blue_label.setStyleSheet("color:#2563eb; font-size:11px;")
        for t in ("状态", "建议"):
            l = QLabel(t)
            l.setStyleSheet("color:#94a3b8; font-size:11px;")
            strip.addWidget(l)
        strip.addWidget(self._state_badge)
        strip.addWidget(self._suggest_badge)
        strip.addStretch()
        strip.addWidget(self._bar_label)
        strip.addWidget(self._dot_label)
        strip.addWidget(self._blue_label)
        prev_outer.addLayout(strip)
        left_layout.addWidget(preview_card, stretch=1)

        # ── 日志（左列下方） ──
        log_card = Card("日志")
        log_lay = log_card.layout()
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(80)
        self.log_output.setMaximumHeight(200)
        log_lay.addWidget(self.log_output)
        left_layout.addWidget(log_card)

        hsplit.addWidget(left)

        # ── 右侧面板 ──
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
        hsplit.addWidget(right_scroll)

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
        btn_bar = QPushButton("绿条 + 黄点区域")
        btn_bar.clicked.connect(lambda: self._select_roi("bar_area", "选择绿条和黄点搜索区域"))
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
        btn_test.clicked.connect(self.test_recognition)
        run_lay.addWidget(btn_test)

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
        right_layout.addWidget(run)

        # ── 使用教程 ──
        tut = Card("使用教程")
        tut_lay = tut.layout()
        btn_tut = QPushButton("打开教程")
        btn_tut.clicked.connect(self._show_tutorial)
        tut_lay.addWidget(btn_tut)
        right_layout.addWidget(tut)

        right_layout.addStretch()

        hsplit.setSizes([740, 270])

    # ════════════════════════════════════════════════════
    #  Hotkey
    # ════════════════════════════════════════════════════
    def _bind_hotkey(self) -> None:
        self.stop_requested.connect(self.stop_worker)
        try:
            from pynput.keyboard import GlobalHotKeys
            self.hotkey_listener = GlobalHotKeys(
                {"<f8>": lambda: self.stop_requested.emit("F8 热键触发停止")}
            )
            self.hotkey_listener.start()
            self.log("F8 热键已启用")
        except Exception as exc:
            self.log(f"全局热键不可用: {exc}")

    # ════════════════════════════════════════════════════
    #  Monitor
    # ════════════════════════════════════════════════════
    def _on_monitor_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._monitors):
            return
        m = self._monitors[index]
        self.screen_size = (int(m["width"]), int(m["height"]))
        config = self._ensure_config()
        config.capture.monitor_index = index + 1
        config.screen.width = self.screen_size[0]
        config.screen.height = self.screen_size[1]
        self.last_capture_roi = ROI(name="screen", x=0, y=0, w=self.screen_size[0], h=self.screen_size[1])
        if self.worker is not None:
            self.worker.update_config(config)
        self.log(f"已切换到显示器 {index + 1}: {m['width']}x{m['height']}")

    # ════════════════════════════════════════════════════
    #  Config
    # ════════════════════════════════════════════════════
    def _refresh_config_list(self) -> None:
        self._config_files = list_configs(self.base_path)
        self.config_combo.clear()
        for p in self._config_files:
            self.config_combo.addItem(p.stem, str(p))

    def _on_load_config(self) -> None:
        path_str = self.config_combo.currentData()
        if not path_str:
            QMessageBox.information(self, "提示", "没有可加载的配置")
            return
        path = Path(path_str)
        self.config = load_config(path, self.screen_size)
        self._sync_ui_from_config()
        self.log(f"已加载: {path.name} · 显示器 {self.config.capture.monitor_index} · {self.config.screen.width}x{self.config.screen.height}")

    def _sync_ui_from_config(self) -> None:
        if self.config is None:
            return
        target_mss_index = self.config.capture.monitor_index
        self.monitor_combo.blockSignals(True)
        for i in range(self.monitor_combo.count()):
            if self.monitor_combo.itemData(i) == target_mss_index:
                self.monitor_combo.setCurrentIndex(i)
                break
        self.monitor_combo.blockSignals(False)
        self.screen_size = (self.config.screen.width, self.config.screen.height)
        strength_pct = int(self.config.control.strength * 100)
        self.strength_slider.blockSignals(True)
        self.strength_slider.setValue(strength_pct)
        self.strength_slider.blockSignals(False)
        self.strength_label.setText(f"{strength_pct}%")

    def _on_save_config(self) -> None:
        if self.config is None:
            self.config = AppConfig.from_dict({}, self.screen_size)
        name, ok = QFileDialog.getSaveFileName(
            self, "保存配置", str(self.base_path / CONFIGS_DIR / "config.json"), "JSON (*.json)"
        )
        if not ok or not name:
            return
        save_config(self.config, Path(name))
        self._refresh_config_list()
        self.log(f"已保存: {Path(name).name}")

    def _on_strength_changed(self, value: int) -> None:
        self.strength_label.setText(f"{value}%")
        if self.config is not None:
            self.config.control.strength = value / 100.0
            if self.worker is not None:
                self.worker.update_config(self.config)

    def _on_new_config(self) -> None:
        self.config = AppConfig.from_dict({}, self.screen_size)
        self._setup_images.clear()
        self._setup_index = 0
        self.image_combo.clear()
        self.log("已新建配置，请上传截图并框选区域")

    def _ensure_config(self) -> AppConfig:
        if self.config is None:
            self.config = AppConfig.from_dict({}, self.screen_size)
        return self.config

    # ════════════════════════════════════════════════════
    #  Image
    # ════════════════════════════════════════════════════
    def _on_upload_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "上传截图", str(self.base_path / "image"), "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not files:
            return
        for f in files:
            img = cv2.imread(f)
            if img is not None:
                self._setup_images.append(img)
                self.image_combo.addItem(Path(f).name)
        if self._setup_images:
            self.image_combo.setCurrentIndex(len(self._setup_images) - 1)
            self.log(f"已上传 {len(files)} 张截图")

    def _on_capture_screen(self) -> None:
        ret = QMessageBox.information(
            self, "延时截图",
            "点击「确定」后，窗口将最小化并在 5 秒后自动截取当前屏幕。\n\n"
            "请确保 5 秒内游戏画面已准备好。",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        )
        if ret != QMessageBox.StandardButton.Ok:
            return
        self.showMinimized()
        QTimer.singleShot(5000, self._do_capture)

    def _do_capture(self) -> None:
        config = self._ensure_config()
        monitor_idx = config.capture.monitor_index
        self.log(f"正在截取显示器 {monitor_idx}...")
        frame, source_roi = self.capture.grab_fullscreen(monitor_idx)
        self.last_capture_frame = frame
        self.last_capture_roi = source_roi
        self._setup_images.append(frame)
        self.image_combo.addItem(f"截图 #{len(self._setup_images)}")
        self.image_combo.setCurrentIndex(len(self._setup_images) - 1)
        self._set_preview(frame, source_roi)
        self.showNormal()
        self.activateWindow()
        self.log("已截取当前屏幕")

    def _on_image_switched(self, index: int) -> None:
        if 0 <= index < len(self._setup_images):
            self._setup_index = index
            frame = self._setup_images[index]
            roi = ROI(name="screen", x=0, y=0, w=frame.shape[1], h=frame.shape[0])
            self.last_capture_frame = frame
            self.last_capture_roi = roi
            self._set_preview(frame, roi, None)

    # ════════════════════════════════════════════════════
    #  ROI
    # ════════════════════════════════════════════════════
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
        dialog = ROISelectionDialog(frame, title, current_roi)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        roi = dialog.selected_roi(key)
        if not roi.valid():
            self.log(f"未保存 {key}，框选区域无效")
            return
        config.rois.set(key, roi)
        self.log(f"已保存区域 {key}: x={roi.x} y={roi.y} w={roi.w} h={roi.h}")
        self._set_preview(frame, self.last_capture_roi)

    # ════════════════════════════════════════════════════
    #  Test recognition
    # ════════════════════════════════════════════════════
    def test_recognition(self) -> None:
        config = self.config
        if config is None:
            QMessageBox.information(self, "提示", "请先加载或新建配置")
            return
        self._on_capture_screen()
        frame = self.last_capture_frame
        source_roi = self.last_capture_roi
        if frame is None:
            return

        observation = analyze_frame(frame, source_roi, config)
        suggestion, _ = compute_ad_pulse(observation.bar, observation.dot, config.control)

        for name, roi_key, thresh_key in [
            ("绿条", "bar_area", "bar"),
            ("黄点", "bar_area", "dot"),
            ("蓝圈", "blue_circle_area", "blue_circle"),
        ]:
            roi = config.rois.get(roi_key)
            thresh = getattr(config.hsv_thresholds, thresh_key)
            info = debug_detect(frame, source_roi, roi, thresh, config.control.min_area)
            if not info["roi_valid"]:
                self.log(f"  [{name}] ROI 未设置或无效")
            else:
                self.log(
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
                f" · dot={'YES' if observation.dot_visible else 'no'}"
                f" · blue={'YES' if observation.blue_circle and observation.blue_circle.found else 'no'}"
            ),
        )
        self.current_packet = packet
        self._update_preview_from_packet(packet)
        self.log(packet.message)

    # ════════════════════════════════════════════════════
    #  Worker lifecycle
    # ════════════════════════════════════════════════════
    def start_worker(self) -> None:
        if self.config is None:
            QMessageBox.information(self, "提示", "请先加载或新建配置")
            return
        available, reason = InputController.backend_supported()
        if not available:
            QMessageBox.critical(self, "不可用", f"无法加载 pydirectinput: {reason}")
            return
        if not is_admin():
            QMessageBox.warning(self, "非管理员", "请用管理员身份运行 run_admin.bat")
            return
        self.stop_worker("重新启动", silent=True)
        self.worker = VisionWorker(self.base_path, self.config, self.result_queue)
        self.worker.log_message.connect(self.log)
        self.worker.stopped.connect(self._on_worker_stopped)
        self.worker.failed.connect(self._on_worker_failed)
        self.worker.start()
        self._header_status.set_status("RUN", "#16a34a")
        self.log("识别线程已启动")

    def stop_worker(self, reason: str = "用户停止", silent: bool = False) -> None:
        if self.worker is None:
            if not silent:
                self.log(reason)
            return
        worker = self.worker
        self.worker = None
        worker.stop(reason)
        worker.wait(1500)
        self._header_status.set_status("OFF", "#94a3b8")

    # ════════════════════════════════════════════════════
    #  Result processing
    # ════════════════════════════════════════════════════
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
        sc, st = _STATE_COLORS.get(packet.state, ("#64748b", packet.state.value))
        uc, ut = _SUGGESTION_COLORS.get(packet.suggestion, ("#94a3b8", packet.suggestion.value))
        self._state_badge.set_status(st, sc)
        self._suggest_badge.set_status(ut, uc)
        self._bar_label.setText(f"绿条: {packet.observation.bar.center if packet.observation.bar else '-'}")
        self._dot_label.setText(f"黄点: {packet.observation.dot.x if packet.observation.dot else '-'}")
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
                pt = (obs.dot.x - source_roi.x, obs.dot.y - source_roi.y)
                cv2.circle(overlay, pt, 5, (234, 88, 12), 2)
            if obs.blue_circle is not None and obs.blue_circle.found and obs.blue_circle.bbox is not None:
                rel = obs.blue_circle.bbox.relative_to(source_roi.x, source_roi.y)
                cv2.rectangle(overlay, (rel.x, rel.y), (rel.x + rel.w, rel.y + rel.h), (37, 99, 235), 2)
            cv2.putText(overlay, f"State: {packet.state.value}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(overlay, f"Suggest: {packet.suggestion.value}", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        qi = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qi).scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(pix)
        self._refresh_status()

    def _refresh_status(self) -> None:
        config = self._ensure_config()
        mon_idx = config.capture.monitor_index
        self.statusBar().showMessage(
            f"显示器 {mon_idx}: {self.screen_size[0]}x{self.screen_size[1]}  ·  F8 停止"
        )

    def _on_worker_stopped(self, reason: str) -> None:
        self.log(reason)
        self._header_status.set_status("OFF", "#94a3b8")

    def _on_worker_failed(self, error_message: str) -> None:
        self.log(f"识别线程异常: {error_message}")
        self.stop_worker("异常停止", silent=True)

    def log(self, message: str) -> None:
        self.log_output.appendPlainText(message)

    # ════════════════════════════════════════════════════
    #  Tutorial
    # ════════════════════════════════════════════════════
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
