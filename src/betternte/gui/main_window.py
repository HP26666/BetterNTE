from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from betternte.core.capture import ScreenCapture
from betternte.core.config_manager import ConfigManager
from betternte.core.input import admin_status_text, is_admin
from betternte.gui.components import Badge, STYLE
from betternte.gui.settings import GlobalSettingsPanel
from betternte.gui.sidebar import SidebarNav
from betternte.tasks.fishing.settings import FishingSettingsPanel


class MainWindow(QMainWindow):
    """主窗口：header + 侧边栏导航 + 内容区域。"""

    stop_requested = Signal(str)

    def __init__(self, base_path: Path) -> None:
        super().__init__()
        self.base_path = base_path
        self.setWindowTitle("BetterNTE")
        self.resize(1200, 720)
        self.setMinimumSize(900, 560)

        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(STYLE)

        # 初始化全局配置
        cfg_mgr = ConfigManager.instance()
        cfg_mgr.set_config_path(base_path / "configs" / "global_config.json")

        self.capture = ScreenCapture()
        self._monitors = self.capture.list_monitors()
        self.hotkey_listener = None

        self._build_ui()
        self._restore_monitor_selection(cfg_mgr.get_monitor_index())
        self._bind_hotkey()
        self._sidebar.select("fishing")

        self._log(f"BetterNTE 已启动 · {admin_status_text()}")

    def closeEvent(self, event) -> None:
        self._fishing_panel.cleanup()
        if self.hotkey_listener is not None:
            try:
                self.hotkey_listener.stop()
            except Exception:
                pass
        return super().closeEvent(event)

    # ════════════════════════════════════════════════════════
    #  Header
    # ════════════════════════════════════════════════════════
    def _build_header(self) -> QWidget:
        hdr = QWidget()
        hdr.setFixedHeight(44)
        hdr.setStyleSheet("background:#ffffff; border-bottom:1px solid #e2e8f0;")
        row = QHBoxLayout(hdr)
        row.setContentsMargins(16, 0, 12, 0)
        row.setSpacing(10)

        logo = QLabel("Better<b>NTE</b>")
        logo.setTextFormat(Qt.TextFormat.RichText)
        logo.setStyleSheet(
            "color:#1e293b; font-size:16px; font-weight:800; letter-spacing:2px; background:transparent;"
        )

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
            f"color:{admin_color}; background:{admin_bg}; border:1px solid {admin_border};"
            f"border-radius:3px; padding:1px 8px; font-size:10px; font-weight:600;"
        )

        f8_badge = QLabel("F8 停止")
        f8_badge.setStyleSheet(
            "color:#dc2626; background:#fee2e2; border:1px solid #fca5a5;"
            "border-radius:3px; padding:1px 8px; font-size:10px; font-weight:600;"
        )

        self._header_status = Badge("OFF", "#94a3b8")

        row.addWidget(logo)
        row.addStretch()
        row.addWidget(self.monitor_combo)
        row.addWidget(self._admin_badge)
        row.addWidget(f8_badge)
        row.addWidget(self._header_status)
        return hdr

    # ════════════════════════════════════════════════════════
    #  UI
    # ════════════════════════════════════════════════════════
    def _build_ui(self) -> None:
        container = QWidget()
        self.setCentralWidget(container)
        root = QVBoxLayout(container)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        root.addWidget(self._build_header())

        # 主体：侧边栏 + 内容区
        body = QWidget()
        body.setStyleSheet("background:#f1f5f9;")
        body_layout = QHBoxLayout(body)
        body_layout.setSpacing(0)
        body_layout.setContentsMargins(0, 0, 0, 0)

        self._sidebar = SidebarNav()
        self._sidebar.add_page("fishing", "自动钓鱼")
        self._sidebar.add_page("settings", "全局设置")
        self._sidebar.page_changed.connect(self._on_page_changed)

        self._stack = QStackedWidget()
        self._stack.setStyleSheet("background:#f1f5f9;")

        self._fishing_panel = FishingSettingsPanel(self.base_path, self._monitors)
        self._fishing_panel.status_changed.connect(self._on_fishing_status_changed)
        self._stack.addWidget(self._fishing_panel)

        self._settings_panel = GlobalSettingsPanel()
        self._settings_panel.settings_changed.connect(self._on_global_settings_changed)
        self._stack.addWidget(self._settings_panel)

        body_layout.addWidget(self._sidebar)
        body_layout.addWidget(self._stack, stretch=1)
        root.addWidget(body, stretch=1)

    def _restore_monitor_selection(self, monitor_index: int) -> None:
        if not self._monitors:
            return
        target_index = 0
        for index, monitor in enumerate(self._monitors):
            if int(monitor.get("_mss_index", index + 1)) == int(monitor_index):
                target_index = index
                break
        self.monitor_combo.blockSignals(True)
        self.monitor_combo.setCurrentIndex(target_index)
        self.monitor_combo.blockSignals(False)
        self._on_monitor_changed(target_index)

    # ════════════════════════════════════════════════════════
    #  Slots
    # ════════════════════════════════════════════════════════
    def _on_page_changed(self, page_id: str) -> None:
        if page_id == "fishing":
            self._stack.setCurrentWidget(self._fishing_panel)
        elif page_id == "settings":
            self._stack.setCurrentWidget(self._settings_panel)

    def _on_monitor_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._monitors):
            return
        m = self._monitors[index]
        cfg_mgr = ConfigManager.instance()
        cfg_mgr.set_monitor_index(int(m.get("_mss_index", index + 1)))
        cfg_mgr.save()
        self._fishing_panel.on_monitor_changed(index, m)

    def _on_fishing_status_changed(self, text: str, color: str) -> None:
        self._header_status.set_status(text, color)

    def _on_global_settings_changed(self) -> None:
        self._fishing_panel.sync_capture_from_global(log_change=True)
        self._log("全局设置已保存。")

    # ════════════════════════════════════════════════════════
    #  Hotkey
    # ════════════════════════════════════════════════════════
    def _bind_hotkey(self) -> None:
        self.stop_requested.connect(lambda reason: self._fishing_panel.stop(reason))
        try:
            from pynput.keyboard import GlobalHotKeys

            self.hotkey_listener = GlobalHotKeys(
                {"<f8>": lambda: self.stop_requested.emit("F8 热键触发停止")}
            )
            self.hotkey_listener.start()
            self._log("F8 热键已启用")
        except Exception as exc:
            self._log(f"全局热键不可用: {exc}")

    # ──────────────────────────────────────────────────────────
    def _log(self, message: str) -> None:
        self._fishing_panel._log(message)
