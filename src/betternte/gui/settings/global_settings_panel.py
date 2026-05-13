"""gui/settings/global_settings_panel.py — 全局设置面板。"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from betternte.core.capture import list_game_windows
from betternte.core.config_manager import ConfigManager
from betternte.core.models import CaptureMode, InputMode
from betternte.gui.components import Card


class _SectionTitle(QLabel):
    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text.upper(), parent)
        self.setStyleSheet(
            "color:#64748b; font-size:10px; font-weight:700; letter-spacing:1.2px;"
            "padding:12px 0 4px 0; background:transparent;"
        )


class _OptionRow(QWidget):
    """一行配置项：左侧标签 + 右侧控件。"""

    def __init__(self, label: str, widget: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        lbl = QLabel(label)
        lbl.setStyleSheet("color:#334155; font-size:12px; background:transparent;")
        lbl.setFixedWidth(120)
        row.addWidget(lbl)
        row.addWidget(widget, stretch=1)


class GlobalSettingsPanel(QWidget):
    """全局设置页：截图模式、输入模式、窗口选择器。"""

    settings_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cfg = ConfigManager.instance()
        self._build_ui()
        self._load_values()

    # ────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 16, 20, 24)
        layout.setSpacing(0)

        # ── 截图模式 ──
        layout.addWidget(_SectionTitle("截图模式"))
        capture_card = Card()
        cap_layout = QVBoxLayout()
        cap_layout.setContentsMargins(12, 8, 12, 12)
        cap_layout.setSpacing(8)
        self._cap_full = QRadioButton("全屏截图（mss）")
        self._cap_bitblt = QRadioButton("窗口截图（BitBlt，支持后台）")
        self._cap_group = QButtonGroup(self)
        self._cap_group.addButton(self._cap_full, 0)
        self._cap_group.addButton(self._cap_bitblt, 1)
        self._cap_full.toggled.connect(self._on_capture_mode_changed)
        cap_layout.addWidget(self._cap_full)
        cap_layout.addWidget(self._cap_bitblt)

        # 窗口选择器（仅 BitBlt 模式使用）
        win_row_widget = QWidget()
        win_row = QHBoxLayout(win_row_widget)
        win_row.setContentsMargins(0, 0, 0, 0)
        win_row.setSpacing(6)
        win_lbl = QLabel("游戏窗口")
        win_lbl.setStyleSheet("color:#334155; font-size:12px; background:transparent;")
        win_lbl.setFixedWidth(80)
        self._window_combo = QComboBox()
        self._window_combo.setMinimumWidth(220)
        self._window_combo.wheelEvent = lambda e: e.ignore()
        refresh_btn = QPushButton("刷新")
        refresh_btn.setFixedWidth(56)
        refresh_btn.clicked.connect(self._refresh_windows)
        win_row.addWidget(win_lbl)
        win_row.addWidget(self._window_combo, stretch=1)
        win_row.addWidget(refresh_btn)
        cap_layout.addWidget(win_row_widget)
        self._win_row_widget = win_row_widget

        # 窗口类名输入
        self._class_edit = QLineEdit()
        self._class_edit.setPlaceholderText("窗口类名（如 UnrealWindow）")
        self._class_edit.textChanged.connect(self._on_class_changed)
        self._class_row_widget = _OptionRow("窗口类名", self._class_edit)
        cap_layout.addWidget(self._class_row_widget)

        capture_card.setLayout(cap_layout)
        layout.addWidget(capture_card)

        # ── 输入模式 ──
        layout.addWidget(_SectionTitle("输入模式"))
        input_card = Card()
        inp_layout = QVBoxLayout()
        inp_layout.setContentsMargins(12, 8, 12, 12)
        inp_layout.setSpacing(8)
        self._inp_direct = QRadioButton("DirectInput（需要管理员权限）")
        self._inp_post = QRadioButton("PostMessage（支持后台，无需管理员）")
        self._inp_group = QButtonGroup(self)
        self._inp_group.addButton(self._inp_direct, 0)
        self._inp_group.addButton(self._inp_post, 1)
        self._inp_direct.toggled.connect(self._on_input_mode_changed)
        inp_layout.addWidget(self._inp_direct)

        # PostMessage 模式说明
        inp_note = QLabel("注意：PostMessage 适用于支持消息注入的游戏，部分反作弊可能拦截。")
        inp_note.setWordWrap(True)
        inp_note.setStyleSheet("color:#64748b; font-size:11px; background:transparent; margin-left:4px;")
        inp_layout.addWidget(self._inp_post)
        inp_layout.addWidget(inp_note)
        input_card.setLayout(inp_layout)
        layout.addWidget(input_card)

        # ── 保存按钮 ──
        layout.addWidget(_SectionTitle(""))
        save_btn = QPushButton("保存设置")
        save_btn.setProperty("cssClass", "primary")
        save_btn.setFixedHeight(34)
        save_btn.clicked.connect(self._save)
        layout.addWidget(save_btn)

        layout.addStretch()
        scroll.setWidget(content)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ────────────────────────────────────────────────────────
    def _load_values(self) -> None:
        mode = self._cfg.get_capture_mode()
        if mode == CaptureMode.WINDOW_BITBLT:
            self._cap_bitblt.setChecked(True)
        else:
            self._cap_full.setChecked(True)
        self._update_window_row_visibility()

        inp_mode = self._cfg.get_input_mode()
        if inp_mode == InputMode.POST_MESSAGE:
            self._inp_post.setChecked(True)
        else:
            self._inp_direct.setChecked(True)

        self._class_edit.setText(self._cfg.get_game_window_class())
        self._refresh_windows()

    def _refresh_windows(self) -> None:
        cls_filter = self._class_edit.text().strip()
        windows = list_game_windows(class_name=cls_filter)
        self._window_combo.clear()
        for hwnd, title, cls in windows:
            self._window_combo.addItem(f"{title}  [{cls}]", hwnd)
        # 恢复上次选中的 title
        saved_title = self._cfg.get_game_window_title()
        if saved_title:
            for i in range(self._window_combo.count()):
                if saved_title.lower() in self._window_combo.itemText(i).lower():
                    self._window_combo.setCurrentIndex(i)
                    break

    def _update_window_row_visibility(self) -> None:
        is_bitblt = self._cap_bitblt.isChecked()
        self._win_row_widget.setVisible(is_bitblt)
        self._class_row_widget.setVisible(is_bitblt)

    def _on_capture_mode_changed(self) -> None:
        self._update_window_row_visibility()

    def _on_input_mode_changed(self) -> None:
        pass  # 实时更新在保存时统一处理

    def _on_class_changed(self, _: str) -> None:
        pass

    def _save(self) -> None:
        if self._cap_bitblt.isChecked():
            self._cfg.set_capture_mode(CaptureMode.WINDOW_BITBLT)
            # 保存窗口标题
            if self._window_combo.currentIndex() >= 0:
                text = self._window_combo.currentText()
                # text 格式: "标题  [类名]"
                title = text.split("[")[0].strip()
                self._cfg.set_game_window_title(title)
            else:
                self._cfg.set_game_window_title("")
        else:
            self._cfg.set_capture_mode(CaptureMode.FULLSCREEN)

        if self._inp_post.isChecked():
            self._cfg.set_input_mode(InputMode.POST_MESSAGE)
        else:
            self._cfg.set_input_mode(InputMode.DIRECT_INPUT)

        self._cfg.set_game_window_class(self._class_edit.text().strip())
        self._cfg.save()
        self.settings_changed.emit()
