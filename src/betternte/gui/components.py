from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from betternte.tasks.base import BaseTask

# ═══════════════════════════════════════════════════════════
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
QPushButton[cssClass="nav"] {
    background: transparent;
    color: #64748b;
    border: none;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px;
    text-align: left;
}
QPushButton[cssClass="nav"]:hover {
    background: #f1f5f9;
    color: #1e293b;
}
QPushButton[cssClass="nav-active"] {
    background: #dbeafe;
    color: #1e40af;
    border: none;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px;
    font-weight: 600;
    text-align: left;
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

QLineEdit, QSpinBox, QDoubleSpinBox {
    background: #ffffff;
    color: #1e293b;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 26px;
}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #94a3b8;
}
QCheckBox {
    color: #334155;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border-radius: 4px;
    border: 1px solid #94a3b8;
    background: #ffffff;
}
QCheckBox::indicator:checked {
    background: #2563eb;
    border-color: #2563eb;
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



# ═══════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════
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


def create_config_widget(
    key: str,
    value: Any,
    type_desc: dict[str, Any] | None = None,
) -> tuple[QWidget, Callable[[], Any]]:
    """根据配置项类型创建对应控件，返回 (widget, getter)。"""

    type_desc = type_desc or {}
    widget_type = type_desc.get("type")

    if widget_type == "drop_down":
        combo = QComboBox()
        combo.wheelEvent = lambda event: event.ignore()
        options = list(type_desc.get("options", []))
        for option in options:
            combo.addItem(str(option), option)
        index = combo.findData(value)
        if index < 0:
            index = combo.findText(str(value))
        combo.setCurrentIndex(max(index, 0))
        return combo, lambda: combo.currentData() if combo.currentData() is not None else combo.currentText()

    if widget_type == "slider":
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        slider = QSlider(Qt.Orientation.Horizontal)
        minimum = int(type_desc.get("min", 0))
        maximum = int(type_desc.get("max", 100))
        slider.setRange(minimum, maximum)
        slider.setValue(int(value))
        label = QLabel(str(int(value)))
        label.setStyleSheet("color:#2563eb; font-weight:700; min-width:34px;")
        slider.valueChanged.connect(lambda current: label.setText(str(current)))
        layout.addWidget(slider, stretch=1)
        layout.addWidget(label)
        return container, lambda: slider.value()

    if widget_type == "switch":
        checkbox = QCheckBox()
        checkbox.setChecked(bool(value))
        return checkbox, checkbox.isChecked

    if widget_type == "spinbox":
        spinbox = QSpinBox()
        spinbox.setRange(int(type_desc.get("min", 0)), int(type_desc.get("max", 999)))
        spinbox.setValue(int(value))
        return spinbox, spinbox.value

    if widget_type == "text_edit":
        line_edit = QLineEdit("" if value is None else str(value))
        return line_edit, line_edit.text

    if isinstance(value, bool):
        checkbox = QCheckBox()
        checkbox.setChecked(bool(value))
        return checkbox, checkbox.isChecked

    if isinstance(value, int):
        spinbox = QSpinBox()
        spinbox.setRange(int(type_desc.get("min", 0)), int(type_desc.get("max", 999)))
        spinbox.setValue(int(value))
        return spinbox, spinbox.value

    if isinstance(value, float):
        spinbox = QDoubleSpinBox()
        spinbox.setDecimals(int(type_desc.get("decimals", 2)))
        spinbox.setRange(float(type_desc.get("min", 0.0)), float(type_desc.get("max", 999.0)))
        spinbox.setSingleStep(float(type_desc.get("step", 0.1)))
        spinbox.setValue(float(value))
        return spinbox, spinbox.value

    if value is None:
        line_edit = QLineEdit("" if value is None else str(value))
        return line_edit, line_edit.text

    line_edit = QLineEdit(str(value))
    return line_edit, line_edit.text


class ConfigCard(Card):
    """根据 Task 的 default_config 和 config_type 自动生成设置卡片。"""

    def __init__(self, task: BaseTask, parent: QWidget | None = None) -> None:
        title = task.display_name or task.name or "配置"
        super().__init__(title, parent)
        self.task = task
        self._getters: dict[str, Callable[[], Any]] = {}
        body = self.layout()
        for key, value in task.default_config.items():
            type_desc = task.config_type.get(key)
            control, getter = create_config_widget(key, value, type_desc)
            self._getters[key] = getter

            row = QWidget(self)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 4, 0, 4)
            row_layout.setSpacing(10)

            label = QLabel((type_desc or {}).get("label", key))
            label.setStyleSheet("color:#64748b; font-size:11px; min-width:86px;")
            label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

            if isinstance(control, (QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox)):
                control.setMinimumWidth(120)

            row_layout.addWidget(label)
            row_layout.addWidget(control, stretch=1)
            body.addWidget(row)
        body.addStretch()

    def get_config(self) -> dict[str, Any]:
        return {key: getter() for key, getter in self._getters.items()}


__all__ = [
    "Badge",
    "Card",
    "ConfigCard",
    "STYLE",
    "create_config_widget",
]
