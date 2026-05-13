from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget, QLabel, QFrame


class SidebarNav(QWidget):
    """左侧导航栏。每个功能组对应一个按钮，点击后发出 page_changed 信号。"""

    page_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(160)
        self.setStyleSheet("background: #ffffff; border-right: 1px solid #e2e8f0;")

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 12, 8, 12)
        self._layout.setSpacing(4)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        nav_title = QLabel("功能")
        nav_title.setStyleSheet(
            "color:#94a3b8; font-size:10px; font-weight:700; letter-spacing:1px;"
            "padding: 4px 4px 8px 4px; background:transparent;"
        )
        self._layout.addWidget(nav_title)

        self._buttons: dict[str, QPushButton] = {}
        self._current: str | None = None

    def add_page(self, page_id: str, label: str, icon: str = "") -> None:
        """添加一个导航项。"""
        text = f"{icon}  {label}" if icon else label
        btn = QPushButton(text)
        btn.setProperty("cssClass", "nav")
        btn.setCheckable(False)
        btn.clicked.connect(lambda _checked, pid=page_id: self._select(pid))
        self._buttons[page_id] = btn
        self._layout.addWidget(btn)

    def _select(self, page_id: str) -> None:
        if self._current == page_id:
            return
        if self._current and self._current in self._buttons:
            self._buttons[self._current].setProperty("cssClass", "nav")
            self._buttons[self._current].style().unpolish(self._buttons[self._current])
            self._buttons[self._current].style().polish(self._buttons[self._current])
        self._current = page_id
        btn = self._buttons.get(page_id)
        if btn:
            btn.setProperty("cssClass", "nav-active")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        self.page_changed.emit(page_id)

    def select(self, page_id: str) -> None:
        """程序化选中某项。"""
        self._select(page_id)
