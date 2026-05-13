from __future__ import annotations

import ctypes
import sys
import warnings
from pathlib import Path

from PySide6.QtWidgets import QApplication

from betternte.config import ensure_configs_dir
from betternte.gui.main_window import MainWindow


def _check_admin() -> None:
    try:
        if not ctypes.windll.shell32.IsUserAnAdmin():
            warnings.warn(
                "未以管理员权限运行，pydirectinput 可能无法正常工作。"
                "请使用 run_admin.bat 启动。"
            )
    except OSError:
        pass


def main() -> int:
    _check_admin()
    base_path = Path.cwd()
    ensure_configs_dir(base_path)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow(base_path)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
