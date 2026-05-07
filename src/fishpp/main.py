from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from fishpp.config import ensure_configs_dir
from fishpp.gui.main_window import MainWindow


def main() -> int:
    base_path = Path.cwd()
    ensure_configs_dir(base_path)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow(base_path)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
