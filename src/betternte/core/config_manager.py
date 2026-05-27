"""core/config.py — 全局配置管理（单例，持久化 JSON）。"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from betternte.core.models import CaptureConfig, CaptureMode, InputMode


_DEFAULT_GLOBAL_CONFIG: dict[str, Any] = {
    "capture_mode": CaptureMode.FULLSCREEN.value,
    "input_mode": InputMode.DIRECT_INPUT.value,
    "game_window_title": "",
    "game_window_class": "UnrealWindow",
    "monitor_index": 1,
}


class ConfigManager:
    """全局配置管理器（线程安全单例）。"""

    _instance: ConfigManager | None = None
    _lock = threading.Lock()

    @classmethod
    def instance(cls) -> ConfigManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls.__new__(cls)
                cls._instance._data: dict[str, Any] = dict(_DEFAULT_GLOBAL_CONFIG)
                cls._instance._config_path: Path | None = None
            return cls._instance

    def set_config_path(self, path: Path) -> None:
        """设置配置文件路径并立即加载。"""
        self._config_path = path
        self.load()

    def load(self) -> None:
        """从磁盘加载配置，缺失键用默认值补全。"""
        with self._lock:
            if self._config_path and self._config_path.exists():
                try:
                    raw = json.loads(self._config_path.read_text(encoding="utf-8"))
                    self._data = {**_DEFAULT_GLOBAL_CONFIG, **raw}
                except Exception:
                    self._data = dict(_DEFAULT_GLOBAL_CONFIG)
            else:
                self._data = dict(_DEFAULT_GLOBAL_CONFIG)

    def save(self) -> None:
        """保存配置到磁盘。"""
        with self._lock:
            if self._config_path:
                self._config_path.parent.mkdir(parents=True, exist_ok=True)
                self._config_path.write_text(
                    json.dumps(self._data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def get_capture_mode(self) -> CaptureMode:
        try:
            return CaptureMode(self._data.get("capture_mode", CaptureMode.FULLSCREEN.value))
        except ValueError:
            return CaptureMode.FULLSCREEN

    def get_input_mode(self) -> InputMode:
        try:
            return InputMode(self._data.get("input_mode", InputMode.DIRECT_INPUT.value))
        except ValueError:
            return InputMode.DIRECT_INPUT

    def get_monitor_index(self) -> int:
        return max(1, int(self._data.get("monitor_index", 1)))

    def get_game_window_title(self) -> str:
        return str(self._data.get("game_window_title", ""))

    def get_game_window_class(self) -> str:
        return str(self._data.get("game_window_class", "UnrealWindow"))

    def get_capture_config(self) -> CaptureConfig:
        return CaptureConfig(
            monitor_index=self.get_monitor_index(),
            mode=self.get_capture_mode(),
            window_title=self.get_game_window_title(),
            window_class=self.get_game_window_class(),
        )

    def set_capture_mode(self, mode: CaptureMode) -> None:
        self.set("capture_mode", mode.value)

    def set_input_mode(self, mode: InputMode) -> None:
        self.set("input_mode", mode.value)

    def set_monitor_index(self, monitor_index: int) -> None:
        self.set("monitor_index", max(1, int(monitor_index)))

    def set_game_window_title(self, title: str) -> None:
        self.set("game_window_title", title)

    def set_game_window_class(self, class_name: str) -> None:
        self.set("game_window_class", class_name)

    def all(self) -> dict[str, Any]:
        """返回配置副本。"""
        with self._lock:
            return dict(self._data)
