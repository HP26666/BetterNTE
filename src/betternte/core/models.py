from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CaptureMode(str, Enum):
    """截图来源模式。"""
    FULLSCREEN = "fullscreen"       # mss 全屏截图
    WINDOW_BITBLT = "window_bitblt"  # BitBlt 窗口截图（支持后台）


class InputMode(str, Enum):
    """输入控制模式。"""
    DIRECT_INPUT = "direct_input"   # pydirectinput（需要管理员）
    POST_MESSAGE = "post_message"   # PostMessage（支持后台）


@dataclass
class ROI:
    name: str
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0

    def valid(self) -> bool:
        return self.w > 0 and self.h > 0

    def to_dict(self) -> dict[str, int | str]:
        return {"name": self.name, "x": self.x, "y": self.y, "w": self.w, "h": self.h}

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any] | None) -> "ROI":
        data = data or {}
        return cls(
            name=data.get("name", name),
            x=int(data.get("x", 0)),
            y=int(data.get("y", 0)),
            w=int(data.get("w", 0)),
            h=int(data.get("h", 0)),
        )

    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    def relative_to(self, origin_x: int, origin_y: int) -> "ROI":
        return ROI(name=self.name, x=self.x - origin_x, y=self.y - origin_y, w=self.w, h=self.h)

    def offset(self, dx: int = 0, dy: int = 0) -> "ROI":
        return ROI(name=self.name, x=self.x + dx, y=self.y + dy, w=self.w, h=self.h)

    def clamp(self, width: int, height: int) -> "ROI":
        x = max(0, min(self.x, width))
        y = max(0, min(self.y, height))
        w = max(0, min(self.w, width - x))
        h = max(0, min(self.h, height - y))
        return ROI(name=self.name, x=x, y=y, w=w, h=h)


@dataclass
class HSVThreshold:
    lower: list[int] = field(default_factory=lambda: [20, 80, 120])
    upper: list[int] = field(default_factory=lambda: [40, 255, 255])

    def to_dict(self) -> dict[str, list[int]]:
        return {"lower": [int(v) for v in self.lower], "upper": [int(v) for v in self.upper]}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, default_lower: list[int], default_upper: list[int]) -> "HSVThreshold":
        data = data or {}
        return cls(
            lower=[int(v) for v in data.get("lower", default_lower)],
            upper=[int(v) for v in data.get("upper", default_upper)],
        )


@dataclass
class ScreenConfig:
    width: int = 1920
    height: int = 1080

    def to_dict(self) -> dict[str, int]:
        return {"width": int(self.width), "height": int(self.height)}


@dataclass
class CaptureConfig:
    monitor_index: int = 1
    mode: CaptureMode = CaptureMode.FULLSCREEN
    window_title: str = ""
    window_class: str = "UnrealWindow"

    def to_dict(self) -> dict[str, Any]:
        return {
            "monitor_index": int(self.monitor_index),
            "mode": self.mode.value,
            "window_title": self.window_title,
            "window_class": self.window_class,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CaptureConfig":
        data = data or {}
        try:
            mode = CaptureMode(data.get("mode", CaptureMode.FULLSCREEN.value))
        except ValueError:
            mode = CaptureMode.FULLSCREEN
        return cls(
            monitor_index=int(data.get("monitor_index", 1)),
            mode=mode,
            window_title=str(data.get("window_title", "")),
            window_class=str(data.get("window_class", "UnrealWindow")),
        )
