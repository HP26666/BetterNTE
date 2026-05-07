from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FishingState(str, Enum):
    IDLE = "IDLE"
    CASTING = "CASTING"
    WAITING_BITE = "WAITING_BITE"
    HOOKING = "HOOKING"
    CONTROLLING = "CONTROLLING"
    FINISHED = "FINISHED"


class Suggestion(str, Enum):
    NONE = "None"
    A = "A"
    D = "D"
    PRESS_F = "PRESS_F"
    CLICK_SCREEN = "CLICK_SCREEN"
    STOP = "STOP"


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

    def to_dict(self) -> dict[str, Any]:
        return {"monitor_index": int(self.monitor_index)}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CaptureConfig":
        data = data or {}
        return cls(monitor_index=int(data.get("monitor_index", 1)))


@dataclass
class ROISet:
    bar_area: ROI = field(default_factory=lambda: ROI(name="bar_area"))
    blue_circle_area: ROI = field(default_factory=lambda: ROI(name="blue_circle_area"))
    cast_click_area: ROI = field(default_factory=lambda: ROI(name="cast_click_area"))

    def to_dict(self) -> dict[str, dict[str, int | str]]:
        return {
            "bar_area": self.bar_area.to_dict(),
            "blue_circle_area": self.blue_circle_area.to_dict(),
            "cast_click_area": self.cast_click_area.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ROISet":
        data = data or {}
        return cls(
            bar_area=ROI.from_dict("bar_area", data.get("bar_area")),
            blue_circle_area=ROI.from_dict("blue_circle_area", data.get("blue_circle_area")),
            cast_click_area=ROI.from_dict("cast_click_area", data.get("cast_click_area")),
        )

    def get(self, key: str) -> ROI:
        return getattr(self, key)

    def set(self, key: str, value: ROI) -> None:
        setattr(self, key, value)


@dataclass
class ThresholdSet:
    # 固化 HSV 阈值（来自实测配置）
    bar: HSVThreshold = field(default_factory=lambda: HSVThreshold(lower=[75, 140, 144], upper=[95, 255, 255]))
    dot: HSVThreshold = field(default_factory=lambda: HSVThreshold(lower=[18, 30, 100], upper=[38, 255, 255]))
    blue_circle: HSVThreshold = field(default_factory=lambda: HSVThreshold(lower=[98, 163, 195], upper=[118, 255, 255]))

    def to_dict(self) -> dict[str, dict[str, list[int]]]:
        return {
            "bar": self.bar.to_dict(),
            "dot": self.dot.to_dict(),
            "blue_circle": self.blue_circle.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ThresholdSet":
        data = data or {}
        return cls(
            bar=HSVThreshold.from_dict(data.get("bar"), [75, 140, 144], [95, 255, 255]),
            dot=HSVThreshold.from_dict(data.get("dot"), [18, 30, 100], [38, 255, 255]),
            blue_circle=HSVThreshold.from_dict(data.get("blue_circle"), [98, 163, 195], [118, 255, 255]),
        )


@dataclass
class ControlConfig:
    margin: int = 10
    threshold: int = 8
    loop_interval_ms: int = 20
    min_area: int = 60
    smoothing: float = 0.35
    strength: float = 0.5  # 0.0~1.0 控制力度

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "margin": int(self.margin),
            "threshold": int(self.threshold),
            "loop_interval_ms": int(self.loop_interval_ms),
            "min_area": int(self.min_area),
            "smoothing": float(self.smoothing),
            "strength": float(self.strength),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ControlConfig":
        data = data or {}
        return cls(
            margin=int(data.get("margin", 10)),
            threshold=int(data.get("threshold", 8)),
            loop_interval_ms=int(data.get("loop_interval_ms", 20)),
            min_area=int(data.get("min_area", 60)),
            smoothing=float(data.get("smoothing", 0.35)),
            strength=float(data.get("strength", 0.5)),
        )


@dataclass
class AppConfig:
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    rois: ROISet = field(default_factory=ROISet)
    hsv_thresholds: ThresholdSet = field(default_factory=ThresholdSet)
    control: ControlConfig = field(default_factory=ControlConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "screen": self.screen.to_dict(),
            "capture": self.capture.to_dict(),
            "rois": self.rois.to_dict(),
            "hsv_thresholds": self.hsv_thresholds.to_dict(),
            "control": self.control.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, screen_size: tuple[int, int] | None = None) -> "AppConfig":
        data = data or {}
        screen_data = data.get("screen") or {}
        if screen_size is None:
            width = int(screen_data.get("width", 1920))
            height = int(screen_data.get("height", 1080))
        else:
            width, height = screen_size
        return cls(
            screen=ScreenConfig(width=width, height=height),
            capture=CaptureConfig.from_dict(data.get("capture")),
            rois=ROISet.from_dict(data.get("rois")),
            hsv_thresholds=ThresholdSet.from_dict(data.get("hsv_thresholds")),
            control=ControlConfig.from_dict(data.get("control")),
        )

    def clone(self) -> "AppConfig":
        return AppConfig.from_dict(self.to_dict(), (self.screen.width, self.screen.height))


@dataclass
class BarDetection:
    left: int
    right: int
    center: int
    bbox: ROI
    area: float


@dataclass
class DotDetection:
    x: int
    y: int
    bbox: ROI
    area: float


@dataclass
class BlueCircleDetection:
    found: bool
    x: int = 0
    y: int = 0
    bbox: ROI | None = None


@dataclass
class Observation:
    bar: BarDetection | None = None
    dot: DotDetection | None = None
    blue_circle: BlueCircleDetection | None = None
    bar_visible: bool = False
    dot_visible: bool = False


@dataclass
class ResultPacket:
    frame: Any
    source_roi: ROI
    observation: Observation
    state: FishingState
    suggestion: Suggestion
    message: str = ""
    dot_x_smoothed: float | None = None
    bar_center_smoothed: float | None = None
