from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from betternte.core.models import CaptureConfig, HSVThreshold, ROI, ScreenConfig


DEFAULT_BAR_THRESHOLD = ([80, 180, 170], [88, 215, 254])
DEFAULT_DOT_THRESHOLD = ([17, 16, 190], [40, 255, 255])
DEFAULT_BLUE_CIRCLE_THRESHOLD = ([101, 154, 187], [110, 228, 255])
DEFAULT_LOOP_INTERVAL_MS = 20
DEFAULT_CAPTURE_FPS = 50


def _threshold_from_defaults(bounds: tuple[list[int], list[int]]) -> HSVThreshold:
    lower, upper = bounds
    return HSVThreshold(lower=list(lower), upper=list(upper))


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
    bar: HSVThreshold = field(default_factory=lambda: _threshold_from_defaults(DEFAULT_BAR_THRESHOLD))
    dot: HSVThreshold = field(default_factory=lambda: _threshold_from_defaults(DEFAULT_DOT_THRESHOLD))
    blue_circle: HSVThreshold = field(default_factory=lambda: _threshold_from_defaults(DEFAULT_BLUE_CIRCLE_THRESHOLD))

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
            bar=HSVThreshold.from_dict(data.get("bar"), DEFAULT_BAR_THRESHOLD[0], DEFAULT_BAR_THRESHOLD[1]),
            dot=HSVThreshold.from_dict(data.get("dot"), DEFAULT_DOT_THRESHOLD[0], DEFAULT_DOT_THRESHOLD[1]),
            blue_circle=HSVThreshold.from_dict(
                data.get("blue_circle"),
                DEFAULT_BLUE_CIRCLE_THRESHOLD[0],
                DEFAULT_BLUE_CIRCLE_THRESHOLD[1],
            ),
        )


@dataclass
class ControlConfig:
    margin: int = 10
    threshold: int = 8
    loop_interval_ms: int = DEFAULT_LOOP_INTERVAL_MS
    min_area: int = 60
    smoothing: float = 0.35
    strength: float = 0.5
    capture_fps_value: int = DEFAULT_CAPTURE_FPS

    def capture_fps(self) -> int:
        return max(1, int(self.capture_fps_value))

    def actual_capture_fps(self) -> int:
        return max(1, int(round(1000.0 / max(1, self.loop_interval_ms))))

    def set_capture_fps(self, fps: int) -> None:
        fps = max(1, int(fps))
        self.capture_fps_value = fps
        self.loop_interval_ms = max(1, int(round(1000.0 / fps)))

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "margin": int(self.margin),
            "threshold": int(self.threshold),
            "loop_interval_ms": int(self.loop_interval_ms),
            "capture_fps": int(self.capture_fps()),
            "actual_capture_fps": int(self.actual_capture_fps()),
            "min_area": int(self.min_area),
            "smoothing": float(self.smoothing),
            "strength": float(self.strength),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ControlConfig":
        data = data or {}
        fps_value = int(data.get("capture_fps", 0) or 0)
        loop_interval_ms = int(data.get("loop_interval_ms", DEFAULT_LOOP_INTERVAL_MS))
        if fps_value > 0 and "loop_interval_ms" not in data:
            loop_interval_ms = max(1, int(round(1000.0 / fps_value)))
        if fps_value <= 0:
            fps_value = max(1, int(round(1000.0 / max(1, loop_interval_ms))))
        return cls(
            margin=int(data.get("margin", 10)),
            threshold=int(data.get("threshold", 8)),
            loop_interval_ms=loop_interval_ms,
            min_area=int(data.get("min_area", 60)),
            smoothing=float(data.get("smoothing", 0.35)),
            strength=float(data.get("strength", 0.5)),
            capture_fps_value=fps_value,
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


STATE_COLORS: dict[FishingState, tuple[str, str]] = {
    FishingState.IDLE:          ("#64748b", "IDLE"),
    FishingState.CASTING:       ("#d97706", "CAST"),
    FishingState.WAITING_BITE:  ("#2563eb", "WAIT"),
    FishingState.HOOKING:       ("#dc2626", "HOOK"),
    FishingState.CONTROLLING:   ("#16a34a", "CTRL"),
    FishingState.FINISHED:      ("#7c3aed", "DONE"),
}
SUGGESTION_COLORS: dict[Suggestion, tuple[str, str]] = {
    Suggestion.NONE:        ("#94a3b8", "-"),
    Suggestion.A:           ("#2563eb", "A"),
    Suggestion.D:           ("#2563eb", "D"),
    Suggestion.PRESS_F:     ("#d97706", "F"),
    Suggestion.CLICK_SCREEN:("#dc2626", "CLK"),
    Suggestion.STOP:        ("#ef4444", "STOP"),
}
