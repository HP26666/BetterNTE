"""向后兼容的 re-export 层。旧代码 from betternte.vision import X 仍可用。"""
from betternte.tasks.fishing.vision import (
    analyze_frame,
    auto_hsv_bounds_from_point,
    crop_global,
    debug_detect,
    detect_bar,
    detect_blue_circle,
    detect_dot,
)

__all__ = [
    "analyze_frame",
    "auto_hsv_bounds_from_point",
    "crop_global",
    "debug_detect",
    "detect_bar",
    "detect_blue_circle",
    "detect_dot",
]
