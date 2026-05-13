from __future__ import annotations

from typing import TYPE_CHECKING, Any

from betternte.core.models import ROI
from betternte.tasks.fishing.models import BarDetection, ControlConfig, DotDetection, Suggestion


def apply_suggestion(
    controller: Any,
    suggestion: Suggestion,
    click_roi: ROI | None = None,
    pulse_duration: float = 0.0,
    hold: bool = False,
) -> None:
    """根据 Suggestion 调用统一的原子操作接口。controller 可以是 InputController 或 PostMessageController。"""
    if suggestion == Suggestion.A:
        if hold:
            controller.key_down("a")
            controller.key_up("d")
        else:
            controller.release_all()
            controller.pulse("a", pulse_duration)
    elif suggestion == Suggestion.D:
        if hold:
            controller.key_down("d")
            controller.key_up("a")
        else:
            controller.release_all()
            controller.pulse("d", pulse_duration)
    elif suggestion == Suggestion.PRESS_F:
        controller.release_all()
        controller.tap("f")
    elif suggestion == Suggestion.CLICK_SCREEN:
        controller.release_all()
        if click_roi and click_roi.valid():
            x, y = click_roi.center()
            controller.click(x, y)
    else:
        controller.release_all()


def compute_ad_pulse(
    bar: BarDetection | None,
    dot: DotDetection | None,
    control: ControlConfig,
    dot_x: float | None = None,
    bar_center: float | None = None,
    prev_error_ratio: float | None = None,
) -> tuple[Suggestion, float]:
    """PD 风格 A/D 控制，返回 (方向, 脉冲时长秒数)。

    控制力度 control.strength (0.0~1.0) 影响：
    - 死区大小（力度越大死区越小）
    - P 增益（力度越大按得越久）
    - D 增益（力度越大对速度越敏感）
    - 最高脉冲上限
    """
    if bar is None or dot is None:
        return Suggestion.NONE, 0.0

    s = min(max(control.strength, 0.0), 1.0)

    half_width = (bar.right - bar.left) / 2.0
    if half_width <= 0:
        return Suggestion.NONE, 0.0

    dx = dot_x if dot_x is not None else dot.x
    bc = bar_center if bar_center is not None else bar.center

    error = dx - bc
    error_ratio = max(-1.5, min(1.5, error / half_width))

    effective_threshold = max(2.0, control.threshold * (1.0 - s * 0.55))
    if abs(error) <= effective_threshold:
        return Suggestion.NONE, 0.0

    if dx < bar.left + control.margin:
        direction = Suggestion.D
    elif dx > bar.right - control.margin:
        direction = Suggestion.A
    elif error > 0:
        direction = Suggestion.A
    else:
        direction = Suggestion.D

    abs_err = min(1.25, abs(error_ratio))
    p_ms = 10.0 + (abs_err ** 1.15) * (48.0 + s * 42.0)

    d_factor = 1.0
    if prev_error_ratio is not None:
        delta_abs = abs(error_ratio) - abs(prev_error_ratio)
        crossed_center = error_ratio * prev_error_ratio < 0
        d_factor = 1.0 + delta_abs * (1.30 + s * 1.80)
        if crossed_center:
            d_factor *= 0.55
        elif delta_abs < 0:
            d_factor *= 0.75
        d_factor = max(0.40, min(1.65, d_factor))

    edge_boost = 1.0
    if abs_err > 0.85:
        edge_boost = 1.10 + s * 0.25

    duration_ms = p_ms * d_factor * edge_boost
    max_ms = 130.0 + s * 20.0
    duration_ms = max(8.0, min(max_ms, duration_ms))

    return direction, duration_ms / 1000.0
