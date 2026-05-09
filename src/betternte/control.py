from __future__ import annotations

import ctypes
import ctypes.wintypes
import time

from betternte.models import BarDetection, ControlConfig, DotDetection, ROI, Suggestion

_user32 = ctypes.windll.user32
_shell32 = ctypes.windll.shell32

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000


def is_admin() -> bool:
    """检查当前进程是否以管理员权限运行。"""
    return bool(_shell32.IsUserAnAdmin())


def admin_status_text() -> str:
    if is_admin():
        return "管理员 OK"
    return "!! 非管理员 - F键可能被UIPI拦截"


def _click_at(x: int, y: int, screen_w: int, screen_h: int) -> None:
    norm_x = int(x * 65535 / max(screen_w - 1, 1))
    norm_y = int(y * 65535 / max(screen_h - 1, 1))

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", ctypes.c_long), ("dy", ctypes.c_long),
            ("mouseData", ctypes.wintypes.DWORD), ("dwFlags", ctypes.wintypes.DWORD),
            ("time", ctypes.wintypes.DWORD), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]
    class INPUT(ctypes.Structure):
        _fields_ = [("type", ctypes.wintypes.DWORD), ("mi", MOUSEINPUT)]

    zero_extra = ctypes.pointer(ctypes.c_ulong(0))

    inp = INPUT(type=0)
    inp.mi = MOUSEINPUT(norm_x, norm_y, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, 0, zero_extra)
    _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    inp.mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, zero_extra)
    _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    time.sleep(0.05)
    inp.mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, zero_extra)
    _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def compute_ad_pulse(
    bar: BarDetection | None,
    dot: DotDetection | None,
    control: ControlConfig,
    dot_x: float | None = None,
    bar_center: float | None = None,
    prev_error_ratio: float | None = None,
) -> tuple[Suggestion, float]:
    """PID 风格 A/D 控制，返回 (方向, 脉冲时长秒数)。

    控制力度 control.strength (0.0~1.0) 影响：
    - 死区大小（力度越大死区越小）
    - P 增益（力度越大按得越久）
    - D 增益（力度越大对速度越敏感）
    - 最高脉冲上限
    """
    import random

    if bar is None or dot is None:
        return Suggestion.NONE, 0.0

    s = control.strength  # 0.0 ~ 1.0

    half_width = (bar.right - bar.left) / 2.0
    if half_width <= 0:
        return Suggestion.NONE, 0.0

    dx = dot_x if dot_x is not None else dot.x
    bc = bar_center if bar_center is not None else bar.center

    error = dx - bc  # >0: dot 在中心右侧 → 需要 A
    error_ratio = error / half_width  # -1.0 ~ 1.0

    # 死区：力度越大死区越小（1.0 时缩小到 30%）
    effective_threshold = control.threshold * (1.0 - s * 0.7)
    if abs(error) <= effective_threshold:
        return Suggestion.NONE, 0.0

    # 方向判断
    if dx < bar.left + control.margin:
        direction = Suggestion.D
    elif dx > bar.right - control.margin:
        direction = Suggestion.A
    elif error > 0:
        direction = Suggestion.A
    else:
        direction = Suggestion.D

    # ── P 项：力度缩放 0.6x ~ 1.4x ──
    abs_err = min(1.0, abs(error_ratio))
    p_gain = 0.6 + s * 0.8
    p_ms = (22.0 + abs_err * 78.0) * p_gain

    # ── D 项：力度越大对变化越敏感 ──
    if prev_error_ratio is not None:
        de = error_ratio - prev_error_ratio  # >0: 误差在扩大（恶化）
        d_gain = 1.5 + s * 3.0  # 1.5 ~ 4.5
        d_factor = 1.0 + de * d_gain
        d_factor = max(0.20, min(2.5, d_factor))
    else:
        d_factor = 1.0

    duration_ms = p_ms * d_factor

    # 脉冲上限随力度提高（120~180ms）
    max_ms = 120.0 + s * 60.0

    # 轻微随机（±10%）
    jitter = random.uniform(-0.10, 0.10) * duration_ms
    duration_ms = max(12.0, min(max_ms, duration_ms + jitter))

    return direction, duration_ms / 1000.0


class InputController:
    def __init__(self) -> None:
        import pydirectinput
        self._di = pydirectinput
        self._pressed: set[str] = set()
        self._last_action_at = 0.0
        self._screen_size: tuple[int, int] = (1920, 1080)
        self._di.FAILSAFE = False

    def test_f_key(self) -> tuple[bool, str]:
        """独立测试 F 键按下，返回 (成功, 诊断信息)。"""
        parts: list[str] = []
        parts.append(f"管理员: {'是' if is_admin() else '否'}")
        try:
            ok_down = self._di.keyDown("f")
            parts.append(f"keyDown={'OK' if ok_down else 'FAIL'}")
            time.sleep(0.1)
            ok_up = self._di.keyUp("f")
            parts.append(f"keyUp={'OK' if ok_up else 'FAIL'}")
            ok = ok_down and ok_up
        except Exception as exc:
            ok = False
            parts.append(f"异常: {exc}")
        return ok, " · ".join(parts)

    @staticmethod
    def backend_supported() -> tuple[bool, str]:
        try:
            import pydirectinput  # noqa: F401
        except Exception as exc:
            return False, str(exc)
        return True, ""

    def set_screen_size(self, size: tuple[int, int]) -> None:
        self._screen_size = size

    def apply(self, suggestion: Suggestion, control_config: ControlConfig, click_roi: ROI | None = None, pulse_duration: float = 0.0, hold: bool = False) -> None:
        if suggestion == Suggestion.A:
            if hold:
                self._press("a")
                self._release("d")
            else:
                self.release_all()
                self._pulse_ad("a", pulse_duration)
        elif suggestion == Suggestion.D:
            if hold:
                self._press("d")
                self._release("a")
            else:
                self.release_all()
                self._pulse_ad("d", pulse_duration)
        elif suggestion == Suggestion.PRESS_F:
            self.release_all()
            self._tap("f")
        elif suggestion == Suggestion.CLICK_SCREEN:
            self.release_all()
            self._click(click_roi)
        else:
            self.release_all()

    def emergency_corner_stop(self, screen_size: tuple[int, int]) -> bool:
        try:
            point = ctypes.wintypes.POINT()
            _user32.GetCursorPos(ctypes.byref(point))
            x, y = point.x, point.y
        except Exception:
            return False
        width, height = screen_size
        return (x <= 1 and y <= 1) or (x >= width - 2 and y <= 1) or (x <= 1 and y >= height - 2) or (x >= width - 2 and y >= height - 2)

    def release_all(self) -> None:
        for key in list(self._pressed):
            self._di.keyUp(key)
            self._pressed.discard(key)

    def _press(self, key: str) -> None:
        if key in self._pressed:
            return
        self._di.keyDown(key)
        self._pressed.add(key)

    def _release(self, key: str) -> None:
        if key not in self._pressed:
            return
        self._di.keyUp(key)
        self._pressed.discard(key)

    def _tap(self, key: str) -> None:
        now = time.monotonic()
        if now - self._last_action_at < 0.3:
            return
        self._last_action_at = now
        self._di.keyDown(key)
        time.sleep(0.08)
        self._di.keyUp(key)

    def _pulse_ad(self, key: str, duration_s: float) -> None:
        """短脉冲 A/D，时长由 compute_ad_pulse 预先计算。"""
        duration_s = max(0.015, min(0.12, duration_s))
        self._di.keyDown(key)
        time.sleep(duration_s)
        self._di.keyUp(key)

    def _click(self, roi: ROI | None) -> None:
        if roi is None or not roi.valid():
            return
        now = time.monotonic()
        if now - self._last_action_at < 0.3:
            return
        self._last_action_at = now
        x, y = roi.center()
        self._di.moveTo(x, y)
        time.sleep(0.02)
        self._di.mouseDown()
        time.sleep(0.05)
        self._di.mouseUp()
