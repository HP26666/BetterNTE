from __future__ import annotations

import ctypes
import ctypes.wintypes
import time
from typing import Any

from betternte.core.models import ROI

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

    # ── 公共原子操作接口（与 PostMessageController 统一） ──

    def key_down(self, key: str) -> None:
        if key in self._pressed:
            return
        self._di.keyDown(key)
        self._pressed.add(key)

    def key_up(self, key: str) -> None:
        if key not in self._pressed:
            return
        self._di.keyUp(key)
        self._pressed.discard(key)

    def tap(self, key: str, duration: float = 0.08) -> None:
        now = time.monotonic()
        if now - self._last_action_at < 0.3:
            return
        self._last_action_at = now
        self._di.keyDown(key)
        time.sleep(duration)
        self._di.keyUp(key)

    def pulse(self, key: str, duration: float) -> None:
        duration = max(0.015, min(0.12, duration))
        self._di.keyDown(key)
        time.sleep(duration)
        self._di.keyUp(key)

    def click(self, x: int, y: int) -> None:
        now = time.monotonic()
        if now - self._last_action_at < 0.3:
            return
        self._last_action_at = now
        self._di.moveTo(x, y)
        time.sleep(0.02)
        self._di.mouseDown()
        time.sleep(0.05)
        self._di.mouseUp()

    def release_all(self) -> None:
        for key in list(self._pressed):
            self._di.keyUp(key)
            self._pressed.discard(key)

    def emergency_corner_stop(self, screen_ref: tuple[int, int] | ROI) -> bool:
        try:
            point = ctypes.wintypes.POINT()
            _user32.GetCursorPos(ctypes.byref(point))
            x, y = point.x, point.y
        except Exception:
            return False
        if isinstance(screen_ref, ROI):
            left = screen_ref.x
            top = screen_ref.y
            width = screen_ref.w
            height = screen_ref.h
        else:
            left = 0
            top = 0
            width, height = screen_ref
        right = left + width - 1
        bottom = top + height - 1
        return (
            (x <= left + 1 and y <= top + 1)
            or (x >= right - 1 and y <= top + 1)
            or (x <= left + 1 and y >= bottom - 1)
            or (x >= right - 1 and y >= bottom - 1)
        )


# ──────────────────────────────────────────────────────────────────────────────
#  PostMessage 模式：向游戏窗口发送消息，无需管理员权限，支持后台运行
# ──────────────────────────────────────────────────────────────────────────────

# Windows 消息常量
_WM_KEYDOWN = 0x0100
_WM_KEYUP = 0x0101
_WM_LBUTTONDOWN = 0x0201
_WM_LBUTTONUP = 0x0202
_WM_MOUSEMOVE = 0x0200

# 常用虚拟键码
_VK_MAP: dict[str, int] = {
    "a": 0x41, "b": 0x42, "c": 0x43, "d": 0x44, "e": 0x45,
    "f": 0x46, "g": 0x47, "h": 0x48, "i": 0x49, "j": 0x4A,
    "k": 0x4B, "l": 0x4C, "m": 0x4D, "n": 0x4E, "o": 0x4F,
    "p": 0x50, "q": 0x51, "r": 0x52, "s": 0x53, "t": 0x54,
    "u": 0x55, "v": 0x56, "w": 0x57, "x": 0x58, "y": 0x59,
    "z": 0x5A,
    "0": 0x30, "1": 0x31, "2": 0x32, "3": 0x33, "4": 0x34,
    "5": 0x35, "6": 0x36, "7": 0x37, "8": 0x38, "9": 0x39,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73,
    "f5": 0x74, "f6": 0x75, "f7": 0x76, "f8": 0x77,
    "enter": 0x0D, "space": 0x20, "escape": 0x1B,
    "shift": 0x10, "ctrl": 0x11, "alt": 0x12,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
}


def _make_lparam(row: int, col: int) -> int:
    """封装鼠标坐标到 lParam。"""
    return (row & 0xFFFF) << 16 | (col & 0xFFFF)


def _key_lparam(vk: int, key_up: bool = False) -> int:
    """封装键盘消息 lParam (scan code 简化版)。"""
    import win32api
    scan = win32api.MapVirtualKey(vk, 0)
    repeat = 1
    extended = 0
    prev_down = 0 if not key_up else 1
    transition = 1 if key_up else 0
    return (transition << 31) | (prev_down << 30) | (extended << 24) | (scan << 16) | repeat


class PostMessageController:
    """通过 PostMessage 向游戏窗口发送键盘/鼠标消息（支持后台，无需管理员）。"""

    def __init__(self, hwnd: int) -> None:
        self.hwnd = hwnd
        self._pressed_vk: set[int] = set()
        self._last_action_at = 0.0

    def _vk(self, key: str) -> int:
        vk = _VK_MAP.get(key.lower())
        if vk is None:
            raise ValueError(f"未知按键: {key!r}")
        return vk

    def _post(self, msg: int, wparam: int, lparam: int) -> None:
        try:
            import win32api
            win32api.PostMessage(self.hwnd, msg, wparam, lparam)
        except Exception:
            pass

    def key_down(self, key: str) -> None:
        vk = self._vk(key)
        if vk in self._pressed_vk:
            return
        self._post(_WM_KEYDOWN, vk, _key_lparam(vk, False))
        self._pressed_vk.add(vk)

    def key_up(self, key: str) -> None:
        vk = self._vk(key)
        self._post(_WM_KEYUP, vk, _key_lparam(vk, True))
        self._pressed_vk.discard(vk)

    def tap(self, key: str, duration: float = 0.08) -> None:
        now = time.monotonic()
        if now - self._last_action_at < 0.3:
            return
        self._last_action_at = now
        self.key_down(key)
        time.sleep(duration)
        self.key_up(key)

    def pulse(self, key: str, duration: float) -> None:
        duration = max(0.015, min(0.12, duration))
        self.key_down(key)
        time.sleep(duration)
        self.key_up(key)

    def release_all(self) -> None:
        for vk in list(self._pressed_vk):
            self._post(_WM_KEYUP, vk, _key_lparam(vk, True))
        self._pressed_vk.clear()

    def _screen_to_client(self, x: int, y: int) -> tuple[int, int]:
        try:
            import win32gui

            return win32gui.ScreenToClient(self.hwnd, (x, y))
        except Exception:
            return x, y

    def click(self, x: int, y: int) -> None:
        """在屏幕坐标 (x,y) 处发送左键点击。"""
        now = time.monotonic()
        if now - self._last_action_at < 0.3:
            return
        self._last_action_at = now
        client_x, client_y = self._screen_to_client(x, y)
        lp = _make_lparam(client_y, client_x)
        self._post(_WM_LBUTTONDOWN, 0x0001, lp)
        time.sleep(0.05)
        self._post(_WM_LBUTTONUP, 0x0000, lp)

    def emergency_corner_stop(self, screen_ref: tuple[int, int] | ROI) -> bool:
        """PostMessage 模式下不支持鼠标角落急停，始终返回 False。"""
        return False
