from __future__ import annotations

import mss
import numpy as np
import cv2

from betternte.core.models import CaptureConfig, CaptureMode, ROI


def find_game_window(title_pattern: str = "", class_name: str = "") -> int | None:
    """通过标题或类名查找游戏窗口 HWND。title_pattern 为子串匹配。"""
    try:
        import win32gui

        result: list[int] = []

        def _enum_cb(hwnd: int, _: None) -> None:
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd)
            cls = win32gui.GetClassName(hwnd)
            title_ok = (not title_pattern) or (title_pattern.lower() in title.lower())
            class_ok = (not class_name) or (class_name.lower() == cls.lower())
            if title_ok and class_ok and title:
                result.append(hwnd)

        win32gui.EnumWindows(_enum_cb, None)
        return result[0] if result else None
    except Exception:
        return None


def list_game_windows(title_pattern: str = "", class_name: str = "") -> list[tuple[int, str, str]]:
    """列出所有匹配的可见窗口，返回 [(hwnd, title, class_name)]。"""
    try:
        import win32gui

        result: list[tuple[int, str, str]] = []

        def _enum_cb(hwnd: int, _: None) -> None:
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd)
            cls = win32gui.GetClassName(hwnd)
            if not title:
                return
            title_ok = (not title_pattern) or (title_pattern.lower() in title.lower())
            class_ok = (not class_name) or (class_name.lower() == cls.lower())
            if title_ok and class_ok:
                result.append((hwnd, title, cls))

        win32gui.EnumWindows(_enum_cb, None)
        return result
    except Exception:
        return []


def grab_window_bitblt(hwnd: int) -> tuple[np.ndarray, ROI] | None:
    """用 BitBlt 截取指定 HWND 的客户区，支持后台窗口（非最小化）。"""
    try:
        import win32gui
        import win32ui
        import win32con

        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        w = right - left
        h = bottom - top
        if w <= 0 or h <= 0:
            return None

        hwnd_dc = win32gui.GetDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
        save_dc.SelectObject(bitmap)

        try:
            save_dc.BitBlt((0, 0), (w, h), mfc_dc, (0, 0), win32con.SRCCOPY)

            bmp_info = bitmap.GetInfo()
            bmp_str = bitmap.GetBitmapBits(True)
            stride = bmp_info.get("bmWidthBytes", bmp_info["bmWidth"])
            img = np.frombuffer(bmp_str, dtype=np.uint8).reshape(h, stride // 4, 4)
            frame = cv2.cvtColor(img[:, :w, :], cv2.COLOR_BGRA2BGR)

            try:
                screen_left, screen_top = win32gui.ClientToScreen(hwnd, (0, 0))
            except Exception:
                screen_left, screen_top = 0, 0

            roi = ROI(name="window", x=screen_left, y=screen_top, w=w, h=h)
            return frame, roi
        finally:
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwnd_dc)
            win32gui.DeleteObject(bitmap.GetHandle())
    except Exception:
        return None


class ScreenCapture:
    def __init__(self) -> None:
        self._sct = mss.mss()

    def list_monitors(self) -> list[dict]:
        """返回物理显示器列表，包含 _mss_index。"""
        result = []
        for i, m in enumerate(self._sct.monitors[1:], start=1):
            entry = dict(m)
            entry["_mss_index"] = i
            result.append(entry)
        return result

    def screen_size(self, monitor_index: int = 1) -> tuple[int, int]:
        monitors = self._sct.monitors
        idx = max(1, min(monitor_index, len(monitors) - 1))
        monitor = monitors[idx]
        return int(monitor["width"]), int(monitor["height"])

    def grab_fullscreen(self, monitor_index: int = 1) -> tuple[np.ndarray, ROI]:
        monitors = self._sct.monitors
        idx = max(1, min(monitor_index, len(monitors) - 1))
        monitor = monitors[idx]
        image = np.array(self._sct.grab(monitor))
        frame = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return frame, ROI(name="screen", x=int(monitor["left"]), y=int(monitor["top"]), w=frame.shape[1], h=frame.shape[0])

    def grab_source(self, capture_config: CaptureConfig) -> tuple[np.ndarray, ROI]:
        """根据 CaptureConfig 模式选择截图方式。"""
        if capture_config.mode == CaptureMode.WINDOW_BITBLT:
            hwnd = find_game_window(capture_config.window_title, capture_config.window_class)
            if hwnd is None:
                raise RuntimeError("窗口截图失败: 未找到匹配的游戏窗口，请检查窗口标题或类名设置")
            result = grab_window_bitblt(hwnd)
            if result is None:
                raise RuntimeError("窗口截图失败: BitBlt 捕获失败，请确认目标窗口未最小化且客户区可见")
            return result
        return self.grab_fullscreen(capture_config.monitor_index)

    def grab_window(self, hwnd: int) -> tuple[np.ndarray, ROI] | None:
        """直接用 HWND 截取窗口。"""
        return grab_window_bitblt(hwnd)

