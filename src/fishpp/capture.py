from __future__ import annotations

import mss
import numpy as np
import cv2

from fishpp.models import CaptureConfig, ROI


class ScreenCapture:
    def __init__(self) -> None:
        self._sct = mss.MSS()

    def list_monitors(self) -> list[dict]:
        """Return physical monitors with mss index embedded."""
        result = []
        # monitors[0] is the virtual combined screen, [1:] are physical
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
        return self.grab_fullscreen(capture_config.monitor_index)
