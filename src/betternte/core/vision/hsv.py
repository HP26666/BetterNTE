"""core/vision/hsv.py — 通用 HSV 处理工具（从 tasks/fishing/vision.py 提取）。"""
from __future__ import annotations

import cv2
import numpy as np

from betternte.core.models import HSVThreshold, ROI


def crop_global(frame: np.ndarray, source_roi: ROI, target_roi: ROI) -> tuple[np.ndarray | None, ROI | None]:
    """从全局坐标裁剪 ROI，返回裁剪后图像和相对坐标 ROI。"""
    if frame is None or not target_roi.valid():
        return None, None
    relative = target_roi.relative_to(source_roi.x, source_roi.y)
    relative = relative.clamp(frame.shape[1], frame.shape[0])
    if not relative.valid():
        return None, None
    view = frame[relative.y : relative.y + relative.h, relative.x : relative.x + relative.w].copy()
    return view, relative


def auto_hsv_bounds_from_point(
    frame: np.ndarray,
    x: int,
    y: int,
    delta_h: int = 10,
    delta_s: int = 60,
    delta_v: int = 60,
) -> tuple[list[int], list[int], tuple[int, int, int]]:
    """根据图像中某点的 HSV 自动计算阈值范围。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sample = hsv[y, x]
    h, s, v = [int(value) for value in sample]
    lower = [max(0, h - delta_h), max(0, s - delta_s), max(0, v - delta_v)]
    upper = [min(179, h + delta_h), 255, 255]
    return lower, upper, (h, s, v)


def threshold_mask(view: np.ndarray, threshold: HSVThreshold, morph: bool = True) -> np.ndarray:
    """对图像应用 HSV 阈值，可选形态学开闭运算去噪。"""
    hsv = cv2.cvtColor(view, cv2.COLOR_BGR2HSV)
    lower = np.array(threshold.lower, dtype=np.uint8)
    upper = np.array(threshold.upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    if morph:
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# 向后兼容的别名
_threshold_mask = threshold_mask
