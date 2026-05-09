from __future__ import annotations

import cv2
import numpy as np

from fishpp.models import (
    AppConfig,
    BarDetection,
    BlueCircleDetection,
    DotDetection,
    HSVThreshold,
    Observation,
    ROI,
)


def crop_global(frame: np.ndarray, source_roi: ROI, target_roi: ROI) -> tuple[np.ndarray | None, ROI | None]:
    if frame is None or not target_roi.valid():
        return None, None
    relative = target_roi.relative_to(source_roi.x, source_roi.y)
    relative = relative.clamp(frame.shape[1], frame.shape[0])
    if not relative.valid():
        return None, None
    view = frame[relative.y : relative.y + relative.h, relative.x : relative.x + relative.w].copy()
    return view, relative


def auto_hsv_bounds_from_point(frame: np.ndarray, x: int, y: int, delta_h: int = 10, delta_s: int = 60, delta_v: int = 60) -> tuple[list[int], list[int], tuple[int, int, int]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sample = hsv[y, x]
    h, s, v = [int(value) for value in sample]
    lower = [max(0, h - delta_h), max(0, s - delta_s), max(0, v - delta_v)]
    upper = [min(179, h + delta_h), 255, 255]
    return lower, upper, (h, s, v)


def _threshold_mask(view: np.ndarray, threshold: HSVThreshold, morph: bool = True) -> np.ndarray:
    hsv = cv2.cvtColor(view, cv2.COLOR_BGR2HSV)
    lower = np.array(threshold.lower, dtype=np.uint8)
    upper = np.array(threshold.upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    if morph:
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_bar(frame: np.ndarray, source_roi: ROI, search_roi: ROI, hsv_threshold: HSVThreshold, min_area: int) -> BarDetection | None:
    """用水平投影找绿条的真实左右边界，而非 blob 包围盒。

    1. HSV 掩码 → 2. 找绿条所在的行范围 → 3. 水平投影找精确左右边界 → 4. 中点 = (左+右)/2
    """
    view, relative = crop_global(frame, source_roi, search_roi)
    if view is None or relative is None:
        return None

    mask = _threshold_mask(view, hsv_threshold)
    total_mask_pixels = int(cv2.countNonZero(mask))
    if total_mask_pixels < min_area:
        return None

    # ── 找绿条的垂直行范围 ──
    row_sums = mask.sum(axis=1).astype(np.int32)
    max_row = int(row_sums.max())
    if max_row == 0:
        return None
    active_rows = np.where(row_sums > max_row * 0.15)[0]
    if len(active_rows) < 2:
        return None
    y1, y2 = int(active_rows[0]), int(active_rows[-1])
    h_bar = y2 - y1 + 1

    # ── 在绿条行范围内做水平投影 ──
    bar_strip = mask[y1 : y2 + 1, :]
    col_sums = bar_strip.sum(axis=0).astype(np.int32)
    max_col = int(col_sums.max())
    if max_col == 0:
        return None

    # 动态阈值：最大列和的 15%
    col_threshold = max(1, int(max_col * 0.15))
    above = np.where(col_sums > col_threshold)[0]
    if len(above) < 3:
        return None

    # ── 找连续的最大段（真正的绿条区间） ──
    # 可能有多个绿色块，取跨度最大的连续段
    gaps = np.diff(above)
    split_points = np.where(gaps > 3)[0]  # >3px 的间隔视为断开
    segments = []
    start = above[0]
    for sp in split_points:
        end = above[sp]
        segments.append((int(start), int(end)))
        start = above[sp + 1]
    segments.append((int(start), int(above[-1])))
    widest_seg = max(segments, key=lambda seg: seg[1] - seg[0])
    left_x, right_x = widest_seg

    # ── 输出 ──
    center_x = (left_x + right_x) // 2
    abs_left = source_roi.x + relative.x + left_x
    abs_right = source_roi.x + relative.x + right_x
    abs_center = source_roi.x + relative.x + center_x
    abs_y = source_roi.y + relative.y + y1
    w = right_x - left_x + 1
    bbox = ROI(name="bar_detection", x=abs_left, y=abs_y, w=w, h=h_bar)
    area = float(w * h_bar)
    return BarDetection(left=abs_left, right=abs_right, center=abs_center, bbox=bbox, area=area)


def detect_dot(frame: np.ndarray, source_roi: ROI, search_roi: ROI, hsv_threshold: HSVThreshold, min_area: int) -> DotDetection | None:
    view, relative = crop_global(frame, source_roi, search_roi)
    if view is None or relative is None:
        return None
    mask = _threshold_mask(view, hsv_threshold, morph=False)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dot_min_area = max(10, min_area // 4)
    filtered = [c for c in contours if cv2.contourArea(c) >= dot_min_area]
    if not filtered:
        return None
    contour = max(filtered, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    abs_x = source_roi.x + relative.x + x
    abs_y = source_roi.y + relative.y + y
    bbox = ROI(name="dot_detection", x=abs_x, y=abs_y, w=w, h=h)
    return DotDetection(x=abs_x + w // 2, y=abs_y + h // 2, bbox=bbox, area=float(cv2.contourArea(contour)))


def detect_blue_circle(frame: np.ndarray, source_roi: ROI, search_roi: ROI, hsv_threshold: HSVThreshold, min_area: int) -> BlueCircleDetection:
    view, relative = crop_global(frame, source_roi, search_roi)
    if view is None or relative is None:
        return BlueCircleDetection(found=False)
    mask = _threshold_mask(view, hsv_threshold)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not filtered:
        return BlueCircleDetection(found=False)
    contour = max(filtered, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    abs_x = source_roi.x + relative.x + x
    abs_y = source_roi.y + relative.y + y
    bbox = ROI(name="blue_circle_detection", x=abs_x, y=abs_y, w=w, h=h)
    return BlueCircleDetection(found=True, x=abs_x + w // 2, y=abs_y + h // 2, bbox=bbox)


def debug_detect(frame: np.ndarray, source_roi: ROI, search_roi: ROI, hsv_threshold: HSVThreshold, min_area: int) -> dict:
    """Return detailed debug info for a single detection."""
    view, relative = crop_global(frame, source_roi, search_roi)
    if view is None or relative is None:
        return {"roi_valid": False, "view_shape": None, "mask_pixels": 0, "contours": 0, "largest_area": 0, "threshold": hsv_threshold.to_dict()}
    mask = _threshold_mask(view, hsv_threshold)
    mask_pixels = int(cv2.countNonZero(mask))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [int(cv2.contourArea(c)) for c in contours]
    areas.sort(reverse=True)
    return {
        "roi_valid": True,
        "view_shape": f"{view.shape[1]}x{view.shape[0]}",
        "mask_pixels": mask_pixels,
        "contours": len(contours),
        "contour_areas": areas[:5],
        "min_area": min_area,
        "threshold": hsv_threshold.to_dict(),
    }


def analyze_frame(frame: np.ndarray, source_roi: ROI, config: AppConfig) -> Observation:
    bar = detect_bar(frame, source_roi, config.rois.bar_area, config.hsv_thresholds.bar, config.control.min_area)
    dot = detect_dot(frame, source_roi, config.rois.bar_area, config.hsv_thresholds.dot, config.control.min_area)
    blue_circle = detect_blue_circle(frame, source_roi, config.rois.blue_circle_area, config.hsv_thresholds.blue_circle, config.control.min_area)

    return Observation(
        bar=bar,
        dot=dot,
        blue_circle=blue_circle,
        bar_visible=bar is not None,
        dot_visible=dot is not None,
    )
