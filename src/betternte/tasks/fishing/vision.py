from __future__ import annotations

import cv2
import numpy as np

from betternte.core.models import HSVThreshold, ROI
from betternte.core.vision.hsv import (
    auto_hsv_bounds_from_point,
    crop_global,
    threshold_mask as _threshold_mask,
)
from betternte.tasks.fishing.models import (
    AppConfig,
    BarDetection,
    BlueCircleDetection,
    DotDetection,
    Observation,
)


def _contour_centroid(contour: np.ndarray) -> tuple[int, int] | None:
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        return None
    return int(round(moments["m10"] / moments["m00"])), int(round(moments["m01"] / moments["m00"]))


def detect_bar(frame: np.ndarray, source_roi: ROI, search_roi: ROI, hsv_threshold: HSVThreshold, min_area: int) -> BarDetection | None:
    """优先锁定细长水平轮廓，再用投影法计算绿条边界，避免天气/场景噪声干扰。"""
    view, relative = crop_global(frame, source_roi, search_roi)
    if view is None or relative is None:
        return None

    mask = _threshold_mask(view, hsv_threshold)
    bridge_kernel = np.ones((3, 9), dtype=np.uint8)
    analysis_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, bridge_kernel)
    total_mask_pixels = int(cv2.countNonZero(analysis_mask))
    if total_mask_pixels < min_area:
        return None

    contours, _ = cv2.findContours(analysis_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_candidate: tuple[float, tuple[int, int, int, int], float] | None = None
    min_width = max(12, int(view.shape[1] * 0.08))
    max_height = max(6, int(view.shape[0] * 0.95))
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_width or h > max_height:
            continue
        aspect = w / max(h, 1)
        if aspect < 2.0:
            continue
        fill_ratio = area / max(1.0, float(w * h))
        score = w * 3.0 + aspect * 18.0 + fill_ratio * 40.0
        candidate = (score, (x, y, w, h), area)
        if best_candidate is None or candidate[0] > best_candidate[0]:
            best_candidate = candidate

    if best_candidate is None:
        return None

    _, (box_x, box_y, box_w, box_h), _ = best_candidate
    roi_mask = analysis_mask[box_y : box_y + box_h, box_x : box_x + box_w]
    if roi_mask.size == 0:
        return None

    row_sums = roi_mask.sum(axis=1).astype(np.int32)
    max_row = int(row_sums.max())
    if max_row == 0:
        return None
    active_rows = np.where(row_sums > max_row * 0.20)[0]
    if len(active_rows) < 2:
        return None
    y1, y2 = int(active_rows[0]), int(active_rows[-1])
    h_bar = y2 - y1 + 1

    bar_strip = roi_mask[y1 : y2 + 1, :]
    col_sums = bar_strip.sum(axis=0).astype(np.int32)
    max_col = int(col_sums.max())
    if max_col == 0:
        return None

    col_threshold = max(1, int(max_col * 0.20))
    above = np.where(col_sums > col_threshold)[0]
    if len(above) < 3:
        return None

    left_x = box_x + int(above[0])
    right_x = box_x + int(above[-1])

    center_x = (left_x + right_x) // 2
    abs_left = source_roi.x + relative.x + left_x
    abs_right = source_roi.x + relative.x + right_x
    abs_center = source_roi.x + relative.x + center_x
    abs_y = source_roi.y + relative.y + box_y + y1
    w = right_x - left_x + 1
    bbox = ROI(name="bar_detection", x=abs_left, y=abs_y, w=w, h=h_bar)
    area = float(w * h_bar)
    return BarDetection(left=abs_left, right=abs_right, center=abs_center, bbox=bbox, area=area)


def _prepare_dot_mask(view: np.ndarray, hsv_threshold: HSVThreshold) -> np.ndarray:
    """黄线检测的 mask 预处理：闭运算 + 膨胀。"""
    mask = _threshold_mask(view, hsv_threshold, morph=False)
    # 垂直方向闭运算，连接竖条断裂区域
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 1), dtype=np.uint8))
    # 水平膨胀，扩大细竖条宽度
    mask = cv2.dilate(mask, np.ones((1, 3), dtype=np.uint8), iterations=1)
    return mask


def _detect_dot_contour(
    mask: np.ndarray,
    view_shape: tuple[int, ...],
    source_roi: ROI,
    relative: ROI,
    min_area: int,
    bar: BarDetection | None = None,
) -> DotDetection | None:
    """轮廓法检测竖条：找最佳竖直轮廓。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dot_min_area = max(10, min_area // 4)
    dot_max_area = max(dot_min_area * 10, int(view_shape[0] * view_shape[1] * 0.08))
    best_match: tuple[float, DotDetection] | None = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < dot_min_area or area > dot_max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        vertical_ratio = h / max(w, 1)
        if vertical_ratio < 1.2:
            continue
        fill_ratio = area / max(1.0, float(w * h))
        if fill_ratio < 0.10:
            continue
        center = _contour_centroid(contour)
        if center is None:
            center = (x + w // 2, y + h // 2)
        center_x, center_y = center
        abs_x = source_roi.x + relative.x + center_x
        abs_y = source_roi.y + relative.y + center_y
        score = vertical_ratio * 38.0 + fill_ratio * 90.0 + area * 0.30
        if vertical_ratio >= 2.0:
            score += 18.0
        if w <= max(6, int(view_shape[1] * 0.06)):
            score += 12.0
        if bar is not None:
            half = max(1.0, (bar.right - bar.left) / 2.0)
            max_vertical_offset = max(12.0, bar.bbox.h * 3.0)
            vertical_offset = abs(abs_y - (bar.bbox.y + bar.bbox.h / 2.0))
            if vertical_offset > max_vertical_offset:
                continue
            if abs_x < bar.left - half * 1.6 or abs_x > bar.right + half * 1.6:
                continue
            horizontal_offset = abs(abs_x - bar.center)
            score -= vertical_offset * 1.5
            score -= horizontal_offset * 0.10
            score += max(0.0, 18.0 - vertical_offset)
        bbox = ROI(name="dot_detection", x=source_roi.x + relative.x + x, y=source_roi.y + relative.y + y, w=w, h=h)
        detection = DotDetection(x=abs_x, y=abs_y, bbox=bbox, area=area)
        if best_match is None or score > best_match[0]:
            best_match = (score, detection)
    return best_match[1] if best_match is not None else None


def _detect_dot_projection(
    mask: np.ndarray,
    source_roi: ROI,
    relative: ROI,
    bar: BarDetection,
) -> DotDetection | None:
    """投影法检测竖条：在 bar 范围内用列求和找竖条峰值。轮廓法失败时的后备。"""
    # 将 bar 坐标转到 view 局部坐标
    bar_left_local = bar.left - source_roi.x - relative.x
    bar_right_local = bar.right - source_roi.x - relative.x
    margin = max(8, int((bar.right - bar.left) * 0.15))
    search_left = max(0, int(bar_left_local - margin))
    search_right = min(mask.shape[1], int(bar_right_local + margin))
    if search_left >= search_right:
        return None

    # 列投影：在 bar 范围内找黄色像素最多的列
    region = mask[:, search_left:search_right]
    col_sums = region.sum(axis=0).astype(np.float64)
    if col_sums.max() < 3:
        return None

    peak_local = int(np.argmax(col_sums))
    peak_col = search_left + peak_local

    # 以峰值列为中心取窄带（3列），做行投影确定竖条高度
    band_left = max(0, peak_col - 1)
    band_right = min(mask.shape[1], peak_col + 2)
    band = mask[:, band_left:band_right]
    row_sums = band.sum(axis=1).astype(np.float64)
    active_rows = np.where(row_sums > 0)[0]
    if len(active_rows) < 3:
        return None

    y1, y2 = int(active_rows[0]), int(active_rows[-1])
    h = y2 - y1 + 1
    area = float(band.sum())
    center_x = peak_col
    center_y = (y1 + y2) // 2

    abs_x = source_roi.x + relative.x + center_x
    abs_y = source_roi.y + relative.y + center_y
    bbox = ROI(
        name="dot_detection",
        x=source_roi.x + relative.x + band_left,
        y=source_roi.y + relative.y + y1,
        w=band_right - band_left,
        h=h,
    )
    return DotDetection(x=abs_x, y=abs_y, bbox=bbox, area=area)


def detect_dot(
    frame: np.ndarray,
    source_roi: ROI,
    search_roi: ROI,
    hsv_threshold: HSVThreshold,
    min_area: int,
    bar: BarDetection | None = None,
) -> DotDetection | None:
    """检测黄线/竖条标记。先尝试轮廓法，失败后在 bar 范围内用投影法搜索。"""
    view, relative = crop_global(frame, source_roi, search_roi)
    if view is None or relative is None:
        return None

    mask = _prepare_dot_mask(view, hsv_threshold)

    # 方法1：轮廓法
    result = _detect_dot_contour(mask, view.shape, source_roi, relative, min_area, bar)
    if result is not None:
        return result

    # 方法2：投影法后备（仅在 bar 已知时）
    if bar is not None:
        return _detect_dot_projection(mask, source_roi, relative, bar)

    return None


def _circularity(contour: np.ndarray) -> float:
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return 0.0
    area = cv2.contourArea(contour)
    return (4.0 * np.pi * area) / (perimeter * perimeter)


def detect_blue_circle(frame: np.ndarray, source_roi: ROI, search_roi: ROI, hsv_threshold: HSVThreshold, min_area: int) -> BlueCircleDetection:
    view, relative = crop_global(frame, source_roi, search_roi)
    if view is None or relative is None:
        return BlueCircleDetection(found=False)
    mask = _threshold_mask(view, hsv_threshold)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in contours if cv2.contourArea(c) >= min_area and _circularity(c) >= 0.45]
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
    dot = detect_dot(frame, source_roi, config.rois.bar_area, config.hsv_thresholds.dot, config.control.min_area, bar=bar)
    blue_circle = detect_blue_circle(frame, source_roi, config.rois.blue_circle_area, config.hsv_thresholds.blue_circle, config.control.min_area)

    return Observation(
        bar=bar,
        dot=dot,
        blue_circle=blue_circle,
        bar_visible=bar is not None,
        dot_visible=dot is not None,
    )
