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


_DOT_HORIZONTAL_DILATE_KERNEL = np.ones((1, 5), dtype=np.uint8)
_DOT_VERTICAL_CLOSE_KERNEL = np.ones((7, 1), dtype=np.uint8)
_DOT_SOLIDIFY_KERNEL = np.ones((3, 3), dtype=np.uint8)


def _contour_centroid(contour: np.ndarray) -> tuple[int, int] | None:
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        return None
    return int(round(moments["m10"] / moments["m00"])), int(round(moments["m01"] / moments["m00"]))


def _mask_edge_touches(mask: np.ndarray) -> dict[str, bool]:
    active = mask > 0
    if active.size == 0:
        return {
            "touches_left": False,
            "touches_right": False,
            "touches_top": False,
            "touches_bottom": False,
        }
    return {
        "touches_left": bool(active[:, 0].any()),
        "touches_right": bool(active[:, -1].any()),
        "touches_top": bool(active[0, :].any()),
        "touches_bottom": bool(active[-1, :].any()),
    }


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


def prepare_dot_mask(view: np.ndarray, hsv_threshold: HSVThreshold) -> np.ndarray:
    """黄线检测的 mask 预处理：闭运算 + 膨胀。"""
    mask = _threshold_mask(view, hsv_threshold, morph=False)
    # 先横向加粗，再纵向闭运算，兼容有一定宽度的 I 形竖条和抗锯齿断裂。
    mask = cv2.dilate(mask, _DOT_HORIZONTAL_DILATE_KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _DOT_VERTICAL_CLOSE_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _DOT_SOLIDIFY_KERNEL)
    return mask


_prepare_dot_mask = prepare_dot_mask


def _serialize_dot_candidate(candidate: DotDetection | None) -> dict[str, float | int] | None:
    if candidate is None:
        return None
    width = int(candidate.bbox.w)
    height = int(candidate.bbox.h)
    return {
        "x": int(candidate.x),
        "y": int(candidate.y),
        "w": width,
        "h": height,
        "area": float(candidate.area),
        "vertical_ratio": round(height / max(width, 1), 2),
    }


def draw_debug_panel(
    image: np.ndarray,
    lines: list[str],
    origin: tuple[int, int] = (8, 8),
    panel_color: tuple[int, int, int] = (15, 23, 42),
    accent_color: tuple[int, int, int] = (234, 179, 8),
    font_scale: float = 0.42,
    thickness: int = 1,
    line_gap: int = 6,
    padding: int = 8,
) -> None:
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    max_width = max((size[0] for size in text_sizes), default=0)
    line_height = max((size[1] for size in text_sizes), default=12)
    width = max_width + padding * 2
    height = len(lines) * line_height + max(0, len(lines) - 1) * line_gap + padding * 2

    x1 = max(0, origin[0])
    y1 = max(0, origin[1])
    x2 = min(image.shape[1] - 1, x1 + width)
    y2 = min(image.shape[0] - 1, y1 + height)
    cv2.rectangle(image, (x1, y1), (x2, y2), panel_color, -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), accent_color, 1)

    baseline_y = y1 + padding + line_height
    for index, line in enumerate(lines):
        text_y = baseline_y + index * (line_height + line_gap)
        if text_y > y2 - 4:
            break
        cv2.putText(image, line, (x1 + padding, text_y), font, font_scale, (248, 250, 252), thickness, cv2.LINE_AA)


def format_edge_flags(edge_touches: dict[str, bool] | None) -> str:
    edge_touches = edge_touches or {}
    return "".join(
        label
        for key, label in (
            ("touches_top", "T"),
            ("touches_bottom", "B"),
            ("touches_left", "L"),
            ("touches_right", "R"),
        )
        if edge_touches.get(key)
    ) or "-"


def dot_debug_lines(dot_info: dict | None, smoothed_x: float | None = None) -> list[str]:
    if not dot_info or not dot_info.get("roi_valid"):
        return []
    candidate = dot_info.get("final_candidate") or dot_info.get("raw_candidate")
    lines = [
        f"DOT raw={dot_info.get('raw_mask_pixels', 0)} prep={dot_info.get('prepared_mask_pixels', 0)} cnt={dot_info.get('prepared_contours', 0)}",
        f"mode={dot_info.get('final_mode', 'none')} proj={'Y' if dot_info.get('projection_found') else 'N'} edge={format_edge_flags(dot_info.get('edge_touches'))}",
    ]
    if candidate is not None:
        lines.append(
            f"x={candidate['x']} y={candidate['y']} w={candidate['w']} h={candidate['h']} r={candidate['vertical_ratio']:.2f}"
        )
    else:
        lines.append("x=- y=- w=- h=- r=-")
    if smoothed_x is not None:
        smooth_line = f"smooth_x={smoothed_x:.1f}"
        visible = dot_info.get("visible")
        if visible is not None:
            smooth_line += f" visible={'Y' if visible else 'N'}"
        lines.append(smooth_line)
    return lines


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
    dot_min_area = max(6, min_area // 6)
    dot_max_area = max(dot_min_area * 10, int(view_shape[0] * view_shape[1] * 0.08))
    best_match: tuple[float, DotDetection] | None = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < dot_min_area or area > dot_max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        if h < max(5, int(view_shape[0] * 0.10)):
            continue
        vertical_ratio = h / max(w, 1)
        if vertical_ratio < 1.0:
            continue
        fill_ratio = area / max(1.0, float(w * h))
        if fill_ratio < 0.06:
            continue
        center = _contour_centroid(contour)
        if center is None:
            center = (x + w // 2, y + h // 2)
        center_x, center_y = center
        abs_x = source_roi.x + relative.x + center_x
        abs_y = source_roi.y + relative.y + center_y
        score = vertical_ratio * 38.0 + fill_ratio * 90.0 + area * 0.30 + h * 0.8
        if vertical_ratio >= 2.0:
            score += 18.0
        preferred_max_width = max(8, int(view_shape[1] * 0.02))
        if w == 1:
            score -= 4.0
        elif w <= preferred_max_width:
            score += 10.0
        else:
            score += max(-8.0, 10.0 - (w - preferred_max_width) * 1.2)
        if bar is not None:
            half = max(1.0, (bar.right - bar.left) / 2.0)
            max_vertical_offset = max(18.0, bar.bbox.h * 4.5)
            vertical_offset = abs(abs_y - (bar.bbox.y + bar.bbox.h / 2.0))
            if vertical_offset > max_vertical_offset:
                continue
            if abs_x < bar.left - half * 2.2 or abs_x > bar.right + half * 2.2:
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
    margin = max(12, int((bar.right - bar.left) * 0.35))
    search_left = max(0, int(bar_left_local - margin))
    search_right = min(mask.shape[1], int(bar_right_local + margin))
    if search_left >= search_right:
        return None

    bar_center_local_y = bar.bbox.y + bar.bbox.h / 2.0 - source_roi.y - relative.y
    row_margin = max(14, int(bar.bbox.h * 4.5))
    search_top = max(0, int(bar_center_local_y - row_margin))
    search_bottom = min(mask.shape[0], int(bar_center_local_y + row_margin + 1))
    if search_top >= search_bottom:
        return None

    # 列投影：在 bar 范围内找黄色像素最多的列
    region = (mask[search_top:search_bottom, search_left:search_right] > 0).astype(np.uint8)
    col_pixels = region.sum(axis=0).astype(np.int32)
    if col_pixels.size == 0 or int(col_pixels.max()) < 2:
        return None

    col_scores = np.convolve(col_pixels, np.array([1, 2, 1], dtype=np.int32), mode="same")
    peak_local = int(np.argmax(col_scores))
    peak_col = search_left + peak_local

    # 以峰值列为中心取窄带（3列），做行投影确定竖条高度
    band_left = max(0, peak_col - 2)
    band_right = min(mask.shape[1], peak_col + 3)
    band = (mask[search_top:search_bottom, band_left:band_right] > 0).astype(np.uint8)
    row_pixels = band.sum(axis=1).astype(np.int32)
    active_rows = np.where(row_pixels > 0)[0]
    if len(active_rows) < 4:
        return None

    y1, y2 = int(active_rows[0]) + search_top, int(active_rows[-1]) + search_top
    h = y2 - y1 + 1
    area = float(int(band.sum()))
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

    mask = prepare_dot_mask(view, hsv_threshold)

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


def debug_detect(
    frame: np.ndarray,
    source_roi: ROI,
    search_roi: ROI,
    hsv_threshold: HSVThreshold,
    min_area: int,
    mode: str = "generic",
    bar: BarDetection | None = None,
) -> dict:
    """Return detailed debug info for a single detection."""
    view, relative = crop_global(frame, source_roi, search_roi)
    if view is None or relative is None:
        return {"roi_valid": False, "view_shape": None, "mask_pixels": 0, "contours": 0, "largest_area": 0, "threshold": hsv_threshold.to_dict()}

    if mode == "dot":
        raw_mask = _threshold_mask(view, hsv_threshold, morph=False)
        prepared_mask = prepare_dot_mask(view, hsv_threshold)
        raw_contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        prepared_contours, _ = cv2.findContours(prepared_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_areas = [int(cv2.contourArea(c)) for c in raw_contours]
        prepared_areas = [int(cv2.contourArea(c)) for c in prepared_contours]
        raw_areas.sort(reverse=True)
        prepared_areas.sort(reverse=True)
        raw_candidate = _detect_dot_contour(prepared_mask, view.shape, source_roi, relative, min_area, bar=None)
        filtered_candidate = _detect_dot_contour(prepared_mask, view.shape, source_roi, relative, min_area, bar=bar)
        projection_found = False
        projection_candidate = None
        if bar is not None:
            projection_candidate = _detect_dot_projection(prepared_mask, source_roi, relative, bar)
            projection_found = projection_candidate is not None
        final_candidate = filtered_candidate or projection_candidate or raw_candidate
        if filtered_candidate is not None:
            final_mode = "contour"
        elif projection_candidate is not None:
            final_mode = "projection"
        elif raw_candidate is not None:
            final_mode = "raw"
        else:
            final_mode = "none"
        return {
            "roi_valid": True,
            "mode": mode,
            "view_shape": f"{view.shape[1]}x{view.shape[0]}",
            "mask_pixels": int(cv2.countNonZero(prepared_mask)),
            "raw_mask_pixels": int(cv2.countNonZero(raw_mask)),
            "prepared_mask_pixels": int(cv2.countNonZero(prepared_mask)),
            "contours": len(prepared_contours),
            "raw_contours": len(raw_contours),
            "prepared_contours": len(prepared_contours),
            "contour_areas": prepared_areas[:5],
            "raw_contour_areas": raw_areas[:5],
            "prepared_contour_areas": prepared_areas[:5],
            "min_area": min_area,
            "edge_touches": _mask_edge_touches(prepared_mask),
            "projection_found": projection_found,
            "raw_candidate": _serialize_dot_candidate(raw_candidate),
            "filtered_candidate": _serialize_dot_candidate(filtered_candidate),
            "projection_candidate": _serialize_dot_candidate(projection_candidate),
            "final_candidate": _serialize_dot_candidate(final_candidate),
            "final_mode": final_mode,
            "threshold": hsv_threshold.to_dict(),
        }

    mask = _threshold_mask(view, hsv_threshold)
    mask_pixels = int(cv2.countNonZero(mask))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [int(cv2.contourArea(c)) for c in contours]
    areas.sort(reverse=True)
    return {
        "roi_valid": True,
        "mode": mode,
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
