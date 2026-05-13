"""COCO 特征模板匹配接口。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np

from betternte.core.image_io import load_image_bgr
from betternte.core.models import ROI


@dataclass(slots=True)
class Box:
    """模板匹配结果。"""

    x: int
    y: int
    w: int
    h: int
    confidence: float
    name: str = ""

    @property
    def bbox(self) -> ROI:
        return ROI(name=self.name or "feature", x=self.x, y=self.y, w=self.w, h=self.h)


@dataclass(slots=True)
class Feature:
    """单个模板特征。"""

    name: str
    template: np.ndarray
    bbox: ROI
    threshold: float = 0.95
    use_gray_scale: bool = False
    scaling: float = 1.0

    @property
    def width(self) -> int:
        return int(self.template.shape[1])

    @property
    def height(self) -> int:
        return int(self.template.shape[0])


@dataclass(slots=True)
class _RawFeature:
    name: str
    image_path: Path
    image_width: int
    image_height: int
    bbox: tuple[float, float, float, float]
    threshold: float
    use_gray_scale: bool


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _normalize_image_path(base_dir: Path, file_name: str) -> Path:
    file_name = (file_name or "").replace("\\", "/")
    if file_name.startswith("/data/local-files/?d="):
        file_name = file_name.split("=", 1)[1]
    path = Path(file_name)
    if path.is_absolute():
        return path
    if file_name.startswith("images/"):
        return base_dir / file_name
    return base_dir / file_name


def _scale_by_anchor(value: float, source_dim: int, target_dim: int, scale: float) -> int:
    if value > source_dim / 2:
        return int(round(target_dim - (source_dim - value) * scale))
    return int(round(value * scale))


def _adjust_bbox(
    x: float,
    y: float,
    w: float,
    h: float,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int, int, int, float]:
    if target_width <= 0 or target_height <= 0 or source_width <= 0 or source_height <= 0:
        return int(round(x)), int(round(y)), max(1, int(round(w))), max(1, int(round(h))), 1.0
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    scale = min(scale_x, scale_y)
    scaled_w = max(1, int(round(w * scale)))
    scaled_h = max(1, int(round(h * scale)))
    scaled_x = _scale_by_anchor(x, source_width, target_width, scale)
    scaled_y = _scale_by_anchor(y, source_height, target_height, scale)
    return scaled_x, scaled_y, scaled_w, scaled_h, scale


def _filter_matches(result: np.ndarray, threshold: float, width: int, height: int) -> list[tuple[tuple[int, int], float]]:
    ys, xs = np.where(result >= threshold)
    matches = sorted(
        [((int(x), int(y)), float(result[y, x])) for x, y in zip(xs, ys)],
        key=lambda item: item[1],
        reverse=True,
    )
    selected: list[tuple[tuple[int, int], float]] = []
    for (x, y), confidence in matches:
        overlaps = False
        for (other_x, other_y), _ in selected:
            if x < other_x + width and x + width > other_x and y < other_y + height and y + height > other_y:
                overlaps = True
                break
        if not overlaps:
            selected.append(((x, y), confidence))
    return selected


class FeatureSet:
    """从 COCO JSON 加载模板，并在截图中执行模板匹配。"""

    def __init__(self, coco_json: str | Path, default_threshold: float = 0.95) -> None:
        self.coco_json = Path(coco_json)
        self.default_threshold = float(default_threshold)
        self._lock = Lock()
        self._raw_features: dict[str, _RawFeature] = {}
        self._features: dict[str, Feature] = {}
        self._source_images: dict[Path, np.ndarray] = {}
        self._target_size: tuple[int, int] | None = None
        self._load_definitions()

    def _load_definitions(self) -> None:
        self._raw_features.clear()
        self._features.clear()
        self._target_size = None
        if not self.coco_json.exists():
            return
        with self.coco_json.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        categories = {int(item["id"]): str(item["name"]) for item in data.get("categories", []) if "id" in item and "name" in item}
        images = {int(item["id"]): item for item in data.get("images", []) if "id" in item}
        base_dir = self.coco_json.parent
        for ann in data.get("annotations", []):
            category_name = categories.get(int(ann.get("category_id", -1)), "")
            image_info = images.get(int(ann.get("image_id", -1)))
            bbox = ann.get("bbox") or []
            if not category_name or image_info is None or len(bbox) != 4:
                continue
            image_path = _normalize_image_path(base_dir, str(image_info.get("file_name", "")))
            width = int(image_info.get("width", 0))
            height = int(image_info.get("height", 0))
            attributes = ann.get("attributes") or {}
            self._raw_features[category_name] = _RawFeature(
                name=category_name,
                image_path=image_path,
                image_width=width,
                image_height=height,
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                threshold=float(attributes.get("threshold", self.default_threshold)),
                use_gray_scale=bool(attributes.get("use_gray_scale", False)),
            )

    def reload(self) -> None:
        with self._lock:
            self._source_images.clear()
            self._load_definitions()

    def empty(self) -> bool:
        return not self._raw_features

    def has_definition(self, name: str) -> bool:
        return name in self._raw_features

    def _read_source_image(self, image_path: Path) -> np.ndarray | None:
        cached = self._source_images.get(image_path)
        if cached is not None:
            return cached
        image = load_image_bgr(image_path)
        if image is None:
            return None
        self._source_images[image_path] = image
        return image

    def _ensure_features(self, frame: np.ndarray) -> None:
        target_size = (int(frame.shape[1]), int(frame.shape[0]))
        with self._lock:
            if self._target_size == target_size and self._features:
                return
            self._features = {}
            for raw in self._raw_features.values():
                source = self._read_source_image(raw.image_path)
                if source is None:
                    continue
                src_x, src_y, src_w, src_h = raw.bbox
                x0 = max(0, int(round(src_x)))
                y0 = max(0, int(round(src_y)))
                x1 = min(source.shape[1], int(round(src_x + src_w)))
                y1 = min(source.shape[0], int(round(src_y + src_h)))
                if x1 <= x0 or y1 <= y0:
                    continue
                template = source[y0:y1, x0:x1].copy()
                if template.size == 0:
                    continue
                adj_x, adj_y, adj_w, adj_h, scale = _adjust_bbox(
                    src_x,
                    src_y,
                    src_w,
                    src_h,
                    raw.image_width or source.shape[1],
                    raw.image_height or source.shape[0],
                    target_size[0],
                    target_size[1],
                )
                if (adj_w, adj_h) != (template.shape[1], template.shape[0]):
                    template = cv2.resize(template, (adj_w, adj_h), interpolation=cv2.INTER_AREA)
                self._features[raw.name] = Feature(
                    name=raw.name,
                    template=template,
                    bbox=ROI(name=raw.name, x=adj_x, y=adj_y, w=adj_w, h=adj_h),
                    threshold=raw.threshold,
                    use_gray_scale=raw.use_gray_scale,
                    scaling=scale,
                )
            self._target_size = target_size

    def get_feature(self, name: str, frame: np.ndarray | None = None) -> Feature | None:
        if frame is not None:
            self._ensure_features(frame)
        return self._features.get(name)

    def find_feature(
        self,
        name: str,
        frame: np.ndarray,
        threshold: float | None = None,
        *,
        use_gray_scale: bool | None = None,
        roi: ROI | None = None,
        limit: int = 0,
        match_method: int = cv2.TM_CCOEFF_NORMED,
    ) -> list[Box]:
        if frame is None or frame.size == 0:
            return []
        self._ensure_features(frame)
        feature = self._features.get(name)
        if feature is None:
            return []

        effective_threshold = float(feature.threshold if threshold is None else threshold)
        effective_gray = feature.use_gray_scale if use_gray_scale is None else use_gray_scale
        search_roi = roi.clamp(frame.shape[1], frame.shape[0]) if roi is not None else ROI(name="frame", x=0, y=0, w=frame.shape[1], h=frame.shape[0])
        if not search_roi.valid():
            return []

        search_area = frame[search_roi.y : search_roi.y + search_roi.h, search_roi.x : search_roi.x + search_roi.w]
        template = feature.template
        if template.shape[1] > search_area.shape[1] or template.shape[0] > search_area.shape[0]:
            return []

        if effective_gray:
            search_area = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if template.ndim == 3 else template

        result = cv2.matchTemplate(search_area, template, match_method)
        result[np.isnan(result)] = 0
        result[np.isinf(result)] = 0

        if limit == 1 and match_method == cv2.TM_CCOEFF_NORMED:
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if float(max_val) < effective_threshold:
                return []
            return [
                Box(
                    x=int(max_loc[0] + search_roi.x),
                    y=int(max_loc[1] + search_roi.y),
                    w=feature.width,
                    h=feature.height,
                    confidence=float(max_val),
                    name=name,
                )
            ]

        matches = _filter_matches(result, effective_threshold, feature.width, feature.height)
        boxes = [
            Box(
                x=match_x + search_roi.x,
                y=match_y + search_roi.y,
                w=feature.width,
                h=feature.height,
                confidence=confidence,
                name=name,
            )
            for (match_x, match_y), confidence in matches
        ]
        if limit > 0:
            return boxes[:limit]
        return boxes

    def find_one(
        self,
        name: str,
        frame: np.ndarray,
        threshold: float | None = None,
        *,
        use_gray_scale: bool | None = None,
        roi: ROI | None = None,
        match_method: int = cv2.TM_CCOEFF_NORMED,
    ) -> Box | None:
        matches = self.find_feature(
            name,
            frame,
            threshold,
            use_gray_scale=use_gray_scale,
            roi=roi,
            limit=1,
            match_method=match_method,
        )
        return matches[0] if matches else None

    def feature_exists(
        self,
        name: str,
        frame: np.ndarray,
        threshold: float | None = None,
        *,
        use_gray_scale: bool | None = None,
        roi: ROI | None = None,
    ) -> bool:
        return self.find_one(name, frame, threshold, use_gray_scale=use_gray_scale, roi=roi) is not None


_DEFAULT_FEATURE_SET: FeatureSet | None = None


def get_default_feature_set() -> FeatureSet:
    global _DEFAULT_FEATURE_SET
    if _DEFAULT_FEATURE_SET is None:
        _DEFAULT_FEATURE_SET = FeatureSet(_project_root() / "assets" / "features.json")
    return _DEFAULT_FEATURE_SET


def find_feature(
    name: str,
    frame: np.ndarray,
    threshold: float | None = None,
    *,
    use_gray_scale: bool | None = None,
    roi: ROI | None = None,
    limit: int = 0,
    match_method: int = cv2.TM_CCOEFF_NORMED,
    feature_set: FeatureSet | None = None,
) -> list[Box]:
    matcher = feature_set or get_default_feature_set()
    return matcher.find_feature(
        name,
        frame,
        threshold,
        use_gray_scale=use_gray_scale,
        roi=roi,
        limit=limit,
        match_method=match_method,
    )


def find_one(
    name: str,
    frame: np.ndarray,
    threshold: float | None = None,
    *,
    use_gray_scale: bool | None = None,
    roi: ROI | None = None,
    match_method: int = cv2.TM_CCOEFF_NORMED,
    feature_set: FeatureSet | None = None,
) -> Box | None:
    matcher = feature_set or get_default_feature_set()
    return matcher.find_one(
        name,
        frame,
        threshold,
        use_gray_scale=use_gray_scale,
        roi=roi,
        match_method=match_method,
    )


def feature_exists(
    name: str,
    frame: np.ndarray,
    threshold: float | None = None,
    *,
    use_gray_scale: bool | None = None,
    roi: ROI | None = None,
    feature_set: FeatureSet | None = None,
) -> bool:
    matcher = feature_set or get_default_feature_set()
    return matcher.feature_exists(name, frame, threshold, use_gray_scale=use_gray_scale, roi=roi)


__all__ = ["Box", "Feature", "FeatureSet", "feature_exists", "find_feature", "find_one", "get_default_feature_set"]