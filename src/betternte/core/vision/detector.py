"""core/vision/detector.py — 目标检测接口（YOLOv8 ONNX，可选依赖）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from betternte.core.models import ROI


@dataclass
class DetectionResult:
    label: str
    confidence: float
    bbox: ROI


class YOLODetector:
    """YOLOv8 ONNX 检测器（需要 onnxruntime）。"""

    def __init__(self) -> None:
        self._session: Any = None
        self._class_names: list[str] = []
        self._input_w = 640
        self._input_h = 640
        self._conf_threshold = 0.5
        self._iou_threshold = 0.45

    def load(self, model_path: str, class_names: list[str] | None = None, conf: float = 0.5) -> None:
        """加载 ONNX 模型。"""
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self._class_names = class_names or []
            self._conf_threshold = conf
        except Exception as exc:
            raise RuntimeError(f"加载 YOLO 模型失败: {exc}") from exc

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        """对帧进行目标检测，返回检测结果列表。"""
        if not self.loaded:
            return []
        try:
            input_tensor = self._preprocess(frame)
            outputs = self._session.run(None, {self._session.get_inputs()[0].name: input_tensor})
            return self._postprocess(outputs[0], frame.shape[1], frame.shape[0])
        except Exception:
            return []

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self._input_w, self._input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def _postprocess(self, output: np.ndarray, orig_w: int, orig_h: int) -> list[DetectionResult]:
        results: list[DetectionResult] = []
        output = output[0].T  # [num_boxes, 4+classes]
        scale_x = orig_w / self._input_w
        scale_y = orig_h / self._input_h
        for row in output:
            scores = row[4:]
            cls_id = int(np.argmax(scores))
            conf = float(scores[cls_id])
            if conf < self._conf_threshold:
                continue
            cx, cy, bw, bh = row[:4]
            x = int((cx - bw / 2) * scale_x)
            y = int((cy - bh / 2) * scale_y)
            w = int(bw * scale_x)
            h = int(bh * scale_y)
            label = self._class_names[cls_id] if cls_id < len(self._class_names) else str(cls_id)
            results.append(DetectionResult(label=label, confidence=conf, bbox=ROI(name=label, x=x, y=y, w=w, h=h)))
        return self._nms(results)

    def _nms(self, results: list[DetectionResult]) -> list[DetectionResult]:
        """非极大值抑制（NMS）。"""
        if not results:
            return results
        keep: list[bool] = [True] * len(results)
        for i in range(len(results)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(results)):
                if not keep[j]:
                    continue
                if results[i].label != results[j].label:
                    continue
                if _iou(results[i].bbox, results[j].bbox) > self._iou_threshold:
                    if results[j].confidence < results[i].confidence:
                        keep[j] = False
                    else:
                        keep[i] = False
        return [r for r, k in zip(results, keep) if k]


def _iou(a: ROI, b: ROI) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = max(1, a.w * a.h + b.w * b.h - inter)
    return inter / union
