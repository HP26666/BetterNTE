"""可选 OCR 接口，默认使用 easyocr。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from betternte.core.models import ROI


@dataclass(slots=True)
class OCRResult:
    """单条 OCR 结果。"""

    text: str
    confidence: float
    bbox: ROI


class OCR:
    """OCR 封装。引擎不可用时返回空结果，不阻塞主流程。"""

    def __init__(self, engine_type: str = "easyocr", languages: list[str] | None = None) -> None:
        self.engine_type = engine_type
        self.languages = languages or ["ch_sim", "en"]
        self._engine: Any = None
        self._engine_loaded = False
        self._engine_attempted = False
        self._engine_error = ""

    @property
    def loaded(self) -> bool:
        return self._engine_loaded and self._engine is not None

    @property
    def last_error(self) -> str:
        return self._engine_error

    def init_engine(self, engine_type: str | None = None) -> bool:
        if engine_type is not None:
            self.engine_type = engine_type
            self._engine = None
            self._engine_loaded = False
            self._engine_attempted = False
            self._engine_error = ""

        if self._engine_attempted:
            return self.loaded

        self._engine_attempted = True
        if self.engine_type != "easyocr":
            self._engine_error = f"不支持的 OCR 引擎: {self.engine_type}"
            return False

        try:
            import easyocr  # type: ignore

            self._engine = easyocr.Reader(self.languages, gpu=False)
            self._engine_loaded = True
        except Exception as exc:
            self._engine = None
            self._engine_loaded = False
            self._engine_error = str(exc)
        return self.loaded

    def ocr(
        self,
        frame: np.ndarray,
        region: ROI | None = None,
        match_text: str | None = None,
        regex: str | None = None,
    ) -> list[OCRResult]:
        if frame is None or frame.size == 0:
            return []

        roi = region.clamp(frame.shape[1], frame.shape[0]) if region is not None else ROI(name="frame", x=0, y=0, w=frame.shape[1], h=frame.shape[0])
        if not roi.valid():
            return []

        if not self.loaded and not self.init_engine():
            return []

        view = frame[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
        if view.size == 0:
            return []

        pattern = re.compile(regex) if regex else None
        target_text = match_text.lower() if match_text else None

        try:
            rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
            raw_results = self._engine.readtext(rgb, detail=1, paragraph=False)
        except Exception as exc:
            self._engine_error = str(exc)
            return []

        results: list[OCRResult] = []
        for item in raw_results:
            if len(item) < 3:
                continue
            points, text, confidence = item[0], str(item[1]).strip(), float(item[2])
            if not text:
                continue
            if target_text and target_text not in text.lower():
                continue
            if pattern and pattern.search(text) is None:
                continue

            xs = [int(round(point[0])) for point in points]
            ys = [int(round(point[1])) for point in points]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            results.append(
                OCRResult(
                    text=text,
                    confidence=confidence,
                    bbox=ROI(name="ocr", x=roi.x + x0, y=roi.y + y0, w=max(1, x1 - x0), h=max(1, y1 - y0)),
                )
            )
        return results


_DEFAULT_OCR: OCR | None = None


def get_default_ocr() -> OCR:
    global _DEFAULT_OCR
    if _DEFAULT_OCR is None:
        _DEFAULT_OCR = OCR()
    return _DEFAULT_OCR


def init_engine(engine_type: str = "easyocr") -> bool:
    return get_default_ocr().init_engine(engine_type)


def ocr(
    frame: np.ndarray,
    region: ROI | None = None,
    match_text: str | None = None,
    regex: str | None = None,
    *,
    engine: OCR | None = None,
) -> list[OCRResult]:
    runner = engine or get_default_ocr()
    return runner.ocr(frame, region=region, match_text=match_text, regex=regex)


__all__ = ["OCR", "OCRResult", "get_default_ocr", "init_engine", "ocr"]