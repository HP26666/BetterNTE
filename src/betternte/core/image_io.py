from __future__ import annotations

import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np


@contextmanager
def _suppress_native_stderr():
    try:
        stderr_fd = sys.stderr.fileno()
    except Exception:
        yield
        return

    saved_stderr_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w", encoding="utf-8", errors="ignore") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)


def load_image_bgr(path: str | Path) -> np.ndarray | None:
    image_path = Path(path)
    try:
        from PIL import Image, ImageOps
    except Exception:
        with _suppress_native_stderr():
            return cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    try:
        with _suppress_native_stderr():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*iCCP: known incorrect sRGB profile.*")
                with Image.open(image_path) as image:
                    image = ImageOps.exif_transpose(image)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    rgb = np.array(image, dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None