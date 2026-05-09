from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import YOLO_IMG_SIZE


@dataclass(frozen=True)
class HandwritingPreprocessOutput:
    original_bgr: np.ndarray
    resized_rgb_float: np.ndarray


def load_and_preprocess_image(image_path: Path) -> HandwritingPreprocessOutput:
    import cv2

    if not image_path.exists():
        raise FileNotFoundError(f"Handwriting image not found: {image_path}")

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    resized_bgr = cv2.resize(bgr, (YOLO_IMG_SIZE, YOLO_IMG_SIZE), interpolation=cv2.INTER_AREA)
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    rgb_float = resized_rgb.astype(np.float32) / 255.0
    return HandwritingPreprocessOutput(original_bgr=bgr, resized_rgb_float=rgb_float)

