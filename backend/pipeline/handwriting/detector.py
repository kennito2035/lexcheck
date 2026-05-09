from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import YOLO_CLASS_NAMES, YOLO_CONF_THRESHOLD, YOLO_IMG_SIZE, YOLO_WEIGHTS_PATH

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LetterDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class HandwritingDetectionResult:
    reversal_count: int
    corrected_count: int
    normal_count: int
    reversal_rate: float
    detections: list[LetterDetection]

    def compute_risk_score(self) -> float:
        total = max(1, self.reversal_count + self.corrected_count + self.normal_count)
        score = (self.reversal_count * 1.0 + self.corrected_count * 0.4) / float(total)
        return float(min(1.0, max(0.0, score)))


def detect_letters_yolo(image_rgb_float: np.ndarray, weights_path: Path | None = None) -> HandwritingDetectionResult:
    weights_path = weights_path or YOLO_WEIGHTS_PATH
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found at {weights_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required to run YOLO inference") from exc

    try:
        model = YOLO(str(weights_path))
        results = model.predict(
            source=(image_rgb_float * 255.0).astype(np.uint8),
            conf=YOLO_CONF_THRESHOLD,
            verbose=False,
        )
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError("YOLO inference failed") from exc

    detections: list[LetterDetection] = []
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
        conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        for bbox, class_id_f, conf_f in zip(xyxy, cls, conf, strict=False):
            class_id = int(class_id_f)
            class_name = YOLO_CLASS_NAMES.get(class_id, f"class_{class_id}")
            detections.append(
                LetterDetection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(conf_f),
                    bbox_xyxy=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                )
            )

    reversal_count = sum(1 for d in detections if d.class_name == "Reversal")
    corrected_count = sum(1 for d in detections if d.class_name == "Corrected")
    normal_count = sum(1 for d in detections if d.class_name == "Normal")
    total = max(1, reversal_count + corrected_count + normal_count)
    reversal_rate = reversal_count / float(total)

    return HandwritingDetectionResult(
        reversal_count=reversal_count,
        corrected_count=corrected_count,
        normal_count=normal_count,
        reversal_rate=float(reversal_rate),
        detections=detections,
    )
