from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import EYE_FEATURES, RF_MODEL_PATH, RISK_LOW_MAX, RISK_MODERATE_MAX

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EyeTrackingResult:
    features: dict[str, float]
    reading_difficulty_prob: float
    severity_cluster: str


def classify_eye_tracking(features: dict[str, float], rf_model_path: Path | None = None) -> EyeTrackingResult:
    rf_model_path = rf_model_path or RF_MODEL_PATH
    if not rf_model_path.exists():
        raise FileNotFoundError(f"RF model not found at {rf_model_path}")

    try:
        import joblib
    except ImportError as exc:
        raise ImportError("joblib is required to load the RF model") from exc

    try:
        model = joblib.load(rf_model_path)
        x = np.array([[float(features[f]) for f in EYE_FEATURES]], dtype=np.float32)
        proba = model.predict_proba(x)
        prob = float(proba[0, 1])
        prob = float(min(1.0, max(0.0, prob)))
        return EyeTrackingResult(
            features=features,
            reading_difficulty_prob=prob,
            severity_cluster=_severity_from_prob(prob),
        )
    except (ValueError, OSError, AttributeError) as exc:
        raise RuntimeError("RF classification failed") from exc


def _severity_from_prob(prob: float) -> str:
    if prob <= RISK_LOW_MAX:
        return "low"
    if prob <= RISK_MODERATE_MAX:
        return "moderate"
    return "high"


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(min(1.0, max(0.0, (value - low) / (high - low))))
