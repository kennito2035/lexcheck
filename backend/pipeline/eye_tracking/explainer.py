from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from config import EYE_FEATURES, RF_MODEL_PATH

logger = logging.getLogger(__name__)


def compute_shap_values(features: dict[str, float], rf_model_path: Path | None = None) -> dict[str, float]:
    rf_model_path = rf_model_path or RF_MODEL_PATH
    if not rf_model_path.exists():
        raise FileNotFoundError(f"RF model not found at {rf_model_path}")

    try:
        import joblib
        import shap
    except ImportError as exc:
        raise ImportError("SHAP explainability requires joblib, shap, and numpy") from exc

    try:
        model = joblib.load(rf_model_path)
        x = np.array([[float(features[f]) for f in EYE_FEATURES]], dtype=np.float32)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(x)

        if isinstance(shap_vals, list) and len(shap_vals) >= 2:
            arr = np.asarray(shap_vals[1])
        elif hasattr(shap_vals, "values"):
            arr = np.asarray(shap_vals.values)
        else:
            arr = np.asarray(shap_vals)

        if arr.ndim == 3:
            class1_vals = arr[0, :, 1]
        elif arr.ndim == 2:
            class1_vals = arr[0, :]
        elif arr.ndim == 1:
            class1_vals = arr
        else:
            raise ValueError(f"Unexpected SHAP values shape: {arr.shape}")

        values = {feat: float(val) for feat, val in zip(EYE_FEATURES, class1_vals.tolist(), strict=False)}
        return _sort_by_abs(values)
    except (ValueError, OSError, AttributeError, TypeError, IndexError) as exc:
        raise RuntimeError("SHAP computation failed") from exc


def _sort_by_abs(values: dict[str, float]) -> dict[str, float]:
    return dict(sorted(values.items(), key=lambda kv: abs(kv[1]), reverse=True))
