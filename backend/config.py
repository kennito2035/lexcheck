from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

_BACKEND_DIR: Path = Path(__file__).resolve().parent


def _resolve_path(env_key: str, default_rel: str) -> Path:
    raw = os.environ.get(env_key, default_rel)
    p = Path(raw)
    if p.is_absolute():
        return p
    return (_BACKEND_DIR / p).resolve()

GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TIMEOUT_S: float = float(os.environ.get("GEMINI_TIMEOUT_S", "30"))
GEMINI_MAX_RETRIES: int = int(os.environ.get("GEMINI_MAX_RETRIES", "3"))
GEMINI_RETRY_BACKOFF_S: float = float(os.environ.get("GEMINI_RETRY_BACKOFF_S", "1.0"))
GEMINI_RETRY_BACKOFF_MAX_S: float = float(os.environ.get("GEMINI_RETRY_BACKOFF_MAX_S", "8.0"))

YOLO_WEIGHTS_PATH: Path = _resolve_path("YOLO_WEIGHTS_PATH", "models/yolov11_dyslexia.pt")
YOLO_CONF_THRESHOLD: float = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.40"))
YOLO_IMG_SIZE: int = int(os.environ.get("YOLO_IMG_SIZE", "640"))
YOLO_CLASS_NAMES: dict[int, str] = {0: "Normal", 1: "Reversal", 2: "Corrected"}

RF_MODEL_PATH: Path = _resolve_path("RF_MODEL_PATH", "models/rf_dyslexia.pkl")

EYE_FEATURES: list[str] = [
    "fixation_count",
    "fixation_mean_duration_ms",
    "fixation_max_duration_ms",
    "fixation_std_duration_ms",
    "saccade_count",
    "saccade_mean_amplitude_deg",
    "saccade_std_amplitude_deg",
    "regression_count",
    "regression_rate",
    "first_pass_fixation_time_ms",
    "skip_rate",
    "word_reading_time_mean_ms",
]

HW_WEIGHT: float = float(os.environ.get("HW_WEIGHT", "0.55"))
ET_WEIGHT: float = float(os.environ.get("ET_WEIGHT", "0.45"))

RISK_LOW_MAX: float = float(os.environ.get("RISK_LOW_MAX", "0.35"))
RISK_MODERATE_MAX: float = float(os.environ.get("RISK_MODERATE_MAX", "0.65"))

OUTPUT_DIR: Path = _resolve_path("OUTPUT_DIR", "outputs")
