from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path

from config import ET_WEIGHT, HW_WEIGHT, OUTPUT_DIR
from pipeline.llm.narrator import narrate_findings
from pipeline.report.generator import (
    DISCLAIMER,
    EyeTrackingIndicators,
    HandwritingIndicators,
    ScreeningReport,
    risk_label_from_score,
)

logger = logging.getLogger(__name__)

BytesLike = bytes | bytearray | memoryview


def _infer_data_consistency_note(*, image_path: Path, csv_path: Path) -> str:
    csv_s = str(csv_path).lower()
    img_s = str(image_path).lower()

    etdd70_match = re.search(r"subject_(\d+)_", csv_path.name, flags=re.IGNORECASE)
    et_subject_id = etdd70_match.group(1) if etdd70_match else None

    image_source = "unknown_image_source"
    if "synthdata_handwriting" in img_s:
        image_source = "synthdata_handwriting"
    elif "dysgraphia" in img_s:
        image_source = "dysgraphia_handwriting_dataset"
    elif "non-dyslexic" in img_s or "non_dyslexic" in img_s:
        image_source = "non_dyslexic_image_folder"
    elif "dyslexic" in img_s:
        image_source = "dyslexic_image_folder"

    csv_source = "unknown_eye_tracking_source"
    if "etdd70" in csv_s:
        csv_source = "etdd70"

    if et_subject_id and et_subject_id in image_path.name:
        return ""

    if csv_source == "etdd70" and image_source != "unknown_image_source":
        return (
            "Potential modality mismatch: eye-tracking input appears to be from ETDD70 "
            f"(subject_id={et_subject_id or 'unknown'}), while the handwriting image appears to be from "
            f"'{image_source}'. Combined risk may not reflect a single individual."
        )

    if et_subject_id and image_source == "unknown_image_source":
        return (
            "Eye-tracking input includes an ETDD70 subject_id, but the handwriting image filename does not. "
            "Unable to verify both modalities are from the same individual."
        )

    return ""


def run_pipeline(
    *,
    image_path: Path,
    csv_path: Path,
    output_dir: Path | None = None,
    enable_llm: bool = True,
    skip_handwriting: bool = False,
    skip_eye_tracking: bool = False,
) -> ScreeningReport:
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if skip_handwriting:
        logger.info("Stage 1/5: Handwriting detection (YOLO) [SKIPPED]")
        logger.info("Stage 2/5: Grad-CAM heatmap generation [SKIPPED]")
        hw_score = 0.0
        hw_reversal_count = 0
        hw_corrected_count = 0
        hw_normal_count = 0
        hw_reversal_rate = 0.0
        hw_detections = []
        gradcam_path = ""
    else:
        logger.info("Stage 1/5: Handwriting detection (YOLO)")
        from pipeline.handwriting.detector import detect_letters_yolo
        from pipeline.handwriting.preprocessor import load_and_preprocess_image

        hw_pre = load_and_preprocess_image(image_path=image_path)
        hw_det = detect_letters_yolo(image_rgb_float=hw_pre.resized_rgb_float)
        hw_score = float(hw_det.compute_risk_score())
        hw_reversal_count = int(hw_det.reversal_count)
        hw_corrected_count = int(hw_det.corrected_count)
        hw_normal_count = int(hw_det.normal_count)
        hw_reversal_rate = float(hw_det.reversal_rate)
        hw_detections = hw_det.detections

        logger.info(
            "Handwriting: reversals=%s corrected=%s normal=%s reversal_rate=%.4f hw_score=%.4f",
            hw_reversal_count,
            hw_corrected_count,
            hw_normal_count,
            hw_reversal_rate,
            hw_score,
        )

        logger.info("Stage 2/5: Grad-CAM heatmap generation")
        from pipeline.handwriting.explainer import generate_gradcam_heatmap

        gradcam_path = str(
            generate_gradcam_heatmap(
                original_bgr=hw_pre.original_bgr,
                resized_rgb_float=hw_pre.resized_rgb_float,
                output_dir=output_dir,
                detections=hw_detections,
            )
        )
        logger.info("Grad-CAM saved: %s", gradcam_path)

    if skip_eye_tracking:
        logger.info("Stage 3/5: Eye-tracking feature extraction + RF classification [SKIPPED]")
        logger.info("Stage 4/5: SHAP values [SKIPPED]")
        features: dict[str, float] = {}
        et_score = 0.0
        et_severity_cluster = "low"
        shap_values: dict[str, float] = {}
        top3: list[tuple[str, float]] = []
        fixation_mean_ms = 0.0
        regression_rate = 0.0
    else:
        logger.info("Stage 3/5: Eye-tracking feature extraction + RF classification")
        from pipeline.eye_tracking.classifier import classify_eye_tracking
        from pipeline.eye_tracking.feature_extractor import extract_eye_tracking_features

        features = extract_eye_tracking_features(csv_path=csv_path)
        et_result = classify_eye_tracking(features=features)
        et_score = float(et_result.reading_difficulty_prob)
        et_severity_cluster = et_result.severity_cluster
        fixation_mean_ms = float(features.get("fixation_mean_duration_ms", 0.0))
        regression_rate = float(features.get("regression_rate", 0.0))

        logger.info(
            "Eye-tracking: fixation_mean=%.1f regression_rate=%.4f prob=%.4f cluster=%s",
            fixation_mean_ms,
            regression_rate,
            et_score,
            et_severity_cluster,
        )

        logger.info("Stage 4/5: SHAP values")
        from pipeline.eye_tracking.explainer import compute_shap_values

        shap_values = compute_shap_values(features=features)
        top3 = list(shap_values.items())[:3]
        logger.info("Top SHAP: %s", top3)

    if skip_handwriting and not skip_eye_tracking:
        risk_score = round(float(et_score), 4)
    elif skip_eye_tracking and not skip_handwriting:
        risk_score = round(float(hw_score), 4)
    else:
        risk_score = round(HW_WEIGHT * float(hw_score) + ET_WEIGHT * float(et_score), 4)
    risk_label = risk_label_from_score(risk_score)
    if skip_handwriting:
        data_consistency_note = "Single-modality run: handwriting stages were skipped."
    elif skip_eye_tracking:
        data_consistency_note = "Single-modality run: eye-tracking stages were skipped."
    else:
        data_consistency_note = _infer_data_consistency_note(image_path=image_path, csv_path=csv_path)

    logger.info("Stage 5/5: LLM narration + report assembly")
    if enable_llm:
        llm_summary = narrate_findings(
            handwriting_reversal_count=hw_reversal_count,
            handwriting_corrected_count=hw_corrected_count,
            handwriting_normal_count=hw_normal_count,
            handwriting_reversal_rate=hw_reversal_rate,
            fixation_mean_duration_ms=fixation_mean_ms,
            regression_rate=regression_rate,
            reading_difficulty_prob=float(et_score),
            severity_cluster=et_severity_cluster,
            top_shap_contributors=top3,
            combined_risk_score=risk_score,
            combined_risk_label=risk_label,
            handwriting_skipped=skip_handwriting,
            eye_tracking_skipped=skip_eye_tracking,
        )
    else:
        llm_summary = (
            f"LLM narration disabled. Combined risk score: {risk_score:.4f} ({risk_label}). "
            "This is a pre-screening indicator requiring specialist assessment."
        )

    hw_severity = "high" if hw_reversal_count >= 3 or hw_reversal_rate >= 0.25 else "moderate"
    if hw_reversal_count == 0 and hw_reversal_rate < 0.10:
        hw_severity = "low"

    report = ScreeningReport(
        risk_score=risk_score,
        risk_label=risk_label,
        handwriting_skipped=bool(skip_handwriting),
        eye_tracking_skipped=bool(skip_eye_tracking),
        handwriting_indicators=HandwritingIndicators(
            reversal_count=hw_reversal_count,
            corrected_count=hw_corrected_count,
            normal_count=hw_normal_count,
            reversal_rate=hw_reversal_rate,
            severity_flag=hw_severity,
        ),
        eye_tracking_indicators=EyeTrackingIndicators(
            fixation_mean_ms=fixation_mean_ms,
            saccade_amp_mean=float(features.get("saccade_mean_amplitude_deg", 0.0)) if features else 0.0,
            regression_count=int(features.get("regression_count", 0.0)) if features else 0,
            regression_rate=regression_rate,
            reading_difficulty_prob=float(et_score),
            severity_cluster=et_severity_cluster,
        ),
        gradcam_heatmap_path=str(gradcam_path),
        shap_values=shap_values,
        llm_summary=llm_summary,
        data_consistency_note=data_consistency_note,
        disclaimer=DISCLAIMER,
    )
    return report


def run_pipeline_from_bytes(
    *,
    image_bytes: BytesLike,
    csv_bytes: BytesLike,
    output_dir: Path | None = None,
    enable_llm: bool = True,
    skip_handwriting: bool = False,
    skip_eye_tracking: bool = False,
    image_filename: str = "input.jpg",
    csv_filename: str = "input.csv",
) -> ScreeningReport:
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        image_path = tmp_path / image_filename
        csv_path = tmp_path / csv_filename
        image_path.write_bytes(bytes(image_bytes))
        csv_path.write_bytes(bytes(csv_bytes))
        return run_pipeline(
            image_path=image_path,
            csv_path=csv_path,
            output_dir=output_dir,
            enable_llm=enable_llm,
            skip_handwriting=skip_handwriting,
            skip_eye_tracking=skip_eye_tracking,
        )
