from __future__ import annotations

import json
import sys
from pathlib import Path


def test_pipeline_end_to_end() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    image_path = project_root / "data" / "sample" / "handwriting_sample.jpg"
    csv_path = project_root / "data" / "sample" / "eye_tracking_sample.csv"
    assert image_path.exists() is True
    assert csv_path.exists() is True

    from pipeline.orchestrator import run_pipeline

    report = run_pipeline(image_path=image_path, csv_path=csv_path)

    assert 0.0 <= report.risk_score <= 1.0
    assert report.risk_label in {"low", "moderate", "high"}
    assert report.handwriting_skipped is False
    assert report.eye_tracking_skipped is False

    assert report.handwriting_indicators.reversal_count >= 0
    assert report.handwriting_indicators.reversal_rate >= 0.0

    assert 0.0 <= report.eye_tracking_indicators.reading_difficulty_prob <= 1.0
    assert report.eye_tracking_indicators.severity_cluster in {"low", "moderate", "high"}

    assert Path(report.gradcam_heatmap_path).exists() is True
    assert len(report.shap_values) == 12

    assert "NOT A" in report.disclaimer
    assert isinstance(report.llm_summary, str) and len(report.llm_summary.strip()) > 0

    payload = json.loads(report.to_json(indent=2))
    assert "risk_score" in payload
    assert "disclaimer" in payload

    txt = report.to_text()
    assert "RISK SCORE" in txt
    assert "SHAP" in txt
    assert "DISCLAIMER" in txt

    print(txt)


if __name__ == "__main__":
    test_pipeline_end_to_end()
