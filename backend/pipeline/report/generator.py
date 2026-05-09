from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from config import RISK_LOW_MAX, RISK_MODERATE_MAX

DISCLAIMER: str = (
    "THIS IS A PRE-SCREENING INDICATOR ONLY. It is not a clinical diagnosis. "
    "A qualified specialist assessment is required. NOT A diagnosis."
)


class HandwritingIndicators(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reversal_count: int
    corrected_count: int
    normal_count: int
    reversal_rate: float
    severity_flag: str


class EyeTrackingIndicators(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixation_mean_ms: float
    saccade_amp_mean: float
    regression_count: int
    regression_rate: float
    reading_difficulty_prob: float
    severity_cluster: str


class ScreeningReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_score: float
    risk_label: str
    handwriting_skipped: bool = Field(default=False)
    eye_tracking_skipped: bool = Field(default=False)
    handwriting_indicators: HandwritingIndicators
    eye_tracking_indicators: EyeTrackingIndicators
    gradcam_heatmap_path: str
    shap_values: dict[str, float]
    llm_summary: str
    data_consistency_note: str = Field(default="")
    disclaimer: str = Field(default=DISCLAIMER)

    @model_validator(mode="after")
    def _inject_disclaimer(self) -> "ScreeningReport":
        object.__setattr__(self, "disclaimer", DISCLAIMER)
        return self

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    def to_text(self) -> str:
        top5 = list(self.shap_values.items())[:5]
        shap_lines = "\n".join([f"- {k}: {v:+.4f}" for k, v in top5]) if top5 else "- (none)"
        consistency = self.data_consistency_note.strip() or "(none detected)"

        return (
            "RISK SCORE\n"
            f"- score: {self.risk_score:.4f}\n"
            f"- label: {self.risk_label}\n"
            "\n"
            "MODALITIES\n"
            f"- handwriting_skipped: {self.handwriting_skipped}\n"
            f"- eye_tracking_skipped: {self.eye_tracking_skipped}\n"
            "\n"
            "HANDWRITING INDICATORS\n"
            f"- reversal_count: {self.handwriting_indicators.reversal_count}\n"
            f"- corrected_count: {self.handwriting_indicators.corrected_count}\n"
            f"- normal_count: {self.handwriting_indicators.normal_count}\n"
            f"- reversal_rate: {self.handwriting_indicators.reversal_rate:.4f}\n"
            f"- severity_flag: {self.handwriting_indicators.severity_flag}\n"
            "\n"
            "EYE-TRACKING INDICATORS\n"
            f"- fixation_mean_ms: {self.eye_tracking_indicators.fixation_mean_ms:.1f}\n"
            f"- saccade_amp_mean: {self.eye_tracking_indicators.saccade_amp_mean:.3f}\n"
            f"- regression_count: {self.eye_tracking_indicators.regression_count}\n"
            f"- regression_rate: {self.eye_tracking_indicators.regression_rate:.4f}\n"
            f"- reading_difficulty_prob: {self.eye_tracking_indicators.reading_difficulty_prob:.4f}\n"
            f"- severity_cluster: {self.eye_tracking_indicators.severity_cluster}\n"
            "\n"
            "SHAP (TOP 5 CONTRIBUTORS)\n"
            f"{shap_lines}\n"
            "\n"
            "GRAD-CAM\n"
            f"- path: {self.gradcam_heatmap_path}\n"
            "\n"
            "DATA CONSISTENCY\n"
            f"- note: {consistency}\n"
            "\n"
            "LLM SUMMARY\n"
            f"{self.llm_summary}\n"
            "\n"
            "DISCLAIMER\n"
            f"{self.disclaimer}\n"
        )


def risk_label_from_score(score: float) -> str:
    if score <= RISK_LOW_MAX:
        return "low"
    if score <= RISK_MODERATE_MAX:
        return "moderate"
    return "high"
