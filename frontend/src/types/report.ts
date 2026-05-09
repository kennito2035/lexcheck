// ── Upload metadata ────────────────────────────────────────────────────────

export type FileMeta = {
  name: string
  size: number
  type: string
}

export type ReportSessionPayload = {
  rid: string
  createdAt: string
  files: {
    handwriting?: FileMeta
    eyeTracking?: FileMeta[]
  }
  results?: {
    handwriting?: unknown
    eyeTracking?: unknown
    report?: ScreeningReport
  }
}

// ── Screening Report (backend response) ────────────────────────────────────

export type HandwritingIndicators = {
  reversal_count: number
  corrected_count: number
  normal_count: number
  reversal_rate: number
  severity_flag: string
}

export type EyeTrackingIndicators = {
  fixation_mean_ms: number
  saccade_amp_mean: number
  regression_count: number
  regression_rate: number
  reading_difficulty_prob: number
  severity_cluster: "low" | "moderate" | "high"
}

export type ScreeningReport = {
  risk_score: number
  risk_label: "low" | "moderate" | "high"
  handwriting_skipped?: boolean
  eye_tracking_skipped?: boolean
  handwriting_indicators: HandwritingIndicators
  eye_tracking_indicators: EyeTrackingIndicators
  gradcam_heatmap_path: string
  shap_values: Record<string, number>
  llm_summary: string
  data_consistency_note?: string
  disclaimer: string
}

// ── API response types ───────────────────────────────────────────────────

export type ApiPollResponse =
  | { status: "ready"; report: ScreeningReport }
  | { status: "processing" }
  | { status: "error"; message: string }

export type UploadResponse = {
  rid: string
  report?: ScreeningReport
}
