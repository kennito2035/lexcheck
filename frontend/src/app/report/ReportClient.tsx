"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import Image from "next/image"

import type { ApiPollResponse, ScreeningReport } from "../../types/report"
import {
  SkeletonBar,
  SkeletonBox,
  SkeletonText,
} from "../components/SkeletonLoader"

// ── Polling configuration ─────────────────────────────────────────────────

const POLL_INTERVAL_MS = 3_000
const POLL_TIMEOUT_MS = 5 * 60 * 1_000 // 5 minutes

// ── Helpers ───────────────────────────────────────────────────────────────

function riskColor(label: string): string {
  switch (label) {
    case "high":
      return "#da1e28"
    case "moderate":
      return "#f1c21b"
    default:
      return "#24a148"
  }
}

function riskBgColor(label: string): string {
  switch (label) {
    case "high":
      return "#fff0f0"
    case "moderate":
      return "#fffce5"
    default:
      return "#f0faf0"
  }
}

function formatPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

function sortTopByAbs(values: Record<string, number>, n: number): Array<[string, number]> {
  return Object.entries(values)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, n)
}

type TroubleshootingCard = {
  title: string
  steps: string[]
  commands?: string[]
}

function buildTroubleshooting(errorMsg: string): TroubleshootingCard | null {
  const msg = (errorMsg || "").toLowerCase()
  const isYoloWeights = msg.includes("yolo weights not found")
  const isRfModel = msg.includes("rf model not found")
  const isUltralyticsMissing =
    msg.includes("ultralytics is required") ||
    msg.includes("no module named 'ultralytics'") ||
    msg.includes("missing grad-cam dependencies")
  const isShapMissing =
    msg.includes("shap explainability requires") ||
    msg.includes("no module named 'shap'") ||
    msg.includes("joblib is required")

  if (isYoloWeights) {
    return {
      title: "Handwriting model weights missing",
      steps: [
        "Confirm LexCheck/backend/models/yolov11_dyslexia.pt exists.",
        "If you moved the file, set YOLO_WEIGHTS_PATH in backend/.env to the correct relative path (from backend/).",
        "Restart the Next.js dev server after changing backend/.env.",
      ],
    }
  }

  if (isRfModel) {
    return {
      title: "Eye-tracking model file missing",
      steps: [
        "Confirm LexCheck/backend/models/rf_dyslexia.pkl exists.",
        "If you moved the file, set RF_MODEL_PATH in backend/.env to the correct relative path (from backend/).",
        "Restart the Next.js dev server after changing backend/.env.",
      ],
    }
  }

  if (isUltralyticsMissing || isShapMissing) {
    const steps = [
      "Activate your Python virtual environment from the LexCheck folder.",
      "Install backend dependencies.",
      "Restart the Next.js dev server and retry.",
    ]
    return {
      title: "Python dependencies missing",
      steps,
      commands: [
        "cd LexCheck",
        ".venv\\Scripts\\activate.bat",
        "pip install -r backend\\requirements.txt",
      ],
    }
  }

  return null
}

// ── Report detail components ──────────────────────────────────────────────

function SectionBox({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div
      style={{
        border: "2px solid #122942",
        borderRadius: 12,
        padding: "24px",
        background: "#ffffff",
        boxShadow: "4px 4px 0px #122942",
      }}
    >
      <h3 style={{ margin: "0 0 16px", fontSize: 20, color: "#111", fontWeight: 800 }}>{title}</h3>
      {children}
    </div>
  )
}

function RiskBadge({ score, label }: { score: number; label: string }) {
  return (
    <div
      style={{
        display: "inline-flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 8,
        padding: "20px 48px",
        borderRadius: 12,
        background: riskBgColor(label),
        border: "2px solid #122942",
        boxShadow: "4px 4px 0px #122942",
      }}
    >
      <span style={{ fontSize: 48, fontWeight: 700, color: riskColor(label), lineHeight: 1 }}>
        {score.toFixed(2)}
      </span>
      <span
        style={{
          fontSize: 14,
          fontWeight: 600,
          textTransform: "uppercase",
          color: riskColor(label),
          letterSpacing: "0.05em",
        }}
      >
        {label} risk
      </span>
    </div>
  )
}

function StatRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        padding: "6px 0",
        borderBottom: "1px solid #f0f0f0",
        fontSize: 14,
      }}
    >
      <span style={{ color: "#525252" }}>{label}</span>
      <span style={{ fontWeight: 600, color: "#111" }}>{value}</span>
    </div>
  )
}

function getCachedReport(rid: string): ScreeningReport | null {
  if (typeof window === "undefined") return null
  try {
    const raw =
      sessionStorage.getItem(`lexcheck.report.${rid}`) ??
      sessionStorage.getItem(`lexchat.report.${rid}`)
    if (!raw) return null
    const parsed = JSON.parse(raw) as { results?: { report?: ScreeningReport } }
    return parsed?.results?.report ?? null
  } catch {
    return null
  }
}

// ── Main component ────────────────────────────────────────────────────────

export default function ReportClient({ rid }: { rid: string | null }) {
  const [report, setReport] = useState<ScreeningReport | null>(() => (rid ? getCachedReport(rid) : null))
  const [status, setStatus] = useState<
    "idle" | "loading" | "processing" | "ready" | "error" | "timeout"
  >(() => (rid ? (getCachedReport(rid) ? "ready" : "processing") : "idle"))
  const [errorMsg, setErrorMsg] = useState<string>("")
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const mountedRef = useRef(true)

  // ── Polling logic ───────────────────────────────────────────────────────

  const poll = useCallback(async () => {
    if (!rid) return
    if (report) return true
    try {
      const res = await fetch(`/api/screening/${encodeURIComponent(rid)}`)
      
      let data: ApiPollResponse
      try {
        data = await res.json()
      } catch (e) {
        if (!res.ok) {
          throw new Error(`Server returned ${res.status} ${res.statusText}`)
        }
        throw new Error("Failed to parse server response")
      }

      if (!mountedRef.current) return

      if (data.status === "ready") {
        setReport(data.report)
        setStatus("ready")
        return true // signal completion
      }

      if (data.status === "error") {
        setErrorMsg(data.message ?? "Pipeline failed")
        setStatus("error")
        return true
      }

      // Still processing — continue polling
      return false
    } catch (err) {
      if (!mountedRef.current) return true
      setErrorMsg(err instanceof Error ? err.message : "Connection error")
      setStatus("error")
      return true
    }
  }, [report, rid])

  useEffect(() => {
    if (!rid) return

    mountedRef.current = true

    // Start polling
    const startTime = Date.now()
    const pollOnce = async () => {
      const done = await poll()
      if (done) {
        if (pollingRef.current) clearInterval(pollingRef.current)
        return
      }

      // Check timeout
      if (Date.now() - startTime > POLL_TIMEOUT_MS) {
        if (mountedRef.current) {
          setStatus("timeout")
          setErrorMsg("The screening is taking longer than expected. Please try again.")
        }
        if (pollingRef.current) clearInterval(pollingRef.current)
      }
    }

    // Initial poll immediately
    pollOnce()
    pollingRef.current = setInterval(pollOnce, POLL_INTERVAL_MS)

    return () => {
      mountedRef.current = false
      if (pollingRef.current) {
        clearInterval(pollingRef.current)
        pollingRef.current = null
      }
    }
  }, [rid, poll])

  // ── Status screens ──────────────────────────────────────────────────────

  if (!rid) {
    return (
      <div style={{ padding: "40px 0", textAlign: "center", color: "#525252" }}>
        <p>No submission selected.</p>
        <p>
          <a href="/test" style={{ color: "#0f62fe", textDecoration: "underline" }}>
            Go to the test page
          </a>{" "}
          to upload files and start a screening.
        </p>
      </div>
    )
  }

  if (status === "processing") {
    return (
      <div style={{ display: "grid", gap: 24, maxWidth: 720, margin: "0 auto" }}>
        <div style={{ textAlign: "center" }}>
          <p style={{ color: "#525252", fontSize: 14, marginBottom: 16 }}>
            Analyzing your samples — this may take a moment…
          </p>
          <div style={{ display: "flex", justifyContent: "center" }}>
            <SkeletonBar width="140px" height="40px" />
          </div>
        </div>

        <SectionBox title="Key Indicators">
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <SkeletonBar width="100%" height="18px" />
            <SkeletonBar width="95%" height="18px" />
            <SkeletonBar width="88%" height="18px" />
          </div>
        </SectionBox>

        <SectionBox title="Visualizations">
          <SkeletonBox width="100%" height="220px" />
        </SectionBox>

        <SectionBox title="Analysis Summary">
          <SkeletonText lines={4} width="100%" />
        </SectionBox>
      </div>
    )
  }

  if (status === "error" || status === "timeout") {
    const troubleshooting = buildTroubleshooting(errorMsg)
    return (
      <div
        style={{
          padding: "40px 24px",
          textAlign: "center",
          maxWidth: 560,
          margin: "0 auto",
        }}
      >
        <h2 style={{ color: "#da1e28", marginBottom: 12 }}>
          {status === "timeout" ? "Timed Out" : "Analysis Failed"}
        </h2>
        <p style={{ color: "#525252", marginBottom: 16 }}>
          {errorMsg ? errorMsg.split("\n")[0] : "An error occurred."}
        </p>

        {troubleshooting ? (
          <div
            style={{
              textAlign: "left",
              background: "#fffce5",
              border: "2px solid #122942",
              borderRadius: 12,
              padding: 16,
              boxShadow: "3px 3px 0px #122942",
              margin: "0 auto 18px",
            }}
          >
            <div style={{ fontWeight: 800, color: "#111", marginBottom: 8 }}>
              Quick fix: {troubleshooting.title}
            </div>
            <ol style={{ margin: "0 0 10px 18px", padding: 0, color: "#111", fontSize: 13, lineHeight: 1.6 }}>
              {troubleshooting.steps.map((s) => (
                <li key={s}>{s}</li>
              ))}
            </ol>
            {troubleshooting.commands ? (
              <pre
                style={{
                  margin: 0,
                  padding: 12,
                  background: "#ffffff",
                  border: "1px solid #e0e0e0",
                  borderRadius: 8,
                  overflowX: "auto",
                  fontSize: 12,
                }}
              >
                {troubleshooting.commands.join("\n")}
              </pre>
            ) : null}
          </div>
        ) : null}

        {errorMsg && errorMsg.includes("\n") ? (
          <details
            style={{
              textAlign: "left",
              margin: "0 auto 18px",
              background: "#f4f4f4",
              border: "1px solid #e0e0e0",
              borderRadius: 10,
              padding: "10px 12px",
            }}
          >
            <summary style={{ cursor: "pointer", fontWeight: 700, color: "#111" }}>Show full error</summary>
            <pre style={{ margin: "10px 0 0", whiteSpace: "pre-wrap", fontSize: 12, color: "#111" }}>
              {errorMsg}
            </pre>
          </details>
        ) : null}
        <div style={{ display: "flex", justifyContent: "center" }}>
          <a
            href="/test"
            style={{
              display: "inline-flex",
              alignItems: "center",
              padding: "14px 28px",
              background: "#55a6fa",
              color: "#ffffff",
              border: "2px solid #122942",
              borderRadius: "999px",
              textDecoration: "none",
              fontWeight: 800,
              fontSize: 16,
              boxShadow: "2px 2px 0px #122942",
              transition: "transform 150ms",
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.transform = "translate(-2px, -2px)"
              e.currentTarget.style.boxShadow = "4px 4px 0px #122942"
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.transform = "none"
              e.currentTarget.style.boxShadow = "2px 2px 0px #122942"
            }}
            onMouseDown={(e) => {
              e.currentTarget.style.transform = "translate(2px, 2px)"
              e.currentTarget.style.boxShadow = "0px 0px 0px #122942"
            }}
            onMouseUp={(e) => {
              e.currentTarget.style.transform = "none"
              e.currentTarget.style.boxShadow = "2px 2px 0px #122942"
            }}
          >
            Try Again
          </a>
        </div>
      </div>
    )
  }

  // ── Report ready ────────────────────────────────────────────────────────
  if (!report) return null

  const hw = report.handwriting_indicators
  const et = report.eye_tracking_indicators
  const topShap = sortTopByAbs(report.shap_values, 8)
  const handwritingSkipped = report.handwriting_skipped === true
  const eyeTrackingSkipped = report.eye_tracking_skipped === true

  return (
    <div style={{ display: "grid", gap: 24, maxWidth: 720, margin: "0 auto" }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 8 }}>
        <h1 style={{ fontSize: 28, margin: "0 0 4px" }}>Screening Report</h1>
      </div>

      {/* Risk Score */}
      <SectionBox title="Overall Risk Score">
        <div style={{ display: "flex", justifyContent: "center", padding: "12px 0" }}>
          <RiskBadge score={report.risk_score} label={report.risk_label} />
        </div>
      </SectionBox>

      {/* Handwriting Indicators */}
      {handwritingSkipped ? (
        <SectionBox title="Handwriting Analysis">
          <p style={{ margin: 0, color: "#525252", fontSize: 13 }}>
            Handwriting analysis was skipped (no handwriting image uploaded).
          </p>
        </SectionBox>
      ) : (
        <SectionBox title="Handwriting Analysis">
          <StatRow label="Normal letters" value={hw.normal_count} />
          <StatRow label="Reversal letters" value={hw.reversal_count} />
          <StatRow label="Corrected letters" value={hw.corrected_count} />
          <StatRow label="Reversal rate" value={formatPct(hw.reversal_rate)} />
          <StatRow label="Severity flag" value={hw.severity_flag} />
        </SectionBox>
      )}

      {/* Eye-Tracking Indicators */}
      {eyeTrackingSkipped ? (
        <SectionBox title="Eye-Tracking Analysis">
          <p style={{ margin: 0, color: "#525252", fontSize: 13 }}>
            Eye-tracking analysis was skipped (no eye-tracking CSV uploaded).
          </p>
        </SectionBox>
      ) : (
        <SectionBox title="Eye-Tracking Analysis">
          <StatRow label="Mean fixation" value={`${et.fixation_mean_ms.toFixed(1)} ms`} />
          <StatRow label="Mean saccade amplitude" value={`${et.saccade_amp_mean.toFixed(2)}°`} />
          <StatRow label="Regression count" value={et.regression_count.toFixed(0)} />
          <StatRow label="Regression rate" value={formatPct(et.regression_rate)} />
          <StatRow
            label="Reading difficulty probability"
            value={formatPct(et.reading_difficulty_prob)}
          />
          <StatRow label="Severity cluster" value={et.severity_cluster} />
        </SectionBox>
      )}

      {/* Grad-CAM */}
      {handwritingSkipped ? null : (
        <SectionBox title="Visualizations">
          <p style={{ fontSize: 13, color: "#525252", margin: "0 0 8px" }}>
            Grad-CAM heatmap overlay highlighting regions that influenced the handwriting analysis.
          </p>
          {report.gradcam_heatmap_path ? (
            <div
              style={{
                width: "100%",
                background: "#ffffff",
                borderRadius: 8,
                border: "1px solid #e0e0e0",
                overflow: "hidden",
              }}
            >
              <div style={{ position: "relative", width: "100%", height: 220 }}>
                <Image
                  src={`/api/screening/${encodeURIComponent(rid)}?asset=gradcam`}
                  alt="Grad-CAM heatmap overlay"
                  fill
                  sizes="100vw"
                  unoptimized
                  style={{ objectFit: "contain" }}
                />
              </div>
            </div>
          ) : (
            <div
              style={{
                width: "100%",
                height: 220,
                background: "#f4f4f4",
                borderRadius: 8,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "#8c8c8c",
                fontSize: 13,
                border: "1px dashed #e0e0e0",
              }}
            >
              No heatmap available
            </div>
          )}
        </SectionBox>
      )}

      {/* SHAP Feature Importance */}
      {eyeTrackingSkipped ? null : (
        <SectionBox title="Top Contributing Factors (SHAP)">
          <div style={{ fontSize: 13, color: "#525252", marginBottom: 12 }}>
            Feature importance scores from the eye-tracking model. Higher absolute values indicate
            greater influence on the result.
          </div>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: "2px solid #e0e0e0" }}>
                  <th style={{ textAlign: "left", padding: "6px 8px", color: "#525252" }}>
                    Feature
                  </th>
                  <th style={{ textAlign: "right", padding: "6px 8px", color: "#525252" }}>
                    SHAP Value
                  </th>
                  <th style={{ textAlign: "right", padding: "6px 8px", color: "#525252" }}>
                    |Impact|
                  </th>
                </tr>
              </thead>
              <tbody>
                {topShap.map(([feat, val]) => (
                  <tr key={feat} style={{ borderBottom: "1px solid #f0f0f0" }}>
                    <td style={{ padding: "6px 8px", color: "#111" }}>
                      {feat.replace(/_/g, " ")}
                    </td>
                    <td style={{ padding: "6px 8px", textAlign: "right", fontFamily: "monospace" }}>
                      {val >= 0 ? "+" : ""}
                      {val.toFixed(4)}
                    </td>
                    <td style={{ padding: "6px 8px", textAlign: "right", fontFamily: "monospace" }}>
                      {Math.abs(val).toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionBox>
      )}

      {/* LLM Summary */}
      <SectionBox title="Analysis Summary">
        <p style={{ fontSize: 14, lineHeight: 1.7, color: "#111", margin: 0, whiteSpace: "pre-wrap" }}>
          {report.llm_summary}
        </p>
      </SectionBox>

      {report.data_consistency_note ? (
        <SectionBox title="Data Consistency">
          <p style={{ fontSize: 13, lineHeight: 1.6, color: "#525252", margin: 0 }}>
            {report.data_consistency_note}
          </p>
        </SectionBox>
      ) : null}

      {/* Disclaimer */}
      <div
        style={{
          padding: "16px 20px",
          background: "#f4f4f4",
          borderRadius: 12,
          border: "2px solid #122942",
          boxShadow: "3px 3px 0px #122942",
          fontSize: 12,
          lineHeight: 1.6,
          color: "#525252",
        }}
      >
        <strong style={{ color: "#da1e28" }}>⚠ Important Disclaimer</strong>
        <p style={{ margin: "8px 0 0" }}>{report.disclaimer}</p>
      </div>

      <div style={{ display: "flex", gap: 16, marginTop: 16, justifyContent: "center" }}>
        <button
          style={{
            padding: "14px 28px",
            background: "#ffffff",
            color: "#122942",
            border: "2px solid #122942",
            borderRadius: "999px",
            cursor: "pointer",
            fontWeight: 800,
            fontSize: 16,
            boxShadow: "2px 2px 0px #122942",
            transition: "transform 150ms",
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.transform = "translate(-2px, -2px)"
            e.currentTarget.style.boxShadow = "4px 4px 0px #122942"
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.transform = "none"
            e.currentTarget.style.boxShadow = "2px 2px 0px #122942"
          }}
          onMouseDown={(e) => {
            e.currentTarget.style.transform = "translate(2px, 2px)"
            e.currentTarget.style.boxShadow = "0px 0px 0px #122942"
          }}
          onMouseUp={(e) => {
            e.currentTarget.style.transform = "none"
            e.currentTarget.style.boxShadow = "2px 2px 0px #122942"
          }}
          onClick={() => window.print()}
        >
          Download PDF
        </button>
        <a
          href="/test"
          style={{
            display: "inline-flex",
            alignItems: "center",
            padding: "14px 28px",
            background: "#55a6fa",
            color: "#ffffff",
            border: "2px solid #122942",
            borderRadius: "999px",
            textDecoration: "none",
            fontWeight: 800,
            fontSize: 16,
            boxShadow: "2px 2px 0px #122942",
            transition: "transform 150ms",
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.transform = "translate(-2px, -2px)"
            e.currentTarget.style.boxShadow = "4px 4px 0px #122942"
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.transform = "none"
            e.currentTarget.style.boxShadow = "2px 2px 0px #122942"
          }}
          onMouseDown={(e) => {
            e.currentTarget.style.transform = "translate(2px, 2px)"
            e.currentTarget.style.boxShadow = "0px 0px 0px #122942"
          }}
          onMouseUp={(e) => {
            e.currentTarget.style.transform = "none"
            e.currentTarget.style.boxShadow = "2px 2px 0px #122942"
          }}
        >
          Start New Test
        </a>
      </div>
    </div>
  )
}
