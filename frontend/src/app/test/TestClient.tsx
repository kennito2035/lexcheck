"use client"

import { useRouter } from "next/navigation"
import { useMemo, useRef, useState } from "react"

import type { FileMeta, ReportSessionPayload, UploadResponse } from "../../types/report"
import styles from "./test.module.css"

function toMeta(file: File): FileMeta {
  return { name: file.name, size: file.size, type: file.type }
}

type ValidationResult =
  | { ok: true }
  | { ok: false; message: string }

function normalizeHeaderCell(cell: string): string {
  return cell.trim().replace(/^["']|["']$/g, "").trim().toLowerCase()
}

async function readCsvHeaderColumns(file: File): Promise<Set<string>> {
  const chunk = await file.slice(0, 64 * 1024).text()
  const lines = chunk.replace(/^\uFEFF/, "").split(/\r?\n/)
  const headerLine = lines.find((l) => l.trim().length > 0) ?? ""
  const cols = headerLine
    .split(",")
    .map(normalizeHeaderCell)
    .filter(Boolean)
  return new Set(cols)
}

function hasAny(cols: Set<string>, candidates: string[]): boolean {
  return candidates.some((c) => cols.has(c))
}

function requireAllGroups(cols: Set<string>, groups: Record<string, string[]>): string[] {
  const missing: string[] = []
  for (const [label, candidates] of Object.entries(groups)) {
    if (!hasAny(cols, candidates)) missing.push(label)
  }
  return missing
}

async function validateEyeTrackingFiles(files: File[]): Promise<ValidationResult> {
  const csvFiles = files.filter((f) => (f.name || "").toLowerCase().endsWith(".csv"))
  if (csvFiles.length !== files.length) {
    return { ok: false, message: "Eye-tracking upload must be CSV (.csv) files only." }
  }

  if (files.length > 2) {
    return { ok: false, message: "Upload either 1 CSV (event rows) or 2 CSVs (fixations + saccades pair)." }
  }

  const isFix = (name: string) => /_fixations\.csv$/i.test(name)
  const isSac = (name: string) => /_saccades\.csv$/i.test(name)

  if (files.length === 1) {
    const f = files[0]!
    const cols = await readCsvHeaderColumns(f)

    const eventRowsMissing = requireAllGroups(cols, {
      event_type: ["event_type", "event", "type", "eventtype"],
      duration_ms: ["duration_ms", "duration", "dur_ms", "ms", "durationms"],
      amplitude_deg: ["amplitude_deg", "amplitude", "amp_deg", "amp", "amplitudedeg"],
      is_regression: ["is_regression", "regression", "isregression", "is_backward", "backward"],
    })

    if (eventRowsMissing.length === 0) return { ok: true }

    const looksLikeFix = hasAny(cols, ["duration_ms"]) && (hasAny(cols, ["fix_x"]) || hasAny(cols, ["fix_y"]))
    const looksLikeSac =
      (hasAny(cols, ["ampl"]) || (hasAny(cols, ["ampl_x"]) && hasAny(cols, ["ampl_y"]))) &&
      hasAny(cols, ["start_x"]) &&
      hasAny(cols, ["end_x"])

    if (looksLikeFix || looksLikeSac || isFix(f.name) || isSac(f.name)) {
      return {
        ok: false,
        message:
          "Paired exports require 2 files: *_fixations.csv (duration_ms + fix_x/fix_y) and *_saccades.csv (ampl or ampl_x/ampl_y + start_x + end_x). Upload both together.",
      }
    }

    return {
      ok: false,
      message: `Event-rows CSV is missing required columns: ${eventRowsMissing.join(", ")}.`,
    }
  }

  const [a, b] = files
  const fix = isFix(a!.name) ? a! : isFix(b!.name) ? b! : null
  const sac = isSac(a!.name) ? a! : isSac(b!.name) ? b! : null

  if (!fix || !sac) {
    return {
      ok: false,
      message: "For paired uploads, file names must end with *_fixations.csv and *_saccades.csv (upload both).",
    }
  }

  const fixCols = await readCsvHeaderColumns(fix)
  const sacCols = await readCsvHeaderColumns(sac)

  const missingFix: string[] = []
  if (!hasAny(fixCols, ["duration_ms"])) missingFix.push("duration_ms")
  if (!(hasAny(fixCols, ["fix_x"]) || hasAny(fixCols, ["fix_y"]))) missingFix.push("fix_x or fix_y")

  const missingSac: string[] = []
  const hasAmpl = hasAny(sacCols, ["ampl"]) || (hasAny(sacCols, ["ampl_x"]) && hasAny(sacCols, ["ampl_y"]))
  if (!hasAmpl) missingSac.push("ampl (or ampl_x + ampl_y)")
  if (!hasAny(sacCols, ["start_x"])) missingSac.push("start_x")
  if (!hasAny(sacCols, ["end_x"])) missingSac.push("end_x")

  if (missingFix.length || missingSac.length) {
    const parts: string[] = []
    if (missingFix.length) parts.push(`fixations missing: ${missingFix.join(", ")}`)
    if (missingSac.length) parts.push(`saccades missing: ${missingSac.join(", ")}`)
    return { ok: false, message: `CSV headers invalid (${parts.join(" | ")}).` }
  }

  return { ok: true }
}

type UploadCardProps = {
  title: string
  subtitle: string
  description: string
  supported: string
  accept: string
  fileCount: number
  fileLabel: string
  multiple?: boolean
  onFiles: (files: File[]) => void | Promise<void>
  dropText: string
  variant: "green" | "blue" | "purple"
}

function UploadCard({ title, subtitle, description, supported, accept, fileCount, fileLabel, multiple, onFiles, dropText, variant }: UploadCardProps) {
  const inputRef = useRef<HTMLInputElement | null>(null)

  return (
    <section className={styles.card} data-variant={variant}>
      <div className={styles.cardInfo}>
        <h2 className={styles.cardTitle}>{title}</h2>
        <p className={styles.cardText}>{subtitle}</p>
        <p className={styles.cardHint}>{description}</p>
        <p className={styles.cardHint}>Supported Format: {supported}</p>
        <p className={styles.fileName}>{fileCount ? `\u2713 Selected: ${fileLabel}` : ""}</p>
      </div>

      <div
        className={styles.dropzone}
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") inputRef.current?.click()
        }}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault()
          const files = Array.from(e.dataTransfer.files ?? [])
          void onFiles(multiple ? files : files.slice(0, 1))
        }}
      >
        <div className={styles.tiltedSquare} />
        <p className={styles.dropzoneText}>{dropText}</p>
        <p className={styles.dropzoneOr}>OR</p>
        <span className={styles.browseButton}>Browse here</span>
        <input
          ref={inputRef}
          className={styles.hiddenInput}
          type="file"
          multiple={multiple}
          accept={accept}
          onChange={(e) => {
            const files = Array.from(e.target.files ?? [])
            void onFiles(multiple ? files : files.slice(0, 1))
          }}
        />
      </div>
    </section>
  )
}

export default function TestClient() {
  const router = useRouter()
  const [handwritingFile, setHandwritingFile] = useState<File | null>(null)
  const [eyeTrackingFiles, setEyeTrackingFiles] = useState<File[]>([])
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [eyeTrackingError, setEyeTrackingError] = useState<string | null>(null)
  const [eyeTrackingValidating, setEyeTrackingValidating] = useState(false)

  const canSubmit = useMemo(() => {
    const hasInputs = Boolean(handwritingFile || eyeTrackingFiles.length)
    const eyeOk = !eyeTrackingFiles.length || (!eyeTrackingError && !eyeTrackingValidating)
    return hasInputs && eyeOk && !uploading
  }, [eyeTrackingError, eyeTrackingFiles.length, eyeTrackingValidating, handwritingFile, uploading])

  async function onEyeTrackingFiles(files: File[]) {
    setEyeTrackingValidating(true)
    setEyeTrackingError(null)
    try {
      const picked = files.slice(0, 2)
      const res = await validateEyeTrackingFiles(picked)
      if (!res.ok) {
        setEyeTrackingFiles([])
        setEyeTrackingError(res.message)
        return
      }
      setEyeTrackingFiles(picked)
    } finally {
      setEyeTrackingValidating(false)
    }
  }

  async function onSubmit() {
    if (!canSubmit) return

    setUploading(true)
    setError(null)

    try {
      const formData = new FormData()

      if (handwritingFile) {
        formData.append("handwriting", handwritingFile)
      }
      for (const f of eyeTrackingFiles) {
        formData.append("eyeTracking", f)
      }

      const res = await fetch("/api/screening", {
        method: "POST",
        body: formData,
      })

      if (!res.ok) {
        const body = await res.json().catch(() => null)
        throw new Error(body?.error ?? `Upload failed (${res.status})`)
      }

      const payload = (await res.json()) as UploadResponse
      const rid = payload.rid
      const report = payload.report

      // Store session metadata for convenience
      const sessionPayload: ReportSessionPayload = {
        rid,
        createdAt: new Date().toISOString(),
        files: {
          handwriting: handwritingFile ? toMeta(handwritingFile) : undefined,
          eyeTracking: eyeTrackingFiles.length ? eyeTrackingFiles.map(toMeta) : undefined,
        },
        results: report ? { report } : undefined,
      }
      sessionStorage.setItem(`lexcheck.report.${rid}`, JSON.stringify(sessionPayload))

      router.push(`/report?rid=${encodeURIComponent(rid)}`)
    } catch (err) {
      const message = err instanceof Error ? err.message : "Something went wrong"
      setError(message)
      setUploading(false)
    }
  }

  return (
    <div className={styles.page}>
      <div className={styles.hero}>
        <h1 className={styles.title}>Let&apos;s start your adventure</h1>
        <p className={styles.subtitle}>Upload handwriting, eye-tracking data, or both</p>
      </div>

      <div className={styles.cards}>
        <UploadCard
          title="Upload handwriting"
          subtitle="Select and upload files"
          description="Analyzes letter formation and writing patterns."
          supported="PNG, JPEG, JPG, SVG"
          accept="image/*"
          fileCount={handwritingFile ? 1 : 0}
          fileLabel={handwritingFile?.name ?? ""}
          onFiles={(files) => setHandwritingFile(files[0] ?? null)}
          dropText="Drag your image here"
          variant="blue"
        />

        <UploadCard
          title="Upload eye-tracking data"
          subtitle="Select and upload CSV files"
          description="Required: either 1 event-rows CSV (event_type, duration_ms, amplitude_deg, is_regression) OR 2 CSVs (*_fixations.csv + *_saccades.csv)."
          supported="CSV (1–2 files)"
          accept=".csv,text/csv"
          fileCount={eyeTrackingFiles.length}
          fileLabel={
            eyeTrackingFiles.length === 1
              ? eyeTrackingFiles[0]!.name
              : eyeTrackingFiles.length > 1
                ? `${eyeTrackingFiles.length} files`
                : ""
          }
          multiple
          onFiles={onEyeTrackingFiles}
          dropText="Drag your CSV file(s) here"
          variant="purple"
        />
      </div>

      {eyeTrackingError ? <p className={styles.error}>{eyeTrackingError}</p> : null}
      {error ? <p className={styles.error}>{error}</p> : null}

      <div className={styles.buttonGroup}>
        <button className={styles.backBtn} type="button" onClick={() => router.back()}>
          Back
        </button>
        <button className={styles.submitBtn} type="button" onClick={onSubmit} disabled={!canSubmit}>
          {uploading ? "Uploading\u2026" : "Submit"}
        </button>
      </div>
    </div>
  )
}
