/*
 * POST /api/screening
 *
 * Accepts multipart/form-data with optional fields:
 *   - handwriting  (image file: JPEG/PNG/WebP) — writing sample
 *   - eyeTracking  (CSV, one or two files)     — eye-tracking CSV (event rows) OR a paired fixations+saccades export
 *
 * At least one file is required.
 *
 * Saves uploaded files, spawns the Python screening pipeline as a child
 * process, and returns a report ID (rid) immediately.  The client polls
 * GET /api/screening/{rid} for the completed report.
 */
import { NextRequest, NextResponse } from "next/server"
import { spawn } from "child_process"
import { writeFile, mkdir, unlink } from "fs/promises"
import { existsSync } from "fs"
import path from "path"
import crypto from "crypto"

export const runtime = "nodejs"
const PIPELINE_TIMEOUT_MS = 5 * 60 * 1_000

// ── Paths ─────────────────────────────────────────────────────────────────

function lexcheckRoot(): string {
  let cur = process.cwd()
  for (let i = 0; i < 8; i++) {
    const backend = path.join(cur, "backend")
    const frontend = path.join(cur, "frontend")
    if (existsSync(backend) && existsSync(frontend)) return cur

    const base = path.basename(cur)
    if (base === "frontend") {
      const parent = path.resolve(cur, "..")
      const backend2 = path.join(parent, "backend")
      const frontend2 = path.join(parent, "frontend")
      if (existsSync(backend2) && existsSync(frontend2)) return parent
    }

    const next = path.resolve(cur, "..")
    if (next === cur) break
    cur = next
  }
  return process.cwd()
}

const LEXCHECK_DIR = lexcheckRoot()
const FRONTEND_DIR = path.join(LEXCHECK_DIR, "frontend")
const BACKEND_DIR = path.join(LEXCHECK_DIR, "backend")
const TMP_DIR = path.join(FRONTEND_DIR, ".tmp")

function pythonExecutable(): string {
  if (process.platform === "win32") {
    const venvPython = path.join(LEXCHECK_DIR, ".venv", "Scripts", "python.exe")
    if (existsSync(venvPython)) return venvPython
    return "python"
  }

  const venvPython = path.join(LEXCHECK_DIR, ".venv", "bin", "python3")
  if (existsSync(venvPython)) return venvPython
  return "python3"
}

function reportsDir() {
  return path.join(TMP_DIR, "reports")
}

function processingPath(rid: string) {
  return path.join(reportsDir(), `${rid}.processing.json`)
}

function uploadsDir(rid: string) {
  return path.join(TMP_DIR, "uploads", rid)
}

function outputDir(rid: string) {
  return path.join(BACKEND_DIR, "outputs", rid)
}

async function writeProcessingReport(rid: string, payload: Record<string, unknown>) {
  const p = processingPath(rid)
  await writeFile(p, JSON.stringify(payload, null, 2))
}

async function clearProcessingReport(rid: string) {
  const p = processingPath(rid)
  if (!existsSync(p)) return
  try {
    await unlink(p)
  } catch {
    // ignore
  }
}

// ── Allowed file types ───────────────────────────────────────────────────

const ALLOWED_IMAGE = ["image/jpeg", "image/png", "image/webp"]

function isAllowedImage(mime: string): boolean {
  return ALLOWED_IMAGE.includes(mime)
}

function isAllowedCsv(file: File): boolean {
  const name = (file.name || "").toLowerCase()
  return name.endsWith(".csv")
}

function safeFilename(name: string): string {
  const base = path.basename(name || "upload.csv")
  const cleaned = base.replace(/[^a-zA-Z0-9._-]/g, "_")
  return cleaned || "upload.csv"
}

type ValidationResult =
  | { ok: true }
  | { ok: false; error: string }

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
  if (!files.length) return { ok: true }

  if (files.length > 2) {
    return { ok: false, error: "Upload either 1 CSV (event rows) or 2 CSVs (fixations + saccades pair)." }
  }

  for (const f of files) {
    if (!isAllowedCsv(f)) return { ok: false, error: "Eye-tracking upload must be CSV (.csv) files only." }
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
        error:
          "Paired exports require 2 files: *_fixations.csv (duration_ms + fix_x/fix_y) and *_saccades.csv (ampl or ampl_x/ampl_y + start_x + end_x). Upload both together.",
      }
    }

    return { ok: false, error: `Event-rows CSV is missing required columns: ${eventRowsMissing.join(", ")}.` }
  }

  const [a, b] = files
  const fix = isFix(a!.name) ? a! : isFix(b!.name) ? b! : null
  const sac = isSac(a!.name) ? a! : isSac(b!.name) ? b! : null

  if (!fix || !sac) {
    return {
      ok: false,
      error: "For paired uploads, file names must end with *_fixations.csv and *_saccades.csv (upload both).",
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
    return { ok: false, error: `CSV headers invalid (${parts.join(" | ")}).` }
  }

  return { ok: true }
}

// ── POST handler ─────────────────────────────────────────────────────────

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const handwritingFile = formData.get("handwriting") as File | null
    const eyeTrackingFiles = formData.getAll("eyeTracking") as File[]

    // At least one file required
    if (!handwritingFile && !eyeTrackingFiles.length) {
      return NextResponse.json(
        { error: "At least one file is required (handwriting or eyeTracking)" },
        { status: 400 },
      )
    }

    // Validate MIME types
    if (handwritingFile && !isAllowedImage(handwritingFile.type)) {
      return NextResponse.json(
        {
          error: `Invalid handwriting file type: ${handwritingFile.type}. Allowed: JPEG, PNG, WebP`,
        },
        { status: 400 },
      )
    }
    for (const f of eyeTrackingFiles) {
      if (!isAllowedCsv(f)) {
        return NextResponse.json(
          { error: "Eye-tracking upload must be CSV (.csv) files only." },
          { status: 400 },
        )
      }
    }

    const validated = await validateEyeTrackingFiles(eyeTrackingFiles)
    if (!validated.ok) {
      return NextResponse.json({ error: validated.error }, { status: 400 })
    }

    const rid = crypto.randomUUID()
    const uploadDir = uploadsDir(rid)
    await mkdir(uploadDir, { recursive: true })

    let imagePath: string | null = null
    let csvPath: string | null = null

    // Save handwriting image
    if (handwritingFile) {
      imagePath = path.join(uploadDir, `handwriting.${handwritingFile.name.split(".").pop() ?? "jpg"}`)
      const bytes = Buffer.from(await handwritingFile.arrayBuffer())
      await writeFile(imagePath, bytes)
    }

    // Save eye-tracking (CSV)
    if (eyeTrackingFiles.length) {
      const sorted = eyeTrackingFiles.slice(0, 2)
      for (const f of sorted) {
        const target = path.join(uploadDir, safeFilename(f.name))
        const bytes = Buffer.from(await f.arrayBuffer())
        await writeFile(target, bytes)
      }

      const preferred = sorted.find((f) => /_fixations\.csv$/i.test(f.name)) ?? sorted[0]!
      csvPath = path.join(uploadDir, safeFilename(preferred.name))
    }

    // Prepare reports directory for output
    await mkdir(reportsDir(), { recursive: true })
    await mkdir(outputDir(rid), { recursive: true })

    // ── Spawn Python pipeline (fire-and-forget) ──────────────────────────
    const imageArg = imagePath ?? "sample"
    const csvArg = csvPath ?? "sample"
    const skipHandwriting = !imagePath
    const skipEyeTracking = !csvPath

    const args: string[] = [
      path.join(BACKEND_DIR, "main.py"),
      "--image", imageArg,
      "--csv", csvArg,
      ...(skipHandwriting ? ["--skip-handwriting"] : []),
      ...(skipEyeTracking ? ["--skip-eye-tracking"] : []),
      "--json",
    ]

    const pythonCmd = pythonExecutable()

    const proc = spawn(pythonCmd, args, {
      cwd: BACKEND_DIR,
      env: {
        ...process.env,
        OUTPUT_DIR: outputDir(rid),
      },
      stdio: ["ignore", "pipe", "pipe"],
    })

    await writeProcessingReport(rid, {
      status: "processing",
      startedAt: new Date().toISOString(),
      pid: proc.pid ?? null,
      skipHandwriting,
      skipEyeTracking,
    })

    let stdout = ""
    let stderr = ""

    proc.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString()
    })

    proc.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString()
    })

    const exitCode = await new Promise<number>((resolve, reject) => {
      const timer = setTimeout(() => {
        try {
          proc.kill()
        } catch {
          // ignore
        }
        resolve(124)
      }, PIPELINE_TIMEOUT_MS)

      proc.on("error", (err) => {
        clearTimeout(timer)
        reject(err)
      })
      proc.on("exit", (code) => {
        clearTimeout(timer)
        resolve(code ?? 1)
      })
    })

    await clearProcessingReport(rid)

    if (exitCode === 124) {
      const excerpt = stderr.slice(0, 8_000) || "Pipeline timed out"
      const message =
        "Pipeline timed out while processing. Check the Next.js terminal logs for Python stderr, " +
        "then retry. " +
        excerpt
      await writeErrorReport(rid, message)
      return NextResponse.json({ error: message }, { status: 504 })
    }

    if (exitCode !== 0) {
      const excerpt = stderr.slice(0, 8_000) || `Process exited with code ${exitCode}`
      await writeErrorReport(rid, excerpt)
      return NextResponse.json({ error: excerpt }, { status: 500 })
    }

    try {
      const reportJson = JSON.parse(stdout)
      const reportPath = path.join(reportsDir(), `${rid}.json`)
      await writeFile(reportPath, JSON.stringify(reportJson, null, 2))
      return NextResponse.json({ rid, report: reportJson }, { status: 200 })
    } catch {
      await writeErrorReport(rid, "Pipeline output was not valid JSON")
      return NextResponse.json({ error: "Pipeline output was not valid JSON" }, { status: 500 })
    }

  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error"
    console.error("[screening] POST handler error:", message)
    return NextResponse.json({ error: message }, { status: 500 })
  }
}

// ── Helper: write an error report ─────────────────────────────────────────

async function writeErrorReport(rid: string, message: string): Promise<void> {
  try {
    const reportPath = path.join(reportsDir(), `${rid}.error.json`)
    await writeFile(
      reportPath,
      JSON.stringify({ status: "error", message, timestamp: new Date().toISOString() }, null, 2),
    )
  } catch {
    // Best-effort — logging is already done above
  }
}
