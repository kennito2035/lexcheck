/*
 * GET /api/screening/{rid}
 *
 * Returns the completed screening report or a processing/error status.
 *
 * Responses:
 *   200 — { status: "ready",   report: ScreeningReport }
 *   202 — { status: "processing" }
 *   404 — { status: "error",   message: "Report not found" }
 *   500 — { status: "error",   message: "..." }
 */
import { NextRequest, NextResponse } from "next/server"
import { readFile, unlink, writeFile } from "fs/promises"
import { existsSync } from "fs"
import path from "path"

export const runtime = "nodejs"

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
const REPORTS_DIR = path.join(FRONTEND_DIR, ".tmp", "reports")
const BACKEND_OUTPUTS_DIR = path.join(LEXCHECK_DIR, "backend", "outputs")
const PIPELINE_TIMEOUT_MS = 5 * 60 * 1_000

// ── GET handler ───────────────────────────────────────────────────────────

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ rid: string }> },
) {
  const { rid } = await params

  if (!rid || typeof rid !== "string" || !/^[\w-]{36,}$/.test(rid)) {
    return NextResponse.json(
      { status: "error", message: "Invalid report ID" },
      { status: 404 },
    )
  }

  const processingPath = path.join(REPORTS_DIR, `${rid}.processing.json`)
  if (existsSync(processingPath)) {
    try {
      const raw = await readFile(processingPath, "utf-8")
      const parsed = JSON.parse(raw) as { startedAt?: string; pid?: number | null }
      const startedAt = String(parsed.startedAt || "")
      const startedMs = startedAt ? Date.parse(startedAt) : NaN
      const ageMs = Number.isFinite(startedMs) ? Date.now() - startedMs : 0

      if (Number.isFinite(startedMs) && ageMs > PIPELINE_TIMEOUT_MS) {
        const pid = typeof parsed.pid === "number" ? parsed.pid : null
        if (pid) {
          try {
            process.kill(pid)
          } catch {
            // ignore
          }
        }

        const message =
          "Pipeline timed out while processing. This usually means the Python process got stuck " +
          "(e.g., missing/slow dependencies, SHAP/torch hanging, or a bad CSV format). " +
          "Re-run and check the Next.js terminal logs for the Python stderr."
        const errorPath = path.join(REPORTS_DIR, `${rid}.error.json`)
        await writeFile(
          errorPath,
          JSON.stringify({ status: "error", message, timestamp: new Date().toISOString() }, null, 2),
        )
        await unlink(processingPath).catch(() => null)
        return NextResponse.json({ status: "error", message }, { status: 500 })
      }
    } catch {
      // ignore and continue
    }
  }

  const url = new URL(request.url)
  const asset = url.searchParams.get("asset")
  if (asset === "gradcam") {
    const reportPath = path.join(REPORTS_DIR, `${rid}.json`)
    if (!existsSync(reportPath)) {
      return NextResponse.json({ status: "error", message: "Report not ready" }, { status: 404 })
    }

    try {
      const raw = await readFile(reportPath, "utf-8")
      const parsed = JSON.parse(raw) as { gradcam_heatmap_path?: string }
      const p = String(parsed.gradcam_heatmap_path || "")
      if (!p) {
        return NextResponse.json({ status: "error", message: "No heatmap available" }, { status: 404 })
      }

      const heatmapPath = path.resolve(p)
      const allowedRoot = path.resolve(path.join(BACKEND_OUTPUTS_DIR, rid))
      const normHeatmap = process.platform === "win32" ? heatmapPath.toLowerCase() : heatmapPath
      const normRoot = process.platform === "win32" ? allowedRoot.toLowerCase() : allowedRoot
      if (!(normHeatmap === normRoot || normHeatmap.startsWith(normRoot + path.sep))) {
        return NextResponse.json({ status: "error", message: "Invalid asset path" }, { status: 404 })
      }
      if (!existsSync(heatmapPath)) {
        return NextResponse.json({ status: "error", message: "Asset not found" }, { status: 404 })
      }

      const bytes = await readFile(heatmapPath)
      return new NextResponse(bytes, {
        status: 200,
        headers: {
          "Content-Type": "image/png",
          "Cache-Control": "no-store",
        },
      })
    } catch {
      return NextResponse.json({ status: "error", message: "Failed to load asset" }, { status: 500 })
    }
  }

  // Check for error report first
  const errorPath = path.join(REPORTS_DIR, `${rid}.error.json`)
  if (existsSync(errorPath)) {
    try {
      const raw = await readFile(errorPath, "utf-8")
      const errorReport = JSON.parse(raw)
      return NextResponse.json(errorReport, { status: 500 })
    } catch {
      // fall through to generic error
    }
    return NextResponse.json(
      { status: "error", message: "Pipeline failed" },
      { status: 500 },
    )
  }

  // Check for completed report
  const reportPath = path.join(REPORTS_DIR, `${rid}.json`)
  if (existsSync(reportPath)) {
    try {
      const raw = await readFile(reportPath, "utf-8")
      const report = JSON.parse(raw)
      return NextResponse.json({ status: "ready", report }, { status: 200 })
    } catch (parseErr) {
      return NextResponse.json(
        {
          status: "error",
          message: "Failed to parse report data",
        },
        { status: 500 },
      )
    }
  }

  // No report yet — still processing
  return NextResponse.json({ status: "processing" }, { status: 202 })
}
