# LexCheck

LexCheck is a research-grade dyslexia pre-screening demo with:

- A Next.js frontend UI (`frontend`)
- A Python backend pipeline (`backend`) that produces:
  - a JSON screening report
  - explainability artifacts (e.g., Grad-CAM heatmap images)

This is a pre-screening indicator only. It is not a diagnosis.

## Repo Layout

- `frontend` — Next.js frontend (upload + report UI)
- `backend` — Python backend (YOLO + RandomForest + SHAP + Gemini narration)

## Setup

### 1) Python backend dependencies

From the `LexCheck` folder:

```bat
py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
python.exe -m pip install --upgrade pip
pip install -r backend\requirements.txt
```

Create `backend\.env` by copying `backend\.env.example` and filling in values as needed.

Notes (Windows):
- Use Python 3.11 or 3.12 (some dependencies are pinned to `python_version < "3.13"`; Python 3.13+ will skip installing torch/ultralytics/SHAP and the pipeline will fail).
- If `pip` is missing inside the venv, run: `python -m ensurepip --upgrade`
- If you see `'.venv\\Scripts\\activate.bat' is not recognized`, you are likely in the wrong folder or your venv folder name is different. Run the `python -m venv .venv` command above inside `LexCheck\` (so the path is `LexCheck\.venv\...`). If your venv folder is named `venv` (no dot), use: `call venv\Scripts\activate.bat`

### 2) Frontend dependencies

Open a second terminal:

```bat
cd LexCheck\frontend
npm install
```

If `npm` fails in PowerShell due to execution policy, run these commands in Command Prompt (cmd.exe) instead.

## Run

### Start the Next.js app

```bat
cd LexCheck\frontend
npm run dev
```

Open:
- http://localhost:3000/test

Upload:
- handwriting image (PNG/JPG), or
- eye-tracking CSV data (1–2 CSV files for a fixations+saccades pair, or a single event-row CSV), or
- both

The UI will trigger the backend pipeline and then show the report.

## Outputs

Each run is assigned a report id (`rid`).

- Report JSON is cached by the frontend under: `frontend/.tmp/reports/<rid>.json`
- Backend artifacts are written under: `backend/outputs/<rid>/`

Note: Run artifacts and uploads are stored under `frontend/.tmp/` and are ignored by git.

## Environment Variables

Backend reads `.env` from `backend/`:

- `GEMINI_API_KEY` (optional; if missing, the report uses a deterministic summary and makes no external API calls)
- `YOLO_WEIGHTS_PATH` (optional override; defaults to `models/yolov11_dyslexia.pt` under `backend/` — the weights file is required when handwriting runs)
- `RF_MODEL_PATH` (optional override; defaults to `models/rf_dyslexia.pkl` under `backend/` — the model file is required when eye-tracking runs)
- `OUTPUT_DIR` (set automatically per run by the frontend integration)

Note: This repo does not include “mock mode” fallbacks. If model files or Python dependencies are missing, the run will fail and the report page will show a quick-fix checklist.
