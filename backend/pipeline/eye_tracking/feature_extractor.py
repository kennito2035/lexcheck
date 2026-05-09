from __future__ import annotations

from pathlib import Path

from config import EYE_FEATURES


def extract_eye_tracking_features(csv_path: Path) -> dict[str, float]:
    import pandas as pd

    if not csv_path.exists():
        raise FileNotFoundError(f"Eye-tracking CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "event_type" in df.columns:
        return _extract_from_event_rows(df)

    if "duration_ms" in df.columns and ("fix_x" in df.columns or "fix_y" in df.columns):
        fix_df = df
        sac_path = _paired_etdd70_path(csv_path, want="saccades")
        if sac_path is None:
            raise ValueError(
                "Detected ETDD70-style fixations file but could not find paired saccades file. "
                f"Expected alongside: {csv_path}"
            )
        sac_df = pd.read_csv(sac_path)
        sac_df.columns = [str(c).strip().lower() for c in sac_df.columns]
        return _extract_from_fixations_saccades(fix_df=fix_df, sac_df=sac_df)

    if "ampl" in df.columns and "start_x" in df.columns and "end_x" in df.columns:
        sac_df = df
        fix_path = _paired_etdd70_path(csv_path, want="fixations")
        if fix_path is None:
            raise ValueError(
                "Detected ETDD70-style saccades file but could not find paired fixations file. "
                f"Expected alongside: {csv_path}"
            )
        fix_df = pd.read_csv(fix_path)
        fix_df.columns = [str(c).strip().lower() for c in fix_df.columns]
        return _extract_from_fixations_saccades(fix_df=fix_df, sac_df=sac_df)

    raise ValueError(
        "Unrecognized eye-tracking CSV schema. Provide either event rows with event_type/duration_ms/amplitude_deg/"
        "is_regression, or an ETDD70 *_fixations.csv / *_saccades.csv file."
    )


def _extract_from_event_rows(df: "pd.DataFrame") -> dict[str, float]:
    import pandas as pd

    aliases: dict[str, list[str]] = {
        "event_type": ["event_type", "event", "type", "eventtype"],
        "duration_ms": ["duration_ms", "duration", "dur_ms", "ms", "durationms"],
        "amplitude_deg": ["amplitude_deg", "amplitude", "amp_deg", "amp", "amplitudedeg"],
        "is_regression": ["is_regression", "regression", "isregression", "is_backward", "backward"],
    }

    resolved: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        for c in candidates:
            if c in df.columns:
                resolved[canonical] = c
                break

    missing = [k for k in aliases if k not in resolved]
    if missing:
        raise ValueError(f"Missing required columns in eye-tracking CSV: {missing}")

    event_col = resolved["event_type"]
    duration_col = resolved["duration_ms"]
    amp_col = resolved["amplitude_deg"]
    reg_col = resolved["is_regression"]

    df[event_col] = df[event_col].astype(str).str.strip().str.lower()

    fixation_mask = df[event_col].isin(["fixation", "fix", "f"])
    saccade_mask = df[event_col].isin(["saccade", "sac", "s"])

    fix = df.loc[fixation_mask].copy()
    sac = df.loc[saccade_mask].copy()

    fix_dur = pd.to_numeric(fix[duration_col], errors="coerce").dropna()
    sac_amp = pd.to_numeric(sac[amp_col], errors="coerce").dropna()
    sac_reg = pd.to_numeric(sac[reg_col], errors="coerce").fillna(0).astype(int)

    fixation_count = int(fixation_mask.sum())
    fixation_mean = float(fix_dur.mean()) if len(fix_dur) else 0.0
    fixation_max = float(fix_dur.max()) if len(fix_dur) else 0.0
    fixation_std = float(fix_dur.std(ddof=0)) if len(fix_dur) else 0.0

    saccade_count = int(saccade_mask.sum())
    saccade_mean_amp = float(sac_amp.mean()) if len(sac_amp) else 0.0
    saccade_std_amp = float(sac_amp.std(ddof=0)) if len(sac_amp) else 0.0

    regression_count = int((sac_reg == 1).sum()) if len(sac_reg) else 0
    regression_rate = float(regression_count / max(1, saccade_count))

    first_pass_fixation_time = float(fix_dur.sum()) if len(fix_dur) else 0.0
    skip_rate = 0.0
    word_reading_time_mean = fixation_mean

    values: dict[str, float] = {
        "fixation_count": float(fixation_count),
        "fixation_mean_duration_ms": float(fixation_mean),
        "fixation_max_duration_ms": float(fixation_max),
        "fixation_std_duration_ms": float(fixation_std),
        "saccade_count": float(saccade_count),
        "saccade_mean_amplitude_deg": float(saccade_mean_amp),
        "saccade_std_amplitude_deg": float(saccade_std_amp),
        "regression_count": float(regression_count),
        "regression_rate": float(regression_rate),
        "first_pass_fixation_time_ms": float(first_pass_fixation_time),
        "skip_rate": float(skip_rate),
        "word_reading_time_mean_ms": float(word_reading_time_mean),
    }

    return {k: float(values.get(k, 0.0)) for k in EYE_FEATURES}


def _paired_etdd70_path(path: Path, *, want: str) -> Path | None:
    p = str(path)
    if want == "saccades":
        candidate = Path(p.replace("_fixations.csv", "_saccades.csv"))
        return candidate if candidate.exists() else None
    if want == "fixations":
        candidate = Path(p.replace("_saccades.csv", "_fixations.csv"))
        return candidate if candidate.exists() else None
    raise ValueError(f"Unknown pair target: {want}")


def _extract_from_fixations_saccades(*, fix_df: "pd.DataFrame", sac_df: "pd.DataFrame") -> dict[str, float]:
    import numpy as np
    import pandas as pd

    if "duration_ms" not in fix_df.columns:
        raise ValueError("Fixations file missing duration_ms column.")

    fix_dur = pd.to_numeric(fix_df["duration_ms"], errors="coerce").dropna()
    fixation_count = int(len(fix_dur))
    fixation_mean = float(fix_dur.mean()) if fixation_count else 0.0
    fixation_max = float(fix_dur.max()) if fixation_count else 0.0
    fixation_std = float(fix_dur.std(ddof=0)) if fixation_count else 0.0

    amplitude_col = "ampl" if "ampl" in sac_df.columns else None
    if amplitude_col is None and "ampl_x" in sac_df.columns and "ampl_y" in sac_df.columns:
        ax = pd.to_numeric(sac_df["ampl_x"], errors="coerce")
        ay = pd.to_numeric(sac_df["ampl_y"], errors="coerce")
        ampl = np.sqrt(ax * ax + ay * ay)
    else:
        ampl = pd.to_numeric(sac_df[amplitude_col] if amplitude_col else pd.Series([], dtype=float), errors="coerce")

    ampl = ampl.dropna()
    saccade_count = int(len(ampl))
    saccade_mean_amp = float(ampl.mean()) if saccade_count else 0.0
    saccade_std_amp = float(ampl.std(ddof=0)) if saccade_count else 0.0

    if "start_x" not in sac_df.columns or "end_x" not in sac_df.columns:
        raise ValueError("Saccades file missing start_x/end_x columns (required for regression detection).")
    start_x = pd.to_numeric(sac_df["start_x"], errors="coerce")
    end_x = pd.to_numeric(sac_df["end_x"], errors="coerce")
    valid = (~start_x.isna()) & (~end_x.isna())
    regression_count = int((end_x[valid] < start_x[valid]).sum())
    regression_rate = float(regression_count / max(1, saccade_count))

    first_pass_fixation_time = float(fix_dur.sum()) if fixation_count else 0.0
    skip_rate = 0.0
    word_reading_time_mean = fixation_mean

    values: dict[str, float] = {
        "fixation_count": float(fixation_count),
        "fixation_mean_duration_ms": float(fixation_mean),
        "fixation_max_duration_ms": float(fixation_max),
        "fixation_std_duration_ms": float(fixation_std),
        "saccade_count": float(saccade_count),
        "saccade_mean_amplitude_deg": float(saccade_mean_amp),
        "saccade_std_amplitude_deg": float(saccade_std_amp),
        "regression_count": float(regression_count),
        "regression_rate": float(regression_rate),
        "first_pass_fixation_time_ms": float(first_pass_fixation_time),
        "skip_rate": float(skip_rate),
        "word_reading_time_mean_ms": float(word_reading_time_mean),
    }
    return {k: float(values.get(k, 0.0)) for k in EYE_FEATURES}
