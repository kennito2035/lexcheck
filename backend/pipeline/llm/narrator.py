from __future__ import annotations

import logging
import json
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import (
    GEMINI_API_KEY,
    GEMINI_MAX_RETRIES,
    GEMINI_MODEL,
    GEMINI_RETRY_BACKOFF_MAX_S,
    GEMINI_RETRY_BACKOFF_S,
    GEMINI_TIMEOUT_S,
)

logger = logging.getLogger(__name__)

SPECIALIST_SENTENCE: str = "This is a pre-screening indicator requiring specialist assessment."
MAX_SUMMARY_WORDS: int = 320

SYSTEM_PROMPT: str = (
    "You are a pre-screening report narrator.\n"
    "Goal: Write an objective, helpful interpretation for teachers/parents based strictly on the provided model outputs.\n"
    "Rules:\n"
    "- Only use facts present in the provided outputs. Do not guess missing information.\n"
    "- Use numeric values exactly as provided (do not change totals).\n"
    "- You may interpret what metrics can indicate in general, using cautious language ('may', 'can be associated with'), without diagnosing.\n"
    "- You may interpret labels and flags in plain language (e.g., explain what a 'high' or 'moderate' label means in the context of this tool), without adding new numeric thresholds.\n"
    "- If a stage was skipped, clearly state that the report reflects only the available modality.\n"
    "- Never diagnose dyslexia or any condition.\n"
    "- Keep it <=320 words.\n"
    "- End with: 'This is a pre-screening indicator requiring specialist assessment.'\n"
    "Format:\n"
    "- Use 2–4 short paragraphs (separated by blank lines).\n"
    "- Paragraph 1: Combined risk score + label, and which modality/modality(ies) were included (explicitly mention if handwriting or eye-tracking was skipped).\n"
    "- Paragraph 2: If eye-tracking ran, interpret fixation mean duration, regression rate, and reading difficulty probability in plain language (no diagnosis).\n"
    "- Paragraph 3: If handwriting ran, summarize totals/reversals/corrections/reversal rate and interpret what that pattern may suggest for pre-screening (no diagnosis).\n"
    "- Paragraph 4: If SHAP is available, list the top two SHAP feature names and give a one-sentence plain-language explanation of what each feature relates to.\n"
    "- Do not introduce any numbers or thresholds that are not in the provided outputs.\n"
)


def narrate_findings(
    *,
    handwriting_reversal_count: int,
    handwriting_corrected_count: int,
    handwriting_normal_count: int,
    handwriting_reversal_rate: float,
    fixation_mean_duration_ms: float,
    regression_rate: float,
    reading_difficulty_prob: float,
    severity_cluster: str,
    top_shap_contributors: list[tuple[str, float]],
    combined_risk_score: float,
    combined_risk_label: str,
    handwriting_skipped: bool = False,
    eye_tracking_skipped: bool = False,
) -> str:
    total_letters = max(0, handwriting_reversal_count + handwriting_corrected_count + handwriting_normal_count)
    contributors_text = ", ".join([f"{k}={v:+.3f}" for k, v in top_shap_contributors[:3]]) or "none"
    lines: list[str] = ["Model outputs:"]
    lines.append(f"- Handwriting stage skipped={handwriting_skipped}")
    if not handwriting_skipped:
        lines.append(
            f"- Handwriting detections: Total={total_letters}, Reversal={handwriting_reversal_count}, "
            f"Corrected={handwriting_corrected_count}, Normal={handwriting_normal_count}, "
            f"Reversal rate={handwriting_reversal_rate:.3f}"
        )
    lines.append(f"- Eye-tracking stage skipped={eye_tracking_skipped}")
    if not eye_tracking_skipped:
        lines.append(
            f"- Eye-tracking: Fixation mean duration (ms)={fixation_mean_duration_ms:.1f}, "
            f"Regression rate={regression_rate:.3f}, Reading difficulty probability={reading_difficulty_prob:.3f}, "
            f"Severity cluster={severity_cluster}"
        )
        lines.append(f"- SHAP top contributors (class-1): {contributors_text}")
    lines.append(f"- Combined ensemble risk score={combined_risk_score:.4f}, label={combined_risk_label}")
    lines.append("")
    lines.append("Write a short summary grounded strictly in these outputs.")
    lines.append("Requirements:")
    lines.append("- Mention the combined risk score and risk label.")
    lines.append("- If a stage is skipped, explicitly say it was skipped and do not imply that it ran.")
    if not handwriting_skipped:
        lines.append("- Mention handwriting total, reversals, corrected, reversal rate.")
        lines.append("- Briefly interpret what the handwriting pattern suggests for pre-screening (objective, no diagnosis).")
    if not eye_tracking_skipped:
        lines.append("- Mention fixation mean, regression rate, reading difficulty probability, severity cluster.")
        lines.append("- Mention the top two SHAP contributors by name (if provided).")
    if eye_tracking_skipped:
        lines.append("- State that eye-tracking metrics and SHAP contributors are unavailable because that stage was skipped.")
    lines.append("- Do not introduce any numbers not present above.")
    user_prompt = "\n".join(lines)

    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY is empty. LLM narration unavailable.")
        return (
            f"Combined risk score={combined_risk_score:.4f}, label={combined_risk_label}. "
            "LLM narration is unavailable (no API key), so this is a minimal summary only. "
            "This is a pre-screening indicator requiring specialist assessment."
        )

    return _narrate_with_gemini(
        user_prompt=user_prompt,
        combined_risk_score=combined_risk_score,
        combined_risk_label=combined_risk_label,
    )


def _narrate_with_gemini(*, user_prompt: str, combined_risk_score: float, combined_risk_label: str) -> str:
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY is empty. LLM narration unavailable.")
        return (
            f"LLM narration unavailable (no API key). Combined risk score: {combined_risk_score:.4f} "
            f"({combined_risk_label}). This is a pre-screening indicator requiring specialist assessment."
        )

    models_to_try = [GEMINI_MODEL, "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3-flash-preview"]

    last_error_text: str | None = None
    for model_name in _dedupe_keep_order(models_to_try):
        for attempt in range(max(1, GEMINI_MAX_RETRIES)):
            try:
                return _call_gemini_generate_content(model=model_name, user_prompt=user_prompt)
            except HTTPError as exc:
                body_text = _read_http_error_body(exc)
                last_error_text = body_text or str(exc)
                if exc.code == 404:
                    break

                if exc.code in {429, 500, 502, 503, 504}:
                    wait_s = _compute_backoff_seconds(exc, attempt=attempt)
                    logger.warning("Gemini API busy (HTTP %s). Retrying in %.2fs (model=%s).", exc.code, wait_s, model_name)
                    time.sleep(wait_s)
                    continue

                logger.error("Gemini API call failed (HTTP %s).", exc.code, exc_info=True)
                return (
                    "LLM narration unavailable due to an error calling the API. "
                    f"Combined risk score: {combined_risk_score:.4f} ({combined_risk_label}). "
                    "This is a pre-screening indicator requiring specialist assessment."
                )
            except (URLError, TimeoutError, ValueError, OSError) as exc:
                last_error_text = str(exc)
                logger.error("Gemini API call failed.", exc_info=True)
                return (
                    "LLM narration unavailable due to an error calling the API. "
                    f"Combined risk score: {combined_risk_score:.4f} ({combined_risk_label}). "
                    "This is a pre-screening indicator requiring specialist assessment."
                )

    logger.error("Gemini API call failed after trying multiple models. Last error: %s", last_error_text)
    return (
        "LLM narration unavailable due to an error calling the API. "
        f"Combined risk score: {combined_risk_score:.4f} ({combined_risk_label}). "
        "This is a pre-screening indicator requiring specialist assessment."
    )


def _call_gemini_generate_content(*, model: str, user_prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 512},
    }

    req = Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    req.add_header("x-goog-api-key", GEMINI_API_KEY)

    with urlopen(req, timeout=float(GEMINI_TIMEOUT_S)) as resp:
        body = resp.read().decode("utf-8")
    data = json.loads(body)
    candidates = data.get("candidates") or []
    if not candidates:
        raise ValueError("No candidates returned from Gemini")
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    if not parts:
        raise ValueError("No content parts returned from Gemini")
    text = str(parts[0].get("text") or "").strip()
    if not text:
        raise ValueError("Empty response from Gemini")
    cleaned = _sanitize_llm_summary(text)
    if _summary_is_too_generic(cleaned) or not _summary_meets_requirements(cleaned, user_prompt=user_prompt):
        return _sanitize_llm_summary(_deterministic_summary_from_prompt(user_prompt))
    return cleaned


def _sanitize_llm_summary(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"(,?\s+and)\.\s*$", ".", text, flags=re.IGNORECASE)
    text = re.sub(r"(,?\s+and)\s*$", ".", text, flags=re.IGNORECASE)
    text = re.sub(r"\.{2,}", ".", text)

    text = _drop_trailing_specialist_sentences(text)
    if not text.lower().endswith(SPECIALIST_SENTENCE.lower()):
        sep = " " if text.endswith((".", "!", "?")) else ". "
        text = f"{text}{sep}{SPECIALIST_SENTENCE}"

    words = text.split()
    if len(words) <= MAX_SUMMARY_WORDS:
        return text

    final_words = SPECIALIST_SENTENCE.split()
    max_main = max(0, MAX_SUMMARY_WORDS - len(final_words))
    main = text[: -len(SPECIALIST_SENTENCE)].strip()
    main_words = main.split()
    main_trunc = " ".join(main_words[:max_main]).rstrip(" ,;:")
    if main_trunc and not main_trunc.endswith((".", "!", "?")):
        main_trunc = f"{main_trunc}."
    if not main_trunc:
        return SPECIALIST_SENTENCE
    return f"{main_trunc} {SPECIALIST_SENTENCE}"


def _drop_trailing_specialist_sentences(text: str) -> str:
    text_stripped = text.strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text_stripped) if s.strip()]
    while sentences:
        tail = sentences[-1].lower()
        if "specialist assessment" in tail and "pre-screening" in tail:
            sentences.pop()
            continue
        if "specialist assessment" in tail:
            sentences.pop()
            continue
        if "pre-screening indicator" in tail and "assessment" in tail:
            sentences.pop()
            continue
        break
    return " ".join(sentences).strip()


def _summary_is_too_generic(text: str) -> bool:
    digits = re.findall(r"\d", text)
    if len(digits) < 4:
        return True
    required_terms = ["risk"]
    lowered = text.lower()
    return any(t not in lowered for t in required_terms)


def _summary_meets_requirements(text: str, *, user_prompt: str) -> bool:
    lowered = text.lower()

    lines = [ln.strip() for ln in user_prompt.splitlines() if ln.strip().startswith("- ")]
    hw_skipped = any(ln.lower().startswith("- handwriting stage skipped=true") for ln in lines)
    et_skipped = any(ln.lower().startswith("- eye-tracking stage skipped=true") for ln in lines)

    has_risk = ("risk score" in lowered or "risk" in lowered) and ("label" in lowered or "low" in lowered or "moderate" in lowered or "high" in lowered)
    if not has_risk:
        return False

    if not hw_skipped:
        if "handwriting" not in lowered:
            return False
        if not any(term in lowered for term in ["reversal", "reversals", "corrected", "total"]):
            return False

    if not et_skipped:
        if "eye-tracking" not in lowered and "eye tracking" not in lowered:
            return False
        if not any(term in lowered for term in ["fixation", "regression", "probability", "severity"]):
            return False
        if "shap" not in lowered:
            return False

    if et_skipped and not any(term in lowered for term in ["skipped", "unavailable", "not included"]):
        return False

    return True


def _deterministic_summary_from_prompt(user_prompt: str) -> str:
    lines = [ln.strip() for ln in user_prompt.splitlines() if ln.strip().startswith("- ")]
    hw_skipped = any(ln.lower().startswith("- handwriting stage skipped=true") for ln in lines)
    et_skipped = any(ln.lower().startswith("- eye-tracking stage skipped=true") for ln in lines)
    hw = next((ln for ln in lines if ln.lower().startswith("- handwriting detections:")), "")
    et = next((ln for ln in lines if ln.lower().startswith("- eye-tracking:")), "")
    shap = next((ln for ln in lines if ln.lower().startswith("- shap top contributors")), "")
    risk = next((ln for ln in lines if ln.lower().startswith("- combined ensemble risk score")), "")

    top_two = ""
    m = re.search(r":\s*(.+)$", shap)
    if m:
        parts = [p.strip() for p in m.group(1).split(",") if p.strip() and p.strip().lower() != "none"]
        if parts:
            names = [p.split("=")[0].strip() for p in parts[:2]]
            top_two = ", ".join(names)

    hw_short = hw.replace("- Handwriting detections:", "Handwriting:").strip()
    et_short = et.replace("- Eye-tracking:", "Eye-tracking:").strip()
    risk_short = risk.replace("- Combined ensemble", "Combined").strip()
    shap_short = f"Top SHAP contributors: {top_two}." if top_two else "Top SHAP contributors: (none)."

    parts_out: list[str] = [f"{risk_short}."]
    if hw_skipped:
        parts_out.append("Handwriting stage was skipped.")
    elif hw_short:
        parts_out.append(f"{hw_short}.")
        parts_out.append(
            "For pre-screening, higher counts of reversals/corrections can be a signal worth discussing with a specialist, "
            "especially if it aligns with observed reading or writing difficulties."
        )
    if et_skipped:
        parts_out.append("Eye-tracking stage was skipped, so eye-tracking metrics and SHAP contributors are not included.")
    elif et_short:
        parts_out.append(f"{et_short}.")
        parts_out.append(shap_short)
    summary = " ".join([p for p in parts_out if p])
    return summary


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _read_http_error_body(err: HTTPError) -> str | None:
    try:
        raw = err.read()
        if not raw:
            return None
        return raw.decode("utf-8", errors="replace")
    except OSError:
        return None


def _compute_backoff_seconds(err: HTTPError, *, attempt: int) -> float:
    retry_after = _retry_after_seconds(err)
    if retry_after is not None:
        return float(min(GEMINI_RETRY_BACKOFF_MAX_S, max(0.0, retry_after)))
    exp = GEMINI_RETRY_BACKOFF_S * (2.0**attempt)
    return float(min(GEMINI_RETRY_BACKOFF_MAX_S, max(0.0, exp)))


def _retry_after_seconds(err: HTTPError) -> float | None:
    headers = getattr(err, "headers", None)
    if headers is None:
        return None
    value = headers.get("Retry-After")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None
