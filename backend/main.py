from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pipeline.orchestrator import run_pipeline


def _resolve_input_path(arg_value: str, sample_path: Path) -> Path:
    if arg_value.strip().lower() == "sample":
        return sample_path
    return Path(arg_value)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dyslexia pre-screening backend CLI (no diagnosis).")
    parser.add_argument("--image", default="sample", help="Path to handwriting image (or 'sample').")
    parser.add_argument("--csv", default="sample", help="Path to eye-tracking CSV (or 'sample').")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write outputs/screening_report.json (under OUTPUT_DIR).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    parser.add_argument("--skip-handwriting", action="store_true", help="Skip handwriting stages (YOLO + heatmap).")
    parser.add_argument("--skip-eye-tracking", action="store_true", help="Skip eye-tracking stages (features + RF + SHAP).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    project_root = Path(__file__).resolve().parent
    if args.skip_handwriting and args.skip_eye_tracking:
        logging.getLogger(__name__).critical("Invalid flags: cannot set both --skip-handwriting and --skip-eye-tracking")
        return 1
    image_path = _resolve_input_path(args.image, project_root / "data" / "sample" / "handwriting_sample.jpg")
    csv_path = _resolve_input_path(args.csv, project_root / "data" / "sample" / "eye_tracking_sample.csv")

    try:
        report = run_pipeline(
            image_path=image_path,
            csv_path=csv_path,
            skip_handwriting=bool(args.skip_handwriting),
            skip_eye_tracking=bool(args.skip_eye_tracking),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logging.getLogger(__name__).critical("Pipeline failed", exc_info=True)
        return 1

    if args.save:
        from config import OUTPUT_DIR

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "screening_report.json"
        output_path.write_text(report.to_json(indent=2), encoding="utf-8")

    if args.json:
        print(report.to_json(indent=2))
    else:
        print(report.to_text())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
