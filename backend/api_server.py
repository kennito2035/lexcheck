from __future__ import annotations

import argparse
import base64
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from config import OUTPUT_DIR
from pipeline.orchestrator import run_pipeline_from_bytes

logger = logging.getLogger(__name__)


def _bool_from_query(qs: dict[str, list[str]], key: str, default: bool) -> bool:
    raw = (qs.get(key) or [str(int(default))])[0].strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


class _Handler(BaseHTTPRequestHandler):
    server_version = "dyslexia-prescreening/1.0"

    def _set_headers(self, status: int, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _send_json(self, status: int, payload: dict) -> None:
        self._set_headers(status, "application/json; charset=utf-8")
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_OPTIONS(self) -> None:
        self._set_headers(204, "text/plain; charset=utf-8")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/health":
            self._send_json(200, {"ok": True})
            return

        if path.startswith("/v1/outputs/"):
            rel = path[len("/v1/outputs/") :]
            rel = unquote(rel)
            if not rel or rel.endswith("/") or rel.startswith(("/", "\\")) or ".." in rel.replace("\\", "/"):
                self._send_json(400, {"error": "invalid output path"})
                return

            target = (OUTPUT_DIR / rel).resolve()
            output_root = OUTPUT_DIR.resolve()
            try:
                target.relative_to(output_root)
            except ValueError:
                self._send_json(400, {"error": "invalid output path"})
                return

            if not target.exists() or not target.is_file():
                self._send_json(404, {"error": "not found"})
                return

            suffix = target.suffix.lower()
            content_type = "application/octet-stream"
            if suffix == ".png":
                content_type = "image/png"
            elif suffix in {".jpg", ".jpeg"}:
                content_type = "image/jpeg"
            elif suffix == ".json":
                content_type = "application/json; charset=utf-8"
            elif suffix == ".txt":
                content_type = "text/plain; charset=utf-8"

            self._set_headers(200, content_type)
            self.wfile.write(target.read_bytes())
            return

        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path != "/v1/screening-report":
            self._send_json(404, {"error": "not found"})
            return

        enable_llm = _bool_from_query(qs, "enable_llm", True)
        skip_handwriting = _bool_from_query(qs, "skip_handwriting", False)
        skip_eye_tracking = _bool_from_query(qs, "skip_eye_tracking", False)

        if skip_handwriting and skip_eye_tracking:
            self._send_json(400, {"error": "cannot set both skip_handwriting and skip_eye_tracking"})
            return

        content_type = (self.headers.get("Content-Type") or "").lower()
        length = int(self.headers.get("Content-Length") or "0")
        raw_body = self.rfile.read(length) if length > 0 else b""

        try:
            if content_type.startswith("application/json"):
                payload = json.loads(raw_body.decode("utf-8"))
                image_b64 = str(payload.get("image_base64") or "")
                csv_b64 = str(payload.get("csv_base64") or "")
                image_filename = str(payload.get("image_filename") or "input.png")
                csv_filename = str(payload.get("csv_filename") or "input.csv")

                if not image_b64 and not skip_handwriting:
                    self._send_json(400, {"error": "image_base64 is required unless skip_handwriting=true"})
                    return
                if not csv_b64 and not skip_eye_tracking:
                    self._send_json(400, {"error": "csv_base64 is required unless skip_eye_tracking=true"})
                    return

                image_bytes = base64.b64decode(image_b64) if image_b64 else b""
                csv_bytes = base64.b64decode(csv_b64) if csv_b64 else b""
            elif "multipart/form-data" in content_type:
                import cgi
                import io

                env = {"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")}
                fs = cgi.FieldStorage(
                    fp=io.BytesIO(raw_body),
                    headers=self.headers,
                    environ=env,
                    keep_blank_values=True,
                )
                img_item = fs["image"] if "image" in fs else None
                csv_item = fs["csv"] if "csv" in fs else None
                if img_item is None and not skip_handwriting:
                    self._send_json(400, {"error": "multipart requires field: image (unless skip_handwriting=true)"})
                    return
                if csv_item is None and not skip_eye_tracking:
                    self._send_json(400, {"error": "multipart requires field: csv (unless skip_eye_tracking=true)"})
                    return

                image_filename = getattr(img_item, "filename", None) or "input.png"
                csv_filename = getattr(csv_item, "filename", None) or "input.csv"
                image_bytes = img_item.file.read() if img_item is not None else b""
                csv_bytes = csv_item.file.read() if csv_item is not None else b""
            else:
                self._send_json(415, {"error": "unsupported content-type"})
                return

            report = run_pipeline_from_bytes(
                image_bytes=image_bytes,
                csv_bytes=csv_bytes,
                enable_llm=enable_llm,
                skip_handwriting=skip_handwriting,
                skip_eye_tracking=skip_eye_tracking,
                image_filename=image_filename,
                csv_filename=csv_filename,
            )
            out = json.loads(report.to_json(indent=2))
            out_text = report.to_text()
            self._send_json(200, {"report": out, "text": out_text})
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            logger.error("Request failed", exc_info=True)
            self._send_json(500, {"error": str(exc)})


def main() -> int:
    parser = argparse.ArgumentParser(description="Local HTTP API wrapper for the dyslexia prescreening backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    logger.info("Server listening on http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
