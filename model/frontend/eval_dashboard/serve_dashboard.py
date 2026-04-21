#!/usr/bin/env python3
"""serve_dashboard.py - Serve the eval dashboard and required local assets only."""

from __future__ import annotations

import argparse
import contextlib
import json
import posixpath
import subprocess
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_ROOT = "/model/frontend/eval_dashboard/"
MODEL_VENV_PYTHON = PROJECT_ROOT / "model" / ".venv" / "bin" / "python"
FULL_INFERENCE_SCRIPT = PROJECT_ROOT / "model" / "frontend" / "eval_dashboard" / "full_source_inference.py"
ALLOWED_PREFIXES = (
    DASHBOARD_ROOT,
    "/api/full-source-inference",
    "/data/raw_data/",
    "/data/landmark_data/",
    "/model/data_fusion/",
    "/model/model_evaluation/pipelines/",
)
FULL_INFERENCE_IMPORT_CHECK = "import cv2, numpy, torch"
FULL_INFERENCE_PYTHON: Path | None = None


class EvalDashboardHandler(SimpleHTTPRequestHandler):
    """Serve only the dashboard and assets it needs."""

    server_version = "JamJamBeatEvalDashboard/1.0"

    def do_GET(self) -> None:  # noqa: N802
        request_path = urlsplit(self.path).path
        if request_path == "/api/full-source-inference":
            self._handle_full_source_inference()
            return
        if request_path == "/":
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", DASHBOARD_ROOT)
            self.end_headers()
            return
        if request_path == DASHBOARD_ROOT.rstrip("/"):
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", DASHBOARD_ROOT)
            self.end_headers()
            return
        if not self._is_allowed(request_path):
            self.send_error(HTTPStatus.FORBIDDEN, "Path is not exposed by the eval dashboard server.")
            return
        super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802
        request_path = urlsplit(self.path).path
        if request_path == "/api/full-source-inference":
            self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
            self.end_headers()
            return
        if request_path in {"/", DASHBOARD_ROOT.rstrip("/")}:
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", DASHBOARD_ROOT)
            self.end_headers()
            return
        if not self._is_allowed(request_path):
            self.send_error(HTTPStatus.FORBIDDEN, "Path is not exposed by the eval dashboard server.")
            return
        super().do_HEAD()

    def translate_path(self, path: str) -> str:
        request_path = urlsplit(path).path
        normalized = posixpath.normpath(unquote(request_path))
        resolved = (PROJECT_ROOT / normalized.lstrip("/")).resolve(strict=False)
        try:
            resolved.relative_to(PROJECT_ROOT)
        except ValueError:
            return str(PROJECT_ROOT / "__forbidden__")
        return str(resolved)

    def log_message(self, format: str, *args: object) -> None:
        return super().log_message(format, *args)

    @staticmethod
    def _is_allowed(path: str) -> bool:
        return any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in ALLOWED_PREFIXES)

    def _handle_full_source_inference(self) -> None:
        try:
            query = parse_qs(urlsplit(self.path).query)
            suite_name = query.get("suite", [None])[0]
            model_id = query.get("model", [None])[0]
            source_file = query.get("source", [None])[0]
            if not suite_name or not model_id or not source_file:
                self.send_error(HTTPStatus.BAD_REQUEST, "suite, model, and source query params are required.")
                return

            payload = self._run_full_source_inference(suite_name, model_id, source_file)
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError as exc:
            self.send_error(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # pragma: no cover - runtime safeguard for local server use
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    @staticmethod
    def _run_full_source_inference(suite_name: str, model_id: str, source_file: str) -> dict[str, object]:
        python_exec = resolve_full_inference_python()
        command = [
            str(python_exec),
            str(FULL_INFERENCE_SCRIPT),
            "--suite",
            suite_name,
            "--model",
            model_id,
            "--source",
            source_file,
        ]
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "Full source inference failed.")
        return json.loads(completed.stdout)


def resolve_full_inference_python() -> Path:
    global FULL_INFERENCE_PYTHON
    if FULL_INFERENCE_PYTHON is not None:
        return FULL_INFERENCE_PYTHON

    if not MODEL_VENV_PYTHON.exists():
        raise RuntimeError(
            f"Model virtualenv python not found: {MODEL_VENV_PYTHON}. "
            "Create/install the model environment first."
        )

    check = subprocess.run(
        [str(MODEL_VENV_PYTHON), "-c", FULL_INFERENCE_IMPORT_CHECK],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if check.returncode != 0:
        raise RuntimeError(
            "model/.venv does not have the dependencies required for full-source inference. "
            f"Expected imports: {FULL_INFERENCE_IMPORT_CHECK}. "
            f"stderr: {check.stderr.strip() or '-'}"
        )

    FULL_INFERENCE_PYTHON = MODEL_VENV_PYTHON
    return FULL_INFERENCE_PYTHON


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the JamJamBeat eval dashboard with an allowlisted file set.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8123, help="Port to bind. Default: 8123")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), EvalDashboardHandler)
    print(f"Serving JamJamBeat eval dashboard on http://{args.host}:{args.port}{DASHBOARD_ROOT}")
    print("Exposed prefixes:")
    for prefix in ALLOWED_PREFIXES:
        print(f"  - {prefix}")
    with contextlib.suppress(KeyboardInterrupt):
        httpd.serve_forever()
    httpd.server_close()


if __name__ == "__main__":
    main()
