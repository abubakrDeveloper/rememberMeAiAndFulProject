from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import urllib.parse
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "output"
SNAPSHOTS_DIR = OUTPUT_DIR / "snapshots"
DASHBOARD_DIR = ROOT_DIR / "dashboard"
AUDIT_LOG_PATH = OUTPUT_DIR / "audit_log.jsonl"


def _append_audit(event: str, details: Dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(),
        "event": event,
        "details": details,
    }
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=True) + "\n")


def _read_audit(limit: int = 100) -> List[Dict[str, Any]]:
    if not AUDIT_LOG_PATH.exists():
        return []

    rows: List[Dict[str, Any]] = []
    with AUDIT_LOG_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return rows[-max(1, limit) :]


def _latest_file(pattern: str) -> Path | None:
    files = sorted(OUTPUT_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _read_csv(path: Path | None) -> List[Dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _read_json(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _snapshot_url(raw_path: str) -> str:
    if not raw_path:
        return ""

    raw = Path(raw_path)
    file_name = raw.name
    if not file_name:
        return ""

    candidate = SNAPSHOTS_DIR / file_name
    if not candidate.exists():
        return ""

    return "/snapshots/" + urllib.parse.quote(file_name)


def _parse_last_updated(file_paths: List[Path]) -> str:
    if not file_paths:
        return ""
    newest = max(file_paths, key=lambda p: p.stat().st_mtime)
    return datetime.fromtimestamp(newest.stat().st_mtime).isoformat()


def build_latest_payload() -> Dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_file = _latest_file("summary_*.json")
    students_file = _latest_file("attendance_students_*.csv")
    teachers_file = _latest_file("attendance_teachers_*.csv")
    incidents_file = _latest_file("incidents_*.csv")

    summary = _read_json(summary_file)
    students = _read_csv(students_file)
    teachers = _read_csv(teachers_file)
    incidents = _read_csv(incidents_file)

    for incident in incidents:
        incident["snapshot_url"] = _snapshot_url(incident.get("snapshot_path", ""))

    class_names = set()
    for row in students + teachers + incidents:
        class_name = (row.get("class_name") or "").strip()
        if class_name:
            class_names.add(class_name)

    summary_class_name = ((summary.get("session") or {}).get("class_name") or "").strip()
    if summary_class_name:
        class_names.add(summary_class_name)

    camera_health = summary.get("camera_health", {}) if isinstance(summary, dict) else {}

    existing_files = [
        p
        for p in [summary_file, students_file, teachers_file, incidents_file]
        if p is not None and p.exists()
    ]

    return {
        "generated_at": datetime.now().isoformat(),
        "last_report_updated_at": _parse_last_updated(existing_files),
        "files": {
            "summary": summary_file.name if summary_file else "",
            "students": students_file.name if students_file else "",
            "teachers": teachers_file.name if teachers_file else "",
            "incidents": incidents_file.name if incidents_file else "",
        },
        "summary": summary,
        "camera_health": camera_health,
        "available_classes": sorted(class_names),
        "students": students,
        "teachers": teachers,
        "incidents": incidents,
        "audit": _read_audit(limit=120),
    }


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def _write_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_snapshot(self, snapshot_name: str) -> None:
        safe_name = Path(snapshot_name).name
        if not safe_name:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid snapshot name")
            return

        target = SNAPSHOTS_DIR / safe_name
        if not target.exists() or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Snapshot not found")
            return

        mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        data = target.read_bytes()

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path in {"/", "/index.html"}:
            _append_audit(
                "dashboard_view",
                {
                    "client_ip": self.client_address[0] if self.client_address else "",
                    "path": parsed.path,
                    "user_agent": self.headers.get("User-Agent", ""),
                },
            )

        if parsed.path == "/api/latest":
            try:
                payload = build_latest_payload()
                self._write_json(payload)
            except Exception as exc:  # noqa: BLE001
                self._write_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if parsed.path.startswith("/snapshots/"):
            snapshot_name = urllib.parse.unquote(parsed.path[len("/snapshots/") :])
            self._serve_snapshot(snapshot_name)
            return

        super().do_GET()


def run_dashboard(host: str, port: int, open_browser: bool) -> None:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((host, port), DashboardHandler)
    url = f"http://{host}:{port}"
    print(f"Dashboard server running on {url}")
    print("Press Ctrl+C to stop.")

    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local admin dashboard for classroom monitor outputs")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--open-browser", action="store_true", help="Open dashboard in default browser")
    args = parser.parse_args()

    run_dashboard(host=args.host, port=args.port, open_browser=args.open_browser)


if __name__ == "__main__":
    main()