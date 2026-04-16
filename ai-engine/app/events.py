from __future__ import annotations

import csv
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

EVENT_COLUMNS = [
    "event_id",
    "timestamp",
    "event_type",
    "track_id",
    "identity",
    "confidence",
    "attention_state",
    "reason",
    "course_id",
    "session_id",
    "source",
    "snapshot_path",
]

_EVENT_DEFAULTS: dict[str, Any] = {
    "event_id": "",
    "timestamp": "",
    "course_id": "general",
    "session_id": "session_1",
    "reason": "",
    "snapshot_path": "",
    "attention_state": "",
}


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {column: row.get(column, _EVENT_DEFAULTS.get(column, "")) for column in EVENT_COLUMNS}
    if not str(normalized["event_id"]).strip():
        normalized["event_id"] = str(uuid.uuid4())
    if not str(normalized["timestamp"]).strip():
        normalized["timestamp"] = datetime.utcnow().isoformat()
    if not str(normalized["course_id"]).strip():
        normalized["course_id"] = "general"
    if not str(normalized["session_id"]).strip():
        normalized["session_id"] = "session_1"
    return normalized


def ensure_events_csv(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS)
            writer.writeheader()
        return

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        existing_columns = reader.fieldnames or []
        if existing_columns == EVENT_COLUMNS:
            return
        rows = list(reader)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_normalize_row(row))


def append_event_row(path: Path, row: dict[str, Any]) -> None:
    unknown_columns = [key for key in row if key not in EVENT_COLUMNS]
    if unknown_columns:
        unknown_str = ", ".join(sorted(unknown_columns))
        raise ValueError(f"Unknown event columns: {unknown_str}")

    normalized_row = _normalize_row(row)

    ensure_events_csv(path)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS)
        writer.writerow(normalized_row)
