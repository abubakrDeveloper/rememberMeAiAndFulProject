from __future__ import annotations

import csv
import re
import uuid
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

ROSTER_COLUMNS = ["student_id", "full_name", "aliases", "course_id"]

REVIEW_ACTION_COLUMNS = [
    "action_id",
    "action_time",
    "action_type",
    "actor_role",
    "target_event_id",
    "new_identity",
    "student_id",
    "attendance_status",
    "session_date",
    "course_id",
    "session_id",
    "notes",
]

ATTENDANCE_STATUSES = [
    "present",
    "late",
    "left_early",
    "late_and_left_early",
    "absent",
    "excused",
]


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_identity(value: str) -> str:
    cleaned = _safe_text(value).lower()
    cleaned = re.sub(r"[\s\-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")


def _split_aliases(raw_aliases: str) -> list[str]:
    if not raw_aliases:
        return []
    parts = re.split(r"[,;|]", raw_aliases)
    return [part.strip() for part in parts if part.strip()]


def load_roster(roster_csv: Path, students_dir: Path) -> pd.DataFrame:
    records: list[dict[str, str]] = []

    if roster_csv.exists():
        try:
            roster_df = pd.read_csv(roster_csv, dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            roster_df = pd.DataFrame(columns=ROSTER_COLUMNS)

        for _, row in roster_df.iterrows():
            student_id = _safe_text(row.get("student_id") or row.get("identity"))
            if not student_id:
                continue
            records.append(
                {
                    "student_id": student_id,
                    "full_name": _safe_text(row.get("full_name")) or student_id,
                    "aliases": _safe_text(row.get("aliases")),
                    "course_id": _safe_text(row.get("course_id")) or "general",
                }
            )
    elif students_dir.exists():
        for child in sorted(students_dir.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            records.append(
                {
                    "student_id": child.name,
                    "full_name": child.name,
                    "aliases": "",
                    "course_id": "general",
                }
            )

    if not records:
        return pd.DataFrame(columns=ROSTER_COLUMNS)

    roster = pd.DataFrame(records)
    roster = roster.drop_duplicates(subset=["student_id"], keep="first")
    return roster[ROSTER_COLUMNS].reset_index(drop=True)


def _ensure_review_actions_csv(path: Path) -> None:
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_ACTION_COLUMNS)
        writer.writeheader()


def load_review_actions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=REVIEW_ACTION_COLUMNS)

    try:
        actions = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        actions = pd.DataFrame(columns=REVIEW_ACTION_COLUMNS)

    for column in REVIEW_ACTION_COLUMNS:
        if column not in actions.columns:
            actions[column] = ""

    if "action_time" in actions.columns:
        actions["action_time"] = pd.to_datetime(actions["action_time"], errors="coerce")

    return actions[REVIEW_ACTION_COLUMNS]


def append_review_action(path: Path, row: dict[str, Any]) -> None:
    unknown_columns = [key for key in row if key not in REVIEW_ACTION_COLUMNS]
    if unknown_columns:
        unknown_str = ", ".join(sorted(unknown_columns))
        raise ValueError(f"Unknown review-action columns: {unknown_str}")

    normalized = {column: _safe_text(row.get(column, "")) for column in REVIEW_ACTION_COLUMNS}
    if not normalized["action_id"]:
        normalized["action_id"] = str(uuid.uuid4())
    if not normalized["action_time"]:
        normalized["action_time"] = datetime.utcnow().isoformat()

    _ensure_review_actions_csv(path)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_ACTION_COLUMNS)
        writer.writerow(normalized)


def build_identity_lookup(roster: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}

    for _, row in roster.iterrows():
        student_id = _safe_text(row.get("student_id"))
        if not student_id:
            continue

        aliases = _split_aliases(_safe_text(row.get("aliases")))
        candidates = [student_id, _safe_text(row.get("full_name")), *aliases]

        for candidate in candidates:
            key = normalize_identity(candidate)
            if key and key not in lookup:
                lookup[key] = student_id

    return lookup


def apply_review_actions(
    events: pd.DataFrame,
    actions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    corrected = events.copy()

    if "identity" not in corrected.columns:
        corrected["identity"] = ""
    if "event_type" not in corrected.columns:
        corrected["event_type"] = ""
    if "event_id" not in corrected.columns:
        corrected["event_id"] = ""

    corrected["corrected_identity"] = corrected["identity"].astype(str)
    corrected["corrected_event_type"] = corrected["event_type"].astype(str)
    corrected["is_false_positive"] = False
    corrected["outsider_approved"] = False

    if actions.empty:
        return corrected, pd.DataFrame(columns=REVIEW_ACTION_COLUMNS)

    actions_sorted = actions.sort_values("action_time", ascending=True, kind="stable")

    for _, action in actions_sorted.iterrows():
        action_type = _safe_text(action.get("action_type"))
        target_event_id = _safe_text(action.get("target_event_id"))
        if not target_event_id:
            continue

        mask = corrected["event_id"].astype(str) == target_event_id
        if not mask.any():
            continue

        if action_type == "relabel_face":
            new_identity = _safe_text(action.get("new_identity"))
            if not new_identity:
                continue
            corrected.loc[mask, "corrected_identity"] = new_identity
            corrected.loc[mask, "corrected_event_type"] = "recognized_student"
        elif action_type == "mark_false_positive":
            corrected.loc[mask, "is_false_positive"] = True
        elif action_type == "approve_outsider_alert":
            corrected.loc[mask, "outsider_approved"] = True

    attendance_overrides = actions_sorted[
        actions_sorted["action_type"].astype(str) == "correct_attendance"
    ].copy()
    return corrected, attendance_overrides


def _session_mask(
    events: pd.DataFrame,
    session_date: date,
    course_id: str,
    session_id: str,
) -> pd.Series:
    ts = pd.to_datetime(events["timestamp"], errors="coerce")
    mask = ts.dt.date == session_date

    if "course_id" in events.columns:
        mask &= events["course_id"].astype(str).fillna("general") == course_id
    if "session_id" in events.columns:
        mask &= events["session_id"].astype(str).fillna("session_1") == session_id

    return mask


def _derive_status(late: bool, early_leave: bool) -> str:
    if late and early_leave:
        return "late_and_left_early"
    if late:
        return "late"
    if early_leave:
        return "left_early"
    return "present"


def build_session_attendance(
    events: pd.DataFrame,
    roster: pd.DataFrame,
    attendance_overrides: pd.DataFrame,
    session_date: date,
    course_id: str,
    session_id: str,
    session_start: time,
    session_end: time,
    late_after_minutes: int,
    early_leave_before_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    if events.empty:
        empty_attendance = pd.DataFrame(
            columns=[
                "student_id",
                "full_name",
                "status",
                "late_arrival",
                "early_leave",
                "first_seen",
                "last_seen",
                "sightings",
                "overridden",
                "override_note",
            ]
        )
        empty_outsiders = pd.DataFrame(columns=["event_id", "timestamp", "identity", "snapshot_path"])
        return empty_attendance, empty_outsiders, {
            "roster_count": 0,
            "present_count": 0,
            "late_count": 0,
            "early_leave_count": 0,
            "attendance_percentage": 0.0,
            "unknown_visitor_count": 0,
            "approved_outsider_count": 0,
        }

    session_events = events[_session_mask(events, session_date, course_id, session_id)].copy()

    if "timestamp" in session_events.columns:
        session_events["timestamp"] = pd.to_datetime(session_events["timestamp"], errors="coerce")

    recognized = session_events[
        (session_events["corrected_event_type"].astype(str) == "recognized_student")
        & (~session_events["is_false_positive"].astype(bool))
    ].copy()

    lookup = build_identity_lookup(roster)
    recognized["matched_student_id"] = recognized["corrected_identity"].astype(str).map(
        lambda value: lookup.get(normalize_identity(value), "")
    )
    recognized = recognized[recognized["matched_student_id"] != ""]

    if not recognized.empty:
        aggregated = (
            recognized.groupby("matched_student_id", as_index=False)
            .agg(
                first_seen=("timestamp", "min"),
                last_seen=("timestamp", "max"),
                sightings=("matched_student_id", "count"),
                max_confidence=("confidence", "max"),
            )
            .rename(columns={"matched_student_id": "student_id"})
        )
    else:
        aggregated = pd.DataFrame(
            columns=["student_id", "first_seen", "last_seen", "sightings", "max_confidence"]
        )

    start_dt = datetime.combine(session_date, session_start)
    end_dt = datetime.combine(session_date, session_end)
    if end_dt <= start_dt:
        end_dt = start_dt + timedelta(hours=1)

    late_cutoff = start_dt + timedelta(minutes=max(0, int(late_after_minutes)))
    early_cutoff = end_dt - timedelta(minutes=max(0, int(early_leave_before_minutes)))

    attendance_rows: list[dict[str, Any]] = []
    for _, roster_row in roster.iterrows():
        student_id = _safe_text(roster_row.get("student_id"))
        full_name = _safe_text(roster_row.get("full_name")) or student_id
        matched = aggregated[aggregated["student_id"] == student_id]

        if matched.empty:
            attendance_rows.append(
                {
                    "student_id": student_id,
                    "full_name": full_name,
                    "status": "absent",
                    "late_arrival": False,
                    "early_leave": False,
                    "first_seen": pd.NaT,
                    "last_seen": pd.NaT,
                    "sightings": 0,
                    "overridden": False,
                    "override_note": "",
                }
            )
            continue

        first_seen = matched.iloc[0]["first_seen"]
        last_seen = matched.iloc[0]["last_seen"]
        late_arrival = bool(pd.notna(first_seen) and first_seen > late_cutoff)
        early_leave = bool(pd.notna(last_seen) and last_seen < early_cutoff)

        attendance_rows.append(
            {
                "student_id": student_id,
                "full_name": full_name,
                "status": _derive_status(late_arrival, early_leave),
                "late_arrival": late_arrival,
                "early_leave": early_leave,
                "first_seen": first_seen,
                "last_seen": last_seen,
                "sightings": int(matched.iloc[0]["sightings"]),
                "overridden": False,
                "override_note": "",
            }
        )

    attendance = pd.DataFrame(
        attendance_rows,
        columns=[
            "student_id",
            "full_name",
            "status",
            "late_arrival",
            "early_leave",
            "first_seen",
            "last_seen",
            "sightings",
            "overridden",
            "override_note",
        ],
    )

    if not attendance_overrides.empty and not attendance.empty:
        scoped = attendance_overrides.copy()
        session_date_str = session_date.isoformat()

        if "session_date" in scoped.columns:
            scoped = scoped[
                (scoped["session_date"].astype(str) == "")
                | (scoped["session_date"].astype(str) == session_date_str)
            ]
        if "course_id" in scoped.columns:
            scoped = scoped[
                (scoped["course_id"].astype(str) == "") | (scoped["course_id"].astype(str) == course_id)
            ]
        if "session_id" in scoped.columns:
            scoped = scoped[
                (scoped["session_id"].astype(str) == "")
                | (scoped["session_id"].astype(str) == session_id)
            ]

        scoped = scoped.sort_values("action_time", ascending=True, kind="stable")
        for _, override in scoped.iterrows():
            student_id = _safe_text(override.get("student_id"))
            status = _safe_text(override.get("attendance_status"))
            if not student_id or status not in ATTENDANCE_STATUSES:
                continue

            idx = attendance.index[attendance["student_id"] == student_id]
            if len(idx) == 0:
                continue

            attendance.loc[idx, "status"] = status
            attendance.loc[idx, "late_arrival"] = status in {"late", "late_and_left_early"}
            attendance.loc[idx, "early_leave"] = status in {"left_early", "late_and_left_early"}
            attendance.loc[idx, "overridden"] = True
            attendance.loc[idx, "override_note"] = _safe_text(override.get("notes"))

    outsiders = session_events[
        (session_events["corrected_event_type"].astype(str) == "outsider_candidate")
        & (~session_events["is_false_positive"].astype(bool))
    ].copy()

    if "timestamp" in outsiders.columns:
        outsiders = outsiders.sort_values("timestamp", ascending=False)

    roster_count = int(len(attendance))
    present_count = int((attendance["status"] != "absent").sum()) if roster_count else 0
    late_count = int(attendance["late_arrival"].sum()) if roster_count else 0
    early_count = int(attendance["early_leave"].sum()) if roster_count else 0
    attendance_pct = float((present_count / roster_count) * 100.0) if roster_count else 0.0
    approved_outsiders = int(outsiders["outsider_approved"].sum()) if not outsiders.empty else 0

    summary = {
        "roster_count": roster_count,
        "present_count": present_count,
        "late_count": late_count,
        "early_leave_count": early_count,
        "attendance_percentage": round(attendance_pct, 2),
        "unknown_visitor_count": int(len(outsiders)),
        "approved_outsider_count": approved_outsiders,
    }

    return attendance, outsiders, summary


def build_daily_report(
    events: pd.DataFrame,
    roster: pd.DataFrame,
    course_id: str,
    session_id: str,
    end_date: date,
    days: int = 7,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    roster_count = int(len(roster))
    lookup = build_identity_lookup(roster)

    frame = events.copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")

    for offset in range(max(1, days)):
        day = end_date - timedelta(days=(days - offset - 1))
        mask = frame["timestamp"].dt.date == day
        if "course_id" in frame.columns:
            mask &= frame["course_id"].astype(str).fillna("general") == course_id
        if "session_id" in frame.columns and session_id:
            mask &= frame["session_id"].astype(str).fillna("session_1") == session_id

        day_events = frame[mask]
        day_recognized = day_events[
            (day_events["corrected_event_type"].astype(str) == "recognized_student")
            & (~day_events["is_false_positive"].astype(bool))
        ]

        matched_ids = {
            lookup.get(normalize_identity(identity), "")
            for identity in day_recognized["corrected_identity"].astype(str).tolist()
        }
        matched_ids.discard("")

        unknown_visitors = day_events[
            (day_events["corrected_event_type"].astype(str) == "outsider_candidate")
            & (~day_events["is_false_positive"].astype(bool))
        ]

        present_count = len(matched_ids)
        attendance_pct = float((present_count / roster_count) * 100.0) if roster_count else 0.0

        rows.append(
            {
                "date": day.isoformat(),
                "present_count": present_count,
                "roster_count": roster_count,
                "attendance_percentage": round(attendance_pct, 2),
                "unknown_visitor_alerts": int(len(unknown_visitors)),
            }
        )

    return pd.DataFrame(rows)


def build_weekly_report(daily_report: pd.DataFrame) -> pd.DataFrame:
    if daily_report.empty:
        return pd.DataFrame(
            columns=[
                "week",
                "avg_attendance_percentage",
                "avg_present_count",
                "total_unknown_visitor_alerts",
            ]
        )

    frame = daily_report.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["week"] = frame["date"].dt.strftime("%G-W%V")

    weekly = (
        frame.groupby("week", as_index=False)
        .agg(
            avg_attendance_percentage=("attendance_percentage", "mean"),
            avg_present_count=("present_count", "mean"),
            total_unknown_visitor_alerts=("unknown_visitor_alerts", "sum"),
        )
        .sort_values("week")
    )

    weekly["avg_attendance_percentage"] = weekly["avg_attendance_percentage"].round(2)
    weekly["avg_present_count"] = weekly["avg_present_count"].round(2)
    return weekly


def export_dataframe(
    frame: pd.DataFrame,
    reports_dir: Path,
    prefix: str,
    session_date: date,
    course_id: str,
    session_id: str,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)

    safe_course = re.sub(r"[^a-zA-Z0-9_-]+", "_", course_id).strip("_") or "general"
    safe_session = re.sub(r"[^a-zA-Z0-9_-]+", "_", session_id).strip("_") or "session_1"
    safe_prefix = re.sub(r"[^a-zA-Z0-9_-]+", "_", prefix).strip("_") or "report"

    output_path = reports_dir / f"{safe_prefix}_{session_date.isoformat()}_{safe_course}_{safe_session}.csv"
    frame.to_csv(output_path, index=False)
    return output_path