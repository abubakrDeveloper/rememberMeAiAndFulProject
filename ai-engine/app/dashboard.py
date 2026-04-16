from datetime import date, datetime, time
from pathlib import Path

import pandas as pd
import streamlit as st

from app.attendance import (
    ATTENDANCE_STATUSES,
    append_review_action,
    apply_review_actions,
    build_daily_report,
    build_identity_lookup,
    build_session_attendance,
    build_weekly_report,
    export_dataframe,
    load_review_actions,
    load_roster,
    normalize_identity,
)
from app.config import DemoConfig
from app.events import EVENT_COLUMNS


def load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)

    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    for column in EVENT_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""

    frame["event_id"] = frame["event_id"].astype(str)
    missing_event_id = frame["event_id"].str.strip() == ""
    if missing_event_id.any():
        frame.loc[missing_event_id, "event_id"] = [
            f"legacy_{index + 1}" for index in frame.index[missing_event_id].tolist()
        ]

    frame["course_id"] = frame["course_id"].astype(str).replace("", "general")
    frame["session_id"] = frame["session_id"].astype(str).replace("", "session_1")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def _show_session_metrics(summary: dict[str, float]) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Roster", int(summary["roster_count"]))
    c2.metric("Present", int(summary["present_count"]))
    c3.metric("Late", int(summary["late_count"]))
    c4.metric("Early Leave", int(summary["early_leave_count"]))
    c5.metric("Attendance %", f"{summary['attendance_percentage']:.2f}%")

    c6, c7 = st.columns(2)
    c6.metric("Unknown Visitor Alerts", int(summary["unknown_visitor_count"]))
    c7.metric("Approved Outsider Alerts", int(summary["approved_outsider_count"]))


def _show_unmatched_recognitions(session_events: pd.DataFrame, roster: pd.DataFrame) -> None:
    if session_events.empty:
        return

    lookup = build_identity_lookup(roster)
    recognized = session_events[
        session_events["corrected_event_type"].astype(str) == "recognized_student"
    ].copy()
    if recognized.empty:
        return

    recognized["matched_student_id"] = recognized["corrected_identity"].astype(str).map(
        lambda value: lookup.get(normalize_identity(value), "")
    )

    unmatched = recognized[
        (recognized["matched_student_id"] == "")
        & (~recognized["is_false_positive"].astype(bool))
    ]
    if unmatched.empty:
        return

    st.subheader("Unmatched Recognitions")
    st.info("These recognized identities did not match the class roster automatically.")
    cols = ["event_id", "timestamp", "corrected_identity", "track_id", "confidence"]
    st.dataframe(unmatched[cols].sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)


def _show_teacher_admin_view(
    attendance: pd.DataFrame,
    outsiders: pd.DataFrame,
    daily_report: pd.DataFrame,
    weekly_report: pd.DataFrame,
    reports_dir: Path,
    session_date: date,
    course_id: str,
    session_id: str,
) -> None:
    st.subheader("Session Attendance")
    if attendance.empty:
        st.warning("No roster rows available. Add data/students/roster.csv or student folders.")
    else:
        st.dataframe(attendance.sort_values(["status", "student_id"]), use_container_width=True, hide_index=True)

    st.subheader("Unknown Visitors")
    if outsiders.empty:
        st.info("No unknown visitor detections for this session.")
    else:
        show_cols = [
            "event_id",
            "timestamp",
            "track_id",
            "confidence",
            "snapshot_path",
            "outsider_approved",
        ]
        st.dataframe(outsiders[show_cols], use_container_width=True, hide_index=True)

    st.subheader("Daily Attendance Report")
    if daily_report.empty:
        st.info("No daily report data available.")
    else:
        st.dataframe(daily_report, use_container_width=True, hide_index=True)
        st.line_chart(daily_report.set_index("date")["attendance_percentage"])

    st.subheader("Weekly Attendance Report")
    if weekly_report.empty:
        st.info("No weekly report data available.")
    else:
        st.dataframe(weekly_report, use_container_width=True, hide_index=True)

    st.subheader("Export Reports")
    st.download_button(
        label="Download Session Attendance CSV",
        data=_csv_bytes(attendance),
        file_name=f"attendance_{session_date}_{course_id}_{session_id}.csv",
        mime="text/csv",
        disabled=attendance.empty,
    )
    st.download_button(
        label="Download Daily Report CSV",
        data=_csv_bytes(daily_report),
        file_name=f"daily_report_{session_date}_{course_id}_{session_id}.csv",
        mime="text/csv",
        disabled=daily_report.empty,
    )
    st.download_button(
        label="Download Weekly Report CSV",
        data=_csv_bytes(weekly_report),
        file_name=f"weekly_report_{session_date}_{course_id}_{session_id}.csv",
        mime="text/csv",
        disabled=weekly_report.empty,
    )

    save_col1, save_col2, save_col3 = st.columns(3)
    if save_col1.button("Save Session Report", disabled=attendance.empty):
        path = export_dataframe(attendance, reports_dir, "attendance", session_date, course_id, session_id)
        st.success(f"Saved: {path}")
    if save_col2.button("Save Daily Report", disabled=daily_report.empty):
        path = export_dataframe(daily_report, reports_dir, "daily_report", session_date, course_id, session_id)
        st.success(f"Saved: {path}")
    if save_col3.button("Save Weekly Report", disabled=weekly_report.empty):
        path = export_dataframe(weekly_report, reports_dir, "weekly_report", session_date, course_id, session_id)
        st.success(f"Saved: {path}")


def _record_action(
    review_actions_csv: Path,
    action_type: str,
    actor_role: str,
    session_date: date,
    course_id: str,
    session_id: str,
    target_event_id: str = "",
    new_identity: str = "",
    student_id: str = "",
    attendance_status: str = "",
    notes: str = "",
) -> None:
    append_review_action(
        review_actions_csv,
        {
            "action_type": action_type,
            "actor_role": actor_role,
            "target_event_id": target_event_id,
            "new_identity": new_identity,
            "student_id": student_id,
            "attendance_status": attendance_status,
            "session_date": session_date.isoformat(),
            "course_id": course_id,
            "session_id": session_id,
            "notes": notes,
        },
    )


def _show_security_tools(
    session_events: pd.DataFrame,
    outsiders: pd.DataFrame,
    review_actions_csv: Path,
    actor_role: str,
    session_date: date,
    course_id: str,
    session_id: str,
) -> None:
    st.subheader("Security / Operator Console")

    pending_outsiders = outsiders[~outsiders["outsider_approved"].astype(bool)].copy()
    if pending_outsiders.empty:
        st.success("No pending outsider alerts.")
    else:
        st.warning(f"Pending outsider alerts: {len(pending_outsiders)}")
        cols = ["event_id", "timestamp", "track_id", "confidence", "snapshot_path"]
        st.dataframe(pending_outsiders[cols], use_container_width=True, hide_index=True)

    if not pending_outsiders.empty:
        with st.form("approve_outsider_form"):
            event_id = st.selectbox(
                "Approve outsider event",
                options=pending_outsiders["event_id"].astype(str).tolist(),
            )
            submitted = st.form_submit_button("Approve Alert")
            if submitted:
                _record_action(
                    review_actions_csv=review_actions_csv,
                    action_type="approve_outsider_alert",
                    actor_role=actor_role,
                    target_event_id=event_id,
                    session_date=session_date,
                    course_id=course_id,
                    session_id=session_id,
                    notes="approved by operator",
                )
                st.success("Outsider alert approved.")
                st.rerun()

    actionable_events = session_events.sort_values("timestamp", ascending=False)
    event_ids = actionable_events["event_id"].astype(str).tolist()

    if event_ids:
        with st.form("false_positive_form"):
            event_id = st.selectbox("Mark false positive for event", options=event_ids)
            notes = st.text_input("False positive note", value="false positive")
            submitted = st.form_submit_button("Mark False Positive")
            if submitted:
                _record_action(
                    review_actions_csv=review_actions_csv,
                    action_type="mark_false_positive",
                    actor_role=actor_role,
                    target_event_id=event_id,
                    session_date=session_date,
                    course_id=course_id,
                    session_id=session_id,
                    notes=notes,
                )
                st.success("False positive was recorded.")
                st.rerun()

        with st.form("relabel_face_form"):
            event_id = st.selectbox("Relabel face event", options=event_ids)
            new_identity = st.text_input("Correct identity")
            submitted = st.form_submit_button("Relabel Face")
            if submitted:
                if not new_identity.strip():
                    st.error("Enter a non-empty identity.")
                else:
                    _record_action(
                        review_actions_csv=review_actions_csv,
                        action_type="relabel_face",
                        actor_role=actor_role,
                        target_event_id=event_id,
                        new_identity=new_identity.strip(),
                        session_date=session_date,
                        course_id=course_id,
                        session_id=session_id,
                        notes="manual relabel",
                    )
                    st.success("Face relabeled.")
                    st.rerun()
    else:
        st.info("No session events available for manual action.")


def _show_admin_attendance_correction(
    attendance: pd.DataFrame,
    review_actions_csv: Path,
    session_date: date,
    course_id: str,
    session_id: str,
) -> None:
    st.subheader("Admin Attendance Corrections")
    if attendance.empty:
        st.info("Attendance table is empty.")
        return

    with st.form("admin_attendance_override_form"):
        student_id = st.selectbox("Student", options=attendance["student_id"].astype(str).tolist())
        status = st.selectbox("Correct status", options=ATTENDANCE_STATUSES)
        notes = st.text_input("Correction note")
        submitted = st.form_submit_button("Apply Attendance Correction")

        if submitted:
            _record_action(
                review_actions_csv=review_actions_csv,
                action_type="correct_attendance",
                actor_role="admin",
                student_id=student_id,
                attendance_status=status,
                session_date=session_date,
                course_id=course_id,
                session_id=session_id,
                notes=notes,
            )
            st.success("Attendance correction saved.")
            st.rerun()


def main() -> None:
    defaults = DemoConfig()

    st.set_page_config(page_title="Classroom Attendance Platform", layout="wide")
    st.title("Classroom Attendance and Safety Platform")
    st.caption("Role-based attendance workflow with manual review, unknown-visitor operations, and reporting")

    with st.sidebar:
        st.header("Role")
        role = st.selectbox("Dashboard Role", options=["teacher", "admin", "security/operator"], index=0)

        st.header("Session")
        course_id = st.text_input("Course ID", value="general").strip() or "general"
        session_id = st.text_input("Session ID", value="session_1").strip() or "session_1"
        session_date = st.date_input("Session Date", value=date.today())
        session_start = st.time_input("Session Start", value=time(hour=9, minute=0))
        session_end = st.time_input("Session End", value=time(hour=10, minute=0))
        late_after_minutes = st.number_input("Late After (minutes)", min_value=0, max_value=120, value=10)
        early_leave_before_minutes = st.number_input(
            "Early Leave Before End (minutes)", min_value=0, max_value=120, value=10
        )

        st.header("Data Sources")
        events_csv = Path(st.text_input("Events CSV", value=str(defaults.events_csv)))
        roster_csv = Path(st.text_input("Roster CSV", value=str(defaults.roster_csv)))
        review_actions_csv = Path(
            st.text_input("Review Actions CSV", value=str(defaults.review_actions_csv))
        )
        reports_dir = Path(st.text_input("Reports Directory", value=str(defaults.reports_dir)))

        if st.button("Refresh"):
            st.rerun()

    events = load_events(events_csv)
    if events.empty:
        st.warning("No events found yet. Start recognition first to populate logs.")
        return

    actions = load_review_actions(review_actions_csv)
    corrected_events, attendance_overrides = apply_review_actions(events, actions)
    corrected_events = corrected_events.sort_values("timestamp", ascending=False)

    roster = load_roster(roster_csv, defaults.students_dir)
    if not roster.empty and "course_id" in roster.columns:
        scoped_roster = roster[
            (roster["course_id"].astype(str) == course_id)
            | (roster["course_id"].astype(str) == "general")
        ].copy()
        if scoped_roster.empty:
            scoped_roster = roster.copy()
    else:
        scoped_roster = roster.copy()

    attendance, outsiders, summary = build_session_attendance(
        events=corrected_events,
        roster=scoped_roster,
        attendance_overrides=attendance_overrides,
        session_date=session_date,
        course_id=course_id,
        session_id=session_id,
        session_start=session_start,
        session_end=session_end,
        late_after_minutes=int(late_after_minutes),
        early_leave_before_minutes=int(early_leave_before_minutes),
    )

    daily_report = build_daily_report(
        events=corrected_events,
        roster=scoped_roster,
        course_id=course_id,
        session_id=session_id,
        end_date=session_date,
        days=7,
    )
    weekly_report = build_weekly_report(daily_report)

    _show_session_metrics(summary)

    session_events = corrected_events[
        (corrected_events["timestamp"].dt.date == session_date)
        & (corrected_events["course_id"].astype(str) == course_id)
        & (corrected_events["session_id"].astype(str) == session_id)
    ].copy()

    _show_unmatched_recognitions(session_events, scoped_roster)

    if role in {"teacher", "admin"}:
        _show_teacher_admin_view(
            attendance=attendance,
            outsiders=outsiders,
            daily_report=daily_report,
            weekly_report=weekly_report,
            reports_dir=reports_dir,
            session_date=session_date,
            course_id=course_id,
            session_id=session_id,
        )

    if role in {"security/operator", "admin"}:
        _show_security_tools(
            session_events=session_events,
            outsiders=outsiders,
            review_actions_csv=review_actions_csv,
            actor_role=role,
            session_date=session_date,
            course_id=course_id,
            session_id=session_id,
        )

    if role == "admin":
        _show_admin_attendance_correction(
            attendance=attendance,
            review_actions_csv=review_actions_csv,
            session_date=session_date,
            course_id=course_id,
            session_id=session_id,
        )

    st.subheader("Recent Session Events")
    if session_events.empty:
        st.info("No events for the selected date/course/session.")
    else:
        show_cols = [
            "event_id",
            "timestamp",
            "event_type",
            "corrected_event_type",
            "identity",
            "corrected_identity",
            "track_id",
            "confidence",
            "reason",
            "is_false_positive",
            "outsider_approved",
        ]
        st.dataframe(session_events[show_cols], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
