from pathlib import Path

import pandas as pd
import streamlit as st

from app.config import DemoConfig


def load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "timestamp",
                "event_type",
                "track_id",
                "identity",
                "confidence",
                "attention_state",
                "source",
                "snapshot_path",
            ]
        )
    df = pd.read_csv(path)
    # Backfill for older logs without attention_state column
    if "attention_state" not in df.columns:
        df["attention_state"] = ""
    return df


def show_metrics(events: pd.DataFrame) -> None:
    total = len(events)
    recognized = int((events["event_type"] == "recognized_student").sum())
    outsider = int((events["event_type"] == "outsider_candidate").sum())
    attention_alerts = int((events["event_type"] == "attention_alert").sum())

    recognized_names = events.loc[
        events["event_type"] == "recognized_student", "identity"
    ].nunique()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Events", total)
    c2.metric("Recognized", recognized)
    c3.metric("Outsider Alerts", outsider)
    c4.metric("Attention Alerts", attention_alerts)
    c5.metric("Unique Students", int(recognized_names))


def show_latest_alerts(events: pd.DataFrame, max_images: int = 6) -> None:
    alert_events = events[events["event_type"] == "outsider_candidate"].copy()
    if alert_events.empty:
        st.info("No outsider candidate alerts yet.")
        return

    alert_events = alert_events.sort_values("timestamp", ascending=False).head(max_images)

    st.subheader("Latest Outsider Snapshots")
    image_columns = st.columns(3)

    for index, (_, row) in enumerate(alert_events.iterrows()):
        image_path = Path(str(row.get("snapshot_path", "")))
        with image_columns[index % 3]:
            if image_path.exists():
                st.image(str(image_path), caption=f"{row['timestamp']} | Track {row['track_id']}")
            else:
                st.warning(f"Missing snapshot: {image_path}")


def main() -> None:
    defaults = DemoConfig()

    st.set_page_config(page_title="Classroom CCTV Demo", layout="wide")
    st.title("Classroom Student Recognition Demo")
    st.caption("Attendance and outsider-candidate events from the recognition pipeline")

    with st.sidebar:
        st.header("Data Sources")
        events_csv = Path(
            st.text_input("Events CSV", value=str(defaults.events_csv))
        )
        _alerts_dir = Path(
            st.text_input("Alerts Directory", value=str(defaults.alerts_dir))
        )
        st.caption("Alerts directory is optional in dashboard; events CSV drives the tables.")
        if st.button("Refresh Now"):
            st.rerun()

    events = load_events(events_csv)

    if events.empty:
        st.warning("No events found yet. Start recognition first to populate logs.")
        return

    if "timestamp" in events.columns:
        events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce")
        events = events.sort_values("timestamp", ascending=False)

    show_metrics(events)

    st.subheader("Recent Events")
    event_filter = st.selectbox(
        "Filter by event type",
        options=["all", "recognized_student", "outsider_candidate", "attention_alert"],
        index=0,
    )

    filtered = events if event_filter == "all" else events[events["event_type"] == event_filter]
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.subheader("Attendance Snapshot")
    recognized_events = events[events["event_type"] == "recognized_student"]
    if recognized_events.empty:
        st.info("No recognized students yet.")
    else:
        attendance = (
            recognized_events.groupby("identity", as_index=False)
            .agg(
                first_seen=("timestamp", "min"),
                last_seen=("timestamp", "max"),
                sightings=("identity", "count"),
                max_confidence=("confidence", "max"),
            )
            .sort_values("last_seen", ascending=False)
        )
        st.dataframe(attendance, use_container_width=True, hide_index=True)

    # --- Attention Breakdown ---
    st.subheader("Attention Summary")
    att_events = events[events["attention_state"].isin(["attentive", "distracted", "sleeping"])]
    if att_events.empty:
        st.info("No attention data recorded yet.")
    else:
        att_by_student = (
            att_events[att_events["identity"] != "unknown"]
            .groupby(["identity", "attention_state"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        if not att_by_student.empty:
            pivot = att_by_student.pivot_table(
                index="identity",
                columns="attention_state",
                values="count",
                fill_value=0,
            )
            for col in ["attentive", "distracted", "sleeping"]:
                if col not in pivot.columns:
                    pivot[col] = 0
            pivot = pivot[["attentive", "distracted", "sleeping"]]
            st.dataframe(pivot, use_container_width=True)
            st.bar_chart(pivot)

        sleeping_events = att_events[att_events["attention_state"] == "sleeping"]
        if not sleeping_events.empty:
            st.warning(
                f"{sleeping_events['identity'].nunique()} student(s) detected sleeping "
                f"({len(sleeping_events)} events)"
            )

    show_latest_alerts(events)


if __name__ == "__main__":
    main()
