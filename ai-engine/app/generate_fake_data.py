import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

import cv2

from app.config import DemoConfig
from app.sample_scene import create_sample_frame


def parse_args() -> argparse.Namespace:
    defaults = DemoConfig()

    parser = argparse.ArgumentParser(description="Generate fake logs and alert snapshots for dashboard rehearsal.")
    parser.add_argument("--events-csv", type=Path, default=defaults.events_csv)
    parser.add_argument("--alerts-dir", type=Path, default=defaults.alerts_dir)
    parser.add_argument("--rows", type=int, default=45)
    parser.add_argument("--source", default="sample")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_csv_header_if_needed(path: Path) -> None:
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "event_type",
                "track_id",
                "identity",
                "confidence",
                "attention_state",
                "source",
                "snapshot_path",
            ],
        )
        writer.writeheader()


def append_row(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "event_type",
                "track_id",
                "identity",
                "confidence",
                "attention_state",
                "source",
                "snapshot_path",
            ],
        )
        writer.writerow(row)


def save_unknown_snapshot(alerts_dir: Path, frame_index: int) -> str:
    frame, tracks = create_sample_frame(frame_index)
    outsider = next((track for track in tracks if track["identity"] is None), None)

    if outsider is None:
        return ""

    x1, y1, x2, y2 = outsider["bbox"]
    crop = frame[y1:y2, x1:x2]
    alerts_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    path = alerts_dir / f"fake_alert_{timestamp}_{frame_index}.jpg"
    cv2.imwrite(str(path), crop)
    return str(path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.overwrite and args.events_csv.exists():
        args.events_csv.unlink()

    write_csv_header_if_needed(args.events_csv)

    student_pool = ["student_001", "student_002", "student_003", "student_004"]
    attention_states = ["attentive", "attentive", "attentive", "distracted", "sleeping"]
    now = datetime.utcnow() - timedelta(minutes=20)

    for index in range(max(1, args.rows)):
        now += timedelta(seconds=random.randint(5, 25))

        is_outsider = random.random() < 0.18
        is_attention_alert = (not is_outsider) and random.random() < 0.20

        if is_outsider:
            snapshot = save_unknown_snapshot(args.alerts_dir, frame_index=index * 11)
            row = {
                "timestamp": now.isoformat(),
                "event_type": "outsider_candidate",
                "track_id": 99,
                "identity": "unknown",
                "confidence": f"{random.uniform(0.20, 0.44):.4f}",
                "attention_state": "distracted",
                "source": args.source,
                "snapshot_path": snapshot,
            }
        elif is_attention_alert:
            student = random.choice(student_pool)
            att = random.choice(["distracted", "sleeping"])
            row = {
                "timestamp": now.isoformat(),
                "event_type": "attention_alert",
                "track_id": random.randint(1, 4),
                "identity": student,
                "confidence": f"{random.uniform(0.62, 0.98):.4f}",
                "attention_state": att,
                "source": args.source,
                "snapshot_path": "",
            }
        else:
            student = random.choice(student_pool)
            att = random.choice(attention_states)
            row = {
                "timestamp": now.isoformat(),
                "event_type": "recognized_student",
                "track_id": random.randint(1, 4),
                "identity": student,
                "confidence": f"{random.uniform(0.62, 0.98):.4f}",
                "attention_state": att,
                "source": args.source,
                "snapshot_path": "",
            }

        append_row(args.events_csv, row)

    print(f"Fake data generated: {args.events_csv}")
    print(f"Alert snapshots directory: {args.alerts_dir}")


if __name__ == "__main__":
    main()
