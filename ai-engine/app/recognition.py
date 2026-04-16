import argparse
import csv
import json
import smtplib
import ssl
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from time import sleep
from urllib import error as urlerror
from urllib import request as urlrequest

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from ultralytics import YOLO

from app.attention import AttentionThresholds, classify_attention, get_ear, get_head_pose
from app.config import DemoConfig
from app.face_registry import find_best_match, l2_normalize, load_registry
from app.preprocessing import compute_upscale_factor, enhance_for_cctv, enhance_roi
from app.sample_scene import SampleTrack, create_sample_frame


@dataclass
class TrackState:
    name: str | None = None
    unknown_streak: int = 0
    phone_streak: int = 0
    sleep_streak: int = 0
    attention_state: str = "attentive"
    last_seen: datetime = field(default_factory=datetime.utcnow)
    last_alert: datetime = field(default_factory=lambda: datetime.min)
    last_admin_alert: datetime = field(default_factory=lambda: datetime.min)
    last_recognition_log: datetime = field(default_factory=lambda: datetime.min)
    last_attention_log: datetime = field(default_factory=lambda: datetime.min)


def is_sample_mode(args: argparse.Namespace) -> bool:
    return args.sample_mode or args.source.strip().lower() == "sample"


def parse_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def build_face_analyzer(model_name: str, det_width: int, det_height: int) -> FaceAnalysis:
    face_analyzer = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
    face_analyzer.prepare(ctx_id=0, det_size=(det_width, det_height))
    return face_analyzer


def append_event(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "timestamp",
            "event_type",
            "track_id",
            "identity",
            "confidence",
            "attention_state",
            "reason",
            "source",
            "snapshot_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def detect_phone_boxes(
    detector: YOLO,
    frame: np.ndarray,
    phone_class_id: int,
    phone_conf: float,
) -> list[tuple[int, int, int, int]]:
    results = detector.predict(
        frame,
        classes=[phone_class_id],
        conf=phone_conf,
        verbose=False,
    )

    if not results:
        return []

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    parsed: list[tuple[int, int, int, int]] = []
    for xyxy in boxes.xyxy.cpu().numpy().tolist():
        x1, y1, x2, y2 = xyxy
        parsed.append((int(x1), int(y1), int(x2), int(y2)))
    return parsed


def phone_near_person(
    person_bbox: tuple[int, int, int, int],
    phone_boxes: list[tuple[int, int, int, int]],
    iou_threshold: float,
) -> bool:
    px1, py1, px2, py2 = person_bbox
    w = px2 - px1
    h = py2 - py1

    # Expand person bbox slightly to include hand area around torso.
    margin_x = int(w * 0.20)
    margin_y = int(h * 0.15)
    expanded = (px1 - margin_x, py1 - margin_y, px2 + margin_x, py2 + margin_y)

    ex1, ey1, ex2, ey2 = expanded
    for phone_box in phone_boxes:
        if iou(expanded, phone_box) >= iou_threshold:
            return True

        fx1, fy1, fx2, fy2 = phone_box
        cx = (fx1 + fx2) // 2
        cy = (fy1 + fy2) // 2
        if ex1 <= cx <= ex2 and ey1 <= cy <= ey2:
            return True

    return False


def save_admin_screenshot(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    alerts_dir: Path,
    source: str,
    track_id: int,
    student_name: str,
    reason: str,
) -> Path:
    alerts_dir.mkdir(parents=True, exist_ok=True)
    annotated = frame.copy()
    x1, y1, x2, y2 = bbox

    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 4)
    cv2.putText(
        annotated,
        f"ADMIN ALERT | {student_name} | {reason}",
        (max(8, x1), max(28, y1 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = student_name.replace(" ", "_")
    file_name = f"admin_{reason}_{safe_name}_{source}_track{track_id}_{timestamp}.jpg"
    file_path = alerts_dir / file_name
    cv2.imwrite(str(file_path), annotated)
    return file_path


def send_admin_webhook(
    webhook_url: str,
    payload: dict,
) -> tuple[bool, str]:
    try:
        body = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            webhook_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlrequest.urlopen(req, timeout=8) as response:
            status_code = getattr(response, "status", 200)
            if 200 <= status_code < 300:
                return True, f"webhook:{status_code}"
            return False, f"webhook_status:{status_code}"
    except (urlerror.URLError, TimeoutError, ValueError) as exc:
        return False, f"webhook_error:{exc}"


def send_admin_email(
    args: argparse.Namespace,
    subject: str,
    body: str,
    screenshot_path: Path,
) -> tuple[bool, str]:
    if not args.smtp_host or not args.smtp_from:
        return False, "smtp_not_configured"

    if not args.admin_email:
        return False, "admin_email_missing"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = args.smtp_from
    msg["To"] = args.admin_email
    msg.set_content(body)

    if screenshot_path.exists():
        with screenshot_path.open("rb") as handle:
            data = handle.read()
        msg.add_attachment(data, maintype="image", subtype="jpeg", filename=screenshot_path.name)

    try:
        if args.smtp_use_tls:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(args.smtp_host, args.smtp_port, context=context, timeout=10) as server:
                if args.smtp_user and args.smtp_password:
                    server.login(args.smtp_user, args.smtp_password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(args.smtp_host, args.smtp_port, timeout=10) as server:
                if args.smtp_user and args.smtp_password:
                    server.login(args.smtp_user, args.smtp_password)
                server.send_message(msg)
        return True, "email_sent"
    except (smtplib.SMTPException, OSError, TimeoutError, ValueError) as exc:
        return False, f"email_error:{exc}"


def notify_admin(
    args: argparse.Namespace,
    student_name: str,
    reason: str,
    confidence: float,
    screenshot_path: Path,
    source: str,
    track_id: int,
) -> None:
    timestamp = datetime.utcnow().isoformat()
    payload = {
        "timestamp": timestamp,
        "student": student_name,
        "reason": reason,
        "confidence": round(confidence, 4),
        "source": source,
        "track_id": track_id,
        "screenshot_path": str(screenshot_path),
    }

    notification_results: list[str] = []

    if args.admin_webhook_url:
        ok, msg = send_admin_webhook(args.admin_webhook_url, payload)
        notification_results.append(msg if ok else f"failed:{msg}")

    subject = f"Classroom Alert: {student_name} ({reason})"
    body = (
        f"Timestamp: {timestamp}\n"
        f"Student: {student_name}\n"
        f"Reason: {reason}\n"
        f"Confidence: {confidence:.4f}\n"
        f"Source: {source}\n"
        f"Track ID: {track_id}\n"
        f"Screenshot: {screenshot_path}"
    )
    if args.admin_email:
        ok, msg = send_admin_email(args, subject, body, screenshot_path)
        notification_results.append(msg if ok else f"failed:{msg}")

    # Always persist an admin notification audit line locally.
    args.events_csv.parent.mkdir(parents=True, exist_ok=True)
    audit_path = args.events_csv.parent / "admin_notifications.log"
    with audit_path.open("a", encoding="utf-8") as handle:
        status = ",".join(notification_results) if notification_results else "stored_local_only"
        handle.write(f"{timestamp} | {student_name} | {reason} | {status} | {screenshot_path}\n")


def save_alert_snapshot(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    alerts_dir: Path,
    source: str,
    track_id: int,
) -> Path:
    alerts_dir.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = bbox

    h, w = frame.shape[:2]
    margin_x = int((x2 - x1) * 0.2)
    margin_y = int((y2 - y1) * 0.2)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w - 1, x2 + margin_x)
    y2 = min(h - 1, y2 + margin_y)

    crop = frame[y1:y2, x1:x2]
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    file_name = f"alert_{source}_track{track_id}_{timestamp}.jpg"
    file_path = alerts_dir / file_name
    cv2.imwrite(str(file_path), crop)
    return file_path


def is_file_video_source(source: int | str) -> bool:
    if not isinstance(source, str):
        return False

    source_path = Path(source)
    if not source_path.exists() or not source_path.is_file():
        return False

    return source_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def resolve_output_video_path(source: int | str, output_video: Path | None) -> Path:
    if output_video is not None:
        resolved = output_video
    else:
        if isinstance(source, str):
            stem = Path(source).stem
        else:
            stem = f"camera_{source}"
        resolved = Path("outputs") / "video" / f"{stem}_annotated.mp4"

    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def create_video_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    for codec in ("mp4v", "avc1"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
        if writer.isOpened():
            return writer

    raise RuntimeError(f"Unable to create output video writer for: {path}")


def process_simulated_track(
    frame: np.ndarray,
    track: SampleTrack,
    states: dict[int, TrackState],
    now: datetime,
    args: argparse.Namespace,
    source_label: str,
) -> None:
    track_id = int(track["track_id"])
    x1, y1, x2, y2 = track["bbox"]
    identity = track["identity"]
    score = float(track["score"])
    att_state = track.get("attention_state", "attentive")

    state = states.setdefault(track_id, TrackState())
    state.last_seen = now
    state.attention_state = att_state

    if identity is None:
        state.unknown_streak += 1
        label = f"unknown ({score:.2f})"
        color = (0, 0, 255)

        should_alert = (
            state.unknown_streak >= args.unknown_frames
            and now - state.last_alert >= timedelta(seconds=args.alert_cooldown)
        )
        if should_alert:
            snapshot_path = save_alert_snapshot(
                frame,
                (x1, y1, x2, y2),
                args.alerts_dir,
                source_label,
                track_id,
            )
            append_event(
                args.events_csv,
                {
                    "timestamp": now.isoformat(),
                    "event_type": "outsider_candidate",
                    "track_id": track_id,
                    "identity": "unknown",
                    "confidence": f"{score:.4f}",
                    "attention_state": att_state,
                    "source": args.source,
                    "snapshot_path": str(snapshot_path),
                },
            )
            state.last_alert = now
    else:
        state.name = identity
        state.unknown_streak = 0
        label = f"{identity} ({score:.2f})"
        color = (0, 180, 0)

        if now - state.last_recognition_log >= timedelta(seconds=args.recognition_cooldown):
            append_event(
                args.events_csv,
                {
                    "timestamp": now.isoformat(),
                    "event_type": "recognized_student",
                    "track_id": track_id,
                    "identity": identity,
                    "confidence": f"{score:.4f}",
                    "attention_state": att_state,
                    "source": args.source,
                    "snapshot_path": "",
                },
            )
            state.last_recognition_log = now

        # Log attention changes in sample mode too
        if att_state != "attentive" and now - state.last_attention_log >= timedelta(seconds=getattr(args, 'attention_cooldown', 10)):
            append_event(
                args.events_csv,
                {
                    "timestamp": now.isoformat(),
                    "event_type": "attention_alert",
                    "track_id": track_id,
                    "identity": identity,
                    "confidence": f"{score:.4f}",
                    "attention_state": att_state,
                    "source": args.source,
                    "snapshot_path": "",
                },
            )
            state.last_attention_log = now

    att_color = (180, 50, 255) if att_state == "sleeping" else (0, 165, 255) if att_state == "distracted" else color
    cv2.rectangle(frame, (x1, y1), (x2, y2), att_color, 2)
    display_label = f"ID {track_id} | {label} [{att_state}]"
    cv2.putText(
        frame,
        display_label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        att_color,
        2,
        cv2.LINE_AA,
    )


def run_sample(args: argparse.Namespace) -> None:
    states: dict[int, TrackState] = {}
    frame_index = 0
    source_label = "sample"

    preview_delay_ms = max(1, int(1000 / max(1, args.sample_fps)))
    should_throttle = bool(args.no_display)

    print("Sample mode started. Press 'q' in the preview window to stop.")

    while True:
        frame, tracks = create_sample_frame(
            frame_index,
            width=args.sample_width,
            height=args.sample_height,
        )
        now = datetime.utcnow()

        for track in tracks:
            process_simulated_track(frame, track, states, now, args, source_label)

        stale_ids = [
            track_id
            for track_id, state in states.items()
            if now - state.last_seen > timedelta(seconds=20)
        ]
        for track_id in stale_ids:
            states.pop(track_id, None)

        if not args.no_display:
            cv2.imshow("Classroom Recognition Demo (Sample Mode)", frame)
            if cv2.waitKey(preview_delay_ms) & 0xFF == ord("q"):
                break
        elif should_throttle:
            sleep(1.0 / max(1, args.sample_fps))

        frame_index += 1
        if args.max_frames > 0 and frame_index >= args.max_frames:
            break

    cv2.destroyAllWindows()
    print("Sample mode stopped.")


def _build_attention_thresholds(args: argparse.Namespace) -> AttentionThresholds:
    return AttentionThresholds(
        yaw_limit=args.yaw_limit,
        pitch_down_limit=args.pitch_down_limit,
        pitch_up_limit=args.pitch_up_limit,
        ear_closed=args.ear_closed,
        sleep_streak_threshold=args.sleep_streak,
    )


def _attention_color(state_name: str) -> tuple[int, int, int]:
    if state_name == "sleeping":
        return (180, 50, 255)   # purple
    if state_name == "distracted":
        return (0, 165, 255)    # orange
    return (0, 180, 0)          # green


def run_real(args: argparse.Namespace) -> None:
    registry = load_registry(args.registry)
    detector = YOLO(args.person_model)
    face_analyzer = build_face_analyzer(args.face_model, args.det_width, args.det_height)
    att_thresholds = _build_attention_thresholds(args)

    source = parse_source(args.source)
    source_is_file_video = is_file_video_source(source)

    if args.offline and not source_is_file_video:
        raise ValueError("Offline mode requires a valid video file source (e.g., .mp4)")

    if args.offline:
        args.no_display = True

    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {args.source}")

    # Auto-compute upscale for low-res feeds
    feed_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    upscale = compute_upscale_factor(feed_w, target_min_width=960)
    input_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if input_fps <= 0:
        input_fps = 20.0

    video_writer: cv2.VideoWriter | None = None
    output_video_path: Path | None = None
    latest_phone_boxes: list[tuple[int, int, int, int]] = []

    states: dict[int, TrackState] = {}
    frame_index = 0
    source_label = str(args.source).replace("/", "_").replace("\\", "_")

    print("Pipeline started. Press 'q' in the preview window to stop.")

    while True:
        ok, raw_frame = capture.read()
        if not ok:
            break

        frame_index += 1
        if args.frame_skip > 1 and frame_index % args.frame_skip != 0:
            continue

        # CCTV enhancement: lighting normalization, denoise, sharpen
        frame = enhance_for_cctv(raw_frame, upscale_factor=upscale)

        if args.enable_phone_detection and frame_index % max(1, args.phone_check_interval) == 0:
            latest_phone_boxes = detect_phone_boxes(
                detector=detector,
                frame=frame,
                phone_class_id=args.phone_class_id,
                phone_conf=args.phone_conf,
            )

        results = detector.track(
            frame,
            persist=True,
            classes=[0],
            tracker="bytetrack.yaml",
            verbose=False,
            conf=args.person_conf,
        )

        now = datetime.utcnow()

        if results:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and boxes.id is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy().astype(int)
                ids = boxes.id.int().cpu().tolist()

                for bbox, track_id in zip(xyxy, ids):
                    x1, y1, x2, y2 = bbox.tolist()
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1] - 1, x2)
                    y2 = min(frame.shape[0] - 1, y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    state = states.setdefault(track_id, TrackState())
                    state.last_seen = now

                    roi = enhance_roi(frame[y1:y2, x1:x2])
                    faces = face_analyzer.get(roi)

                    label = "no-face"
                    color = (255, 255, 0)
                    att_label = ""

                    if faces:
                        face = max(
                            faces,
                            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                        )
                        embedding = l2_normalize(face.embedding)
                        name, score = find_best_match(
                            embedding,
                            registry,
                            threshold=args.match_threshold,
                        )

                        # --- attention classification ---
                        pitch, yaw, _roll = get_head_pose(face)
                        ear = get_ear(face)
                        att_state, state.sleep_streak = classify_attention(
                            pitch=pitch,
                            yaw=yaw,
                            ear=ear,
                            sleep_streak=state.sleep_streak,
                            thresholds=att_thresholds,
                        )
                        state.attention_state = att_state
                        att_label = att_state

                        if name is None:
                            state.unknown_streak += 1
                            label = f"unknown ({score:.2f})"
                            color = (0, 0, 255)

                            should_alert = (
                                state.unknown_streak >= args.unknown_frames
                                and now - state.last_alert >= timedelta(seconds=args.alert_cooldown)
                            )
                            if should_alert:
                                snapshot_path = save_alert_snapshot(
                                    frame,
                                    (x1, y1, x2, y2),
                                    args.alerts_dir,
                                    source_label,
                                    track_id,
                                )
                                append_event(
                                    args.events_csv,
                                    {
                                        "timestamp": now.isoformat(),
                                        "event_type": "outsider_candidate",
                                        "track_id": track_id,
                                        "identity": "unknown",
                                        "confidence": f"{score:.4f}",
                                        "attention_state": att_state,
                                        "reason": "outsider",
                                        "source": args.source,
                                        "snapshot_path": str(snapshot_path),
                                    },
                                )
                                state.last_alert = now
                        else:
                            state.name = name
                            state.unknown_streak = 0
                            label = f"{name} ({score:.2f})"
                            color = (0, 180, 0)

                            if now - state.last_recognition_log >= timedelta(seconds=args.recognition_cooldown):
                                append_event(
                                    args.events_csv,
                                    {
                                        "timestamp": now.isoformat(),
                                        "event_type": "recognized_student",
                                        "track_id": track_id,
                                        "identity": name,
                                        "confidence": f"{score:.4f}",
                                        "attention_state": att_state,
                                        "reason": "",
                                        "source": args.source,
                                        "snapshot_path": "",
                                    },
                                )
                                state.last_recognition_log = now

                            # Log attention changes
                            if att_state != "attentive" and now - state.last_attention_log >= timedelta(seconds=args.attention_cooldown):
                                append_event(
                                    args.events_csv,
                                    {
                                        "timestamp": now.isoformat(),
                                        "event_type": "attention_alert",
                                        "track_id": track_id,
                                        "identity": name,
                                        "confidence": f"{score:.4f}",
                                        "attention_state": att_state,
                                        "reason": att_state,
                                        "source": args.source,
                                        "snapshot_path": "",
                                    },
                                )
                                state.last_attention_log = now

                            # Phone-use detection (COCO class 67) near a recognized student.
                            person_bbox = (x1, y1, x2, y2)
                            phone_detected = (
                                args.enable_phone_detection
                                and phone_near_person(
                                    person_bbox=person_bbox,
                                    phone_boxes=latest_phone_boxes,
                                    iou_threshold=args.phone_iou_threshold,
                                )
                            )
                            if phone_detected:
                                state.phone_streak += 1
                            else:
                                state.phone_streak = 0

                            admin_reason = ""
                            if att_state == "sleeping" and state.sleep_streak >= args.sleep_streak:
                                admin_reason = "sleeping"
                            elif state.phone_streak >= args.phone_streak_threshold:
                                admin_reason = "phone_use"

                            if (
                                admin_reason
                                and now - state.last_admin_alert >= timedelta(seconds=args.admin_alert_cooldown)
                            ):
                                admin_shot_path = save_admin_screenshot(
                                    frame=frame,
                                    bbox=person_bbox,
                                    alerts_dir=args.alerts_dir,
                                    source=source_label,
                                    track_id=track_id,
                                    student_name=name,
                                    reason=admin_reason,
                                )

                                append_event(
                                    args.events_csv,
                                    {
                                        "timestamp": now.isoformat(),
                                        "event_type": "admin_alert",
                                        "track_id": track_id,
                                        "identity": name,
                                        "confidence": f"{score:.4f}",
                                        "attention_state": att_state,
                                        "reason": admin_reason,
                                        "source": args.source,
                                        "snapshot_path": str(admin_shot_path),
                                    },
                                )

                                notify_admin(
                                    args=args,
                                    student_name=name,
                                    reason=admin_reason,
                                    confidence=score,
                                    screenshot_path=admin_shot_path,
                                    source=args.source,
                                    track_id=track_id,
                                )
                                state.last_admin_alert = now

                    # Draw bounding box with attention-aware color
                    draw_color = _attention_color(state.attention_state) if att_label else color
                    cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
                    display_label = f"ID {track_id} | {label}"
                    if att_label:
                        display_label += f" [{att_label}]"
                    cv2.putText(
                        frame,
                        display_label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        draw_color,
                        2,
                        cv2.LINE_AA,
                    )

        stale_ids = [
            track_id
            for track_id, state in states.items()
            if now - state.last_seen > timedelta(seconds=30)
        ]
        for track_id in stale_ids:
            states.pop(track_id, None)

        if args.save_output or args.offline:
            if video_writer is None:
                output_video_path = resolve_output_video_path(source, args.output_video)
                frame_h, frame_w = frame.shape[:2]
                target_fps = args.output_fps if args.output_fps > 0 else input_fps
                video_writer = create_video_writer(
                    path=output_video_path,
                    fps=max(1.0, target_fps),
                    width=frame_w,
                    height=frame_h,
                )
            assert video_writer is not None
            video_writer.write(frame)

        if not args.no_display:
            cv2.imshow("Classroom Recognition Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if args.max_frames > 0 and frame_index >= args.max_frames:
            break

    capture.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    print("Pipeline stopped.")
    if output_video_path is not None:
        print(f"Annotated output saved: {output_video_path}")


def parse_args() -> argparse.Namespace:
    defaults = DemoConfig()

    parser = argparse.ArgumentParser(description="Run live recognition and outsider alerts.")
    parser.add_argument("--source", default="0", help="Camera index like 0 or video path")
    parser.add_argument("--sample-mode", action="store_true", help="Run synthetic no-camera demo mode")
    parser.add_argument("--offline", action="store_true", help="Process a video file offline (no preview window)")
    parser.add_argument("--registry", type=Path, default=defaults.registry_file)
    parser.add_argument("--events-csv", type=Path, default=defaults.events_csv)
    parser.add_argument("--alerts-dir", type=Path, default=defaults.alerts_dir)
    parser.add_argument("--save-output", action="store_true", help="Save annotated output video")
    parser.add_argument("--output-video", type=Path, default=None, help="Output path for annotated video")
    parser.add_argument("--output-fps", type=float, default=0.0, help="Output FPS; 0 uses source FPS")

    parser.add_argument("--person-model", default="yolov8n.pt")
    parser.add_argument("--face-model", default="buffalo_l")
    parser.add_argument("--det-width", type=int, default=640)
    parser.add_argument("--det-height", type=int, default=640)

    parser.add_argument("--person-conf", type=float, default=0.35)
    parser.add_argument("--match-threshold", type=float, default=0.45)
    parser.add_argument("--unknown-frames", type=int, default=12)
    parser.add_argument("--alert-cooldown", type=int, default=20)
    parser.add_argument("--admin-alert-cooldown", type=int, default=30)
    parser.add_argument("--recognition-cooldown", type=int, default=8)
    parser.add_argument("--attention-cooldown", type=int, default=10)

    # Phone-use detection
    parser.add_argument("--enable-phone-detection", action="store_true", help="Enable phone-use detection")
    parser.add_argument("--disable-phone-detection", action="store_true", help="Disable phone-use detection")
    parser.add_argument("--phone-class-id", type=int, default=67, help="COCO class id for cell phone")
    parser.add_argument("--phone-conf", type=float, default=0.25)
    parser.add_argument("--phone-iou-threshold", type=float, default=0.08)
    parser.add_argument("--phone-check-interval", type=int, default=3)
    parser.add_argument("--phone-streak-threshold", type=int, default=4)

    # Admin delivery channels
    parser.add_argument("--admin-webhook-url", default="", help="Webhook URL for admin notifications")
    parser.add_argument("--admin-email", default="", help="Admin email for SMTP notifications")
    parser.add_argument("--smtp-host", default="", help="SMTP host")
    parser.add_argument("--smtp-port", type=int, default=465, help="SMTP port")
    parser.add_argument("--smtp-user", default="", help="SMTP username")
    parser.add_argument("--smtp-password", default="", help="SMTP password")
    parser.add_argument("--smtp-from", default="", help="SMTP sender email")
    parser.add_argument("--smtp-use-tls", action="store_true", help="Use SMTPS/TLS (port 465)")

    parser.set_defaults(enable_phone_detection=True, smtp_use_tls=True)

    # Attention thresholds
    parser.add_argument("--yaw-limit", type=float, default=35.0)
    parser.add_argument("--pitch-down-limit", type=float, default=-25.0)
    parser.add_argument("--pitch-up-limit", type=float, default=30.0)
    parser.add_argument("--ear-closed", type=float, default=0.18)
    parser.add_argument("--sleep-streak", type=int, default=8)

    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--sample-width", type=int, default=960)
    parser.add_argument("--sample-height", type=int, default=540)
    parser.add_argument("--sample-fps", type=int, default=20)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.disable_phone_detection:
        args.enable_phone_detection = False
    if is_sample_mode(args):
        run_sample(args)
    else:
        run_real(args)


if __name__ == "__main__":
    main()
