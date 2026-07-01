from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .attendance import AttendanceManager
from .config import AppConfig
from .face_database import FaceDatabase
from .reporting import Reporter
from .tracking import ByteTrackTracker
from .types import PersonRecord
from .video import VideoInput

ScoredDet = Tuple[Tuple[int, int, int, int], float]

logger = logging.getLogger(__name__)

_YUNET_MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"


class ClassroomMonitorApp:
    def __init__(self, config: AppConfig) -> None:
        self.cfg = config

        self.face_db = FaceDatabase(
            threshold=self.cfg.detection.recognition_threshold,
            margin=self.cfg.detection.recognition_margin,
        )
        self.attendance = AttendanceManager(min_confirm_frames=self.cfg.attendance.min_confirm_frames)
        self.tracker = ByteTrackTracker(
            high_score=max(0.6, self.cfg.detection.min_detection_confidence + 0.1),
            max_age_seconds=self.cfg.runtime.track_max_age_seconds,
        )
        self.video = VideoInput(self.cfg.source)
        self.reporter = Reporter(
            output_dir=self.cfg.paths.output_dir,
            snapshots_dir=self.cfg.paths.snapshots_dir,
            session=self.cfg.session,
            admin_cfg=self.cfg.admin_email,
        )

        self._known_students: List[PersonRecord] = []
        self._known_teachers: List[PersonRecord] = []

    @staticmethod
    def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(ax, bx); iy = max(ay, by)
        ix2 = min(ax + aw, bx + bw); iy2 = min(ay + ah, by + bh)
        iw = max(0, ix2 - ix); ih = max(0, iy2 - iy)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _nms_boxes(
        boxes: List[ScoredDet], iou_threshold: float = 0.35
    ) -> List[ScoredDet]:
        if not boxes:
            return []
        # Sort by detection score so the strongest detection wins each overlap.
        sorted_boxes = sorted(boxes, key=lambda b: b[1], reverse=True)
        kept: List[ScoredDet] = []
        for box, score in sorted_boxes:
            x1, y1, w1, h1 = box
            suppressed = False
            for (kx, ky, kw, kh), _ks in kept:
                ix = max(x1, kx); iy = max(y1, ky)
                ix2 = min(x1 + w1, kx + kw); iy2 = min(y1 + h1, ky + kh)
                if ix2 <= ix or iy2 <= iy:
                    continue
                inter = (ix2 - ix) * (iy2 - iy)
                union = w1 * h1 + kw * kh - inter
                if union > 0 and inter / union > iou_threshold:
                    suppressed = True
                    break
            if not suppressed:
                kept.append((box, score))
        return kept

    @staticmethod
    def _yunet_detect_scaled(
        detector,
        bgr: np.ndarray,
        det_w: int,
        frame_w: int,
        frame_h: int,
        min_px: int = 30,
    ) -> List[ScoredDet]:
        """Run YuNet on a frame resized to det_w wide and scale boxes back to full res.

        Returns ((x, y, w, h), score) — the YuNet confidence is kept so the tracker
        can do ByteTrack-style two-stage association.
        """
        scale = det_w / frame_w
        det_h = int(frame_h * scale)
        small = cv2.resize(bgr, (det_w, det_h), interpolation=cv2.INTER_AREA)
        detector.setInputSize((det_w, det_h))
        _, faces = detector.detect(small)
        if faces is None:
            return []
        inv = 1.0 / scale
        boxes: List[ScoredDet] = []
        for face in faces:
            x = int(face[0] * inv); y = int(face[1] * inv)
            w = int(face[2] * inv); h = int(face[3] * inv)
            x = max(0, x); y = max(0, y)
            w = min(w, frame_w - x); h = min(h, frame_h - y)
            ar = w / h if h > 0 else 0
            score = float(face[-1])
            if w >= min_px and h >= min_px and 0.4 <= ar <= 2.5:
                boxes.append(((x, y, w, h), score))
        return boxes

    @staticmethod
    def _draw_box(
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        label: str,
        color: Tuple[int, int, int],
        dimmed: bool = False,
    ) -> None:
        """Draw a corner-bracket box with a filled, readable label chip.

        `dimmed` is used for last-known-position boxes (person not detected this
        frame): thinner brackets, no chip background, so they read as "stale".
        """
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return
        x2, y2 = x + w, y + h
        # Bracket length and thickness scale with box size for consistent weight.
        ln = max(8, int(min(w, h) * 0.22))
        thick = 1 if dimmed else max(2, int(round(min(w, h) / 80)) + 1)

        def corner(p1, p2, p3):
            cv2.line(frame, p1, p2, color, thick, cv2.LINE_AA)
            cv2.line(frame, p1, p3, color, thick, cv2.LINE_AA)

        corner((x, y), (x + ln, y), (x, y + ln))            # top-left
        corner((x2, y), (x2 - ln, y), (x2, y + ln))         # top-right
        corner((x, y2), (x + ln, y2), (x, y2 - ln))         # bottom-left
        corner((x2, y2), (x2 - ln, y2), (x2, y2 - ln))      # bottom-right

        if not label:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        fscale = 0.5
        fthick = 1
        (tw, th), base = cv2.getTextSize(label, font, fscale, fthick)
        pad = 4
        chip_w = tw + 2 * pad
        chip_h = th + base + 2 * pad
        cx = x
        cy = y - chip_h - 2
        if cy < 0:  # not enough room above — drop chip just inside the top edge
            cy = y + 2
        if dimmed:
            # No filled chip for stale boxes — just text, so they recede visually.
            cv2.putText(frame, label, (cx + pad, cy + th + pad), font, fscale, color, fthick, cv2.LINE_AA)
            return
        # Filled chip background for readability on busy/dark frames.
        cv2.rectangle(frame, (cx, cy), (cx + chip_w, cy + chip_h), color, -1)
        cv2.putText(frame, label, (cx + pad, cy + th + pad), font, fscale, (20, 20, 20), fthick, cv2.LINE_AA)

    @staticmethod
    def _draw_recognized_panel(
        frame: np.ndarray,
        seen_people: dict,
        active_ids: set,
    ) -> None:
        """Draw a sidebar panel listing all recognized students."""
        if not seen_people:
            return
        h_frame, w_frame = frame.shape[:2]
        row_h = 24
        header_h = 26
        panel_w = 280
        panel_h = len(seen_people) * row_h + header_h + 12
        px = w_frame - panel_w - 10
        py = h_frame - panel_h - 10  # bottom-right corner
        # semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (px - 4, py - 4), (px + panel_w, py + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        font = cv2.FONT_HERSHEY_SIMPLEX
        active_count = sum(1 for pid in seen_people if pid in active_ids)
        cv2.putText(frame, f"Recognized  {active_count}/{len(seen_people)} active",
                    (px + 8, py + 17), font, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.line(frame, (px, py + header_h - 2), (px + panel_w - 6, py + header_h - 2),
                 (90, 90, 90), 1, cv2.LINE_AA)
        for i, (pid, (_bbox, label, p_color, _ts)) in enumerate(seen_people.items()):
            y_pos = py + header_h + 6 + i * row_h
            is_active = pid in active_ids
            dot_color = p_color if is_active else (140, 140, 140)
            # dot indicator: filled = active, hollow = last seen
            if is_active:
                cv2.circle(frame, (px + 8, y_pos + 7), 5, dot_color, -1)
            else:
                cv2.circle(frame, (px + 8, y_pos + 7), 5, dot_color, 1)
            # show only the name part (strip score/state clutter)
            name = label.split("|")[0].split("(")[0].strip()
            text_color = (245, 245, 245) if is_active else (150, 150, 150)
            cv2.putText(frame, name, (px + 20, y_pos + 12),
                        font, 0.48, text_color, 1, cv2.LINE_AA)

    def _load_roster(self) -> None:
        if self.cfg.runtime.app_mode != "classroom":
            roster_dir = self.cfg.paths.roster_dir
            stats = self.face_db.load_people_dir(roster_dir)
            logger.info("Roster loaded | people=%d", stats['people'])
            if not self.face_db.has_people():
                logger.warning("No roster faces loaded — all faces will show as Unknown.")
            return

        stats = self.face_db.load(
            students_dir=self.cfg.paths.roster_students_dir,
            teachers_dir=self.cfg.paths.roster_teachers_dir,
        )
        self._known_students = self.face_db.list_people("student")
        self._known_teachers = self.face_db.list_people("teacher")
        logger.info(
            "Roster loaded | students=%d teachers=%d people=%d",
            stats['students'], stats['teachers'], stats['people'],
        )
        if not self.face_db.has_people():
            logger.warning("No roster faces loaded — attendance won't work.")

    def run(self) -> None:
        self._load_roster()
        self.video.open()

        frame_count = 0
        start_ts = time.time()
        last_process_ts = 0.0
        _fps_report_ts = start_ts  # for periodic FPS logging
        last_recognize_ts: float = -999.0  # controls the recognition interval
        # Tuning knobs (promoted to config; values below are the runtime copies).
        RECOGNIZE_INTERVAL = self.cfg.runtime.recognize_interval_seconds
        last_tiled_det_ts: float = -999.0   # controls tiled detection interval
        TILED_DET_INTERVAL = self.cfg.runtime.tiled_det_interval_seconds
        _cached_boxes: List[ScoredDet] = []  # last tiled result
        # Persistent overlay: person_id -> (bbox, label, color, last_seen_ts)
        # Once a student is recognized we keep drawing their box every frame,
        # dimmed when not actively detected in the current frame.
        _seen_people: dict = {}
        # Option B: per-track vote buffer — identity locked only after VOTE_NEEDED agreements
        _vote_buffer: Dict[int, Dict[str, int]] = {}
        VOTE_NEEDED = self.cfg.detection.votes_needed
        # A returning known face matched via the re-ID gallery locks faster than a
        # cold enrollment match (it was already confirmed moments ago).
        REID_VOTES_NEEDED = max(1, min(3, VOTE_NEEDED))
        # Option C: distance must be this good (cosine similarity) to count toward attendance
        ATTEND_MIN_SCORE = self.cfg.detection.attend_min_score
        # Stale box TTL: stop drawing a last-known-position box after this many seconds
        STALE_BOX_TTL = self.cfg.runtime.stale_box_ttl_seconds
        # Per-track recognition cooldown for unconfirmed tracks (avoids running ResNet
        # every frame on faces that haven't been matched yet)
        _track_recog_ts: Dict[int, float] = {}
        UNCONFIRMED_RECOG_INTERVAL = self.cfg.runtime.unconfirmed_recog_interval_seconds

        writer = None
        if self.cfg.runtime.save_annotated_video:
            width, height = self.video.frame_size()
            fps = self.video.fps() or 15.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = str(Path(self.cfg.paths.output_dir) / "annotated_output.mp4")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        face_detector = cv2.FaceDetectorYN.create(
            _YUNET_MODEL_PATH,
            "",
            (640, 640),
            score_threshold=self.cfg.detection.min_detection_confidence,
            nms_threshold=0.3,
            top_k=max(self.cfg.detection.max_faces, 50),
        )

        while True:
            ok, frame = self.video.read()
            if not ok or frame is None:
                # File mode: False means EOF — exit the loop.
                # Live/webcam mode: False means no new frame yet (buffer
                # was already consumed or capture thread is reconnecting).
                # Sleep briefly and retry instead of exiting.
                if self.cfg.source.mode != "file":
                    time.sleep(0.01)
                    continue
                break

            now_ts = time.time()
            if self.cfg.runtime.max_fps > 0:
                min_gap = 1.0 / self.cfg.runtime.max_fps
                if now_ts - last_process_ts < min_gap:
                    continue
                last_process_ts = now_ts

            h_frame, w_frame = frame.shape[:2]
            # RGB conversion is deferred: only computed when recognition actually runs.
            _rgb: np.ndarray | None = None

            # Check if it is time to run the expensive recognition pipeline.
            run_recognition = (now_ts - last_recognize_ts >= RECOGNIZE_INTERVAL)
            if run_recognition:
                last_recognize_ts = now_ts

            # --- Face detection: two-tier strategy (mirrors the old MediaPipe approach) ---
            # Fast path (every frame): single 640px-wide pass — keeps tracker alive.
            # Slow path (every 2s): 2×2 tiled pass at 960px per tile — finds small/
            #   angled faces that the single-pass misses at the back of the room.
            _DET_W_FAST = 640
            _DET_W_TILED = 960

            single_boxes = self._yunet_detect_scaled(
                face_detector, frame, _DET_W_FAST, w_frame, h_frame
            )

            if now_ts - last_tiled_det_ts >= TILED_DET_INTERVAL:
                last_tiled_det_ts = now_ts
                tiled: List[ScoredDet] = []
                tile_w = int(w_frame * 0.60)
                tile_h = int(h_frame * 0.60)
                stride_x = w_frame - tile_w
                stride_y = h_frame - tile_h
                for row in range(2):
                    for col in range(2):
                        ox = col * stride_x
                        oy = row * stride_y
                        x2 = min(w_frame, ox + tile_w)
                        y2 = min(h_frame, oy + tile_h)
                        tile_bgr = frame[oy:y2, ox:x2]
                        tw, th = tile_bgr.shape[1], tile_bgr.shape[0]
                        if tw < 20 or th < 20:
                            continue
                        tile_boxes = self._yunet_detect_scaled(
                            face_detector, tile_bgr, _DET_W_TILED, tw, th
                        )
                        for (bx, by, bw, bh), sc in tile_boxes:
                            tiled.append(((ox + bx, oy + by, bw, bh), sc))
                _cached_boxes = self._nms_boxes(tiled)

            boxes = self._nms_boxes(single_boxes + _cached_boxes)

            # --- track faces (always runs to keep bboxes accurate) ---
            tracks = self.tracker.update(boxes, now_ts)

            for track in tracks:
                x, y, w, h = track.bbox
                if w < 30 or h < 30:
                    continue

                _role = "student" if self.cfg.runtime.app_mode == "classroom" else "person"

                # --- Recognition: only when interval elapsed OR track not yet locked ---
                unconfirmed_due = (
                    not track.person_id
                    and now_ts - _track_recog_ts.get(track.track_id, 0.0) >= UNCONFIRMED_RECOG_INTERVAL
                )
                if run_recognition or unconfirmed_due:
                    _track_recog_ts[track.track_id] = now_ts
                    # Pad the YuNet box slightly for better ResNet accuracy.
                    _PAD = max(20, int(min(w, h) * 0.15))
                    x1 = max(0, x - _PAD)
                    y1 = max(0, y - _PAD)
                    x2 = min(w_frame, x + w + _PAD)
                    y2 = min(h_frame, y + h + _PAD)
                    # Lazy RGB conversion — only on the first track that needs recognition
                    if _rgb is None:
                        _rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_rgb = _rgb[y1:y2, x1:x2]
                    if face_rgb.size == 0:
                        continue

                    # Skip MTCNN re-detection; YuNet already found this face.
                    _emb = self.face_db.encode_crop(face_rgb)
                    match_person, score, via_reid = None, 0.0, False
                    if _emb is not None:
                        match_person, score = self.face_db.match_embedding(_emb)
                        if match_person is None:
                            # Enrollment match failed — try re-ID against confirmed recent frames
                            match_person, score = self.face_db.reid_match(_emb)
                            via_reid = match_person is not None

                    locked_id = (
                        track.person_id
                        if track.person_id and not track.person_id.startswith("unknown_")
                        else ""
                    )

                    if locked_id:
                        # STICKY IDENTITY: once recognized, the name stays attached to this
                        # track for its lifetime. A failed match this frame (head turned,
                        # blurry) must NOT relabel it to Unknown.
                        person = self.face_db._records.get(
                            locked_id,
                            PersonRecord(person_id=locked_id, name=track.person_name, role=_role),
                        )
                        agrees = match_person is not None and match_person.person_id == locked_id
                        score = score if agrees else 0.0
                        # Refresh the re-ID gallery only when this frame agrees with the lock.
                        if _emb is not None and agrees:
                            self.face_db.update_gallery(locked_id, _emb)
                        if self.cfg.runtime.app_mode == "classroom":
                            self.attendance.mark_seen(person, datetime.now(), frame_id=frame_count)
                    elif match_person is not None:
                        # Not locked yet — accumulate votes. A strong re-ID hit (a face that
                        # was confirmed moments ago) locks faster than a cold enrollment match.
                        votes = _vote_buffer.setdefault(track.track_id, {})
                        votes[match_person.person_id] = votes.get(match_person.person_id, 0) + 1
                        need = REID_VOTES_NEEDED if via_reid else VOTE_NEEDED
                        top_pid = max(votes, key=votes.get)
                        if votes[top_pid] >= need:
                            locked = self.face_db._records.get(top_pid)
                            if locked:
                                track.person_id = locked.person_id
                                track.person_name = locked.name

                        if track.person_id and not track.person_id.startswith("unknown_"):
                            person = self.face_db._records.get(track.person_id, match_person)
                            if _emb is not None:
                                self.face_db.update_gallery(track.person_id, _emb)
                            if (self.cfg.runtime.app_mode == "classroom"
                                    and score >= ATTEND_MIN_SCORE):
                                self.attendance.mark_seen(person, datetime.now(), frame_id=frame_count)
                        else:
                            # Still gathering votes — show as Unknown until confirmed
                            person = PersonRecord(
                                person_id=f"unknown_{track.track_id}", name="Unknown", role=_role,
                            )
                    else:
                        person = PersonRecord(
                            person_id=f"unknown_{track.track_id}", name="Unknown", role=_role,
                        )
                else:
                    # Reuse cached (vote-confirmed) identity from the track
                    if track.person_id and not track.person_id.startswith("unknown_"):
                        # Look up the real record so we preserve the correct role
                        person = self.face_db._records.get(
                            track.person_id,
                            PersonRecord(
                                person_id=track.person_id,
                                name=track.person_name,
                                role=_role,
                            ),
                        )
                        score = 0.0
                        # Identity was already vote-confirmed — safe to mark attendance
                        if self.cfg.runtime.app_mode == "classroom":
                            self.attendance.mark_seen(person, datetime.now(), frame_id=frame_count)
                    else:
                        person = PersonRecord(
                            person_id=f"unknown_{track.track_id}", name="Unknown", role=_role,
                        )
                        score = 0.0

                # Label: name+score for a fresh match; just the name when maintaining a
                # locked identity (score 0) so it doesn't read as a bad "(0.00)" match.
                if person.name == "Unknown":
                    label = "Unknown"
                elif score > 0:
                    label = f"{person.name} ({score:.2f})"
                else:
                    label = person.name
                color = (0, 200, 0) if person.name != "Unknown" else (0, 165, 255)

                self._draw_box(frame, track.bbox, label, color)
                # Update persistent roster for recognized students.
                # Always refresh the bbox from the live track so it never
                # drifts to a stale position, even between recognition runs.
                if person.name != "Unknown":
                    # Reuse the previous label/color if this frame is between
                    # recognition runs (label/color already set above from track).
                    _seen_people[person.person_id] = (track.bbox, label, color, now_ts)
                elif person.person_id in _seen_people:
                    # Track is active but identity shows Unknown (still voting) —
                    # keep the previously confirmed entry but update its bbox position.
                    prev_bbox, prev_label, prev_color, prev_ts = _seen_people[person.person_id]
                    _seen_people[person.person_id] = (track.bbox, prev_label, prev_color, prev_ts)

            # Draw the recognized-students sidebar panel.
            active_ids = {t.person_id for t in tracks if t.person_id}

            # Expire stale entries — remove any person not actively tracked for
            # longer than STALE_BOX_TTL so their frozen box never lingers on the
            # wrong desk or an empty seat.
            expired = [
                pid for pid, (_b, _l, _c, ts) in _seen_people.items()
                if pid not in active_ids and (now_ts - ts) > STALE_BOX_TTL
            ]
            for pid in expired:
                _seen_people.pop(pid, None)

            # Draw a dimmed box at the last-known position for recently-seen
            # students who are temporarily not detected (e.g. looked away).
            active_boxes = [t.bbox for t in tracks]
            for pid, (p_bbox, p_label, p_color, _ts) in _seen_people.items():
                if pid in active_ids:
                    continue
                # Safety net: never draw a dimmed "kept" box over a live box, so a
                # brief detection glitch can't show a double box (kept + new Unknown).
                if any(self._bbox_iou(p_bbox, ab) > 0.3 for ab in active_boxes):
                    continue
                self._draw_box(frame, p_bbox, p_label, p_color, dimmed=True)

            frame_count += 1

            # Periodic FPS counter (every 100 frames)
            if frame_count % 100 == 0:
                elapsed = now_ts - _fps_report_ts
                if elapsed > 0:
                    logger.info("FPS: %.1f  (frames %d, elapsed %.1fs)",
                                100.0 / elapsed, frame_count, now_ts - start_ts)
                _fps_report_ts = now_ts

            if writer is not None:
                writer.write(frame)

            if self.cfg.runtime.display:
                window_title = "Classroom Monitor" if self.cfg.runtime.app_mode == "classroom" else "Face Recognition"
                # Scale down for display so the window fits on screen.
                # Boxes are already drawn on `frame` at full resolution, so
                # resizing here keeps them correctly aligned with the image.
                _DISP_MAX_W = 1280
                if w_frame > _DISP_MAX_W:
                    _disp_scale = _DISP_MAX_W / w_frame
                    display_frame = cv2.resize(
                        frame,
                        (_DISP_MAX_W, int(h_frame * _disp_scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    display_frame = frame
                self._draw_recognized_panel(display_frame, _seen_people, active_ids)
                cv2.imshow(window_title, display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if self.cfg.runtime.stop_after_seconds > 0 and (now_ts - start_ts) >= self.cfg.runtime.stop_after_seconds:
                break

        if writer is not None:
            writer.release()

        camera_health = self.video.get_health()
        self.video.release()
        if self.cfg.runtime.display:
            cv2.destroyAllWindows()

        if self.cfg.runtime.app_mode == "classroom":
            students = self.attendance.finalize_students(
                self._known_students,
                include_absent=self.cfg.attendance.absent_if_not_seen,
            )
            teachers = self.attendance.finalize_teachers(
                self._known_teachers,
                include_absent=self.cfg.attendance.absent_if_not_seen,
            )

            student_csv, teacher_csv = self.reporter.export_attendance(students, teachers)
            incident_csv = self.reporter.export_incidents_csv()
            summary_path = self.reporter.export_session_summary(
                students, teachers, self.cfg.source.source, camera_health,
            )

            logger.info("Processed %d frames", frame_count)
            logger.info("  Students present : %d/%d",
                        sum(1 for s in students if s.status == 'present'), len(students))
            logger.info("  Incidents logged : %d", len(self.reporter._incident_log))
            logger.info("  Reports saved to : %s/", self.cfg.paths.output_dir)

            if self.cfg.admin_email.enabled:
                attachments = [student_csv, teacher_csv, incident_csv, summary_path]
                attachments.extend(self.reporter.get_snapshot_paths(limit=5))
                self.reporter.maybe_send_email(
                    subject=f"Session Report — {self.cfg.session.class_name}",
                    body=f"Automated report for {self.cfg.session.class_name}.",
                    attachments=attachments,
                )

            logger.info("Processing completed.")
            logger.info("  Student attendance : %s", student_csv)
            logger.info("  Teacher attendance : %s", teacher_csv)
            logger.info("  Incidents          : %s", incident_csv)
            logger.info("  Summary            : %s", summary_path)
        else:
            logger.info("Processed %d frames", frame_count)
            logger.info("Processing completed.")