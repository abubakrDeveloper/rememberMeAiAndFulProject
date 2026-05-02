from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .attendance import AttendanceManager
from .behavior import BehaviorAnalyzer
from .config import AppConfig
from .face_database import FaceDatabase
from .reporting import Reporter
from .tracking import CentroidTracker
from .types import Incident, PersonRecord
from .video import VideoInput

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection


class ClassroomMonitorApp:
    def __init__(self, config: AppConfig) -> None:
        self.cfg = config

        self.face_db = FaceDatabase(tolerance=self.cfg.detection.recognition_tolerance)
        self.attendance = AttendanceManager(min_confirm_frames=self.cfg.attendance.min_confirm_frames)
        self.behavior = BehaviorAnalyzer(self.cfg.behavior)
        self.tracker = CentroidTracker(max_distance=80)
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
    def _abs_box(det, w: int, h: int) -> Tuple[int, int, int, int]:
        """Convert MediaPipe relative bbox to absolute (x, y, w, h)."""
        bb = det.location_data.relative_bounding_box
        x = max(0, int(bb.xmin * w))
        y = max(0, int(bb.ymin * h))
        bw = min(int(bb.width * w), w - x)
        bh = min(int(bb.height * h), h - y)
        return x, y, bw, bh

    @staticmethod
    def _nms_boxes(
        boxes: List[Tuple[int, int, int, int]], iou_threshold: float = 0.35
    ) -> List[Tuple[int, int, int, int]]:
        """Non-maximum suppression: remove duplicate boxes based on IoU."""
        if not boxes:
            return []
        # Sort by descending box area so larger (usually better) detections are kept first.
        sorted_boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        kept: List[Tuple[int, int, int, int]] = []
        for box in sorted_boxes:
            x1, y1, w1, h1 = box
            suppressed = False
            for kx, ky, kw, kh in kept:
                ix = max(x1, kx)
                iy = max(y1, ky)
                ix2 = min(x1 + w1, kx + kw)
                iy2 = min(y1 + h1, ky + kh)
                if ix2 <= ix or iy2 <= iy:
                    continue
                inter = (ix2 - ix) * (iy2 - iy)
                union = w1 * h1 + kw * kh - inter
                if union > 0 and inter / union > iou_threshold:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(box)
        return kept

    def _detect_faces_tiled(
        self,
        rgb: np.ndarray,
        detector,
        tile_cols: int = 2,
        tile_rows: int = 2,
        tile_det_width: int = 960,
    ) -> List[Tuple[int, int, int, int]]:
        """Detect faces by running the detector on a 2×2 grid of overlapping tiles.

        Each tile covers ~60% of the frame dimension so adjacent tiles overlap by ~20%.
        This makes faces appear ~2× larger in each tile vs. a single 1280-wide downscale,
        which substantially improves detection of small or partially-visible faces.
        """
        h_frame, w_frame = rgb.shape[:2]
        # Tile dimensions: 60% of frame so that two tiles cover the full frame with ~20% overlap.
        tile_w = int(w_frame * 0.60)
        tile_h = int(h_frame * 0.60)
        # Stride: moves forward by 40% so col0=(0..60%) col1=(40%..100%)
        stride_x = w_frame - tile_w   # puts right tile flush with right edge
        stride_y = h_frame - tile_h   # puts bottom tile flush with bottom edge

        all_boxes: List[Tuple[int, int, int, int]] = []

        for row in range(tile_rows):
            for col in range(tile_cols):
                ox = col * stride_x
                oy = row * stride_y
                x2 = min(w_frame, ox + tile_w)
                y2 = min(h_frame, oy + tile_h)
                tile = rgb[oy:y2, ox:x2]
                th, tw = tile.shape[:2]
                if tw < 20 or th < 20:
                    continue
                # Scale tile to tile_det_width for MediaPipe.
                det_scale = min(tile_det_width, tw) / tw
                det_w = int(tw * det_scale)
                det_h = int(th * det_scale)
                small = cv2.resize(tile, (det_w, det_h), interpolation=cv2.INTER_AREA)
                try:
                    result = detector.process(small)
                except Exception:
                    continue
                if not result.detections:
                    continue
                inv = 1.0 / det_scale
                for det in result.detections:
                    sx, sy, sw, sh = self._abs_box(det, det_w, det_h)
                    # Project back to full-frame coordinates.
                    fx = ox + int(sx * inv)
                    fy = oy + int(sy * inv)
                    fw = int(sw * inv)
                    fh = int(sh * inv)
                    fx = max(0, min(fx, w_frame - 1))
                    fy = max(0, min(fy, h_frame - 1))
                    fw = min(fw, w_frame - fx)
                    fh = min(fh, h_frame - fy)
                    ar = fw / fh if fh > 0 else 0
                    if fw >= 50 and fh >= 50 and 0.4 <= ar <= 2.5:
                        all_boxes.append((fx, fy, fw, fh))

        # Also run a single full-frame pass at tile_det_width — catches faces that
        # happen to fall exactly at a tile boundary and are missed by every tile.
        full_scale = tile_det_width / w_frame
        full_w = int(w_frame * full_scale)
        full_h = int(h_frame * full_scale)
        small_full = cv2.resize(rgb, (full_w, full_h), interpolation=cv2.INTER_AREA)
        try:
            result_full = detector.process(small_full)
            if result_full.detections:
                inv_full = 1.0 / full_scale
                for det in result_full.detections:
                    sx, sy, sw, sh = self._abs_box(det, full_w, full_h)
                    fx = int(sx * inv_full)
                    fy = int(sy * inv_full)
                    fw = int(sw * inv_full)
                    fh = int(sh * inv_full)
                    fx = max(0, min(fx, w_frame - 1))
                    fy = max(0, min(fy, h_frame - 1))
                    fw = min(fw, w_frame - fx)
                    fh = min(fh, h_frame - fy)
                    ar = fw / fh if fh > 0 else 0
                    if fw >= 50 and fh >= 50 and 0.4 <= ar <= 2.5:
                        all_boxes.append((fx, fy, fw, fh))
        except Exception:
            pass

        return self._nms_boxes(all_boxes)

    @staticmethod
    def _draw_box(frame: np.ndarray, bbox: Tuple[int, int, int, int], label: str, color: Tuple[int, int, int]) -> None:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, max(y - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
        panel_w = 280
        panel_h = len(seen_people) * row_h + 12
        px = w_frame - panel_w - 10
        py = h_frame - panel_h - 10  # bottom-right corner
        # semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (px - 4, py - 4), (px + panel_w, py + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        for i, (pid, (_bbox, label, p_color, _ts)) in enumerate(seen_people.items()):
            y_pos = py + 8 + i * row_h
            is_active = pid in active_ids
            dot_color = p_color if is_active else (80, 140, 200)
            # dot indicator: filled = active, hollow = last seen
            if is_active:
                cv2.circle(frame, (px + 8, y_pos + 7), 5, dot_color, -1)
            else:
                cv2.circle(frame, (px + 8, y_pos + 7), 5, dot_color, 1)
            # show only the name part (strip score/state clutter)
            name = label.split("|")[0].split("(")[0].strip()
            cv2.putText(frame, name, (px + 20, y_pos + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, dot_color, 1, cv2.LINE_AA)

    def _get_face_landmarks(
        self,
        mesh_results,
        bbox: Tuple[int, int, int, int],
        img_w: int,
        img_h: int,
    ) -> Optional[np.ndarray]:
        """Find the Face Mesh result whose nose tip is inside bbox."""
        if not mesh_results or not mesh_results.multi_face_landmarks:
            return None
        bx, by, bw, bh = bbox
        for face_lm in mesh_results.multi_face_landmarks:
            nose = face_lm.landmark[1]
            nx, ny = int(nose.x * img_w), int(nose.y * img_h)
            if bx <= nx <= bx + bw and by <= ny <= by + bh:
                lm = np.array(
                    [[l.x, l.y, l.z] for l in face_lm.landmark],
                    dtype=np.float32,
                )
                return lm
        return None

    def _load_roster(self) -> None:
        if self.cfg.runtime.app_mode != "classroom":
            roster_dir = self.cfg.paths.roster_dir
            stats = self.face_db.load_people_dir(roster_dir)
            print(f"Roster loaded | people={stats['people']}")
            if not self.face_db.has_people():
                print("Warning: no roster faces loaded — all faces will show as Unknown.")
            return

        stats = self.face_db.load(
            students_dir=self.cfg.paths.roster_students_dir,
            teachers_dir=self.cfg.paths.roster_teachers_dir,
        )
        self._known_students = self.face_db.list_people("student")
        self._known_teachers = self.face_db.list_people("teacher")
        print(
            f"Roster loaded | students={stats['students']} "
            f"teachers={stats['teachers']} people={stats['people']}"
        )
        if not self.face_db.has_people():
            print("Warning: no roster faces loaded — attendance won't work.")

    def run(self) -> None:
        self._load_roster()
        self.video.open()

        frame_count = 0
        start_ts = time.time()
        last_process_ts = 0.0
        last_recognize_ts: float = -999.0  # controls 30-second recognition interval
        RECOGNIZE_INTERVAL = 30.0  # seconds between recognition runs per track
        last_tiled_det_ts: float = -999.0   # controls tiled detection interval
        TILED_DET_INTERVAL = 2.0            # seconds between expensive tiled passes
        _cached_boxes: List[Tuple[int, int, int, int]] = []  # last tiled result
        # Persistent overlay: person_id -> (bbox, label, color, last_seen_ts)
        # Once a student is recognized we keep drawing their box every frame,
        # dimmed when not actively detected in the current frame.
        _seen_people: dict = {}
        # Option B: per-track vote buffer — identity locked only after VOTE_NEEDED agreements
        _vote_buffer: Dict[int, Dict[str, int]] = {}
        VOTE_NEEDED = 5
        # Option C: distance must be this good (score = 1-dist) to count toward attendance
        ATTEND_MIN_SCORE = 0.40  # corresponds to recognition distance ≤ 0.60
        # Stale box TTL: stop drawing a last-known-position box after this many seconds
        STALE_BOX_TTL = 5.0

        writer = None
        if self.cfg.runtime.save_annotated_video:
            width, height = self.video.frame_size()
            fps = self.video.fps() or 15.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = str(Path(self.cfg.paths.output_dir) / "annotated_output.mp4")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        face_det = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=self.cfg.detection.min_detection_confidence,
        )
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.cfg.detection.max_faces,
            refine_landmarks=True,
            min_detection_confidence=self.cfg.detection.min_detection_confidence,
            min_tracking_confidence=0.4,
        )

        while True:
            ok, frame = self.video.read()
            if not ok or frame is None:
                break

            now_ts = time.time()
            if self.cfg.runtime.max_fps > 0:
                min_gap = 1.0 / self.cfg.runtime.max_fps
                if now_ts - last_process_ts < min_gap:
                    continue
                last_process_ts = now_ts

            h_frame, w_frame = frame.shape[:2]
            incidents_to_record: List[Incident] = []
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Check if it is time to run the expensive recognition pipeline.
            run_recognition = (now_ts - last_recognize_ts >= RECOGNIZE_INTERVAL)
            if run_recognition:
                last_recognize_ts = now_ts

            # --- Face detection: two-tier strategy for speed vs. coverage ---
            # Fast path (every frame): single full-frame pass at 960px — 1 inference,
            #   keeps tracker alive and bboxes current between tiled runs.
            # Slow path (every 2s): full tiled 2×2 pass — finds small/angled faces
            #   that the single pass misses, cached until the next tiled run.
            run_tiled = (now_ts - last_tiled_det_ts >= TILED_DET_INTERVAL)

            # Always run the cheap single-pass for fresh tracking boxes.
            _det_scale = 960 / w_frame
            _det_w = 960
            _det_h = int(h_frame * _det_scale)
            rgb_small_det = cv2.resize(rgb, (_det_w, _det_h), interpolation=cv2.INTER_AREA)
            single_boxes: List[Tuple[int, int, int, int]] = []
            try:
                det_result = face_det.process(rgb_small_det)
                if det_result.detections:
                    inv = 1.0 / _det_scale
                    for det in det_result.detections:
                        sx, sy, sw, sh = self._abs_box(det, _det_w, _det_h)
                        fx = int(sx * inv); fy = int(sy * inv)
                        fw = int(sw * inv); fh = int(sh * inv)
                        ar = fw / fh if fh > 0 else 0
                        if fw >= 50 and fh >= 50 and 0.4 <= ar <= 2.5:
                            single_boxes.append((fx, fy, fw, fh))
            except Exception as exc:  # noqa: BLE001
                print(f"[frame {frame_count}] FaceDetection error: {exc}")

            if run_tiled:
                last_tiled_det_ts = now_ts
                try:
                    _cached_boxes = self._detect_faces_tiled(rgb, face_det)
                except Exception as exc:  # noqa: BLE001
                    print(f"[frame {frame_count}] Tiled detection error: {exc}")
                    _cached_boxes = []

            # Merge: start from the tiled cache, then add any single-pass boxes
            # not already covered by a cached box (NMS-style).
            detected_boxes = self._nms_boxes(single_boxes + _cached_boxes)

            boxes: List[Tuple[int, int, int, int]] = detected_boxes

            # Use the 1280px downscale for FaceMesh (consistent with detection)
            _mesh_w = min(1280, w_frame)
            _mesh_scale = _mesh_w / w_frame
            rgb_small_mesh = cv2.resize(rgb, (_mesh_w, int(h_frame * _mesh_scale)), interpolation=cv2.INTER_AREA)

            # --- face mesh for all faces at once ---
            mesh_result = None
            try:
                mesh_result = face_mesh.process(rgb_small_mesh)
            except Exception as exc:  # noqa: BLE001
                print(f"[frame {frame_count}] FaceMesh error: {exc}")

            # --- track faces (always runs to keep bboxes accurate) ---
            tracks = self.tracker.update(boxes, now_ts)

            for track in tracks:
                x, y, w, h = track.bbox
                if w < 50 or h < 50:
                    continue

                # --- Recognition: only when interval has elapsed OR track not yet locked ---
                if run_recognition or not track.person_id:
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(w_frame, x + w)
                    y2 = min(h_frame, y + h)
                    face_rgb = rgb[y1:y2, x1:x2]
                    if face_rgb.size == 0:
                        continue

                    person, score = self.face_db.match(face_rgb)
                    if person:
                        # Option B: accumulate votes; lock identity only after VOTE_NEEDED agreements
                        if not track.person_id:
                            votes = _vote_buffer.setdefault(track.track_id, {})
                            votes[person.person_id] = votes.get(person.person_id, 0) + 1
                            top_pid = max(votes, key=votes.get)
                            if votes[top_pid] >= VOTE_NEEDED:
                                locked = self.face_db._records.get(top_pid)
                                if locked:
                                    track.person_id = locked.person_id
                                    track.person_name = locked.name

                        # Identity confirmed (locked now or was locked before)
                        if track.person_id and not track.person_id.startswith("unknown_"):
                            person = self.face_db._records.get(track.person_id, person)
                            # Option C: mark attendance only for strong matches
                            if (self.cfg.runtime.app_mode == "classroom"
                                    and score >= ATTEND_MIN_SCORE):
                                self.attendance.mark_seen(person, datetime.now())
                        else:
                            # Still gathering votes — show as Unknown until confirmed
                            person = PersonRecord(
                                person_id=f"unknown_{track.track_id}",
                                name="Unknown",
                                role="student" if self.cfg.runtime.app_mode == "classroom" else "person",
                            )
                    else:
                        person = PersonRecord(
                            person_id=f"unknown_{track.track_id}",
                            name="Unknown",
                            role="student" if self.cfg.runtime.app_mode == "classroom" else "person",
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
                                role="student",
                            ),
                        )
                        score = 0.0
                        # Identity was already vote-confirmed — safe to mark attendance
                        if self.cfg.runtime.app_mode == "classroom":
                            self.attendance.mark_seen(person, datetime.now())
                    else:
                        person = PersonRecord(
                            person_id=f"unknown_{track.track_id}",
                            name="Unknown",
                            role="student" if self.cfg.runtime.app_mode == "classroom" else "person",
                        )
                        score = 0.0

                # Get per-face landmarks (mesh was run on downscaled image)
                det_w = rgb_small_mesh.shape[1]
                det_h_px = rgb_small_mesh.shape[0]
                scaled_bbox = (
                    int(x * _mesh_scale), int(y * _mesh_scale),
                    int(w * _mesh_scale), int(h * _mesh_scale),
                )
                landmarks = self._get_face_landmarks(mesh_result, scaled_bbox, det_w, det_h_px)

                if self.cfg.runtime.app_mode == "classroom":
                    state_label, incidents = self.behavior.analyze(
                        key=track.person_id or f"track_{track.track_id}",
                        person=person,
                        track_id=track.track_id,
                        bbox=track.bbox,
                        landmarks=landmarks,
                        now_ts=now_ts,
                    )
                    label = f"{person.name} | {state_label} | {score:.2f}"
                    if state_label in {"distracted", "sleeping"}:
                        color = (0, 0, 255)
                    elif state_label == "focused":
                        color = (0, 200, 0)
                    else:
                        color = (0, 255, 255)
                    incidents_to_record.extend([i for i in incidents if i.role == "student"])
                else:
                    incidents = []
                    label = f"{person.name} ({score:.2f})" if person.name != "Unknown" else "Unknown"
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
            for pid, (p_bbox, p_label, p_color, _ts) in _seen_people.items():
                if pid not in active_ids:
                    self._draw_box(frame, p_bbox, p_label, (80, 140, 200))

            # Panel is drawn on display_frame after resize (bottom-right, sharp text).

            for inc in incidents_to_record:
                inc.snapshot_path = self.reporter.save_incident_snapshot(frame, inc, [inc.bbox])
                self.reporter.record_incident(inc)

            frame_count += 1

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

        face_det.close()
        face_mesh.close()

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

            print(f"\nProcessed {frame_count} frames")
            print(f"  Students present : {sum(1 for s in students if s.status == 'present')}/{len(students)}")
            print(f"  Incidents logged : {len(self.reporter._incident_log)}")
            print(f"  Reports saved to : {self.cfg.paths.output_dir}/")

            if self.cfg.admin_email.enabled:
                attachments = [student_csv, teacher_csv, incident_csv, summary_path]
                attachments.extend(self.reporter.get_snapshot_paths(limit=5))
                self.reporter.maybe_send_email(
                    subject=f"Session Report — {self.cfg.session.class_name}",
                    body=f"Automated report for {self.cfg.session.class_name}.",
                    attachments=attachments,
                )

            print("Processing completed.")
            print(f"  Student attendance : {student_csv}")
            print(f"  Teacher attendance : {teacher_csv}")
            print(f"  Incidents          : {incident_csv}")
            print(f"  Summary            : {summary_path}")
        else:
            print(f"\nProcessed {frame_count} frames")
            print("Processing completed.")