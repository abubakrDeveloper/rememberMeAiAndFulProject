from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

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
        self.tracker = CentroidTracker()
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
        row_h = 30
        panel_w = 320
        panel_h = len(seen_people) * row_h + 12
        px = w_frame - panel_w - 10
        py = 10
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
        # Persistent overlay: person_id -> (bbox, label, color, last_seen_ts)
        # Once a student is recognized we keep drawing their box every frame,
        # dimmed when not actively detected in the current frame.
        _seen_people: dict = {}

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

            # --- Detection: runs every processed frame for accurate bboxes ---
            _SCALES = [1280, 640] if w_frame > 1280 else [w_frame]
            detected_boxes: List[Tuple[int, int, int, int]] = []
            for _DET_W in _SCALES:
                det_scale = _DET_W / w_frame
                det_h = int(h_frame * det_scale)
                rgb_small = cv2.resize(rgb, (_DET_W, det_h), interpolation=cv2.INTER_AREA)
                try:
                    det_result = face_det.process(rgb_small)
                    if det_result.detections:
                        inv = 1.0 / det_scale
                        for det in det_result.detections:
                            sx, sy, sw, sh = self._abs_box(det, _DET_W, det_h)
                            detected_boxes.append((
                                int(sx * inv), int(sy * inv),
                                int(sw * inv), int(sh * inv),
                            ))
                except Exception as exc:  # noqa: BLE001
                    print(f"[frame {frame_count}] FaceDetection error at scale {_DET_W}: {exc}")

            # Deduplicate boxes with >50% overlap (same face found at both scales)
            boxes: List[Tuple[int, int, int, int]] = []
            for bx, by, bw, bh in detected_boxes:
                overlap = False
                for ex, ey, ew, eh in boxes:
                    ix1, iy1 = max(bx, ex), max(by, ey)
                    ix2, iy2 = min(bx+bw, ex+ew), min(by+bh, ey+eh)
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2-ix1) * (iy2-iy1)
                        area = min(bw*bh, ew*eh)
                        if area > 0 and inter / area > 0.5:
                            overlap = True
                            break
                if not overlap:
                    boxes.append((bx, by, bw, bh))

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
                if w < 10 or h < 10:
                    continue

                # --- Recognition: only when interval has elapsed OR track is new ---
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
                        track.person_id = person.person_id
                        track.person_name = person.name
                        if self.cfg.runtime.app_mode == "classroom":
                            self.attendance.mark_seen(person, datetime.now())
                    else:
                        person = PersonRecord(
                            person_id=f"unknown_{track.track_id}",
                            name="Unknown",
                            role="student" if self.cfg.runtime.app_mode == "classroom" else "person",
                        )
                else:
                    # Reuse cached identity from the track
                    if track.person_id and not track.person_id.startswith("unknown_"):
                        person = PersonRecord(
                            person_id=track.person_id,
                            name=track.person_name,
                            role="student",
                        )
                        score = 0.0
                        # Count attendance every frame, not just on recognition runs
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
                # Update persistent roster for recognized students
                if person.name != "Unknown":
                    _seen_people[person.person_id] = (track.bbox, label, color, now_ts)

            # Draw the recognized-students sidebar panel
            active_ids = {t.person_id for t in tracks if t.person_id}
            self._draw_recognized_panel(frame, _seen_people, active_ids)

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