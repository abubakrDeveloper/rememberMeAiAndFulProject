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
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- detect faces ---
            boxes: List[Tuple[int, int, int, int]] = []
            try:
                det_result = face_det.process(rgb)
                if det_result.detections:
                    for det in det_result.detections:
                        boxes.append(self._abs_box(det, w_frame, h_frame))
            except Exception as exc:  # noqa: BLE001
                print(f"[frame {frame_count}] FaceDetection error: {exc}")

            # --- face mesh for all faces at once ---
            mesh_result = None
            try:
                mesh_result = face_mesh.process(rgb)
            except Exception as exc:  # noqa: BLE001
                print(f"[frame {frame_count}] FaceMesh error: {exc}")

            # --- track faces ---
            tracks = self.tracker.update(boxes, now_ts)
            incidents_to_record: List[Incident] = []

            for track in tracks:
                x, y, w, h = track.bbox
                if w < 10 or h < 10:
                    continue

                # Crop face for recognition (RGB for face_recognition lib)
                face_rgb = rgb[y : y + h, x : x + w]
                if face_rgb.size == 0:
                    continue

                person, score = self.face_db.match(face_rgb)
                if person:
                    track.person_id = person.person_id
                    track.person_name = person.name
                    self.attendance.mark_seen(person, datetime.now())
                else:
                    person = PersonRecord(
                        person_id=f"unknown_{track.track_id}",
                        name="Unknown",
                        role="student",
                    )

                # Get per-face landmarks
                landmarks = self._get_face_landmarks(mesh_result, track.bbox, w_frame, h_frame)

                state_label, incidents = self.behavior.analyze(
                    key=track.person_id or f"track_{track.track_id}",
                    person=person,
                    track_id=track.track_id,
                    bbox=track.bbox,
                    landmarks=landmarks,
                    now_ts=now_ts,
                )

                # Draw
                label = f"{person.name} | {state_label} | {score:.2f}"
                if state_label in {"distracted", "sleeping"}:
                    color = (0, 0, 255)
                elif state_label == "focused":
                    color = (0, 200, 0)
                else:
                    color = (0, 255, 255)
                self._draw_box(frame, track.bbox, label, color)

                incidents_to_record.extend([i for i in incidents if i.role == "student"])

            for inc in incidents_to_record:
                inc.snapshot_path = self.reporter.save_incident_snapshot(frame, inc, [inc.bbox])
                self.reporter.record_incident(inc)

            frame_count += 1

            if writer is not None:
                writer.write(frame)

            if self.cfg.runtime.display:
                cv2.imshow("Classroom Monitor", frame)
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