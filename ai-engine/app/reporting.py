from __future__ import annotations

import csv
import json
import smtplib
from dataclasses import asdict
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2

from .config import AdminEmailConfig, SessionConfig
from .types import AttendanceEntry, Incident


class Reporter:
    def __init__(
        self,
        output_dir: str,
        snapshots_dir: str,
        session: SessionConfig,
        admin_cfg: AdminEmailConfig,
        retention_days: int = 3,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.snapshots_dir = Path(snapshots_dir)
        self.session = session
        self.admin_cfg = admin_cfg
        self.retention_days = max(1, int(retention_days))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log_path = self.output_dir / "audit_log.jsonl"
        self._incident_log: List[Incident] = []
        self.cleanup_old_files()

    @staticmethod
    def _confidence_band(score: float) -> str:
        if score >= 0.75:
            return "high"
        if score >= 0.4:
            return "medium"
        return "low"

    def _append_audit(self, event: str, details: Dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "details": details,
        }
        with self.audit_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")

    def cleanup_old_files(self) -> None:
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        deleted_count = 0

        report_patterns = [
            "attendance_*.csv",
            "incidents_*.csv",
            "summary_*.json",
            "annotated_*.mp4",
        ]
        for pattern in report_patterns:
            for path in self.output_dir.glob(pattern):
                if datetime.fromtimestamp(path.stat().st_mtime) < cutoff:
                    path.unlink(missing_ok=True)
                    deleted_count += 1

        for path in self.snapshots_dir.glob("*.jpg"):
            if datetime.fromtimestamp(path.stat().st_mtime) < cutoff:
                path.unlink(missing_ok=True)
                deleted_count += 1

        for path in self.snapshots_dir.glob("*.png"):
            if datetime.fromtimestamp(path.stat().st_mtime) < cutoff:
                path.unlink(missing_ok=True)
                deleted_count += 1

        self._append_audit(
            "retention_cleanup",
            {
                "retention_days": self.retention_days,
                "deleted_files": deleted_count,
            },
        )

    def save_incident_snapshot(
        self,
        frame,
        incident: Incident,
        highlighted_boxes: Sequence[Tuple[int, int, int, int]],
    ) -> str:
        snapshot = frame.copy()
        for x, y, w, h in highlighted_boxes:
            cv2.rectangle(snapshot, (x, y), (x + w, y + h), (0, 0, 255), 2)

        title = f"{incident.incident_type} | {incident.person_name} ({incident.person_id})"
        cv2.putText(snapshot, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        stamp = incident.timestamp.strftime("%Y%m%d_%H%M%S")
        file_name = f"incident_{stamp}_{incident.person_id}_{incident.incident_type}.jpg"
        path = self.snapshots_dir / file_name
        cv2.imwrite(str(path), snapshot)
        return str(path)

    def record_incident(self, incident: Incident) -> None:
        self._incident_log.append(incident)

    def get_snapshot_paths(self, limit: int = 20) -> List[str]:
        paths: List[str] = []
        seen = set()
        for incident in self._incident_log:
            if not incident.snapshot_path:
                continue
            if incident.snapshot_path in seen:
                continue
            seen.add(incident.snapshot_path)
            paths.append(incident.snapshot_path)
        return paths[: max(0, limit)]

    def export_incidents_csv(self) -> str:
        file_name = f"incidents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = self.output_dir / file_name
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "timestamp",
                    "incident_type",
                    "person_id",
                    "person_name",
                    "role",
                    "class_name",
                    "confidence",
                    "confidence_band",
                    "track_id",
                    "bbox",
                    "snapshot_path",
                ]
            )
            for item in self._incident_log:
                writer.writerow(
                    [
                        item.timestamp.isoformat(),
                        item.incident_type,
                        item.person_id,
                        item.person_name,
                        item.role,
                        self.session.class_name,
                        f"{item.confidence:.3f}",
                        self._confidence_band(item.confidence),
                        item.track_id,
                        list(item.bbox),
                        item.snapshot_path,
                    ]
                )
        self._append_audit(
            "report_generated",
            {
                "type": "incidents_csv",
                "path": str(path),
                "rows": len(self._incident_log),
            },
        )
        return str(path)

    def _export_attendance(self, entries: List[AttendanceEntry], role: str) -> str:
        file_name = f"attendance_{role}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = self.output_dir / file_name
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "person_id",
                    "name",
                    "role",
                    "class_name",
                    "status",
                    "confidence",
                    "confidence_band",
                    "first_seen",
                    "last_seen",
                ]
            )
            for item in entries:
                writer.writerow(
                    [
                        item.person_id,
                        item.name,
                        item.role,
                        self.session.class_name,
                        item.status,
                        f"{item.confidence:.3f}",
                        item.confidence_band,
                        item.first_seen.isoformat() if item.first_seen else "",
                        item.last_seen.isoformat() if item.last_seen else "",
                    ]
                )
        self._append_audit(
            "report_generated",
            {
                "type": f"attendance_{role}_csv",
                "path": str(path),
                "rows": len(entries),
            },
        )
        return str(path)

    def export_session_summary(
        self,
        student_entries: List[AttendanceEntry],
        teacher_entries: List[AttendanceEntry],
        source_label: str,
        camera_health: Dict[str, Any] | None = None,
    ) -> str:
        summary = {
            "generated_at": datetime.now().isoformat(),
            "session": asdict(self.session),
            "source": source_label,
            "camera_health": camera_health or {},
            "student_totals": {
                "present": len([x for x in student_entries if x.status == "present"]),
                "absent": len([x for x in student_entries if x.status == "absent"]),
            },
            "teacher_totals": {
                "present": len([x for x in teacher_entries if x.status == "present"]),
                "absent": len([x for x in teacher_entries if x.status == "absent"]),
            },
            "incident_totals": {
                "all": len(self._incident_log),
                "distracted": len([x for x in self._incident_log if x.incident_type == "distracted"]),
                "sleeping": len([x for x in self._incident_log if x.incident_type == "sleeping"]),
            },
        }

        file_name = f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = self.output_dir / file_name
        with path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        self._append_audit(
            "report_generated",
            {
                "type": "summary_json",
                "path": str(path),
            },
        )
        return str(path)

    def export_attendance(
        self,
        student_entries: List[AttendanceEntry],
        teacher_entries: List[AttendanceEntry],
    ) -> Tuple[str, str]:
        return self._export_attendance(student_entries, "students"), self._export_attendance(teacher_entries, "teachers")

    def maybe_send_email(self, subject: str, body: str, attachments: List[str]) -> bool:
        cfg = self.admin_cfg
        if not cfg.enabled:
            self._append_audit(
                "email_skipped",
                {
                    "reason": "disabled",
                    "recipient": cfg.recipient_email,
                },
            )
            return False

        required = [cfg.sender_email, cfg.recipient_email, cfg.smtp_host, str(cfg.smtp_port)]
        if any(not x for x in required):
            self._append_audit(
                "email_skipped",
                {
                    "reason": "missing_configuration",
                    "recipient": cfg.recipient_email,
                },
            )
            return False

        message = EmailMessage()
        message["From"] = cfg.sender_email
        message["To"] = cfg.recipient_email
        message["Subject"] = subject
        message.set_content(body)

        for file_path in attachments:
            p = Path(file_path)
            if not p.exists() or not p.is_file():
                continue
            data = p.read_bytes()
            message.add_attachment(
                data,
                maintype="application",
                subtype="octet-stream",
                filename=p.name,
            )

        try:
            with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=20) as server:
                if cfg.use_tls:
                    server.starttls()
                if cfg.username:
                    server.login(cfg.username, cfg.password)
                server.send_message(message)
        except Exception as exc:  # noqa: BLE001
            self._append_audit(
                "email_sent",
                {
                    "success": False,
                    "recipient": cfg.recipient_email,
                    "subject": subject,
                    "attachments": len(attachments),
                    "error": str(exc),
                },
            )
            return False

        self._append_audit(
            "email_sent",
            {
                "success": True,
                "recipient": cfg.recipient_email,
                "subject": subject,
                "attachments": len(attachments),
            },
        )
        return True