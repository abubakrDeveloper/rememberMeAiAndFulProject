from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class SourceConfig:
    mode: str = "file"  # file | rtsp | webcam
    source: str = ""
    reconnect_seconds: float = 3.0


@dataclass
class DetectionConfig:
    min_detection_confidence: float = 0.5
    max_faces: int = 20
    recognition_tolerance: float = 0.5


@dataclass
class BehaviorConfig:
    distracted_seconds: float = 4.0
    sleeping_seconds: float = 8.0
    incident_cooldown_seconds: float = 20.0
    ear_threshold: float = 0.21
    yaw_threshold_deg: float = 30.0
    smoothing_window_frames: int = 8
    smoothing_required_frames: int = 5


@dataclass
class AttendanceConfig:
    min_confirm_frames: int = 5
    absent_if_not_seen: bool = True


@dataclass
class RuntimeConfig:
    max_fps: float = 10.0
    display: bool = True
    save_annotated_video: bool = False
    stop_after_seconds: float = 0.0
    app_mode: str = "classroom"  # classroom | general


@dataclass
class PathsConfig:
    roster_students_dir: str = "roster/students"
    roster_teachers_dir: str = "roster/teachers"
    roster_dir: str = "roster/people"
    output_dir: str = "output"
    snapshots_dir: str = "output/snapshots"


@dataclass
class SessionConfig:
    institute_name: str = "Institute"
    class_name: str = "Class"
    teacher_id: str = ""
    admin_name: str = "Admin"


@dataclass
class AdminEmailConfig:
    enabled: bool = False
    sender_email: str = ""
    recipient_email: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    use_tls: bool = True


@dataclass
class AppConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    attendance: AttendanceConfig = field(default_factory=AttendanceConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    admin_email: AdminEmailConfig = field(default_factory=AdminEmailConfig)


def _merge_dict(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _to_dataclass(config_dict: Dict[str, Any]) -> AppConfig:
    return AppConfig(
        source=SourceConfig(**config_dict.get("source", {})),
        detection=DetectionConfig(**config_dict.get("detection", {})),
        behavior=BehaviorConfig(**config_dict.get("behavior", {})),
        attendance=AttendanceConfig(**config_dict.get("attendance", {})),
        runtime=RuntimeConfig(**config_dict.get("runtime", {})),
        paths=PathsConfig(**config_dict.get("paths", {})),
        session=SessionConfig(**config_dict.get("session", {})),
        admin_email=AdminEmailConfig(**config_dict.get("admin_email", {})),
    )


def load_config(config_path: Optional[str] = None) -> AppConfig:
    default_dict = {
        "source": SourceConfig().__dict__,
        "detection": DetectionConfig().__dict__,
        "behavior": BehaviorConfig().__dict__,
        "attendance": AttendanceConfig().__dict__,
        "runtime": RuntimeConfig().__dict__,
        "paths": PathsConfig().__dict__,
        "session": SessionConfig().__dict__,
        "admin_email": AdminEmailConfig().__dict__,
    }

    if not config_path:
        return _to_dataclass(default_dict)

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as fh:
        user_config = json.load(fh)

    merged = _merge_dict(default_dict, user_config)
    return _to_dataclass(merged)