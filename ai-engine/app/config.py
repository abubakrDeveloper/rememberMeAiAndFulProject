from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class SourceConfig:
    mode: str = "file"  # file | rtsp | webcam
    source: str = ""
    reconnect_seconds: float = 3.0


@dataclass
class DetectionConfig:
    min_detection_confidence: float = 0.5
    max_faces: int = 20
    # Direct cosine-similarity gate for a face to count as a match (higher = stricter).
    # NOTE: replaces the old inverted `recognition_tolerance` (threshold = 1 - tolerance);
    # an old config carrying `recognition_tolerance` is auto-converted on load.
    recognition_threshold: float = 0.35
    # A match must beat the runner-up by at least this cosine margin to be accepted.
    recognition_margin: float = 0.15
    # Minimum match score required before a confirmed identity counts toward attendance.
    attend_min_score: float = 0.65
    # Number of agreeing recognition votes before a track's identity is locked.
    votes_needed: int = 10

    def __post_init__(self) -> None:
        if not 0.0 <= self.recognition_threshold <= 1.0:
            raise ValueError(f"recognition_threshold must be in [0,1], got {self.recognition_threshold}")
        if not 0.0 <= self.recognition_margin <= 1.0:
            raise ValueError(f"recognition_margin must be in [0,1], got {self.recognition_margin}")
        if not 0.0 <= self.attend_min_score <= 1.0:
            raise ValueError(f"attend_min_score must be in [0,1], got {self.attend_min_score}")
        if self.votes_needed < 1:
            raise ValueError(f"votes_needed must be >= 1, got {self.votes_needed}")
        if self.max_faces < 1:
            raise ValueError(f"max_faces must be >= 1, got {self.max_faces}")


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
    # Recognition/detection pacing (seconds). Promoted from hardcoded loop constants.
    recognize_interval_seconds: float = 30.0      # full recognition pass cadence per track
    tiled_det_interval_seconds: float = 2.0       # expensive tiled detection cadence
    unconfirmed_recog_interval_seconds: float = 0.5  # retry cadence for unlocked tracks
    stale_box_ttl_seconds: float = 5.0            # how long a last-seen box lingers
    # How long a track (and its locked identity) survives with no fresh detection —
    # e.g. while a recognized person turns their head away. Keep aligned with
    # stale_box_ttl_seconds so the dimmed name lingers exactly as long as the track.
    track_max_age_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.max_fps < 0:
            raise ValueError(f"max_fps must be >= 0, got {self.max_fps}")
        if self.app_mode not in {"classroom", "general"}:
            raise ValueError(f"app_mode must be 'classroom' or 'general', got {self.app_mode!r}")
        if self.track_max_age_seconds <= 0:
            raise ValueError(f"track_max_age_seconds must be > 0, got {self.track_max_age_seconds}")


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


def _normalize_detection(detection: Dict[str, Any]) -> Dict[str, Any]:
    """Back-compat: convert a legacy `recognition_tolerance` into `recognition_threshold`."""
    detection = dict(detection)
    if "recognition_tolerance" in detection and "recognition_threshold" not in detection:
        tol = float(detection.pop("recognition_tolerance"))
        detection["recognition_threshold"] = max(0.0, min(1.0, 1.0 - tol))
        logger.warning(
            "config: 'recognition_tolerance'=%.2f is deprecated; converted to "
            "'recognition_threshold'=%.2f. Please update config.json.",
            tol, detection["recognition_threshold"],
        )
    return detection


def _warn_unknown_keys(config_dict: Dict[str, Any]) -> None:
    known_sections = {
        "source", "detection", "behavior", "attendance",
        "runtime", "paths", "session", "admin_email",
    }
    for key in config_dict:
        if key not in known_sections:
            logger.warning("config: ignoring unknown top-level section %r", key)


def _to_dataclass(config_dict: Dict[str, Any]) -> AppConfig:
    return AppConfig(
        source=SourceConfig(**config_dict.get("source", {})),
        detection=DetectionConfig(**_normalize_detection(config_dict.get("detection", {}))),
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

    _warn_unknown_keys(user_config)
    # Normalize the user's detection block before merging so a legacy
    # `recognition_tolerance` key never collides with the default `recognition_threshold`.
    if "detection" in user_config:
        user_config["detection"] = _normalize_detection(user_config["detection"])

    merged = _merge_dict(default_dict, user_config)
    return _to_dataclass(merged)