from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple


@dataclass
class PersonRecord:
    person_id: str
    name: str
    role: str  # student | teacher


@dataclass
class AttendanceEntry:
    person_id: str
    name: str
    role: str
    status: str
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    confidence: float = 0.0
    confidence_band: str = "low"


@dataclass
class Incident:
    timestamp: datetime
    incident_type: str
    person_id: str
    person_name: str
    role: str
    confidence: float
    track_id: int
    bbox: Tuple[int, int, int, int]
    snapshot_path: str = ""


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    last_seen_ts: float
    person_id: str = ""
    person_name: str = ""


@dataclass
class FrameAnalysis:
    annotated_boxes: List[Tuple[int, int, int, int, str]]
    incidents: List[Incident]