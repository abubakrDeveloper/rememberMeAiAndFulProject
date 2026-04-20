from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import BehaviorConfig
from .types import BehaviorState, Incident, PersonRecord

# MediaPipe Face Mesh landmark indices for EAR calculation
# Right eye: [33, 160, 158, 133, 153, 144]
# Left eye:  [362, 385, 387, 263, 373, 380]
_RIGHT_EYE = [33, 160, 158, 133, 153, 144]
_LEFT_EYE = [362, 385, 387, 263, 373, 380]


def _ear(landmarks: np.ndarray, indices: List[int]) -> float:
    """Eye Aspect Ratio — high when open, near zero when closed."""
    p = landmarks[indices]
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])
    if h < 1e-6:
        return 0.0
    return float((v1 + v2) / (2.0 * h))


def _head_yaw(landmarks: np.ndarray) -> float:
    """Approximate head yaw in degrees from nose-tip vs midpoint of ears."""
    nose = landmarks[1]
    left_ear = landmarks[234]
    right_ear = landmarks[454]
    mid = (left_ear + right_ear) / 2.0
    dx = nose[0] - mid[0]
    ref = np.linalg.norm(left_ear - right_ear) / 2.0
    if ref < 1e-6:
        return 0.0
    return float(math.degrees(math.atan2(dx, ref)))


class BehaviorAnalyzer:
    def __init__(self, cfg: BehaviorConfig) -> None:
        self.cfg = cfg
        self._states: Dict[str, BehaviorState] = {}

    def _can_fire(self, state: BehaviorState, incident_type: str, now_ts: float) -> bool:
        last = state.last_incident_at.get(incident_type, 0.0)
        if now_ts - last >= self.cfg.incident_cooldown_seconds:
            state.last_incident_at[incident_type] = now_ts
            return True
        return False

    def _smoothed_label(self, state: BehaviorState, raw_label: str) -> str:
        window = max(1, int(self.cfg.smoothing_window_frames))
        required = max(1, min(int(self.cfg.smoothing_required_frames), window))
        state.recent_labels.append(raw_label)
        if len(state.recent_labels) > window:
            state.recent_labels = state.recent_labels[-window:]
        if raw_label == "focused":
            return "focused"
        label_count = sum(1 for item in state.recent_labels if item == raw_label)
        return raw_label if label_count >= required else "focused"

    def analyze(
        self,
        key: str,
        person: PersonRecord,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        landmarks: Optional[np.ndarray],
        now_ts: float,
    ) -> Tuple[str, List[Incident]]:
        """Classify behavior from 468 Face Mesh landmarks.

        Args:
            landmarks: (468, 3) array of normalised x/y/z from MediaPipe, or None
                       if no mesh was detected for this face.
        """
        state = self._states.setdefault(key, BehaviorState())
        incidents: List[Incident] = []

        # --- compute signals from landmarks ---
        eyes_open = True
        looking_away = False

        if landmarks is not None:
            ear_left = _ear(landmarks, _LEFT_EYE)
            ear_right = _ear(landmarks, _RIGHT_EYE)
            avg_ear = (ear_left + ear_right) / 2.0
            eyes_open = avg_ear >= self.cfg.ear_threshold

            yaw = _head_yaw(landmarks)
            looking_away = abs(yaw) > self.cfg.yaw_threshold_deg
        else:
            # No mesh → treat as not visible (don't fire false alerts)
            state.current_label = "unknown"
            return state.current_label, incidents

        # --- timing logic ---
        if eyes_open and not looking_away:
            state.no_eye_since = None
            state.still_since = None
            state.current_label = self._smoothed_label(state, "focused")
            return state.current_label, incidents

        if state.no_eye_since is None:
            state.no_eye_since = now_ts
        closed_duration = now_ts - state.no_eye_since

        # Classify
        raw_label = "focused"
        raw_confidence = 0.0

        if not eyes_open and closed_duration >= self.cfg.sleeping_seconds:
            raw_label = "sleeping"
            raw_confidence = min(1.0, closed_duration / max(self.cfg.sleeping_seconds, 1.0))
        elif looking_away and closed_duration >= self.cfg.distracted_seconds:
            raw_label = "distracted"
            raw_confidence = min(1.0, closed_duration / max(self.cfg.distracted_seconds, 1.0))
        elif not eyes_open and closed_duration >= self.cfg.distracted_seconds:
            raw_label = "distracted"
            raw_confidence = min(1.0, closed_duration / max(self.cfg.distracted_seconds, 1.0))

        state.current_label = self._smoothed_label(state, raw_label)

        if state.current_label in {"sleeping", "distracted"}:
            if self._can_fire(state, state.current_label, now_ts):
                incidents.append(
                    Incident(
                        timestamp=datetime.now(),
                        incident_type=state.current_label,
                        person_id=person.person_id,
                        person_name=person.name,
                        role=person.role,
                        confidence=raw_confidence,
                        track_id=track_id,
                        bbox=bbox,
                    )
                )

        return state.current_label, incidents