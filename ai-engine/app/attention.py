"""Attention and alertness classification from InsightFace landmarks and head pose.

Uses the 3D head-pose angles (pitch, yaw, roll) that InsightFace already computes
and the eye-aspect-ratio (EAR) from 2D landmarks to classify each detected face
into one of three states:

- attentive   : facing roughly forward, eyes open
- distracted  : looking significantly away (phone, neighbor, window)
- sleeping    : head pitched down AND eyes closed for sustained frames
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Thresholds (tunable via CLI or config later)
# ---------------------------------------------------------------------------

@dataclass
class AttentionThresholds:
    # Head pose (degrees)
    yaw_limit: float = 35.0          # beyond this = looking away
    pitch_down_limit: float = -25.0   # below this = head down (sleeping candidate)
    pitch_up_limit: float = 30.0      # above this = looking up (distracted)

    # Eye aspect ratio
    ear_closed: float = 0.18          # below this = eyes likely closed
    ear_open: float = 0.24            # above this = eyes definitely open

    # Sustained sleeping requires N consecutive "sleep candidate" frames
    sleep_streak_threshold: int = 8


# ---------------------------------------------------------------------------
# Eye Aspect Ratio (EAR)
# ---------------------------------------------------------------------------

def _eye_aspect_ratio(landmarks: np.ndarray, eye_indices: Sequence[int]) -> float:
    """Compute EAR from 6-point eye landmarks.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    pts = landmarks[list(eye_indices)]
    if len(pts) < 6:
        return 1.0  # default open if not enough points

    vertical_a = np.linalg.norm(pts[1] - pts[5])
    vertical_b = np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])

    if horizontal < 1e-6:
        return 1.0

    return float((vertical_a + vertical_b) / (2.0 * horizontal))


def compute_ear_from_5pt(landmarks_5: np.ndarray) -> float:
    """Approximate EAR from InsightFace 5-point landmarks.

    5-point layout: [left_eye, right_eye, nose, left_mouth, right_mouth]
    With only one point per eye we cannot compute true EAR.
    We use a heuristic: if the vertical distance between eye and nose is
    abnormally small relative to inter-eye distance, eyes are likely closing.
    """
    if landmarks_5 is None or len(landmarks_5) < 5:
        return 1.0

    left_eye = landmarks_5[0]
    right_eye = landmarks_5[1]
    nose = landmarks_5[2]

    inter_eye = np.linalg.norm(left_eye - right_eye)
    if inter_eye < 1e-6:
        return 1.0

    eye_center = (left_eye + right_eye) / 2.0
    eye_nose_dist = np.linalg.norm(eye_center - nose)

    # Ratio of eye-nose distance to inter-eye distance.
    # When head drops / eyes close this shrinks.
    ratio = eye_nose_dist / inter_eye
    # Map to a pseudo-EAR scale (0.0 .. ~0.45)
    return float(np.clip(ratio * 0.55, 0.0, 1.0))


def compute_ear_from_landmarks(landmarks: np.ndarray) -> float:
    """Compute EAR from InsightFace 106-point or 68-point landmarks."""
    n = len(landmarks)

    if n >= 106:
        # InsightFace 106-pt: left eye 33-41, right eye 87-95
        left_indices = [33, 34, 35, 36, 37, 38]
        right_indices = [87, 88, 89, 90, 91, 92]
    elif n >= 68:
        # 68-pt dlib-style
        left_indices = [36, 37, 38, 39, 40, 41]
        right_indices = [42, 43, 44, 45, 46, 47]
    else:
        return 1.0  # not enough landmarks

    left_ear = _eye_aspect_ratio(landmarks, left_indices)
    right_ear = _eye_aspect_ratio(landmarks, right_indices)
    return (left_ear + right_ear) / 2.0


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_attention(
    *,
    pitch: float,
    yaw: float,
    ear: float,
    sleep_streak: int,
    thresholds: AttentionThresholds | None = None,
) -> tuple[str, int]:
    """Return (state, updated_sleep_streak).

    state is one of: "attentive", "distracted", "sleeping"
    """
    if thresholds is None:
        thresholds = AttentionThresholds()

    # Sleeping: head pitched down AND eyes appear closed
    head_down = pitch < thresholds.pitch_down_limit
    eyes_closed = ear < thresholds.ear_closed

    if head_down and eyes_closed:
        new_streak = sleep_streak + 1
        if new_streak >= thresholds.sleep_streak_threshold:
            return "sleeping", new_streak
        # not enough frames yet — still call it distracted
        return "distracted", new_streak

    # Reset sleep streak if head is up or eyes open
    new_streak = 0

    # Distracted: looking too far left/right or too far up
    if abs(yaw) > thresholds.yaw_limit or pitch > thresholds.pitch_up_limit:
        return "distracted", new_streak

    return "attentive", new_streak


def get_head_pose(face) -> tuple[float, float, float]:
    """Extract pitch, yaw, roll from an InsightFace face object.

    InsightFace stores pose as a 3-element array on face.pose.
    Returns (pitch, yaw, roll) in degrees. Defaults to (0, 0, 0) if unavailable.
    """
    pose = getattr(face, "pose", None)
    if pose is not None and len(pose) >= 3:
        return float(pose[0]), float(pose[1]), float(pose[2])
    return 0.0, 0.0, 0.0


def get_ear(face) -> float:
    """Best-effort EAR from whatever landmarks InsightFace provides."""
    # Try detailed landmarks first (106 or 68 point)
    landmark_2d_106 = getattr(face, "landmark_2d_106", None)
    if landmark_2d_106 is not None:
        return compute_ear_from_landmarks(landmark_2d_106)

    landmark = getattr(face, "landmark", None)
    if landmark is not None and len(landmark) >= 68:
        return compute_ear_from_landmarks(landmark)

    # Fall back to 5-point
    kps = getattr(face, "kps", None)
    if kps is not None:
        return compute_ear_from_5pt(kps)

    return 1.0  # assume open if nothing available
