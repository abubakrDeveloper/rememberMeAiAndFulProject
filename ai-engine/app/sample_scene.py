from __future__ import annotations

from typing import List, TypedDict

import cv2
import numpy as np


class SampleTrack(TypedDict):
    track_id: int
    bbox: tuple[int, int, int, int]
    identity: str | None
    score: float
    attention_state: str


def _draw_background(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    for y in range(height):
        intensity = int(210 - (80 * y / max(1, height - 1)))
        frame[y, :, :] = (intensity, intensity + 10, intensity + 25)

    cv2.rectangle(frame, (30, 40), (width - 30, 260), (190, 210, 230), -1)
    cv2.rectangle(frame, (0, 300), (width, height), (135, 155, 170), -1)
    cv2.putText(
        frame,
        "CLASSROOM - DEMO MODE",
        (45, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (35, 55, 75),
        2,
        cv2.LINE_AA,
    )


def _make_bbox(x: int, y: int, width: int, height: int, limit_w: int, limit_h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(limit_w - 2, x))
    y1 = max(0, min(limit_h - 2, y))
    x2 = max(x1 + 1, min(limit_w - 1, x + width))
    y2 = max(y1 + 1, min(limit_h - 1, y + height))
    return (x1, y1, x2, y2)


def _draw_person(frame: np.ndarray, bbox: tuple[int, int, int, int], color: tuple[int, int, int], tag: str) -> None:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    body_top = y1 + 28

    cv2.circle(frame, (cx, y1 + 14), 12, color, -1)
    cv2.rectangle(frame, (x1 + 10, body_top), (x2 - 10, y2), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (45, 45, 45), 2)
    cv2.putText(
        frame,
        tag,
        (x1, max(20, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (35, 35, 35),
        1,
        cv2.LINE_AA,
    )


def _pick_attention(frame_index: int, track_id: int) -> str:
    """Cycle through attention states for demo variety."""
    cycle = (frame_index + track_id * 47) % 300
    if cycle < 180:
        return "attentive"
    if cycle < 240:
        return "distracted"
    return "sleeping"


def _attention_tag_color(state: str) -> tuple[int, int, int]:
    if state == "sleeping":
        return (180, 50, 255)
    if state == "distracted":
        return (0, 165, 255)
    return (80, 165, 70)


def create_sample_frame(frame_index: int, width: int = 960, height: int = 540) -> tuple[np.ndarray, List[SampleTrack]]:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    _draw_background(frame)

    tracks: List[SampleTrack] = []
    stride = max(180, width - 220)

    students = [
        (1, "student_001", 95, 3),
        (2, "student_002", 215, 4),
        (3, "student_003", 335, 5),
    ]

    for track_id, name, y, speed in students:
        x = 40 + ((frame_index * speed + track_id * 65) % stride)
        bbox = _make_bbox(x, y, 90, 170, width, height)
        att = _pick_attention(frame_index, track_id)
        tracks.append(
            {
                "track_id": track_id,
                "bbox": bbox,
                "identity": name,
                "score": 0.87 + (track_id * 0.02),
                "attention_state": att,
            }
        )
        color = _attention_tag_color(att)
        _draw_person(frame, bbox, color, f"{name} [{att}]")

    cycle = frame_index % 360
    if 95 <= cycle <= 275:
        x = width - 140 - ((frame_index * 6) % max(220, width - 180))
        bbox = _make_bbox(x, 250, 95, 175, width, height)
        tracks.append(
            {
                "track_id": 99,
                "bbox": bbox,
                "identity": None,
                "score": 0.31,
                "attention_state": "distracted",
            }
        )
        _draw_person(frame, bbox, (65, 65, 215), "unknown")

    return frame, tracks
