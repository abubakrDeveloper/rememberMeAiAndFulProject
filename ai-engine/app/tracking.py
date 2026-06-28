from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import Track

Bbox = Tuple[int, int, int, int]  # (x, y, w, h)
ScoredDet = Tuple[Bbox, float]    # ((x, y, w, h), score)


def _bbox_to_z(bbox: Bbox) -> np.ndarray:
    """(x,y,w,h) -> measurement [cx, cy, area, aspect]."""
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    area = float(max(w, 1) * max(h, 1))
    aspect = w / float(h) if h > 0 else 1.0
    return np.array([cx, cy, area, aspect], dtype=float)


def _z_to_bbox(state: np.ndarray) -> Bbox:
    """state [cx, cy, area, aspect, ...] -> (x,y,w,h) with non-negative size."""
    cx, cy, area, aspect = float(state[0]), float(state[1]), float(state[2]), float(state[3])
    area = max(area, 1.0)
    aspect = max(aspect, 1e-3)
    w = np.sqrt(area * aspect)
    h = area / w if w > 0 else 0.0
    return (int(round(cx - w / 2.0)), int(round(cy - h / 2.0)), int(round(w)), int(round(h)))


def _iou(a: Bbox, b: Bbox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx)
    iy = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    iw = max(0, ix2 - ix)
    ih = max(0, iy2 - iy)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class _KalmanBoxTracker:
    """Constant-velocity Kalman filter on [cx, cy, area, aspect] (classic SORT model)."""

    def __init__(self, bbox: Bbox, track_id: int, now_ts: float) -> None:
        # State: [cx, cy, area, aspect, vcx, vcy, varea]
        self.x = np.zeros((7, 1), dtype=float)
        self.x[:4, 0] = _bbox_to_z(bbox)

        self.F = np.eye(7)
        for i in range(3):
            self.F[i, i + 4] = 1.0  # position += velocity (dt = 1 frame)
        self.H = np.zeros((4, 7))
        for i in range(4):
            self.H[i, i] = 1.0

        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0  # high initial velocity uncertainty
        self.Q = np.eye(7)
        self.Q[4:, 4:] *= 0.01
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0  # area/aspect measurements are noisier

        self.time_since_update = 0
        self.hits = 1
        # The Track object is owned here and returned by reference so that identity
        # fields (person_id/person_name) set by the caller persist across frames and
        # survive short occlusions.
        self.track = Track(
            track_id=track_id,
            bbox=_z_to_bbox(self.x[:, 0]),
            last_seen_ts=now_ts,
        )

    def predict(self) -> Bbox:
        # Guard against negative area blowing up the model.
        if self.x[2, 0] + self.x[6, 0] <= 0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        self.track.bbox = _z_to_bbox(self.x[:, 0])
        return self.track.bbox

    def update(self, bbox: Bbox, now_ts: float) -> None:
        z = _bbox_to_z(bbox).reshape(4, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1
        self.track.bbox = _z_to_bbox(self.x[:, 0])
        self.track.last_seen_ts = now_ts


class ByteTrackTracker:
    """SORT/ByteTrack-style tracker: Kalman prediction + two-stage IoU association.

    Public surface matches the old CentroidTracker so the processor is unaffected:
    `update(detections, now_ts) -> List[Track]`. Detections carry a confidence score
    so high-score detections are matched first and low-score ones recover coasted
    tracks (the ByteTrack idea), improving identity stability through occlusions.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        low_iou_threshold: float = 0.5,
        max_age_seconds: float = 2.0,
        high_score: float = 0.6,
        low_score: float = 0.1,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.low_iou_threshold = low_iou_threshold
        self.max_age_seconds = max_age_seconds
        self.high_score = high_score
        self.low_score = low_score
        self._trackers: List[_KalmanBoxTracker] = []
        self._next_id = 1

    @staticmethod
    def _associate(
        det_boxes: Sequence[Bbox],
        trackers: Sequence[_KalmanBoxTracker],
        iou_threshold: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Hungarian IoU matching. Returns (matches, unmatched_dets, unmatched_trks)."""
        if not det_boxes or not trackers:
            return [], list(range(len(det_boxes))), list(range(len(trackers)))

        iou_matrix = np.zeros((len(det_boxes), len(trackers)), dtype=float)
        for d, db in enumerate(det_boxes):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = _iou(db, trk.track.bbox)

        # Hungarian on cost = -IoU (maximize overlap).
        det_idx, trk_idx = linear_sum_assignment(-iou_matrix)

        matches: List[Tuple[int, int]] = []
        matched_dets = set()
        matched_trks = set()
        for d, t in zip(det_idx, trk_idx):
            if iou_matrix[d, t] >= iou_threshold:
                matches.append((d, t))
                matched_dets.add(d)
                matched_trks.add(t)
        unmatched_dets = [d for d in range(len(det_boxes)) if d not in matched_dets]
        unmatched_trks = [t for t in range(len(trackers)) if t not in matched_trks]
        return matches, unmatched_dets, unmatched_trks

    def update(self, detections: Sequence[ScoredDet], now_ts: float) -> List[Track]:
        # Predict all existing tracks forward one step.
        for trk in self._trackers:
            trk.predict()

        high = [(b, s) for (b, s) in detections if s >= self.high_score]
        low = [(b, s) for (b, s) in detections if self.low_score <= s < self.high_score]

        # --- Stage 1: high-score detections vs all tracks ---
        high_boxes = [b for b, _ in high]
        matches, un_dets_high, un_trks = self._associate(
            high_boxes, self._trackers, self.iou_threshold
        )
        for d, t in matches:
            self._trackers[t].update(high_boxes[d], now_ts)

        # --- Stage 2: low-score detections vs tracks still unmatched ---
        remaining = [self._trackers[t] for t in un_trks]
        low_boxes = [b for b, _ in low]
        matches2, _un_dets_low, un_trks2 = self._associate(
            low_boxes, remaining, self.low_iou_threshold
        )
        for d, t in matches2:
            remaining[t].update(low_boxes[d], now_ts)

        # --- New tracks from unmatched high-score detections ---
        for d in un_dets_high:
            trk = _KalmanBoxTracker(high_boxes[d], self._next_id, now_ts)
            self._next_id += 1
            self._trackers.append(trk)

        # --- Remove stale tracks (coasted longer than max_age) ---
        self._trackers = [
            trk for trk in self._trackers
            if now_ts - trk.track.last_seen_ts <= self.max_age_seconds
        ]

        # Return tracks that were updated/created this frame (time_since_update == 0),
        # de-duplicated and in stable id order.
        seen_ids = set()
        results: List[Track] = []
        for trk in self._trackers:
            if trk.time_since_update == 0 and trk.track.track_id not in seen_ids:
                seen_ids.add(trk.track.track_id)
                results.append(trk.track)
        return results
