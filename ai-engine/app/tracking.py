from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .types import Track


class CentroidTracker:
    def __init__(self, max_distance: float = 80.0, max_track_age: float = 2.0) -> None:
        self.max_distance = max_distance
        self.max_track_age = max_track_age
        self._next_id = 1
        self._tracks: Dict[int, Track] = {}

    @staticmethod
    def _center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x, y, w, h = bbox
        return x + w // 2, y + h // 2

    @staticmethod
    def _distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def update(self, detections: List[Tuple[int, int, int, int]], now_ts: float) -> List[Track]:
        assigned_track_ids = set()
        results: List[Track] = []

        track_items = list(self._tracks.items())

        for det_bbox in detections:
            det_center = self._center(det_bbox)
            best_tid = None
            best_dist = self.max_distance

            for tid, track in track_items:
                if tid in assigned_track_ids:
                    continue
                if now_ts - track.last_seen_ts > self.max_track_age:
                    continue
                track_center = self._center(track.bbox)
                dist = self._distance(det_center, track_center)
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid is None:
                best_tid = self._next_id
                self._next_id += 1
                self._tracks[best_tid] = Track(track_id=best_tid, bbox=det_bbox, last_seen_ts=now_ts)
            else:
                self._tracks[best_tid].bbox = det_bbox
                self._tracks[best_tid].last_seen_ts = now_ts

            assigned_track_ids.add(best_tid)
            results.append(self._tracks[best_tid])

        stale_ids = [tid for tid, t in self._tracks.items() if now_ts - t.last_seen_ts > self.max_track_age]
        for tid in stale_ids:
            self._tracks.pop(tid, None)

        return results