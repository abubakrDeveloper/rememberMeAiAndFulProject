from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import SourceConfig


class VideoInput:
    def __init__(self, source_cfg: SourceConfig) -> None:
        self.cfg = source_cfg
        self.cap: Optional[cv2.VideoCapture] = None
        self.reconnect_attempts = 0
        self.reconnect_successes = 0
        self.last_frame_ts = 0.0
        self.last_reconnect_ts = 0.0
        self.last_error = ""
        self.status = "offline"

    def _resolve_source(self):
        if self.cfg.mode == "webcam":
            try:
                return int(self.cfg.source) if self.cfg.source else 0
            except ValueError:
                return 0
        return self.cfg.source

    def open(self) -> None:
        src = self._resolve_source()
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            self.status = "offline"
            self.last_error = f"Unable to open source: {self.cfg.source}"
            raise RuntimeError(f"Unable to open source: {self.cfg.source}")
        self.status = "online"
        self.last_error = ""

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None:
            self.open()

        assert self.cap is not None
        ok, frame = self.cap.read()
        if ok:
            self.last_frame_ts = time.time()
            self.status = "online"
            self.last_error = ""
            return True, frame

        if self.cfg.mode == "file":
            self.status = "offline"
            self.last_error = "Video file ended or frame read failed"
            return False, None

        self.status = "offline"
        self.last_error = "Live stream read failed, reconnecting"
        self.release()
        self.reconnect_attempts += 1
        self.last_reconnect_ts = time.time()
        time.sleep(max(0.5, self.cfg.reconnect_seconds))
        try:
            self.open()
        except RuntimeError:
            self.last_error = "Reconnect attempt failed"
            return False, None

        assert self.cap is not None
        ok, frame = self.cap.read()
        if ok:
            self.reconnect_successes += 1
            self.last_frame_ts = time.time()
            self.status = "online"
            self.last_error = ""
            return True, frame

        self.status = "offline"
        self.last_error = "Reconnect opened but frame read still failed"
        return False, None

    def fps(self) -> float:
        if self.cap is None:
            return 0.0
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        return fps if fps > 0 else 0.0

    def frame_size(self) -> Tuple[int, int]:
        if self.cap is None:
            return 0, 0
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.status != "error":
            self.status = "offline"

    def get_health(self) -> dict:
        return {
            "status": self.status,
            "last_frame_ts": self.last_frame_ts,
            "last_reconnect_ts": self.last_reconnect_ts,
            "reconnect_attempts": self.reconnect_attempts,
            "reconnect_successes": self.reconnect_successes,
            "last_error": self.last_error,
            "source_mode": self.cfg.mode,
            "source": self.cfg.source,
        }