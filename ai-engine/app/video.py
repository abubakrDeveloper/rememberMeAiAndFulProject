from __future__ import annotations

import threading
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
        # Async capture (live/webcam only)
        self._lock = threading.Lock()
        self._buffer: Optional[np.ndarray] = None
        self._stopped: bool = False
        self._thread: Optional[threading.Thread] = None

    def _resolve_source(self):
        if self.cfg.mode == "webcam":
            try:
                return int(self.cfg.source) if self.cfg.source else 0
            except ValueError:
                return 0
        return self.cfg.source

    def _open_cap(self) -> None:
        """Open the underlying VideoCapture without starting a thread."""
        src = self._resolve_source()
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            self.status = "offline"
            self.last_error = f"Unable to open source: {self.cfg.source}"
            raise RuntimeError(f"Unable to open source: {self.cfg.source}")
        self.status = "online"
        self.last_error = ""

    def open(self) -> None:
        """Open capture and, for live/webcam streams, start a background read thread."""
        self._open_cap()
        if self.cfg.mode != "file":
            self._stopped = False
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

    def _capture_loop(self) -> None:
        """Background thread: continuously fills _buffer with the latest camera frame.

        Handles reconnection automatically so the main processing loop never blocks
        on network I/O for RTSP or webcam sources.
        """
        while not self._stopped:
            if self.cap is None:
                time.sleep(0.05)
                continue
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._buffer = frame
                self.last_frame_ts = time.time()
                self.status = "online"
                self.last_error = ""
            else:
                self.status = "offline"
                self.last_error = "Live stream read failed, reconnecting"
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                self.reconnect_attempts += 1
                self.last_reconnect_ts = time.time()
                time.sleep(max(0.5, self.cfg.reconnect_seconds))
                try:
                    self._open_cap()
                except RuntimeError:
                    self.last_error = "Reconnect attempt failed"
                    time.sleep(1.0)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the latest frame.

        File mode: synchronous — each frame delivered once in order.
        Live/webcam: returns immediately from the background capture buffer,
        eliminating I/O stall from the main processing loop.
        """
        if self.cfg.mode == "file":
            if self.cap is None:
                return False, None
            ok, frame = self.cap.read()
            if ok:
                self.last_frame_ts = time.time()
                self.status = "online"
                self.last_error = ""
                return True, frame
            self.status = "offline"
            self.last_error = "Video file ended or frame read failed"
            return False, None

        # Live/webcam — non-blocking: return whatever is in the capture buffer
        with self._lock:
            frame = self._buffer
        if frame is None:
            return False, None
        if self._stopped:
            return False, None
        return True, frame

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
        self._stopped = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None
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