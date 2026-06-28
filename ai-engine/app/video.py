from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import SourceConfig

logger = logging.getLogger(__name__)


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
        self._new_frame = threading.Event()
        self._stopped: bool = False
        self._thread: Optional[threading.Thread] = None
        # Staleness watchdog: warn once when the feed goes stale, reset on recovery.
        self._stale_warned: bool = False

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
        if self.cfg.mode == "rtsp":
            # Force TCP transport — avoids the corrupted/green frames seen with the
            # default UDP on real CCTV networks.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            # 5s open/read timeouts so a stalled link fails fast (and the reconnect
            # loop kicks in) instead of hanging on OpenCV's 30s default. These must
            # be passed as construction params, not via the env var above.
            _RTSP_TIMEOUT_MS = 5000
            self.cap = cv2.VideoCapture(
                src,
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, _RTSP_TIMEOUT_MS,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, _RTSP_TIMEOUT_MS,
                ],
            )
        else:
            self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            self.status = "offline"
            self.last_error = f"Unable to open source: {self.cfg.source}"
            raise RuntimeError(f"Unable to open source: {self.cfg.source}")
        if self.cfg.mode == "rtsp":
            # Keep only the latest frame in the driver buffer to minimize latency.
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.status = "online"
        self.last_error = ""

    def open(self) -> None:
        """Open capture and, for live/webcam streams, start a background read thread.

        For live modes an initial connection failure is non-fatal: the capture
        thread is started anyway and its reconnect logic keeps retrying until the
        camera appears, so a brief blip at launch never crashes the app. File mode
        still raises on failure (a missing file is a real error).
        """
        if self.cfg.mode == "file":
            self._open_cap()
            return

        try:
            self._open_cap()
        except RuntimeError:
            self.status = "offline"
            self.last_error = "Initial connection failed, retrying in background"
            logger.warning(
                "Could not open live source on first attempt; capture thread will "
                "keep reconnecting: %s", self.cfg.source,
            )
        self._stopped = False
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        """Background thread: continuously fills _buffer with the latest camera frame.

        Handles reconnection automatically so the main processing loop never blocks
        on network I/O for RTSP or webcam sources.
        """
        stale_after = max(5.0, 2.0 * self.cfg.reconnect_seconds)
        while not self._stopped:
            # Watchdog: warn once if the feed has been stale beyond the threshold,
            # so an operator sees a stalled camera during a live session.
            if (
                self.last_frame_ts > 0.0
                and not self._stale_warned
                and time.time() - self.last_frame_ts > stale_after
            ):
                logger.warning(
                    "No fresh frame for %.1fs — live feed appears stalled.",
                    time.time() - self.last_frame_ts,
                )
                self._stale_warned = True
            if self.cap is None:
                time.sleep(0.05)
                continue
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._buffer = frame
                    self._new_frame.set()
                self.last_frame_ts = time.time()
                self.status = "online"
                self.last_error = ""
                self._stale_warned = False
            else:
                self.status = "offline"
                self.last_error = "Live stream read failed, reconnecting"
                logger.warning("Live stream read failed, attempting reconnect...")
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                self.reconnect_attempts += 1
                self.last_reconnect_ts = time.time()
                time.sleep(max(0.5, self.cfg.reconnect_seconds))
                try:
                    self._open_cap()
                    self.reconnect_successes += 1
                    logger.info("Reconnect successful (attempt #%d)", self.reconnect_attempts)
                except RuntimeError:
                    self.last_error = "Reconnect attempt failed"
                    logger.error("Reconnect attempt #%d failed", self.reconnect_attempts)
                    time.sleep(1.0)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the latest frame.

        File mode: synchronous — each frame delivered once in order.
        Live/webcam: returns the most recent frame from the background
        capture buffer, then clears it so the same frame is never
        processed twice.
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

        # Live/webcam — non-blocking: return the latest frame and clear buffer
        # so duplicate processing is avoided.
        with self._lock:
            frame = self._buffer
            self._buffer = None
            self._new_frame.clear()
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