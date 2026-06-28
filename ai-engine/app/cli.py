from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
from pathlib import Path

from .config import AppConfig, load_config
from .processor import ClassroomMonitorApp, _YUNET_MODEL_PATH

_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

logger = logging.getLogger(__name__)


def _ensure_yunet_model() -> None:
    model_path = Path(_YUNET_MODEL_PATH)
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading YuNet face detection model to %s ...", model_path)
    try:
        urllib.request.urlretrieve(_YUNET_URL, model_path)
        logger.info("Download complete.")
    except Exception as exc:
        logger.error("Could not download YuNet model: %s", exc)
        logger.error("Please manually download it from:\n  %s", _YUNET_URL)
        logger.error("and place it at: %s", model_path)
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classroom CCTV monitoring and attendance system")
    parser.add_argument("--config", default="config.json", help="Path to JSON config")
    parser.add_argument("--mode", choices=["file", "rtsp", "webcam"], help="Input mode override")
    parser.add_argument("--source", help="Input source override (file path, RTSP URL, or webcam index)")
    parser.add_argument("--display", action="store_true", help="Force display window on")
    parser.add_argument("--no-display", action="store_true", help="Force display window off")
    parser.add_argument("--stop-after", type=float, default=None, help="Optional stop after N seconds")
    parser.add_argument("--general", action="store_true", help="General face recognition mode (skips classroom-only features)")
    parser.add_argument("--roster", help="Path to directory of face images for general mode (default: roster/people)")
    return parser


def apply_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.mode:
        cfg.source.mode = args.mode
    if args.source:
        cfg.source.source = args.source
    if args.display:
        cfg.runtime.display = True
    if args.no_display:
        cfg.runtime.display = False
    if args.stop_after is not None:
        cfg.runtime.stop_after_seconds = max(0.0, args.stop_after)
    if args.general:
        cfg.runtime.app_mode = "general"
    if args.roster:
        cfg.paths.roster_dir = args.roster
    return cfg


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = build_parser()
    args = parser.parse_args()

    _ensure_yunet_model()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    app = ClassroomMonitorApp(cfg)
    app.run()