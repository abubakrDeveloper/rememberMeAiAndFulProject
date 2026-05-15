from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from .config import AppConfig, load_config
from .processor import ClassroomMonitorApp, _YUNET_MODEL_PATH

_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def _ensure_yunet_model() -> None:
    model_path = Path(_YUNET_MODEL_PATH)
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading YuNet face detection model to {model_path} ...")
    try:
        urllib.request.urlretrieve(_YUNET_URL, model_path)
        print("Download complete.")
    except Exception as exc:
        print(f"ERROR: Could not download YuNet model: {exc}", file=sys.stderr)
        print(f"Please manually download it from:\n  {_YUNET_URL}", file=sys.stderr)
        print(f"and place it at: {model_path}", file=sys.stderr)
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
    parser = build_parser()
    args = parser.parse_args()

    _ensure_yunet_model()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)
    cfg = apply_overrides(cfg, args)

    app = ClassroomMonitorApp(cfg)
    app.run()