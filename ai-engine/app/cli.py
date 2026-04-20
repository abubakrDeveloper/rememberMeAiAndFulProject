from __future__ import annotations

import argparse

from .config import AppConfig, load_config
from .processor import ClassroomMonitorApp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classroom CCTV monitoring and attendance system")
    parser.add_argument("--config", default="config.json", help="Path to JSON config")
    parser.add_argument("--mode", choices=["file", "rtsp", "webcam"], help="Input mode override")
    parser.add_argument("--source", help="Input source override (file path, RTSP URL, or webcam index)")
    parser.add_argument("--display", action="store_true", help="Force display window on")
    parser.add_argument("--no-display", action="store_true", help="Force display window off")
    parser.add_argument("--stop-after", type=float, default=None, help="Optional stop after N seconds")
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
    return cfg


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    app = ClassroomMonitorApp(cfg)
    app.run()