import argparse
from pathlib import Path

import cv2

from app.sample_scene import create_sample_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic classroom demo video.")
    parser.add_argument("--output", type=Path, default=Path("data/sample/classroom_demo.mp4"))
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_frames = max(1, args.seconds * args.fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, float(args.fps), (args.width, args.height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video file: {args.output}")

    for frame_index in range(total_frames):
        frame, _ = create_sample_frame(frame_index, width=args.width, height=args.height)
        writer.write(frame)

    writer.release()
    print(f"Sample video generated at: {args.output}")


if __name__ == "__main__":
    main()
