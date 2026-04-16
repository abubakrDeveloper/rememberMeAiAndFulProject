import argparse
import subprocess
import sys
import time
from pathlib import Path


def _run_blocking(command: list[str]) -> int:
    print(f"\nRunning: {' '.join(command)}")
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def _start_process(command: list[str]) -> subprocess.Popen:
    print(f"Starting: {' '.join(command)}")
    return subprocess.Popen(command)


def _stop_process(process: subprocess.Popen | None, name: str) -> None:
    if process is None or process.poll() is not None:
        return

    print(f"Stopping {name}...")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def _choice(prompt: str, options: dict[str, str], default: str) -> str:
    options_label = ", ".join(f"{key}:{label}" for key, label in options.items())
    while True:
        raw = input(f"{prompt} ({options_label}) [{default}]: ").strip().lower()
        value = raw or default
        if value in options:
            return value
        print("Invalid choice. Try again.")


def _ask_yes_no(prompt: str, default: bool = False) -> bool:
    default_mark = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{default_mark}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes"}


def _ask_int(prompt: str, default: int) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        if raw.isdigit():
            return int(raw)
        print("Please enter a valid integer.")


def _build_recognition_args(
    mode: str,
    source: str,
    course_id: str,
    session_id: str,
    show_preview: bool,
    save_output: bool,
    max_frames: int,
) -> list[str]:
    args = [
        sys.executable,
        "-m",
        "app.recognition",
        "--source",
        source,
        "--course-id",
        course_id,
        "--session-id",
        session_id,
    ]

    if mode == "sample":
        args.append("--sample-mode")
    if mode == "video":
        args.extend(["--offline", "--save-output"])
    if save_output and "--save-output" not in args:
        args.append("--save-output")
    if not show_preview:
        args.append("--no-display")
    if max_frames > 0:
        args.extend(["--max-frames", str(max_frames)])

    return args


def run_enrollment() -> int:
    return _run_blocking([sys.executable, "-m", "app.enrollment"])


def run_fake_data() -> int:
    rows = _ask_int("How many fake rows?", 45)
    course_id = input("Course ID [general]: ").strip() or "general"
    session_id = input("Session ID [session_1]: ").strip() or "session_1"
    return _run_blocking(
        [
            sys.executable,
            "-m",
            "app.generate_fake_data",
            "--overwrite",
            "--rows",
            str(rows),
            "--course-id",
            course_id,
            "--session-id",
            session_id,
        ]
    )


def run_recognition_interactive() -> int:
    mode = _choice(
        "Recognition mode",
        {"sample": "Synthetic sample", "camera": "Webcam", "video": "Video file"},
        default="sample",
    )

    if mode == "sample":
        source = "sample"
    elif mode == "camera":
        source = input("Camera source index [0]: ").strip() or "0"
    else:
        source = input("Video file path: ").strip()
        if not source:
            print("Video path is required.")
            return 1
        if not Path(source).exists():
            print(f"Video path not found: {source}")
            return 1

    show_preview = _ask_yes_no("Show preview window?", default=(mode != "video"))
    save_output = _ask_yes_no("Save annotated output video?", default=(mode == "video"))
    max_frames = _ask_int("Max frames (0 = unlimited)", 0)
    course_id = input("Course ID [general]: ").strip() or "general"
    session_id = input("Session ID [session_1]: ").strip() or "session_1"

    command = _build_recognition_args(
        mode=mode,
        source=source,
        course_id=course_id,
        session_id=session_id,
        show_preview=show_preview,
        save_output=save_output,
        max_frames=max_frames,
    )
    return _run_blocking(command)


def run_dashboard() -> int:
    return _run_blocking([sys.executable, "-m", "streamlit", "run", "app/dashboard.py"])


def run_demo_stack_interactive() -> int:
    mode = _choice(
        "Demo mode",
        {"sample": "Synthetic sample", "camera": "Webcam", "video": "Video file"},
        default="sample",
    )

    if mode == "sample":
        source = "sample"
    elif mode == "camera":
        source = input("Camera source index [0]: ").strip() or "0"
    else:
        source = input("Video file path: ").strip()
        if not source:
            print("Video path is required.")
            return 1
        if not Path(source).exists():
            print(f"Video path not found: {source}")
            return 1

    seed_fake_data = _ask_yes_no("Seed fake data first?", default=False)
    show_preview = _ask_yes_no("Show recognition preview window?", default=(mode != "video"))
    course_id = input("Course ID [general]: ").strip() or "general"
    session_id = input("Session ID [session_1]: ").strip() or "session_1"

    if seed_fake_data:
        fake_code = _run_blocking(
            [
                sys.executable,
                "-m",
                "app.generate_fake_data",
                "--overwrite",
                "--rows",
                "28",
                "--course-id",
                course_id,
                "--session-id",
                session_id,
            ]
        )
        if fake_code != 0:
            return fake_code

    recognition_cmd = _build_recognition_args(
        mode=mode,
        source=source,
        course_id=course_id,
        session_id=session_id,
        show_preview=show_preview,
        save_output=(mode == "video"),
        max_frames=0,
    )
    dashboard_cmd = [sys.executable, "-m", "streamlit", "run", "app/dashboard.py"]

    recognition_process: subprocess.Popen | None = None
    dashboard_process: subprocess.Popen | None = None

    try:
        recognition_process = _start_process(recognition_cmd)
        dashboard_process = _start_process(dashboard_cmd)
        print("Dashboard URL: http://localhost:8501")
        print("Press Ctrl+C to stop both processes.")

        while True:
            if recognition_process.poll() is not None:
                print("Recognition process exited.")
                break
            if dashboard_process.poll() is not None:
                print("Dashboard process exited.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStop requested.")
    finally:
        _stop_process(recognition_process, "recognition")
        _stop_process(dashboard_process, "dashboard")

    return 0


def run_menu() -> int:
    actions = {
        "1": ("Enroll students", run_enrollment),
        "2": ("Generate fake data", run_fake_data),
        "3": ("Run recognition", run_recognition_interactive),
        "4": ("Run dashboard", run_dashboard),
        "5": ("Run recognition + dashboard", run_demo_stack_interactive),
        "6": ("Exit", lambda: 0),
    }

    print("\n=== Classroom Demo Launcher ===")
    for key, (label, _) in actions.items():
        print(f"{key}. {label}")

    while True:
        pick = input("Select an option [5]: ").strip() or "5"
        if pick in actions:
            label, fn = actions[pick]
            if label == "Exit":
                return 0
            print(f"\nSelected: {label}")
            return fn()
        print("Invalid option. Choose 1-6.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified console launcher for the classroom demo")
    parser.add_argument(
        "--action",
        choices=["menu", "enroll", "fake", "recognition", "dashboard", "demo"],
        default="menu",
        help="Run one action directly or open interactive menu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.action == "menu":
        code = run_menu()
    elif args.action == "enroll":
        code = run_enrollment()
    elif args.action == "fake":
        code = run_fake_data()
    elif args.action == "recognition":
        code = run_recognition_interactive()
    elif args.action == "dashboard":
        code = run_dashboard()
    else:
        code = run_demo_stack_interactive()

    raise SystemExit(code)


if __name__ == "__main__":
    main()