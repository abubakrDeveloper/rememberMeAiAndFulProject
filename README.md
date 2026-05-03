# rememberMe

**rememberMe** is a face-recognition-based classroom monitoring system for schools and colleges, built on Python 3.11. It runs on live camera streams, RTSP feeds, or pre-recorded MP4 files and produces structured attendance and behavior reports.

## Features

**Attendance tracking**

- Per-student and per-teacher present/absent determination
- Configurable minimum confirmation frames before marking a person present
- Confidence bands (low / medium / high) on all attendance and incident records

**Behavior detection** (classroom mode)

- States: focused, distracted, sleeping, possible phone use
- Eye Aspect Ratio (EAR) via MediaPipe Face Mesh for sleep detection
- Head-yaw estimation for distraction detection
- Temporal smoothing — requires N positive detections within the last M frames to fire
- Per-incident cooldown to suppress duplicate alerts

**Video input**

- MP4 / video file playback
- RTSP stream with automatic reconnect
- Webcam (by index)

**Reporting and output**

- CSV exports: student attendance, teacher attendance, incidents
- Session summary JSON for dashboard or external integration
- Incident screenshots saved to `output/snapshots/` with distracted faces highlighted in red
- Optional annotated video export
- Automatic 3-day retention cleanup of old reports and snapshots

**Admin dashboard**

- Local HTTP server (`admin_dashboard.py`) serving `dashboard/index.html`
- Browse latest reports, view snapshots, download CSVs
- Audit log (`audit_log.jsonl`) records every dashboard view, report download, and email event

**Email notifications**

- Optional SMTP email to the admin with the session summary (TLS supported)

---

## Requirements

- Python 3.11
- A camera, RTSP stream, or MP4 file

Dependencies (installed via `requirements.txt`):

| Package | Purpose |
|---|---|
| `opencv-python` / `opencv-contrib-python` | Video capture and frame processing |
| `mediapipe` | Face detection, Face Mesh, landmark extraction |
| `face_recognition` + `dlib` | 128-d face embeddings and recognition |
| `numpy` | Numerical operations |

---

## Setup

**1. Create a virtual environment (Windows PowerShell)**

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install -r requirements.txt
```

**2. Add roster images**

```
roster/
  students/    ← student face images
  teachers/    ← teacher face images
```

Filename format:

- `S001_Ali_Khan.jpg` — student (ID underscore name)
- `T001_Ms_Sara.jpg`  — teacher (ID underscore name)

The app detects the largest face in each image and stores a 128-d embedding.

**3. Edit `config.json`**

Key sections:

| Section | Purpose |
|---|---|
| `source` | Input mode (`file` / `rtsp` / `webcam`) and source path or URL |
| `detection` | Face detection confidence and recognition tolerance |
| `behavior` | Thresholds for distraction, sleep, EAR, head yaw, smoothing |
| `attendance` | Minimum confirmation frames |
| `runtime` | FPS cap, display window, annotated video export, stop-after timer |
| `session` | Institute name, class name, teacher ID, admin name |
| `admin_email` | SMTP credentials and toggle |

---

## Running

**Quick launcher (Windows)**

```bat
run.bat
```

Presents an interactive menu to pick mode and source.

**Command line**

```bash
# Classroom mode — default config
py -3.11 main.py --config config.json

# General face recognition mode
py -3.11 main.py --general --mode webcam
py -3.11 main.py --general --mode file --source "videos/people.mp4"

# Overrides
py -3.11 main.py --config config.json --mode rtsp --source "rtsp://user:pass@ip/stream"
py -3.11 main.py --config config.json --mode file --source "videos/classroom.mp4" --no-display
py -3.11 main.py --config config.json --stop-after 300
```

Press `q` to quit when the display window is open.

**Admin dashboard**

```bash
py -3.11 admin_dashboard.py
```

Opens the dashboard in the default browser at `http://localhost:8000`.

---

## Output

```
output/
  attendance_students_<timestamp>.csv
  attendance_teachers_<timestamp>.csv
  incidents_<timestamp>.csv
  summary_<timestamp>.json
  audit_log.jsonl
  snapshots/
    incident_<timestamp>.jpg    ← distracted faces boxed in red
```

---

## 6) Tiny Admin Dashboard (HTML)

This project includes a tiny desktop dashboard page that reads latest CSV and JSON outputs and shows:
- Attendance cards
- Student and teacher attendance tables
- Incident distribution bars
- Incident snapshots gallery
- Camera online/offline, last frame timestamp, and reconnect status
- Filters: date, class, incident type, attendance status, and name/ID search
- Audit log table

Run dashboard server:

```bash
.\.venv311\Scripts\python.exe admin_dashboard.py --open-browser
```

If you want a custom port:

```bash
.\.venv311\Scripts\python.exe admin_dashboard.py --port 9000 --open-browser
```

Then open:

```text
http://127.0.0.1:8765
```

The dashboard reads the newest files in `output/`:
- `attendance_students_*.csv`
- `attendance_teachers_*.csv`
- `incidents_*.csv`
- `summary_*.json`

Also generated in `output/`:
- `audit_log.jsonl` (view/report/email audit events)

## 7) Retention Policy

On each run, files older than 3 days are automatically deleted:
- snapshots in `output/snapshots/`
- report files in `output/` (`attendance_*.csv`, `incidents_*.csv`, `summary_*.json`, `annotated_*.mp4`)

## 8) Reducing False Alerts

Behavior detection includes temporal smoothing to reduce one-frame spikes:
- `behavior.smoothing_window_frames`: number of recent frames to track (M)
- `behavior.smoothing_required_frames`: minimum matching frames needed to alert (N)

Example currently configured:
- window = 8
- required = 5

This means an incident label must appear in at least 5 of the last 8 frames before firing.