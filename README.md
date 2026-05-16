# rememberMe

**rememberMe** is a face-recognition-based classroom monitoring system for schools and colleges, built on Python 3.11. It runs on live camera streams, RTSP feeds, or pre-recorded MP4 files and produces structured attendance and behavior reports.

---

## Features

### AI Engine (Python)

**Attendance tracking**
- Per-student and per-teacher present/absent determination
- Configurable minimum confirmation frames before marking a person present
- Confidence bands (low / medium / high) on all records

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

**Email notifications**
- Optional SMTP email to the admin with the session summary (TLS supported)

### REST API (Node.js)

- **Auth**: sign-up and sign-in with JWT and bcrypt password hashing
- **Users**: role-based model — `Student`, `Teacher`, `Admin` — with optional face image field
- **Attendance**: per-user records with status (`Present`, `Absent`, `Late`, `Excused`), group, lesson order, and AI capture image
- **Groups**: department/course groups with lifecycle status (`Active`, `Graduated`, `Inactive`)
- **Static file serving**: uploaded face images served from `/uploads`

### Admin Dashboard (HTML)

- Attendance cards and student/teacher attendance tables
- Incident distribution bars and snapshot gallery
- Camera online/offline status and last-frame timestamp
- Filters: date, class, incident type, attendance status, name/ID search
- Audit log table

---

## Requirements

### AI Engine
- Python 3.11
- A camera, RTSP stream, or MP4 file

| Package | Purpose |
|---|---|
| `opencv-python` / `opencv-contrib-python` | Video capture and frame processing |
| `mediapipe` | Face detection, Face Mesh, landmark extraction |
| `face_recognition` + `dlib` | 128-d face embeddings and recognition |
| `numpy` | Numerical operations |

### REST API
- Node.js 18+
- MongoDB instance (local or Atlas)

| Package | Purpose |
|---|---|
| `express` | HTTP server and routing |
| `mongoose` | MongoDB ODM |
| `jsonwebtoken` | JWT auth tokens |
| `bcrypt` / `bcryptjs` | Password hashing |
| `dotenv` | Environment variable loading |
| `cors` | Cross-origin resource sharing |

---

## Setup

### 1. AI Engine

**Create a virtual environment (Windows PowerShell)**

```powershell
cd ai-engine
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Add roster images**

```
roster/
  students/    ← student face images  (e.g. S001_Ali_Khan.jpg)
  teachers/    ← teacher face images  (e.g. T001_Ms_Sara.jpg)
```

Filename format: `<ID>_<Name>.jpg` — the app detects the largest face in each image and stores a 128-d embedding.

**Edit `config.json`**

| Section | Purpose |
|---|---|
| `source` | Input mode (`file` / `rtsp` / `webcam`) and source path or URL |
| `detection` | Face detection confidence and recognition tolerance |
| `behavior` | Thresholds for distraction, sleep, EAR, head yaw, smoothing |
| `attendance` | Minimum confirmation frames |
| `runtime` | FPS cap, display window, annotated video export, stop-after timer |
| `session` | Institute name, class name, teacher ID, admin name |
| `admin_email` | SMTP credentials and toggle |

### 2. REST API

```bash
cd server
npm install
```

Create a `.env` file in `server/`:

```env
MONGO_URI=mongodb://localhost:27017/rememberme
JWT_SECRET=your_jwt_secret_here
PORT=4000
```

---

## Running

### AI Engine

**Quick launcher (Windows)**

```bat
cd ai-engine
run.bat
```

**Command line**

```bash
# Classroom mode
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
py -3.11 admin_dashboard.py --open-browser
# Custom port:
py -3.11 admin_dashboard.py --port 9000 --open-browser
```

Opens at `http://127.0.0.1:8765`.

### REST API

```bash
cd server
npm start          # production
npm run dev        # development (nodemon)
```

Server starts on `http://localhost:4000`.

---

## Output

```
ai-engine/output/
  attendance_students_<timestamp>.csv
  attendance_teachers_<timestamp>.csv
  incidents_<timestamp>.csv
  summary_<timestamp>.json
  audit_log.jsonl
  snapshots/
    incident_<timestamp>.jpg    ← distracted faces boxed in red
```

---

## Retention Policy

On each AI engine run, files older than 3 days are automatically deleted:
- Snapshots in `output/snapshots/`
- Report files: `attendance_*.csv`, `incidents_*.csv`, `summary_*.json`, `annotated_*.mp4`

## 8) Reducing False Alerts

Behavior detection includes temporal smoothing to reduce one-frame spikes:
- `behavior.smoothing_window_frames`: number of recent frames to track (M)
- `behavior.smoothing_required_frames`: minimum matching frames needed to alert (N)

Example currently configured:
- window = 8
- required = 5

This means an incident label must appear in at least 5 of the last 8 frames before firing.