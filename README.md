# Classroom CCTV Monitor (Python 3.11)

This is a practical CCTV classroom monitoring application for schools and colleges.

It supports:
- Student attendance (present/absent)
- Teacher attendance (present/absent)
- Real-time camera streams (RTSP or webcam)
- Offline MP4 processing
- Basic behavior detection: focused, distracted, sleeping, possible phone use
- Camera health and reconnect status tracking
- Incident screenshots with distracted students highlighted in red rectangles
- Exporting reports for admin (CSV + JSON)
- Optional email sending to admin
- Confidence bands (low/medium/high) for attendance and incidents
- Audit log for dashboard views and report/email events
- Automatic retention cleanup of old reports/snapshots (3 days)
- Temporal smoothing for distraction alerts (N of last M frames)

## 1) Requirements

- Python 3.11
- A camera stream or MP4 file

Recommended project environment (Windows PowerShell):

```bash
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
```

Install packages:

```bash
py -3.11 -m pip install -r requirements.txt
```

## 2) Roster Setup

Create folders:

```text
roster/
  students/
  teachers/
```

Add one or more face images per person.

Filename format:

- `studentId_studentName.jpg`  (example: `S001_Ali_Khan.jpg`)
- `teacherId_teacherName.jpg`  (example: `T001_Ms_Sara.jpg`)

The app detects the largest face in each image and builds a simple face profile.

## 3) Configuration

Use `config.json`.

Input modes:
- File mode: `"mode": "file"`, source is an MP4 path
- RTSP mode: `"mode": "rtsp"`, source is an RTSP URL
- Webcam mode: `"mode": "webcam"`, source can be `"0"`

Enable admin email by filling `admin_email` and setting `enabled` to `true`.

## 4) Run

```bash
py -3.11 main.py --config config.json
```

Optional overrides:

```bash
py -3.11 main.py --config config.json --mode rtsp --source "rtsp://user:pass@ip/stream"
py -3.11 main.py --config config.json --mode file --source "videos/classroom.mp4" --no-display
py -3.11 main.py --config config.json --stop-after 300
```

Press `q` to stop when display is enabled.

## 5) Output

Generated in `output/`:
- `attendance_students_*.csv`
- `attendance_teachers_*.csv`
- `incidents_*.csv`
- `summary_*.json`

Generated in `output/snapshots/`:
- Incident screenshots with red-highlighted distracted student boxes

## 6) Notes and Useful Additions Included

- RTSP reconnect support
- Cooldown for repeated alerts to avoid spam
- Session summary JSON for dashboard integration
- Optional annotated video export (`runtime.save_annotated_video`)

## 7) Important Limitations

- Face recognition is intentionally lightweight (OpenCV histogram matching) for minimal dependency setup.
- Behavior classification is heuristic and should be tuned per camera angle and class layout.
- `possible_phone_use` is an indicator, not proof.

For production-grade accuracy, replace the face and behavior logic with stronger ML models.


## 8) Retention Policy

On each run, files older than 3 days are automatically deleted:
- snapshots in `output/snapshots/`
- report files in `output/` (`attendance_*.csv`, `incidents_*.csv`, `summary_*.json`, `annotated_*.mp4`)

## 9) Reducing False Alerts

Behavior detection includes temporal smoothing to reduce one-frame spikes:
- `behavior.smoothing_window_frames`: number of recent frames to track (M)
- `behavior.smoothing_required_frames`: minimum matching frames needed to alert (N)

Example currently configured:
- window = 8
- required = 5

This means an incident label must appear in at least 5 of the last 8 frames before firing.
