# Drowsiness Detection (Flask + MediaPipe)

Live webcam drowsiness detector using MediaPipe Face Mesh. Tracks:
- **Eye closures / blinks** in a rolling time window (via EAR — Eye Aspect Ratio)
- **Yawns** in a rolling time window (via MAR — Mouth Aspect Ratio)
- **Drowsiness alert** (on-screen banner + beep) when eyes stay closed too long

## Setup

```bash
cd drowsiness_detection
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser. Grant webcam access to the browser
tab is not needed — OpenCV grabs the webcam directly on the server side, so this must
be run on the machine that has the camera (fine for local dev; for real deployment
you'd want to move capture to the browser via `getUserMedia` and send frames to the
server, since cloud hosts don't have a physical webcam).

## How it works

- `detector.py` — `DrowsinessDetector` class. For every frame: runs MediaPipe Face Mesh,
  computes EAR from 6 eye landmarks per eye and MAR from 4 inner-lip landmarks, then
  runs a small state machine:
  - EAR below threshold for N consecutive frames → eyes "closed". When they reopen,
    one blink/closure event is logged with a timestamp.
  - EAR closed continuously for `DROWSY_CONSEC_FRAMES` → `drowsy_alert = True`.
  - MAR above threshold for `YAWN_CONSEC_FRAMES` consecutive frames → one yawn logged.
  - Timestamps older than `WINDOW_SECONDS` are pruned, so counts always reflect "last N seconds".
- `app.py` — Flask routes:
  - `/` dashboard page
  - `/video_feed` MJPEG stream (frame drawn with landmarks + EAR/MAR overlay)
  - `/status` JSON with live counts, used by the frontend to update the dashboard and
    trigger the beep (generated client-side with the Web Audio API — no audio file needed)
  - `/reset` clears counters without restarting the server
- `templates/index.html` — dashboard UI, polls `/status` every 500ms.

## Tuning

All thresholds are constants at the top of `detector.py`:

| Constant | Meaning | Default |
|---|---|---|
| `EAR_THRESHOLD` | EAR below this = eye considered closed | 0.21 |
| `DROWSY_CONSEC_FRAMES` | consecutive closed frames before alert fires | 20 |
| `YAWN_MAR_THRESHOLD` | MAR above this = mouth wide open | 0.6 |
| `YAWN_CONSEC_FRAMES` | consecutive open-mouth frames to count a yawn | 12 |
| `WINDOW_SECONDS` | rolling window for reported counts | 60 |

Webcam frame rate varies by hardware, so `DROWSY_CONSEC_FRAMES` and `YAWN_CONSEC_FRAMES`
are frame counts, not seconds — recalibrate them against your actual FPS if the alert
feels too twitchy or too slow. Print `cv2.getTickFrequency()`-based FPS or just log
frame timestamps for a few seconds to check.

## CLI / OpenCV-window version

For local testing without a browser, run:

```bash
python cli.py
```

This opens a plain OpenCV window with the same landmark overlay, EAR/MAR readout,
and blink/yawn counts as the Flask version. Controls:
- `q` — quit
- `r` — reset counters

Beep behavior:
- On **Windows**, it uses the built-in `winsound.Beep()` (no extra install needed).
- On **macOS/Linux**, it falls back to the terminal bell (`\a`), which is audible in
  most terminal apps. If you want a real alarm sound instead, swap `_play_beep()` in
  `cli.py` for something like:
  ```python
  from playsound import playsound
  playsound("alarm.wav")
  ```
  (requires `pip install playsound` and an actual `.wav` file).
- Beeps are fired on a background thread with a cooldown so the video loop never
  freezes and it doesn't spam beeps every single frame while drowsy.
