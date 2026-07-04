"""
Standalone local version - opens a styled OpenCV window instead of running
through Flask/browser. Good for quick testing without spinning up a server.

Controls:
  q - quit
  r - reset blink/yawn counters
"""
import platform
import threading
import time

import cv2

from detector import DrowsinessDetector
from dashboard_ui import DashboardRenderer

IS_WINDOWS = platform.system() == "Windows"
if IS_WINDOWS:
    import winsound

BEEP_COOLDOWN = 0.7  # seconds between beeps while alert stays active
_last_beep_time = 0.0


def _play_beep():
    if IS_WINDOWS:
        winsound.Beep(1000, 200)  # frequency Hz, duration ms
    else:
        # Terminal bell - audible in most Linux/macOS terminals.
        # Swap this for `playsound("alarm.wav")` if you want a real audio file instead.
        print('\a', end='', flush=True)


def beep_async():
    """Fires the beep on a background thread so the video loop never blocks/freezes."""
    global _last_beep_time
    now = time.time()
    if now - _last_beep_time < BEEP_COOLDOWN:
        return
    _last_beep_time = now
    threading.Thread(target=_play_beep, daemon=True).start()


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Could not open webcam. Check it's connected and not in use by another app.")
        return

    detector = DrowsinessDetector()
    dashboard = DashboardRenderer()
    window_name = "Vigilance Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("Drowsiness detection running. Press 'q' to quit, 'r' to reset counters.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror view
        frame = detector.process_frame(frame)
        stats = detector.get_stats()

        display = dashboard.render(frame, stats)

        if stats["drowsy_alert"]:
            beep_async()

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.blink_timestamps.clear()
            detector.yawn_timestamps.clear()
            print("Counters reset.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
