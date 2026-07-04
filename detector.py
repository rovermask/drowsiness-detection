import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp


# MediaPipe Face Mesh landmark indices (6-point EAR formula, Soukupová & Čech)
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# Inner-lip points for MAR
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

# --- Tunable thresholds ---
EAR_THRESHOLD = 0.21           # below this = eye considered closed
DROWSY_CONSEC_FRAMES = 20      # ~0.6-0.8s at typical webcam fps = sustained closure -> alert
YAWN_MAR_THRESHOLD = 0.6       # above this = mouth considered open wide
YAWN_CONSEC_FRAMES = 12        # frames mouth must stay open to count as a yawn (not just talking)
WINDOW_SECONDS = 60            # rolling window for reporting blink/yawn counts


def _euclidean(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def eye_aspect_ratio(landmarks, idxs, w, h):
    p1, p2, p3, p4, p5, p6 = [(landmarks[i].x * w, landmarks[i].y * h) for i in idxs]
    vertical = _euclidean(p2, p6) + _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)
    if horizontal == 0:
        return 0.0
    return vertical / (2.0 * horizontal)


def mouth_aspect_ratio(landmarks, w, h):
    top = (landmarks[MOUTH_TOP].x * w, landmarks[MOUTH_TOP].y * h)
    bottom = (landmarks[MOUTH_BOTTOM].x * w, landmarks[MOUTH_BOTTOM].y * h)
    left = (landmarks[MOUTH_LEFT].x * w, landmarks[MOUTH_LEFT].y * h)
    right = (landmarks[MOUTH_RIGHT].x * w, landmarks[MOUTH_RIGHT].y * h)
    horizontal = _euclidean(left, right)
    if horizontal == 0:
        return 0.0
    return _euclidean(top, bottom) / horizontal


class DrowsinessDetector:
    """Wraps MediaPipe Face Mesh and tracks blink / yawn / drowsiness state."""

    def __init__(self,
                 ear_threshold=EAR_THRESHOLD,
                 drowsy_consec_frames=DROWSY_CONSEC_FRAMES,
                 yawn_mar_threshold=YAWN_MAR_THRESHOLD,
                 yawn_consec_frames=YAWN_CONSEC_FRAMES,
                 window_seconds=WINDOW_SECONDS):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.ear_threshold = ear_threshold
        self.drowsy_consec_frames = drowsy_consec_frames
        self.yawn_mar_threshold = yawn_mar_threshold
        self.yawn_consec_frames = yawn_consec_frames
        self.window_seconds = window_seconds

        # Eye state
        self.eye_closed_frames = 0
        self.eye_is_closed = False
        self.blink_timestamps = deque()
        self.drowsy_alert = False

        # Mouth state
        self.mouth_open_frames = 0
        self.mouth_is_open = False
        self.yawn_timestamps = deque()

        self.last_ear = 0.0
        self.last_mar = 0.0
        self.face_found = False

    def _prune(self, dq):
        cutoff = time.time() - self.window_seconds
        while dq and dq[0] < cutoff:
            dq.popleft()

    def process_frame(self, frame):
        """Processes one BGR frame in place, draws overlay, updates counters."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        self.drowsy_alert = False
        self.face_found = bool(results.multi_face_landmarks)

        if self.face_found:
            landmarks = results.multi_face_landmarks[0].landmark

            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_IDX, w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, w, h)
            ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(landmarks, w, h)
            self.last_ear, self.last_mar = ear, mar

            self._update_eyes(ear)
            self._update_mouth(mar)
            self._draw_overlay(frame, landmarks, w, h)
        else:
            # Don't penalize the user for a momentarily lost face (e.g. looked away)
            self.eye_closed_frames = 0
            self.mouth_open_frames = 0

        self._prune(self.blink_timestamps)
        self._prune(self.yawn_timestamps)
        return frame

    def _update_eyes(self, ear):
        if ear < self.ear_threshold:
            self.eye_closed_frames += 1
            self.eye_is_closed = True
            if self.eye_closed_frames >= self.drowsy_consec_frames:
                self.drowsy_alert = True
        else:
            if self.eye_is_closed:
                # Eyes just reopened -> one completed closure/blink event
                self.blink_timestamps.append(time.time())
            self.eye_closed_frames = 0
            self.eye_is_closed = False

    def _update_mouth(self, mar):
        if mar > self.yawn_mar_threshold:
            self.mouth_open_frames += 1
            if self.mouth_open_frames == self.yawn_consec_frames and not self.mouth_is_open:
                self.mouth_is_open = True
                self.yawn_timestamps.append(time.time())
        else:
            self.mouth_open_frames = 0
            self.mouth_is_open = False

    def _draw_overlay(self, frame, landmarks, w, h):
        color_eye = (0, 0, 255) if self.eye_is_closed else (0, 255, 0)
        color_mouth = (0, 0, 255) if self.mouth_is_open else (0, 255, 0)

        for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
            x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 2, color_eye, -1)

        for idx in (MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT):
            x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 2, color_mouth, -1)

        cv2.putText(frame, f"EAR: {self.last_ear:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {self.last_mar:.2f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if self.drowsy_alert:
            cv2.putText(frame, "DROWSINESS ALERT!", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    def get_stats(self):
        self._prune(self.blink_timestamps)
        self._prune(self.yawn_timestamps)
        return {
            "face_found": self.face_found,
            "ear": round(self.last_ear, 3),
            "mar": round(self.last_mar, 3),
            "eye_closed_now": self.eye_is_closed,
            "mouth_open_now": self.mouth_is_open,
            "drowsy_alert": self.drowsy_alert,
            "blinks_in_window": len(self.blink_timestamps),
            "yawns_in_window": len(self.yawn_timestamps),
            "window_seconds": self.window_seconds,
        }
