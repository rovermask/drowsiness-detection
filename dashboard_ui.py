"""
Drawing helpers that render a cockpit-style telemetry dashboard next to the
raw video frame, entirely with OpenCV primitives (no external UI toolkit).

Layout: [ video feed with HUD corner brackets ] [ side panel: status badge,
EAR/MAR sparklines, blink/yawn counters, session clock, key hints ].
"""
import time
from collections import deque

import cv2
import numpy as np

# ---- Palette (BGR, since OpenCV draws in BGR) ----
BG = (18, 13, 10)
PANEL_BORDER = (54, 42, 35)
TEXT = (241, 236, 232)
MUTED = (133, 118, 107)
SAFE = (160, 216, 52)     # teal-green
WARN = (32, 176, 255)     # amber
ALERT = (87, 71, 255)     # red

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

PANEL_WIDTH = 300
TRACE_LEN = 90


class DashboardRenderer:
    def __init__(self):
        self.ear_history = deque([0.3] * TRACE_LEN, maxlen=TRACE_LEN)
        self.mar_history = deque([0.1] * TRACE_LEN, maxlen=TRACE_LEN)
        self.start_time = time.time()

    # ---------- public API ----------
    def render(self, frame, stats):
        """Returns a new BGR image: video (with HUD overlay) + side panel."""
        self.ear_history.append(stats["ear"])
        self.mar_history.append(stats["mar"])

        state, color, label = self._state(stats)

        h, w = frame.shape[:2]
        canvas = np.full((h, w + PANEL_WIDTH, 3), BG, dtype=np.uint8)

        video = frame.copy()
        self._draw_corners(video, w, h, color)
        canvas[0:h, 0:w] = video

        panel = canvas[0:h, w:w + PANEL_WIDTH]
        self._draw_panel(panel, h, stats, state, color, label)

        return canvas

    # ---------- internals ----------
    @staticmethod
    def _state(stats):
        if not stats["face_found"]:
            return "warn", WARN, "NO FACE"
        if stats["drowsy_alert"]:
            return "alert", ALERT, "DROWSY ALERT"
        return "safe", SAFE, "NOMINAL"

    @staticmethod
    def _draw_corners(img, w, h, color, size=26, thickness=2):
        pts = [
            ((0, 0), (1, 1)),
            ((w, 0), (-1, 1)),
            ((0, h), (1, -1)),
            ((w, h), (-1, -1)),
        ]
        for (x, y), (dx, dy) in pts:
            cv2.line(img, (x, y), (x + dx * size, y), color, thickness)
            cv2.line(img, (x, y), (x, y + dy * size), color, thickness)

    def _draw_panel(self, panel, h, stats, state, color, label):
        panel[:] = BG
        cv2.rectangle(panel, (0, 0), (PANEL_WIDTH - 1, h - 1), PANEL_BORDER, 1)

        pad = 18

        # --- Title ---
        cv2.putText(panel, "MONITOR", (pad, 32), FONT_BOLD, 0.55, TEXT, 1, cv2.LINE_AA)
        cv2.putText(panel, "FACE MESH TELEMETRY", (pad, 50), FONT, 0.4, MUTED, 1, cv2.LINE_AA)
        cv2.line(panel, (pad, 62), (PANEL_WIDTH - pad, 62), PANEL_BORDER, 1)

        # --- Status badge ---
        badge_top, badge_bottom = 76, 108
        dim_color = tuple(c // 4 for c in color)
        cv2.rectangle(panel, (pad, badge_top), (PANEL_WIDTH - pad, badge_bottom), dim_color, -1)
        cv2.rectangle(panel, (pad, badge_top), (PANEL_WIDTH - pad, badge_bottom), color, 1)
        cv2.circle(panel, (pad + 14, (badge_top + badge_bottom) // 2), 4, color, -1)
        cv2.putText(panel, label, (pad + 28, badge_bottom - 12), FONT_BOLD, 0.55, color, 1, cv2.LINE_AA)

        # --- EAR trace ---
        y = 132
        self._labeled_value(panel, pad, y, "EAR", f"{stats['ear']:.2f}", color if state == "alert" else TEXT)
        self._sparkline(panel, pad, y + 8, PANEL_WIDTH - 2 * pad, 42, list(self.ear_history),
                         max_val=0.45, color=ALERT if state == "alert" else SAFE)

        # --- MAR trace ---
        y = 206
        mar_color = WARN if stats.get("mouth_open_now") else TEXT
        self._labeled_value(panel, pad, y, "MAR", f"{stats['mar']:.2f}", mar_color)
        self._sparkline(panel, pad, y + 8, PANEL_WIDTH - 2 * pad, 42, list(self.mar_history),
                         max_val=1.0, color=WARN if stats.get("mouth_open_now") else SAFE)

        cv2.line(panel, (pad, 280), (PANEL_WIDTH - pad, 280), PANEL_BORDER, 1)

        # --- Blink / yawn counters ---
        box_w = (PANEL_WIDTH - 2 * pad - 10) // 2
        self._count_box(panel, pad, 296, box_w, 66, stats["blinks_in_window"], "EYE CLOSURES", SAFE)
        self._count_box(panel, pad + box_w + 10, 296, box_w, 66, stats["yawns_in_window"], "YAWNS", WARN)

        cv2.putText(panel, f"rolling {stats['window_seconds']}s window", (pad, 380),
                    FONT, 0.38, MUTED, 1, cv2.LINE_AA)

        # --- Footer: session clock + key hints ---
        elapsed = int(time.time() - self.start_time)
        clock_str = f"{elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}"
        cv2.putText(panel, f"SESSION {clock_str}", (pad, h - 46), FONT, 0.42, MUTED, 1, cv2.LINE_AA)
        cv2.line(panel, (pad, h - 34), (PANEL_WIDTH - pad, h - 34), PANEL_BORDER, 1)
        cv2.putText(panel, "[Q] QUIT   [R] RESET", (pad, h - 14), FONT, 0.4, MUTED, 1, cv2.LINE_AA)

    @staticmethod
    def _labeled_value(panel, x, y, label, value, value_color):
        cv2.putText(panel, label, (x, y), FONT, 0.42, MUTED, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(value, FONT_BOLD, 0.55, 1)
        cv2.putText(panel, value, (PANEL_WIDTH - 18 - tw, y), FONT_BOLD, 0.55, value_color, 1, cv2.LINE_AA)

    @staticmethod
    def _sparkline(panel, x, y, w, h, values, max_val, color):
        n = len(values)
        if n < 2:
            return
        pts = []
        for i, v in enumerate(values):
            px = x + int((i / (n - 1)) * w)
            norm = min(max(v / max_val, 0.0), 1.0)
            py = y + h - int(norm * h)
            pts.append((px, py))
        cv2.polylines(panel, [np.array(pts, dtype=np.int32)], False, color, 1, cv2.LINE_AA)

    @staticmethod
    def _count_box(panel, x, y, w, h, value, label, color):
        cv2.rectangle(panel, (x, y), (x + w, y + h), PANEL_BORDER, 1)
        text = str(value)
        (tw, th), _ = cv2.getTextSize(text, FONT_BOLD, 1.0, 2)
        cv2.putText(panel, text, (x + (w - tw) // 2, y + 10 + th), FONT_BOLD, 1.0, color, 2, cv2.LINE_AA)
        (lw, _), _ = cv2.getTextSize(label, FONT, 0.35, 1)
        cv2.putText(panel, label, (x + (w - lw) // 2, y + h - 10), FONT, 0.35, MUTED, 1, cv2.LINE_AA)
