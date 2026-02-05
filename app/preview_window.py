#!/usr/bin/env python3
"""
Preview helpers
- draw_overlays(): annotates a BGR frame with detection visuals (pure OpenCV, no GUI)
- frame_to_nsimage(): converts a BGR numpy frame to an AppKit NSImage
"""

import cv2

# ── Overlay drawing (called from camera thread — no GUI calls) ─────────────

# Colors (BGR)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
ORANGE = (0, 165, 255)
GRAY = (180, 180, 180)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def draw_overlays(bgr_frame, face_landmarks, hand_landmarks,
                  detector_statuses, enabled_detectors):
    """Draw all debug overlays onto a copy of the frame. No GUI calls."""
    frame = bgr_frame.copy()
    h, w = frame.shape[:2]

    if face_landmarks:
        _draw_face_points(frame, face_landmarks, w, h)
        if 'mouth_breathing' in enabled_detectors:
            _draw_mouth_overlay(frame, face_landmarks, w, h)
        if 'blink_rate' in enabled_detectors:
            _draw_eye_overlay(frame, face_landmarks, w, h)
        if 'face_touching' in enabled_detectors:
            _draw_face_oval(frame, face_landmarks, w, h,
                            hand_landmarks is not None)
        if 'eye_rubbing' in enabled_detectors:
            _draw_eye_zones(frame, face_landmarks, w, h)

    if hand_landmarks:
        _draw_hands(frame, hand_landmarks, w, h)

    _draw_status_panel(frame, detector_statuses, enabled_detectors)
    return frame


# ── NSImage conversion (called from main thread) ──────────────────────────

def frame_to_nsimage(bgr_frame):
    """Convert a BGR numpy array to an AppKit NSImage via JPEG encoding."""
    from AppKit import NSImage
    from Foundation import NSData

    ok, jpg = cv2.imencode('.jpg', bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    ns_data = NSData.dataWithBytes_length_(jpg.tobytes(), len(jpg))
    return NSImage.alloc().initWithData_(ns_data)


# ── Internal drawing helpers ──────────────────────────────────────────────

def _draw_face_points(frame, lm, w, h):
    for idx in [1, 13, 14, 33, 61, 263, 291]:
        cv2.circle(frame, (int(lm[idx].x * w), int(lm[idx].y * h)), 3, CYAN, -1)


def _draw_mouth_overlay(frame, lm, w, h):
    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in [13, 14, 61, 291, 78, 308, 95, 88]]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    pad = 10
    x1, x2, y1, y2 = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad

    hor = abs(lm[291].x - lm[61].x)
    mar = abs(lm[13].y - lm[14].y) / hor if hor > 0.001 else 0
    color = RED if mar > 0.05 else GREEN
    label = f"{'OPEN' if mar > 0.05 else 'CLOSED'} ({mar:.3f})"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def _draw_eye_overlay(frame, lm, w, h):
    for indices in [[159, 145, 23, 130], [386, 374, 253, 359]]:
        pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in indices]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        cv2.rectangle(frame, (min(xs) - 5, min(ys) - 5),
                      (max(xs) + 5, max(ys) + 5), GREEN, 1)


def _draw_eye_zones(frame, lm, w, h):
    for idx in [33, 263]:
        cv2.circle(frame, (int(lm[idx].x * w), int(lm[idx].y * h)),
                   int(0.02 * w), ORANGE, 1)


def _draw_face_oval(frame, lm, w, h, has_hands):
    cx, cy = int(lm[1].x * w), int(lm[1].y * h)
    cv2.circle(frame, (cx, cy), 6, YELLOW, -1)
    cv2.ellipse(frame, (cx, cy), (int(0.12 * w), int(0.35 * h)),
                0, 0, 360, GREEN if has_hands else GRAY, 2)


def _draw_hands(frame, hands, w, h):
    for hand in hands:
        for pt in hand:
            cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 4, MAGENTA, -1)
        for si, ei in HAND_CONNECTIONS:
            s, e = hand[si], hand[ei]
            cv2.line(frame, (int(s.x * w), int(s.y * h)),
                     (int(e.x * w), int(e.y * h)), MAGENTA, 2)


def _draw_status_panel(frame, statuses, enabled):
    h, w = frame.shape[:2]
    y = h - 12
    for name in reversed(list(statuses.keys())):
        on = name in enabled
        label = f"{'[ON]' if on else '[OFF]'} {name}: {statuses[name]}"
        cv2.putText(frame, label, (6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE if on else GRAY, 1)
        y -= 18
