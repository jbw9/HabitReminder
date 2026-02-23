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
                  detector_statuses, enabled_detectors,
                  mouth_counter=0, mouth_threshold=450,
                  nail_counter=0, nail_threshold=60, fps=30):
    """Draw all debug overlays onto a copy of the frame. No GUI calls."""
    frame = bgr_frame.copy()
    h, w = frame.shape[:2]

    if face_landmarks:
        _draw_face_points(frame, face_landmarks, w, h)
        if 'mouth_breathing' in enabled_detectors:
            _draw_mouth_overlay(frame, face_landmarks, w, h)
        if 'nail_biting' in enabled_detectors and hand_landmarks:
            _draw_nail_biting_overlay(frame, face_landmarks, hand_landmarks, w, h)

    if hand_landmarks:
        _draw_hands(frame, hand_landmarks, w, h)

    # Show countdown for whichever detector has made more progress
    mouth_progress = mouth_counter / mouth_threshold if mouth_threshold > 0 else 0
    nail_progress = nail_counter / nail_threshold if nail_threshold > 0 else 0

    if mouth_progress >= nail_progress and mouth_counter > 0:
        _draw_countdown(frame, mouth_counter, mouth_threshold, fps, w, h, 'mouth')
    elif nail_counter > 0:
        _draw_countdown(frame, nail_counter, nail_threshold, fps, w, h, 'nail')

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


def _draw_nail_biting_overlay(frame, face_lm, hand_lms, w, h):
    """Draw lines from fingertips to mouth center, showing distance."""
    import math

    # Calculate mouth center
    mouth_indices = [13, 14, 61, 291]
    mouth_x = sum(face_lm[i].x for i in mouth_indices) / len(mouth_indices)
    mouth_y = sum(face_lm[i].y for i in mouth_indices) / len(mouth_indices)
    mouth_px, mouth_py = int(mouth_x * w), int(mouth_y * h)

    # Draw mouth center
    cv2.circle(frame, (mouth_px, mouth_py), 6, CYAN, -1)

    # Fingertip indices
    fingertips = [4, 8, 12, 16, 20]

    min_distance = float('inf')
    closest_tip = None

    # For each hand
    for hand in hand_lms:
        # For each fingertip
        for tip_idx in fingertips:
            tip = hand[tip_idx]
            tip_px, tip_py = int(tip.x * w), int(tip.y * h)

            # Calculate normalized distance
            dx = tip.x - mouth_x
            dy = tip.y - mouth_y
            distance = math.sqrt(dx * dx + dy * dy)

            if distance < min_distance:
                min_distance = distance
                closest_tip = (tip_px, tip_py)

    # Draw line from closest fingertip to mouth
    if closest_tip:
        color = RED if min_distance < 0.08 else YELLOW if min_distance < 0.15 else GREEN
        cv2.line(frame, closest_tip, (mouth_px, mouth_py), color, 2)

        # Draw distance label at midpoint
        mid_x = (closest_tip[0] + mouth_px) // 2
        mid_y = (closest_tip[1] + mouth_py) // 2
        label = f"{min_distance:.3f}"
        cv2.putText(frame, label, (mid_x, mid_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


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


def _draw_countdown(frame, counter, threshold, fps, w, h, detector_type='mouth'):
    """Draw countdown bar; red !!! banner once threshold hit."""
    # Set messages based on detector type
    if detector_type == 'nail':
        alert_msg = "!!! STOP NAIL BITING !!!"
        countdown_prefix = "MOVE HAND AWAY"
    else:  # mouth
        alert_msg = "!!! CLOSE YOUR MOUTH !!!"
        countdown_prefix = "CLOSE MOUTH"

    if counter >= threshold:
        # ── Alert state: red tint + !!! banner ──────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
        (tw, th), _ = cv2.getTextSize(alert_msg, font, scale, thick)
        tx, ty = (w - tw) // 2, h // 2 + th // 2
        # Black shadow for legibility
        cv2.putText(frame, alert_msg, (tx + 1, ty + 1), font, scale, (0, 0, 0), thick + 1)
        cv2.putText(frame, alert_msg, (tx, ty), font, scale, (0, 0, 255), thick)
        return

    # ── Counting down: progress bar ─────────────────────────────────────
    seconds_remaining = max(0.0, (threshold - counter) / fps)
    progress = min(1.0, counter / threshold)

    bar_h = 18
    bar_y = h - bar_h - 28
    cv2.rectangle(frame, (0, bar_y), (w, bar_y + bar_h), (40, 40, 40), -1)

    fill_w = int(w * progress)
    if progress < 0.5:
        color = (0, int(255 * (1 - progress * 2)), 255)          # blue -> yellow
    else:
        color = (0, 0, int(255 * (progress - 0.5) * 2 + 100))   # yellow -> red
    if fill_w > 0:
        cv2.rectangle(frame, (0, bar_y), (fill_w, bar_y + bar_h), color, -1)

    label = f"{countdown_prefix}  {seconds_remaining:.1f}s"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    tx = (w - tw) // 2
    ty = bar_y + bar_h - (bar_h - th) // 2 - 2
    cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)


def _draw_status_panel(frame, statuses, enabled):
    h, w = frame.shape[:2]
    y = h - 12
    for name in reversed(list(statuses.keys())):
        on = name in enabled
        label = f"{'[ON]' if on else '[OFF]'} {name}: {statuses[name]}"
        cv2.putText(frame, label, (6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE if on else GRAY, 1)
        y -= 18
