#!/usr/bin/env python3
"""
Face Touching Detector
Detects frequent face touching (stress indicator, hygiene concern).
Uses oval/ellipse formula for face zone detection.
Calibrated thresholds: horizontal=0.12, vertical=0.35.
Alert: >= 5 touches in 2 minutes.
"""

import time
import numpy as np
from detectors.base_detector import BaseDetector


class FaceTouchingDetector(BaseDetector):
    """Detects frequent face touching using oval zone detection."""

    # Hand points to check (wrist + fingertips)
    HAND_CHECK_POINTS = [0, 4, 8, 12, 16, 20]

    def __init__(self, horizontal_threshold=0.12, vertical_threshold=0.35,
                 max_touches=5, check_period=120):
        """
        Args:
            horizontal_threshold: Oval horizontal radius (normalized)
            vertical_threshold: Oval vertical radius (normalized)
            max_touches: Max touches allowed in check_period
            check_period: Time window in seconds (default 2 minutes)
        """
        super().__init__(
            name="Face Touching",
            alert_message="Stop touching your face! Reduce stress and hygiene risk.",
        )
        self.horizontal_threshold = horizontal_threshold
        self.vertical_threshold = vertical_threshold
        self.max_touches = max_touches
        self.check_period = check_period

        self._touches = []
        self._was_touching = False

    def on_enable(self):
        self._touches = []
        self._was_touching = False

    def on_disable(self):
        self._touches = []
        self._was_touching = False

    def _is_inside_oval(self, hand_x, hand_y, face_center_x, face_center_y):
        """Check if point is inside face oval using ellipse formula."""
        dx = (hand_x - face_center_x) / self.horizontal_threshold
        dy = (hand_y - face_center_y) / self.vertical_threshold
        return (dx * dx + dy * dy) < 1.0

    def detect(self, face_landmarks, hand_landmarks, frame_width, frame_height):
        if not face_landmarks:
            self._status = "No face detected"
            self._was_touching = False
            return False

        if not hand_landmarks:
            self._status = "No hands detected"
            self._was_touching = False
            return False

        # Face center at nose tip
        nose = face_landmarks[1]
        face_cx, face_cy = nose.x, nose.y

        # Check if any hand point is inside face oval
        hand_near_face = False
        for hand in hand_landmarks:
            for point_idx in self.HAND_CHECK_POINTS:
                point = hand[point_idx]
                if self._is_inside_oval(point.x, point.y, face_cx, face_cy):
                    hand_near_face = True
                    break
            if hand_near_face:
                break

        # Detect new touch event (transition: not touching -> touching)
        if hand_near_face and not self._was_touching:
            self._touches.append(time.time())

        self._was_touching = hand_near_face

        # Remove old touches outside the time window
        now = time.time()
        self._touches = [t for t in self._touches if now - t < self.check_period]

        touch_count = len(self._touches)

        if touch_count >= self.max_touches:
            self._status = f"Frequent touching! {touch_count} in 2min"
            return True
        elif touch_count > 0:
            self._status = f"Face touches: {touch_count}/{self.max_touches} in 2min"
        else:
            self._status = "No face touching"

        return False
