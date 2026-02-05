#!/usr/bin/env python3
"""
Blink Rate Detector
Detects insufficient blinking which leads to dry eyes and eye strain.
Normal: 15-20 blinks/min. Computer use often drops to 5-7.
Calibrated threshold: 0.012 EAR, alert if < 6 blinks/min in 60s window.
"""

import time
import numpy as np
from detectors.base_detector import BaseDetector


class BlinkDetector(BaseDetector):
    """Detects low blink rate."""

    # Eye landmark indices (MediaPipe Face Mesh)
    LEFT_EYE_UPPER = [159, 145]
    LEFT_EYE_LOWER = [23, 130]
    RIGHT_EYE_UPPER = [386, 374]
    RIGHT_EYE_LOWER = [253, 359]

    def __init__(self, ear_threshold=0.012, low_blink_threshold=6, check_interval=60):
        """
        Args:
            ear_threshold: Eye Aspect Ratio threshold for closed eye
            low_blink_threshold: Minimum blinks per minute before alert
            check_interval: Seconds between blink rate checks
        """
        super().__init__(
            name="Blink Rate",
            alert_message="Blink more! You're not blinking enough.",
        )
        self.ear_threshold = ear_threshold
        self.low_blink_threshold = low_blink_threshold
        self.check_interval = check_interval

        self._blink_count = 0
        self._last_check_time = time.time()
        self._was_eyes_closed = False

    def on_enable(self):
        self._blink_count = 0
        self._last_check_time = time.time()
        self._was_eyes_closed = False

    def on_disable(self):
        self._blink_count = 0
        self._was_eyes_closed = False

    def _calculate_ear(self, landmarks, top_indices, bottom_indices):
        """Calculate Eye Aspect Ratio (simplified vertical distance)."""
        top_points = [landmarks[idx] for idx in top_indices]
        bottom_points = [landmarks[idx] for idx in bottom_indices]

        vertical_distances = []
        for top in top_points:
            for bottom in bottom_points:
                vertical_distances.append(abs(top.y - bottom.y))

        return np.mean(vertical_distances)

    def detect(self, face_landmarks, hand_landmarks, frame_width, frame_height):
        if not face_landmarks:
            self._status = "No face detected"
            return False

        # Calculate EAR for both eyes
        left_ear = self._calculate_ear(
            face_landmarks, self.LEFT_EYE_UPPER, self.LEFT_EYE_LOWER
        )
        right_ear = self._calculate_ear(
            face_landmarks, self.RIGHT_EYE_UPPER, self.RIGHT_EYE_LOWER
        )
        avg_ear = (left_ear + right_ear) / 2.0

        # Detect blink (eyes close then open)
        eyes_closed = avg_ear < self.ear_threshold
        if eyes_closed and not self._was_eyes_closed:
            self._blink_count += 1
        self._was_eyes_closed = eyes_closed

        # Check blink rate at interval
        current_time = time.time()
        elapsed = current_time - self._last_check_time

        if elapsed >= self.check_interval:
            blinks_per_minute = (self._blink_count / elapsed) * 60

            low_rate = blinks_per_minute < self.low_blink_threshold
            if low_rate:
                self._status = f"Low blink rate: {blinks_per_minute:.1f}/min"
            else:
                self._status = f"Blink rate OK: {blinks_per_minute:.1f}/min"

            # Reset for next interval
            self._blink_count = 0
            self._last_check_time = current_time
            return low_rate
        else:
            self._status = f"Tracking: {self._blink_count} blinks in {elapsed:.0f}s"

        return False
