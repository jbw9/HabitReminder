#!/usr/bin/env python3
"""
Eye Rubbing Detector
Detects when user rubs their eyes (indicates fatigue or irritation).
Requires hand landmark detection.
Calibrated threshold: 0.02 proximity, 30 frames (~1s).
"""

import numpy as np
from detectors.base_detector import BaseDetector


class EyeRubbingDetector(BaseDetector):
    """Detects eye rubbing behavior via hand-to-eye proximity."""

    # Eye landmark indices
    LEFT_EYE = 33
    RIGHT_EYE = 263

    # Hand points to check (wrist + fingertips)
    HAND_CHECK_POINTS = [0, 4, 8, 12, 16, 20]

    def __init__(self, proximity_threshold=0.02, frames_threshold=30):
        super().__init__(
            name="Eye Rubbing",
            alert_message="Stop rubbing your eyes! This can cause irritation.",
        )
        self.proximity_threshold = proximity_threshold
        self.frames_threshold = frames_threshold
        self._counter = 0

    def on_enable(self):
        self._counter = 0

    def on_disable(self):
        self._counter = 0

    def detect(self, face_landmarks, hand_landmarks, frame_width, frame_height):
        if not face_landmarks:
            self._status = "No face detected"
            self._counter = 0
            return False

        if not hand_landmarks:
            self._status = "No hands detected"
            self._counter = 0
            return False

        left_eye = face_landmarks[self.LEFT_EYE]
        right_eye = face_landmarks[self.RIGHT_EYE]

        for hand in hand_landmarks:
            for point_idx in self.HAND_CHECK_POINTS:
                point = hand[point_idx]
                hand_x, hand_y = point.x, point.y

                dist_left = np.sqrt(
                    (hand_x - left_eye.x) ** 2 + (hand_y - left_eye.y) ** 2
                )
                dist_right = np.sqrt(
                    (hand_x - right_eye.x) ** 2 + (hand_y - right_eye.y) ** 2
                )

                if dist_left < self.proximity_threshold or dist_right < self.proximity_threshold:
                    self._counter += 1
                    self._status = f"Hand near eyes ({self._counter}/{self.frames_threshold})"
                    if self._counter >= self.frames_threshold:
                        self._counter = 0
                        return True
                    return False

        self._counter = 0
        self._status = "No eye rubbing detected"
        return False
