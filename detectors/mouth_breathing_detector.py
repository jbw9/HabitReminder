#!/usr/bin/env python3
"""
Mouth Breathing Detector
Detects when user breathes through mouth instead of nose.
Uses Mouth Aspect Ratio (MAR) calculation.
Uses inner lip landmarks (82, 87) for accurate detection.
Calibrated threshold: 0.15 MAR, 120 frames (~4s at 30fps).
"""

from detectors.base_detector import BaseDetector


class MouthBreathingDetector(BaseDetector):
    """Detects mouth breathing patterns."""

    # Mouth landmark indices (MediaPipe Face Mesh)
    # Using INNER lip edges (82, 87) instead of outer (13, 14)
    # Inner edges touch when mouth is closed, giving MAR ≈ 0
    UPPER_LIP_INNER = 82
    LOWER_LIP_INNER = 87
    LEFT_CORNER = 61
    RIGHT_CORNER = 291

    def __init__(self, mar_threshold=0.15, frames_threshold=120):
        super().__init__(
            name="Mouth Breathing",
            alert_message="Close your mouth! Breathe through your nose.",
        )
        self.mar_threshold = mar_threshold
        self.frames_threshold = frames_threshold
        self._counter = 0

    def on_enable(self):
        self._counter = 0

    def on_disable(self):
        self._counter = 0

    def _calculate_mar(self, landmarks):
        """Calculate Mouth Aspect Ratio (MAR) using inner lip edges."""
        try:
            upper_lip = landmarks[self.UPPER_LIP_INNER]
            lower_lip = landmarks[self.LOWER_LIP_INNER]
            vertical = abs(upper_lip.y - lower_lip.y)

            left_corner = landmarks[self.LEFT_CORNER]
            right_corner = landmarks[self.RIGHT_CORNER]
            horizontal = abs(right_corner.x - left_corner.x)

            if horizontal < 0.001:
                return 0
            return vertical / horizontal
        except Exception:
            return 0

    def detect(self, face_landmarks, hand_landmarks, frame_width, frame_height):
        if not face_landmarks:
            self._status = "No face detected"
            self._counter = 0
            return False

        mar = self._calculate_mar(face_landmarks)

        if mar > self.mar_threshold:
            self._counter += 1
            self._status = f"MOUTH OPEN (MAR:{mar:.3f}) [{self._counter}/{self.frames_threshold}]"
            if self._counter >= self.frames_threshold:
                self._counter = 0
                return True
        else:
            self._counter = 0
            self._status = f"Mouth closed (MAR:{mar:.3f})"

        return False
