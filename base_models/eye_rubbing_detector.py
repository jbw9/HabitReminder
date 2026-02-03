#!/usr/bin/env python3
"""
Eye Rubbing Detector
Detects when user rubs their eyes (indicates fatigue or irritation).
Requires hand landmark detection.
"""

import cv2
import numpy as np
from base_models.base_detector import BaseDetector


class EyeRubbingDetector(BaseDetector):
    """Detects eye rubbing behavior."""

    def __init__(self, proximity_threshold=0.15, frames_threshold=30):
        """
        Args:
            proximity_threshold: Distance threshold for hand near eye
            frames_threshold: Frames before warning (~1 second)
        """
        super().__init__(
            name="Eye Rubbing",
            warning_message="STOP RUBBING YOUR EYES!\nThis can cause irritation",
            warning_threshold=frames_threshold,
            warning_duration=3
        )
        self.proximity_threshold = proximity_threshold

    def detect(self, face_landmarks, frame_width, frame_height, hand_landmarks=None):
        """Detect eye rubbing using hand and face landmarks."""
        if not face_landmarks:
            self.status = "No face detected"
            return False

        if not hand_landmarks or len(hand_landmarks) == 0:
            self.status = "No hands detected"
            return False

        try:
            # Get eye positions
            left_eye = face_landmarks[33]  # Left eye
            right_eye = face_landmarks[263]  # Right eye

            # Check each hand
            for hand in hand_landmarks:
                # Get hand center (wrist + middle finger tip)
                wrist = hand.landmark[0]
                fingertip = hand.landmark[9]  # Middle finger tip

                hand_center_x = (wrist.x + fingertip.x) / 2
                hand_center_y = (wrist.y + fingertip.y) / 2

                # Check distance to eyes
                dist_left = np.sqrt((hand_center_x - left_eye.x)**2 +
                                   (hand_center_y - left_eye.y)**2)
                dist_right = np.sqrt((hand_center_x - right_eye.x)**2 +
                                    (hand_center_y - right_eye.y)**2)

                if dist_left < self.proximity_threshold or dist_right < self.proximity_threshold:
                    self.status = "HAND NEAR EYES - Possible rubbing"
                    return True

            self.status = "No eye rubbing detected"
            return False

        except Exception:
            self.status = "Error in detection"
            return False

    def draw_overlay(self, frame, face_landmarks, frame_width, frame_height, hand_landmarks=None):
        """Draw eye regions."""
        if not face_landmarks:
            return

        try:
            # Draw circles around eyes
            for eye_idx in [33, 263]:  # Left and right eye
                landmark = face_landmarks[eye_idx]
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                color = (0, 0, 255) if self.counter > 0 else (0, 255, 0)
                cv2.circle(frame, (x, y), 30, color, 2)

        except Exception:
            pass
