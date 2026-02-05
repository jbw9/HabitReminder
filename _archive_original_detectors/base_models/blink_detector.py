#!/usr/bin/env python3
"""
Blink Rate Detector
Detects insufficient blinking which can lead to dry eyes and eye strain.
Normal: 15-20 blinks/min, Computer use: often drops to 5-7 blinks/min
"""

import cv2
import time
from base_models.base_detector import BaseDetector


class BlinkDetector(BaseDetector):
    """Detects low blink rate."""

    # Eye landmark indices for MediaPipe Face Mesh
    LEFT_EYE_UPPER = [159, 145]
    LEFT_EYE_LOWER = [23, 130]
    RIGHT_EYE_UPPER = [386, 374]
    RIGHT_EYE_LOWER = [253, 359]

    def __init__(self, low_blink_threshold=6, check_interval=60):
        """
        Args:
            low_blink_threshold: Minimum blinks per minute
            check_interval: Check interval in seconds
        """
        super().__init__(
            name="Blink Rate",
            warning_message="BLINK MORE!\nYou're not blinking enough",
            warning_threshold=1,  # Trigger immediately when low
            warning_duration=3
        )
        self.low_blink_threshold = low_blink_threshold
        self.check_interval = check_interval

        # Blink tracking
        self.blink_count = 0
        self.last_check_time = time.time()
        self.was_eyes_closed = False
        self.ear_threshold = 0.012  # Eye Aspect Ratio threshold (balanced sensitivity)

    def calculate_eye_aspect_ratio(self, landmarks, top_indices, bottom_indices):
        """Calculate Eye Aspect Ratio (EAR) - simplified version."""
        try:
            import numpy as np

            # Get top and bottom points
            top_points = [landmarks[idx] for idx in top_indices]
            bottom_points = [landmarks[idx] for idx in bottom_indices]

            # Calculate average vertical distance
            vertical_distances = []
            for top in top_points:
                for bottom in bottom_points:
                    dist = abs(top.y - bottom.y)
                    vertical_distances.append(dist)

            avg_vertical = np.mean(vertical_distances)
            return avg_vertical
        except Exception:
            return 1.0

    def detect(self, face_landmarks, frame_width, frame_height):
        """Detect blink rate."""
        if not face_landmarks:
            self.status = "No face detected"
            return False

        # Calculate EAR for both eyes
        left_ear = self.calculate_eye_aspect_ratio(
            face_landmarks,
            self.LEFT_EYE_UPPER,
            self.LEFT_EYE_LOWER
        )
        right_ear = self.calculate_eye_aspect_ratio(
            face_landmarks,
            self.RIGHT_EYE_UPPER,
            self.RIGHT_EYE_LOWER
        )

        # Average EAR
        avg_ear = (left_ear + right_ear) / 2.0

        # Detect blink (eye closed then opened)
        eyes_closed = avg_ear < self.ear_threshold

        if eyes_closed and not self.was_eyes_closed:
            # Blink detected
            self.blink_count += 1

        self.was_eyes_closed = eyes_closed

        # Check blink rate periodically
        current_time = time.time()
        elapsed = current_time - self.last_check_time

        if elapsed >= self.check_interval:
            # Calculate blinks per minute
            blinks_per_minute = (self.blink_count / elapsed) * 60

            if blinks_per_minute < self.low_blink_threshold:
                self.status = f"Low blink rate: {blinks_per_minute:.1f}/min (need {self.low_blink_threshold}+)"
                # Reset for next interval
                self.blink_count = 0
                self.last_check_time = current_time
                return True
            else:
                self.status = f"Blink rate OK: {blinks_per_minute:.1f}/min"

            # Reset for next interval
            self.blink_count = 0
            self.last_check_time = current_time

        else:
            self.status = f"Tracking blinks: {self.blink_count} in {elapsed:.0f}s"

        return False

    def draw_overlay(self, frame, face_landmarks, frame_width, frame_height):
        """Draw eye indicators."""
        if not face_landmarks:
            return

        try:
            # Draw rectangles around eyes
            for eye_points in [self.LEFT_EYE_UPPER + self.LEFT_EYE_LOWER,
                             self.RIGHT_EYE_UPPER + self.RIGHT_EYE_LOWER]:
                points = []
                for idx in eye_points:
                    landmark = face_landmarks[idx]
                    x = int(landmark.x * frame_width)
                    y = int(landmark.y * frame_height)
                    points.append((x, y))

                if points:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    x_min, x_max = min(xs) - 5, max(xs) + 5
                    y_min, y_max = min(ys) - 5, max(ys) + 5

                    color = (0, 255, 0) if not self.was_eyes_closed else (255, 0, 0)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        except Exception:
            pass
