#!/usr/bin/env python3
"""
Mouth Breathing Detector
Detects when user breathes through mouth instead of nose.
"""

import cv2
from base_models.base_detector import BaseDetector


class MouthBreathingDetector(BaseDetector):
    """Detects mouth breathing patterns."""

    def __init__(self, threshold=0.05, frames_threshold=120):
        super().__init__(
            name="Mouth Breathing",
            warning_message="CLOSE YOUR MOUTH!\nBreathe through your nose",
            warning_threshold=frames_threshold,
            warning_duration=3
        )
        self.mar_threshold = threshold

    def calculate_mouth_aspect_ratio(self, landmarks):
        """Calculate Mouth Aspect Ratio (MAR)."""
        try:
            # Get vertical distance (lip separation)
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            vertical = abs(upper_lip.y - lower_lip.y)

            # Get horizontal distance (mouth width)
            left_corner = landmarks[61]
            right_corner = landmarks[291]
            horizontal = abs(right_corner.x - left_corner.x)

            if horizontal < 0.001:
                return 0

            return vertical / horizontal
        except Exception:
            return 0

    def detect(self, face_landmarks, frame_width, frame_height):
        """Detect if mouth is open."""
        if not face_landmarks:
            self.status = "No face detected"
            return False

        mar = self.calculate_mouth_aspect_ratio(face_landmarks)

        if mar > self.mar_threshold:
            self.status = f"MOUTH OPEN ({self.counter} frames)"
            return True
        else:
            self.status = "Mouth closed - Good!"
            return False

    def draw_overlay(self, frame, face_landmarks, frame_width, frame_height):
        """Draw mouth detection box."""
        if not face_landmarks:
            return

        try:
            # Draw circles on key mouth points
            for idx in [13, 14, 61, 291]:
                landmark = face_landmarks[idx]
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # Draw rectangle around mouth
            mouth_points = []
            for idx in [61, 291, 13, 14, 78, 308, 95, 88]:
                landmark = face_landmarks[idx]
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                mouth_points.append((x, y))

            if mouth_points:
                xs = [p[0] for p in mouth_points]
                ys = [p[1] for p in mouth_points]
                x_min, x_max = min(xs) - 10, max(xs) + 10
                y_min, y_max = min(ys) - 10, max(ys) + 10

                mar = self.calculate_mouth_aspect_ratio(face_landmarks)
                color = (0, 255, 0) if mar < self.mar_threshold else (0, 0, 255)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                label = "CLOSED" if mar < self.mar_threshold else "OPEN"
                cv2.putText(frame, label, (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception:
            pass
