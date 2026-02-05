#!/usr/bin/env python3
"""
Focus/Attention Detector
Detects when user is looking away from screen for extended periods.
"""

import cv2
import numpy as np
from base_models.base_detector import BaseDetector


class FocusDetector(BaseDetector):
    """Detects lack of focus on screen."""

    def __init__(self, gaze_threshold=0.3, frames_threshold=150):
        """
        Args:
            gaze_threshold: Threshold for off-screen gaze
            frames_threshold: Frames before warning (~5 seconds)
        """
        super().__init__(
            name="Screen Focus",
            warning_message="FOCUS ON YOUR WORK!\nYou've been distracted",
            warning_threshold=frames_threshold,
            warning_duration=3
        )
        self.gaze_threshold = gaze_threshold

    def estimate_gaze_direction(self, landmarks):
        """Estimate if user is looking at screen or away."""
        try:
            # Get nose tip and eye centers
            nose = landmarks[1]
            left_eye = landmarks[33]
            right_eye = landmarks[263]

            # Calculate eye center
            eye_center_x = (left_eye.x + right_eye.x) / 2
            eye_center_y = (left_eye.y + right_eye.y) / 2

            # Calculate nose to eye center vector (gaze approximation)
            # If nose is significantly off-center from eyes, likely looking away
            gaze_offset_x = abs(nose.x - eye_center_x)
            gaze_offset_y = abs(nose.y - eye_center_y)

            # Looking away if nose is too far from eye center horizontally
            looking_away = gaze_offset_x > self.gaze_threshold

            return looking_away, gaze_offset_x

        except Exception:
            return False, 0

    def detect(self, face_landmarks, frame_width, frame_height):
        """Detect if user is not focused on screen."""
        if not face_landmarks:
            self.status = "No face detected - Not at desk?"
            # Count this as not focused
            return True

        looking_away, offset = self.estimate_gaze_direction(face_landmarks)

        if looking_away:
            self.status = f"LOOKING AWAY (offset: {offset:.2f})"
            return True
        else:
            self.status = f"Focused on screen (offset: {offset:.2f})"
            return False

    def draw_overlay(self, frame, face_landmarks, frame_width, frame_height):
        """Draw gaze indicator."""
        if not face_landmarks:
            return

        try:
            # Draw line from eyes to nose to visualize gaze
            left_eye = face_landmarks[33]
            right_eye = face_landmarks[263]
            nose = face_landmarks[1]

            eye_center_x = int(((left_eye.x + right_eye.x) / 2) * frame_width)
            eye_center_y = int(((left_eye.y + right_eye.y) / 2) * frame_height)

            nose_x = int(nose.x * frame_width)
            nose_y = int(nose.y * frame_height)

            looking_away, _ = self.estimate_gaze_direction(face_landmarks)
            color = (0, 0, 255) if looking_away else (0, 255, 0)

            cv2.line(frame, (eye_center_x, eye_center_y),
                    (nose_x, nose_y), color, 2)
            cv2.circle(frame, (eye_center_x, eye_center_y), 5, color, -1)
            cv2.circle(frame, (nose_x, nose_y), 5, color, -1)

        except Exception:
            pass
