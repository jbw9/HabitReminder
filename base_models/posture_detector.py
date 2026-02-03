#!/usr/bin/env python3
"""
Posture Detector
Monitors head position and distance from screen.
Alerts when too close or head is tilted.
"""

import cv2
import numpy as np
from base_models.base_detector import BaseDetector


class PostureDetector(BaseDetector):
    """Detects poor posture (too close to screen or head tilt)."""

    def __init__(self, min_face_width_ratio=0.35, max_tilt_angle=15, frames_threshold=90):
        """
        Args:
            min_face_width_ratio: Minimum face width relative to frame (closer = larger ratio)
            max_tilt_angle: Maximum acceptable head tilt in degrees
            frames_threshold: Frames before warning (~3 seconds)
        """
        super().__init__(
            name="Posture",
            warning_message="ADJUST YOUR POSTURE!\nSit back or straighten your head",
            warning_threshold=frames_threshold,
            warning_duration=3
        )
        self.min_face_width_ratio = min_face_width_ratio
        self.max_tilt_angle = max_tilt_angle
        self.issue_type = None

    def calculate_head_tilt(self, landmarks):
        """Calculate head tilt angle using eye landmarks."""
        try:
            # Use eye corners to calculate tilt
            left_eye = landmarks[33]  # Left eye outer corner
            right_eye = landmarks[263]  # Right eye outer corner

            # Calculate angle
            dx = right_eye.x - left_eye.x
            dy = right_eye.y - left_eye.y
            angle = np.degrees(np.arctan2(dy, dx))

            return abs(angle)
        except Exception:
            return 0

    def calculate_face_width_ratio(self, landmarks, frame_width):
        """Calculate face width relative to frame width."""
        try:
            # Use face outer points
            left_face = landmarks[234]  # Left face edge
            right_face = landmarks[454]  # Right face edge

            face_width = abs(right_face.x - left_face.x)
            return face_width
        except Exception:
            return 0

    def detect(self, face_landmarks, frame_width, frame_height):
        """Detect posture issues."""
        if not face_landmarks:
            self.status = "No face detected"
            return False

        # Check distance (face width)
        face_width_ratio = self.calculate_face_width_ratio(face_landmarks, frame_width)
        too_close = face_width_ratio > self.min_face_width_ratio

        # Check head tilt
        tilt_angle = self.calculate_head_tilt(face_landmarks)
        tilted = tilt_angle > self.max_tilt_angle

        if too_close:
            self.status = f"TOO CLOSE! Move back (face width: {face_width_ratio:.2f})"
            self.issue_type = "distance"
            return True
        elif tilted:
            self.status = f"HEAD TILTED! Straighten up (tilt: {tilt_angle:.1f}°)"
            self.issue_type = "tilt"
            return True
        else:
            self.status = f"Posture OK (distance: {face_width_ratio:.2f}, tilt: {tilt_angle:.1f}°)"
            self.issue_type = None
            return False

    def draw_overlay(self, frame, face_landmarks, frame_width, frame_height):
        """Draw posture indicators."""
        if not face_landmarks:
            return

        try:
            # Draw line between eyes to show tilt
            left_eye = face_landmarks[33]
            right_eye = face_landmarks[263]

            left_point = (int(left_eye.x * frame_width), int(left_eye.y * frame_height))
            right_point = (int(right_eye.x * frame_width), int(right_eye.y * frame_height))

            tilt_angle = self.calculate_head_tilt(face_landmarks)
            face_width_ratio = self.calculate_face_width_ratio(face_landmarks, frame_width)

            # Color based on issues
            if face_width_ratio > self.min_face_width_ratio or tilt_angle > self.max_tilt_angle:
                color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 0)  # Green

            cv2.line(frame, left_point, right_point, color, 2)

            # Draw reference horizontal line
            mid_y = (left_point[1] + right_point[1]) // 2
            cv2.line(frame, (left_point[0] - 20, mid_y),
                    (right_point[0] + 20, mid_y), (255, 255, 255), 1)

        except Exception:
            pass
