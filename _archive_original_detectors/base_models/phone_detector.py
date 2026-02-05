#!/usr/bin/env python3
"""
Phone/Distraction Detector
Detects when user is likely looking at their phone (head down, hands low).
"""

import cv2
import numpy as np
import time
from base_models.base_detector import BaseDetector


class PhoneDetector(BaseDetector):
    """Detects phone usage patterns."""

    def __init__(self, head_down_threshold=0.15, frames_threshold=90):
        """
        Args:
            head_down_threshold: Threshold for head tilt down
            frames_threshold: Frames before warning (~3 seconds)
        """
        super().__init__(
            name="Phone Distraction",
            warning_message="PUT YOUR PHONE AWAY!\nFocus on your work",
            warning_threshold=frames_threshold,
            warning_duration=4
        )
        self.head_down_threshold = head_down_threshold
        self.phone_instances = []
        self.check_period = 300  # 5 minutes

    def detect_head_down(self, landmarks):
        """Detect if head is tilted down (looking at phone)."""
        try:
            # Compare nose and forehead positions
            nose = landmarks[1]
            forehead = landmarks[10]

            # If nose is significantly higher than forehead (in y-coordinate),
            # head is tilted down
            head_down = (nose.y - forehead.y) > self.head_down_threshold

            return head_down

        except Exception:
            return False

    def detect(self, face_landmarks, frame_width, frame_height, hand_landmarks=None):
        """Detect phone usage."""
        if not face_landmarks:
            self.status = "No face detected"
            return False

        head_down = self.detect_head_down(face_landmarks)

        # Additional heuristic: hands below face (typical phone holding position)
        hands_low = False
        if hand_landmarks:
            try:
                face_bottom = face_landmarks[152].y  # Chin

                for hand in hand_landmarks:
                    # Check if hand is below face
                    hand_y = hand.landmark[0].y  # Wrist position

                    if hand_y > face_bottom:
                        hands_low = True
                        break
            except Exception:
                pass

        # Phone usage likely if head is down (and optionally hands are low)
        phone_likely = head_down

        if phone_likely:
            self.status = "HEAD DOWN - Possible phone use"
            return True
        else:
            self.status = "Head position normal"
            return False

    def draw_overlay(self, frame, face_landmarks, frame_width, frame_height, hand_landmarks=None):
        """Draw head position indicator."""
        if not face_landmarks:
            return

        try:
            # Draw line from forehead to nose
            nose = face_landmarks[1]
            forehead = face_landmarks[10]

            nose_point = (int(nose.x * frame_width), int(nose.y * frame_height))
            forehead_point = (int(forehead.x * frame_width), int(forehead.y * frame_height))

            head_down = self.detect_head_down(face_landmarks)
            color = (0, 0, 255) if head_down else (0, 255, 0)

            cv2.line(frame, forehead_point, nose_point, color, 3)

            if head_down:
                cv2.putText(frame, "HEAD DOWN", (nose_point[0] + 10, nose_point[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        except Exception:
            pass
