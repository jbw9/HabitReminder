#!/usr/bin/env python3
"""
Face Touching Detector
Detects when user touches their face frequently (stress indicator, hygiene concern).
"""

import cv2
import numpy as np
import time
from base_models.base_detector import BaseDetector


class FaceTouchingDetector(BaseDetector):
    """Detects frequent face touching."""

    def __init__(self, proximity_threshold=0.2, max_touches=5):
        """
        Args:
            proximity_threshold: Distance threshold for hand near face
            max_touches: Maximum touches in 2 minutes
        """
        super().__init__(
            name="Face Touching",
            warning_message="STOP TOUCHING YOUR FACE!\nReduce stress and hygiene risk",
            warning_threshold=1,
            warning_duration=3
        )
        self.proximity_threshold = proximity_threshold
        self.max_touches = max_touches
        self.touches_in_period = []
        self.check_period = 120  # 2 minutes
        self.was_touching = False

    def detect(self, face_landmarks, frame_width, frame_height, hand_landmarks=None):
        """Detect face touching."""
        if not face_landmarks:
            self.status = "No face detected"
            return False

        if not hand_landmarks or len(hand_landmarks) == 0:
            self.status = "No hands detected"
            self.was_touching = False
            return False

        try:
            # Get face center and bounds
            nose = face_landmarks[1]  # Nose tip
            face_center_x = nose.x
            face_center_y = nose.y

            # Check if hand is near face
            hand_near_face = False

            for hand in hand_landmarks:
                # Check multiple hand points
                for point_idx in [0, 4, 8, 12, 16, 20]:  # Wrist and fingertips
                    point = hand.landmark[point_idx]

                    dist = np.sqrt((point.x - face_center_x)**2 +
                                 (point.y - face_center_y)**2)

                    if dist < self.proximity_threshold:
                        hand_near_face = True
                        break

                if hand_near_face:
                    break

            # Detect new touch event (transition from not touching to touching)
            if hand_near_face and not self.was_touching:
                # New touch detected
                self.touches_in_period.append(time.time())

            self.was_touching = hand_near_face

            # Remove old touches
            current_time = time.time()
            self.touches_in_period = [t for t in self.touches_in_period
                                     if current_time - t < self.check_period]

            touch_count = len(self.touches_in_period)

            if touch_count >= self.max_touches:
                self.status = f"FREQUENT FACE TOUCHING! {touch_count} touches in 2min"
                return True
            elif touch_count > 0:
                self.status = f"Face touches: {touch_count}/2min"
            else:
                self.status = "No excessive face touching"

            return False

        except Exception:
            self.status = "Error in detection"
            return False

    def draw_overlay(self, frame, face_landmarks, frame_width, frame_height, hand_landmarks=None):
        """Draw face boundary."""
        if not face_landmarks:
            return

        try:
            # Draw circle around face center
            nose = face_landmarks[1]
            x = int(nose.x * frame_width)
            y = int(nose.y * frame_height)

            radius = int(self.proximity_threshold * frame_width)
            color = (0, 0, 255) if self.was_touching else (0, 255, 0)
            cv2.circle(frame, (x, y), radius, color, 2)

            if self.was_touching:
                cv2.putText(frame, "TOUCHING", (x - 40, y - radius - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        except Exception:
            pass
