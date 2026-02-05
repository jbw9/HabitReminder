#!/usr/bin/env python3
"""
Fatigue Detector
Detects yawning as an indicator of fatigue/tiredness.
"""

import cv2
import time
from base_models.base_detector import BaseDetector


class FatigueDetector(BaseDetector):
    """Detects fatigue through yawning."""

    def __init__(self, yawn_threshold=0.6, yawn_duration_frames=20, max_yawns=3):
        """
        Args:
            yawn_threshold: MAR threshold for yawn detection (higher than mouth breathing)
            yawn_duration_frames: Minimum frames for yawn
            max_yawns: Maximum yawns in 5 minutes before warning
        """
        super().__init__(
            name="Fatigue",
            warning_message="YOU LOOK TIRED!\nConsider taking a break",
            warning_threshold=1,  # Trigger immediately
            warning_duration=5
        )
        self.yawn_threshold = yawn_threshold
        self.yawn_duration_frames = yawn_duration_frames
        self.max_yawns = max_yawns

        # Yawn tracking
        self.yawn_counter = 0
        self.yawns_in_period = []
        self.check_period = 300  # 5 minutes

    def calculate_mouth_aspect_ratio(self, landmarks):
        """Calculate MAR for yawn detection."""
        try:
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            vertical = abs(upper_lip.y - lower_lip.y)

            left_corner = landmarks[61]
            right_corner = landmarks[291]
            horizontal = abs(right_corner.x - left_corner.x)

            if horizontal < 0.001:
                return 0

            return vertical / horizontal
        except Exception:
            return 0

    def detect(self, face_landmarks, frame_width, frame_height):
        """Detect yawning."""
        if not face_landmarks:
            self.status = "No face detected"
            return False

        mar = self.calculate_mouth_aspect_ratio(face_landmarks)

        # Check for yawn (wide mouth opening)
        if mar > self.yawn_threshold:
            self.yawn_counter += 1

            if self.yawn_counter == self.yawn_duration_frames:
                # Yawn detected!
                self.yawns_in_period.append(time.time())
        else:
            self.yawn_counter = 0

        # Remove old yawns outside time window
        current_time = time.time()
        self.yawns_in_period = [t for t in self.yawns_in_period
                                if current_time - t < self.check_period]

        # Check if too many yawns
        yawn_count = len(self.yawns_in_period)

        if yawn_count >= self.max_yawns:
            self.status = f"FATIGUE DETECTED! {yawn_count} yawns in 5 min"
            return True
        elif yawn_count > 0:
            self.status = f"Yawns detected: {yawn_count}/5min"
        else:
            self.status = "No fatigue detected"

        return False

    def draw_overlay(self, frame, face_landmarks, frame_width, frame_height):
        """Draw yawn indicator."""
        if not face_landmarks:
            return

        try:
            # Draw mouth region
            mouth_points = []
            for idx in [61, 291, 13, 14]:
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

                if mar > self.yawn_threshold and self.yawn_counter > 5:
                    color = (0, 165, 255)  # Orange - yawning
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(frame, "YAWN", (x_min, y_min - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception:
            pass
