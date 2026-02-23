#!/usr/bin/env python3
"""
Nail Biting Detector
Detects when user brings fingers/hands close to their mouth.
Calculates distance between fingertips and mouth center.
"""

import math
from detectors.base_detector import BaseDetector


class NailBitingDetector(BaseDetector):
    """Detects nail biting by tracking hand-to-mouth proximity."""

    # Fingertip landmark indices (MediaPipe Hand model has 21 landmarks per hand)
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

    # Face landmarks for mouth center calculation
    UPPER_LIP = 13
    LOWER_LIP = 14
    LEFT_CORNER = 61
    RIGHT_CORNER = 291

    def __init__(self, distance_threshold=0.08, frames_threshold=60):
        """
        Args:
            distance_threshold: Normalized distance threshold (0-1 range, default 0.08)
                               Fingertips closer than this to mouth trigger detection
            frames_threshold: Number of consecutive frames before alert (default 60 = ~2s at 30fps)
        """
        super().__init__(
            name="Nail Biting",
            alert_message="Stop biting your nails! Take your hand away from your mouth.",
        )
        self.distance_threshold = distance_threshold
        self.frames_threshold = frames_threshold
        self._counter = 0
        self._alert_triggered = False  # fires exactly once per proximity event
        self._last_min_distance = 1.0  # for debugging

    def on_enable(self):
        self._counter = 0
        self._alert_triggered = False
        self._last_min_distance = 1.0

    def on_disable(self):
        self._counter = 0
        self._alert_triggered = False
        self._last_min_distance = 1.0

    def _get_mouth_center(self, face_landmarks):
        """
        Calculate the center point of the mouth.
        Uses average of key mouth landmarks for robustness.

        Returns:
            tuple: (x, y) normalized coordinates, or None if calculation fails
        """
        try:
            upper_lip = face_landmarks[self.UPPER_LIP]
            lower_lip = face_landmarks[self.LOWER_LIP]
            left_corner = face_landmarks[self.LEFT_CORNER]
            right_corner = face_landmarks[self.RIGHT_CORNER]

            # Average position of these 4 mouth points
            center_x = (upper_lip.x + lower_lip.x + left_corner.x + right_corner.x) / 4
            center_y = (upper_lip.y + lower_lip.y + left_corner.y + right_corner.y) / 4

            return (center_x, center_y)
        except Exception:
            return None

    def _calculate_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two 2D points.

        Args:
            point1: tuple (x, y)
            point2: tuple (x, y)

        Returns:
            float: Euclidean distance
        """
        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]
        return math.sqrt(dx * dx + dy * dy)

    def _get_min_fingertip_distance(self, hand_landmarks_list, mouth_center):
        """
        Find the minimum distance between any fingertip and the mouth center.

        Args:
            hand_landmarks_list: List of hand landmark sets (one per detected hand)
            mouth_center: tuple (x, y) of mouth center position

        Returns:
            float: Minimum distance found, or infinity if no hands detected
        """
        min_distance = float('inf')

        for hand_landmarks in hand_landmarks_list:
            for tip_idx in self.FINGERTIPS:
                try:
                    fingertip = hand_landmarks[tip_idx]
                    fingertip_pos = (fingertip.x, fingertip.y)
                    distance = self._calculate_distance(fingertip_pos, mouth_center)
                    min_distance = min(min_distance, distance)
                except Exception:
                    continue

        return min_distance

    def detect(self, face_landmarks, hand_landmarks, frame_width, frame_height):
        """
        Detect if hand is too close to mouth for sustained period.

        Returns:
            bool: True if alert should fire
        """
        # Need both face and hand detection
        if not face_landmarks:
            self._status = "No face detected"
            self._counter = 0
            self._alert_triggered = False
            self._last_min_distance = 1.0
            return False

        if not hand_landmarks or len(hand_landmarks) == 0:
            self._status = "No hands detected"
            self._counter = 0
            self._alert_triggered = False
            self._last_min_distance = 1.0
            return False

        # Get mouth center position
        mouth_center = self._get_mouth_center(face_landmarks)
        if mouth_center is None:
            self._status = "Cannot detect mouth center"
            self._counter = 0
            self._alert_triggered = False
            return False

        # Find minimum distance from any fingertip to mouth
        min_distance = self._get_min_fingertip_distance(hand_landmarks, mouth_center)
        self._last_min_distance = min_distance

        if min_distance < self.distance_threshold:
            # Hand is close to mouth
            if self._counter < self.frames_threshold:
                self._counter += 1

            self._status = f"HAND NEAR MOUTH (dist:{min_distance:.3f}) [{self._counter}/{self.frames_threshold}]"

            # Fire exactly once when threshold is first crossed
            if self._counter >= self.frames_threshold and not self._alert_triggered:
                self._alert_triggered = True
                return True
        else:
            # Hand is far from mouth — reset everything
            self._counter = 0
            self._alert_triggered = False
            self._status = f"Hand away (dist:{min_distance:.3f})"

        return False
