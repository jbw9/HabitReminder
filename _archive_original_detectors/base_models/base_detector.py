#!/usr/bin/env python3
"""
Base Detector Class
All health monitoring detectors inherit from this base class.
"""

import time
from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """Base class for all health monitoring detectors."""

    def __init__(self, name, warning_message, warning_threshold, warning_duration=3):
        """
        Initialize base detector.

        Args:
            name: Display name of the detector
            warning_message: Message to show when warning triggers
            warning_threshold: Frames/count before warning triggers
            warning_duration: How long warning displays (seconds)
        """
        self.name = name
        self.warning_message = warning_message
        self.warning_threshold = warning_threshold
        self.warning_duration = warning_duration

        # State tracking
        self.counter = 0
        self.warning_active = False
        self.warning_start_time = 0
        self.enabled = True
        self.status = "Initializing..."

    @abstractmethod
    def detect(self, face_landmarks, frame_width, frame_height):
        """
        Perform detection on the current frame.

        Args:
            face_landmarks: MediaPipe face landmarks
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels

        Returns:
            bool: True if issue detected, False otherwise
        """
        pass

    @abstractmethod
    def draw_overlay(self, frame, face_landmarks, frame_width, frame_height):
        """
        Draw visual indicators on the frame.

        Args:
            frame: OpenCV frame
            face_landmarks: MediaPipe face landmarks
            frame_width: Frame width
            frame_height: Frame height
        """
        pass

    def update(self, detection_result):
        """
        Update detector state based on detection result.

        Args:
            detection_result: Boolean result from detect()
        """
        if detection_result:
            self.counter += 1
            if self.counter >= self.warning_threshold:
                self.trigger_warning()
        else:
            self.counter = 0

    def trigger_warning(self):
        """Trigger warning display."""
        if not self.warning_active:
            self.warning_active = True
            self.warning_start_time = time.time()

    def check_warning_timeout(self):
        """Check if warning should be dismissed."""
        if self.warning_active and (time.time() - self.warning_start_time) > self.warning_duration:
            self.warning_active = False

    def get_status(self):
        """Get current status string."""
        return self.status

    def is_warning_active(self):
        """Check if warning is currently active."""
        return self.warning_active

    def reset(self):
        """Reset detector state."""
        self.counter = 0
        self.warning_active = False
        self.status = "Ready"
