#!/usr/bin/env python3
"""
Base Detector Class
Abstract base class for all health monitoring detectors.
Thread-safe design for use with background camera processing.
"""

import time
import threading
from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """Base class for all health monitoring detectors."""

    def __init__(self, name, alert_message, alert_cooldown=60):
        """
        Args:
            name: Display name of the detector
            alert_message: Message shown when alert triggers
            alert_cooldown: Seconds between repeated alerts
        """
        self.name = name
        self.alert_message = alert_message
        self.alert_cooldown = alert_cooldown

        # State
        self._enabled = False
        self._status = "Disabled"
        self._last_alert_time = 0
        self._lock = threading.Lock()

    def enable(self):
        """Enable this detector."""
        with self._lock:
            self._enabled = True
            self._status = "Monitoring"
            self.on_enable()

    def disable(self):
        """Disable this detector and reset state."""
        with self._lock:
            self._enabled = False
            self._status = "Disabled"
            self.on_disable()

    def is_enabled(self):
        """Check if detector is enabled."""
        return self._enabled

    def on_enable(self):
        """Hook called when detector is enabled. Override to reset state."""
        pass

    def on_disable(self):
        """Hook called when detector is disabled. Override to clean up."""
        pass

    @abstractmethod
    def detect(self, face_landmarks, hand_landmarks, frame_width, frame_height):
        """
        Run detection on the current frame's landmarks.

        Args:
            face_landmarks: List of MediaPipe face landmarks (or None)
            hand_landmarks: List of hand landmark lists (or None)
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels

        Returns:
            bool: True if the problematic condition is detected
        """
        pass

    def should_alert(self):
        """Check if an alert should be fired (respects cooldown)."""
        now = time.time()
        if now - self._last_alert_time >= self.alert_cooldown:
            return True
        return False

    def mark_alerted(self):
        """Record that an alert was just sent."""
        self._last_alert_time = time.time()

    def get_alert_message(self):
        """Return the alert message for notifications."""
        return self.alert_message

    def get_severity(self):
        """Return alert severity. Override in subclasses for custom severity."""
        return "normal"

    def get_status(self):
        """Get current human-readable status string."""
        return self._status
