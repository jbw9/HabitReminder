#!/usr/bin/env python3
"""
Hydration Reminder Detector
Time-based reminder to drink water at regular intervals.
Default interval: 45 minutes.
"""

import time
from detectors.base_detector import BaseDetector


class HydrationDetector(BaseDetector):
    """Reminds user to drink water at regular intervals."""

    def __init__(self, interval_minutes=45):
        super().__init__(
            name="Hydration",
            alert_message="Time to hydrate! Drink some water.",
        )
        self.interval_seconds = interval_minutes * 60
        self._last_reminder_time = time.time()
        self._next_reminder_time = self._last_reminder_time + self.interval_seconds

    def on_enable(self):
        self._last_reminder_time = time.time()
        self._next_reminder_time = self._last_reminder_time + self.interval_seconds

    def on_disable(self):
        pass

    def detect(self, face_landmarks, hand_landmarks, frame_width, frame_height):
        now = time.time()
        time_until_next = self._next_reminder_time - now

        if now >= self._next_reminder_time:
            # Time to remind
            self._last_reminder_time = now
            self._next_reminder_time = now + self.interval_seconds
            self._status = "Drink water now!"
            return True

        minutes_left = int(time_until_next / 60)
        seconds_left = int(time_until_next % 60)
        self._status = f"Next reminder in {minutes_left}m {seconds_left}s"
        return False

    def reset_timer(self):
        """Reset the hydration timer (call when user drinks water)."""
        self._last_reminder_time = time.time()
        self._next_reminder_time = self._last_reminder_time + self.interval_seconds
        self._status = "Timer reset"
