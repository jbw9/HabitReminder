#!/usr/bin/env python3
"""
Hydration Reminder Detector
Time-based reminder to drink water regularly.
"""

import cv2
import time
from base_models.base_detector import BaseDetector


class HydrationDetector(BaseDetector):
    """Reminds user to drink water at regular intervals."""

    def __init__(self, interval_minutes=45):
        """
        Args:
            interval_minutes: Minutes between hydration reminders
        """
        super().__init__(
            name="Hydration",
            warning_message="TIME TO HYDRATE!\nDrink some water",
            warning_threshold=1,
            warning_duration=5
        )
        self.interval_seconds = interval_minutes * 60
        self.last_reminder_time = time.time()
        self.next_reminder_time = self.last_reminder_time + self.interval_seconds

    def detect(self, face_landmarks, frame_width, frame_height):
        """Check if it's time for hydration reminder."""
        current_time = time.time()
        time_since_last = current_time - self.last_reminder_time
        time_until_next = self.next_reminder_time - current_time

        if current_time >= self.next_reminder_time:
            # Time to remind!
            self.last_reminder_time = current_time
            self.next_reminder_time = current_time + self.interval_seconds
            self.status = "DRINK WATER NOW!"
            return True
        else:
            minutes_left = int(time_until_next / 60)
            seconds_left = int(time_until_next % 60)
            self.status = f"Next reminder in {minutes_left}m {seconds_left}s"
            return False

    def draw_overlay(self, frame, face_landmarks, frame_width, frame_height):
        """Draw hydration status."""
        # Show a small water drop icon position
        try:
            current_time = time.time()
            time_until_next = self.next_reminder_time - current_time

            if time_until_next < 60:  # Last minute warning
                # Draw water drop indicator
                center_x = frame_width - 50
                center_y = 50

                cv2.circle(frame, (center_x, center_y), 20, (255, 200, 0), 2)
                cv2.putText(frame, "H2O", (center_x - 15, center_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
        except Exception:
            pass

    def reset_timer(self):
        """Reset the hydration timer (call after user drinks)."""
        self.last_reminder_time = time.time()
        self.next_reminder_time = self.last_reminder_time + self.interval_seconds
        self.status = "Timer reset - stay hydrated!"
