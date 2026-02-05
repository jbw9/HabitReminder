#!/usr/bin/env python3
"""
Alert System
Handles macOS notifications and audio alerts with cooldown management.
"""

import os
import subprocess
import time


class AlertSystem:
    """Displays visual and audio alerts for detector violations."""

    def __init__(self, cooldown_seconds=60, notifications_enabled=True,
                 audio_enabled=True):
        self.cooldown_seconds = cooldown_seconds
        self.notifications_enabled = notifications_enabled
        self.audio_enabled = audio_enabled

        self._cooldowns = {}  # detector_name -> last_alert_time
        self._pygame_initialized = False

    def send_alert(self, alert):
        """
        Process an alert dict and send notification + sound.

        Args:
            alert: dict with 'detector', 'message', 'severity' keys
        """
        detector_name = alert['detector']

        # Check cooldown
        if self._is_on_cooldown(detector_name):
            return

        # Send notification
        if self.notifications_enabled:
            self._show_notification("Habit Tracker", alert['message'])

        # Play sound
        if self.audio_enabled:
            self._play_system_sound(alert.get('severity', 'normal'))

        # Record cooldown
        self._cooldowns[detector_name] = time.time()

    def _is_on_cooldown(self, detector_name):
        """Check if a detector's alert is still on cooldown."""
        if detector_name not in self._cooldowns:
            return False
        elapsed = time.time() - self._cooldowns[detector_name]
        return elapsed < self.cooldown_seconds

    def _show_notification(self, title, message):
        """Display a macOS notification using osascript."""
        # Escape quotes in the message
        safe_message = message.replace('"', '\\"')
        safe_title = title.replace('"', '\\"')
        script = f'display notification "{safe_message}" with title "{safe_title}"'
        try:
            subprocess.run(
                ['osascript', '-e', script],
                capture_output=True, timeout=5
            )
        except Exception:
            pass

    def _play_system_sound(self, severity):
        """Play a macOS system sound using afplay."""
        # Use built-in macOS sounds - no external files needed
        if severity == 'high':
            sound = '/System/Library/Sounds/Sosumi.aiff'
        else:
            sound = '/System/Library/Sounds/Glass.aiff'

        if os.path.exists(sound):
            try:
                subprocess.Popen(
                    ['afplay', sound],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
