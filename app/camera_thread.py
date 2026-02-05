#!/usr/bin/env python3
"""
Camera Background Thread
Captures frames, processes them through DetectorManager, sends alerts
via queue, and optionally prepares annotated preview frames.
"""

import threading
import time
import cv2

from app.preview_window import draw_overlays

# Preview frame size (for inline menu bar preview)
PREVIEW_WIDTH = 320
PREVIEW_HEIGHT = 180


class CameraThread:
    """Background thread for camera capture and detector processing."""

    def __init__(self, detector_manager, alert_queue, fps=30,
                 camera_width=1280, camera_height=720):
        self.detector_manager = detector_manager
        self.alert_queue = alert_queue
        self.fps = fps
        self.camera_width = camera_width
        self.camera_height = camera_height

        self._thread = None
        self._running = False
        self._camera = None
        self._frame_timestamp_ms = 0

        # Preview state — main thread reads latest_preview_frame
        self._preview_enabled = False
        self.latest_preview_frame = None  # BGR numpy array (320x180)

    @property
    def running(self):
        return self._running

    @property
    def preview_enabled(self):
        return self._preview_enabled

    def set_preview(self, enabled):
        """Toggle preview frame generation on/off."""
        self._preview_enabled = enabled
        if not enabled:
            self.latest_preview_frame = None

    def start(self):
        """Start the camera thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._frame_timestamp_ms = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the camera thread and release resources."""
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self.latest_preview_frame = None

    def _initialize_camera(self):
        self._camera = cv2.VideoCapture(0)
        if not self._camera.isOpened():
            raise RuntimeError("Could not open camera")
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

    def _release_camera(self):
        if self._camera is not None:
            self._camera.release()
            self._camera = None

    def _run(self):
        """Main camera loop (background thread)."""
        try:
            self._initialize_camera()
        except RuntimeError as e:
            self.alert_queue.put({
                'detector': 'system',
                'message': str(e),
                'severity': 'high',
            })
            self._running = False
            return

        frame_delay = 1.0 / self.fps

        while self._running:
            loop_start = time.time()

            ret, frame = self._camera.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._frame_timestamp_ms += int(frame_delay * 1000)

            # Run detectors
            alerts = self.detector_manager.process_frame(
                rgb_frame, self._frame_timestamp_ms
            )
            for alert in alerts:
                self.alert_queue.put(alert)

            # Build preview frame if enabled
            if self._preview_enabled:
                annotated = draw_overlays(
                    frame,
                    self.detector_manager.last_face_landmarks,
                    self.detector_manager.last_hand_landmarks,
                    self.detector_manager.get_all_statuses(),
                    self.detector_manager.enabled_detector_keys(),
                )
                small = cv2.resize(annotated, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
                self.latest_preview_frame = small

            # Frame rate control
            elapsed = time.time() - loop_start
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._release_camera()
