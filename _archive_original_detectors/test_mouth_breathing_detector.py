#!/usr/bin/env python3
"""
Test Mouth Breathing Detector
Debug version to visualise MAR values and tune detection threshold.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import sys
import os

# Allow importing from sibling package when running from this directory
sys.path.insert(0, os.path.dirname(__file__))
from base_models.mouth_breathing_detector import MouthBreathingDetector


class MouthBreathingDetectorDebug:
    """Debug wrapper that renders live MAR values on the camera feed."""

    # Landmark indices used for MAR
    UPPER_LIP = 13
    LOWER_LIP = 14
    LEFT_CORNER = 61
    RIGHT_CORNER = 291

    # Extra landmarks used for the bounding-box overlay
    MOUTH_BOX_INDICES = [61, 291, 13, 14, 78, 308, 95, 88]

    def __init__(self, mar_threshold=0.05, frames_threshold=120):
        # MediaPipe face landmarker
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            running_mode=vision.RunningMode.VIDEO
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.frame_timestamp_ms = 0

        # Camera
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Detector under test
        self.detector = MouthBreathingDetector(
            threshold=mar_threshold,
            frames_threshold=frames_threshold
        )

        # MAR history for range analysis
        self.mar_history = []
        self.min_mar_seen = float('inf')
        self.max_mar_seen = 0.0

        self.alert_count = 0
        self.last_alert_time = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_landmark_px(self, landmarks, idx, w, h):
        lm = landmarks[idx]
        return int(lm.x * w), int(lm.y * h)

    def _draw_info(self, frame, mar, mouth_open, w, h):
        """Render all debug text onto the frame."""
        threshold = self.detector.mar_threshold
        counter = self.detector.counter
        frames_threshold = self.detector.warning_threshold

        # MAR value
        cv2.putText(frame, f"MAR: {mar:.4f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Threshold currently in use
        cv2.putText(frame, f"THRESHOLD: {threshold:.4f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Mouth state
        state = "OPEN" if mouth_open else "CLOSED"
        state_color = (0, 0, 255) if mouth_open else (0, 255, 0)
        cv2.putText(frame, f"MOUTH: {state}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)

        # Persistent open frame counter
        cv2.putText(frame, f"OPEN frames: {counter} / {frames_threshold}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # MAR range (calibration aid)
        min_mar = self.min_mar_seen if self.min_mar_seen != float('inf') else 0.0
        suggested = (min_mar + self.max_mar_seen) / 2.0 if self.max_mar_seen > min_mar else threshold
        cv2.putText(frame, f"MIN MAR seen (closed): {min_mar:.4f}",
                    (w - 360, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, f"MAX MAR seen (open):   {self.max_mar_seen:.4f}",
                    (w - 360, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"SUGGESTED threshold:   {suggested:.4f}",
                    (w - 360, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Alert count
        cv2.putText(frame, f"Alerts triggered: {self.alert_count}",
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Instructions
        cv2.putText(frame, "Press 'q' to quit",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Warning banner
        if self.detector.warning_active:
            cv2.rectangle(frame, (0, h // 2 - 40), (w, h // 2 + 40), (0, 0, 200), -1)
            cv2.putText(frame, "CLOSE YOUR MOUTH!", (w // 2 - 180, h // 2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    def _draw_mouth_overlay(self, frame, landmarks, mar, w, h):
        """Draw key landmark dots and bounding box around the mouth."""
        # Key point dots
        for idx in [self.UPPER_LIP, self.LOWER_LIP, self.LEFT_CORNER, self.RIGHT_CORNER]:
            x, y = self._get_landmark_px(landmarks, idx, w, h)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        # Bounding box
        mouth_pts = [self._get_landmark_px(landmarks, idx, w, h)
                     for idx in self.MOUTH_BOX_INDICES]
        if mouth_pts:
            xs = [p[0] for p in mouth_pts]
            ys = [p[1] for p in mouth_pts]
            x_min, x_max = min(xs) - 10, max(xs) + 10
            y_min, y_max = min(ys) - 10, max(ys) + 10
            color = (0, 0, 255) if mar > self.detector.mar_threshold else (0, 255, 0)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            label = "OPEN" if mar > self.detector.mar_threshold else "CLOSED"
            cv2.putText(frame, label, (x_min, y_min - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        print("Starting Mouth Breathing Detector Debug...")
        print("Open / close your mouth to see MAR values change.")
        print("Press 'q' to quit.")
        print("-" * 60)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Camera read failed.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                self.frame_timestamp_ms += 33
                result = self.landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)

                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]

                    # Run detection
                    mouth_open = self.detector.detect(landmarks, w, h)
                    mar = self.detector.calculate_mouth_aspect_ratio(landmarks)

                    # Update MAR range tracking
                    self.mar_history.append(mar)
                    if len(self.mar_history) > 150:
                        self.mar_history.pop(0)
                    if mar > 0:
                        self.min_mar_seen = min(self.min_mar_seen, mar)
                    self.max_mar_seen = max(self.max_mar_seen, mar)

                    # Advance internal counter / warning state
                    prev_warning = self.detector.warning_active
                    self.detector.update(mouth_open)
                    self.detector.check_warning_timeout()

                    # Count new alerts
                    if self.detector.warning_active and not prev_warning:
                        self.alert_count += 1
                        self.last_alert_time = time.time()
                        print(f"[ALERT] Mouth open for {self.detector.warning_threshold} frames! "
                              f"(MAR: {mar:.4f})")

                    # Draw overlays
                    self._draw_mouth_overlay(frame, landmarks, mar, w, h)
                    self._draw_info(frame, mar, mouth_open, w, h)

                else:
                    cv2.putText(frame, "No face detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()

            cv2.imshow("Mouth Breathing Detector Debug", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    debug = MouthBreathingDetectorDebug(
        mar_threshold=0.05,   # Adjust if you get false positives/negatives
        frames_threshold=120  # ~4 seconds at 30 fps before alert fires
    )
    debug.run()


if __name__ == "__main__":
    main()
