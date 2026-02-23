#!/usr/bin/env python3
"""
Detector Manager
Central management of all detectors and MediaPipe models.
Handles initialization, enable/disable, and per-frame processing.
"""

import os
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from detectors.mouth_breathing_detector import MouthBreathingDetector
from detectors.nail_biting_detector import NailBitingDetector


class DetectorManager:
    """Manages detector lifecycle and frame processing."""

    # Map of detector keys to human-readable names
    DETECTOR_NAMES = {
        'mouth_breathing': 'Mouth Breathing',
        'nail_biting': 'Nail Biting',
    }

    def __init__(self, model_dir=None):
        """
        Args:
            model_dir: Directory containing .task model files.
                       Defaults to project root.
        """
        if model_dir is None:
            if getattr(sys, 'frozen', False):
                # Running inside a PyInstaller bundle
                model_dir = sys._MEIPASS
            else:
                model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = model_dir

        self.detectors = {}
        self.face_detector = None
        self.hand_detector = None

        # Last detection results (for preview overlay)
        self.last_face_landmarks = None
        self.last_hand_landmarks = None

    def initialize_mediapipe(self):
        """Load MediaPipe face and hand landmarker models."""
        face_model = os.path.join(self.model_dir, 'face_landmarker.task')
        hand_model = os.path.join(self.model_dir, 'hand_landmarker.task')

        face_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=face_model),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.face_detector = vision.FaceLandmarker.create_from_options(face_options)

        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=hand_model),
            num_hands=2,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    def initialize_detectors(self):
        """Create all detectors (start disabled)."""
        self.detectors = {
            'mouth_breathing': MouthBreathingDetector(),
            'nail_biting': NailBitingDetector(),
        }

    def enable_detector(self, name):
        """Enable a detector by key name."""
        if name in self.detectors:
            self.detectors[name].enable()

    def disable_detector(self, name):
        """Disable a detector by key name."""
        if name in self.detectors:
            self.detectors[name].disable()

    def is_enabled(self, name):
        """Check if a detector is enabled."""
        if name in self.detectors:
            return self.detectors[name].is_enabled()
        return False

    def any_enabled(self):
        """Check if any detector is currently enabled."""
        return any(d.is_enabled() for d in self.detectors.values())

    def enabled_detector_keys(self):
        """Return set of currently enabled detector key names."""
        return {name for name, d in self.detectors.items() if d.is_enabled()}

    def process_frame(self, rgb_frame, timestamp_ms):
        """
        Run all enabled detectors on a single frame.

        Args:
            rgb_frame: RGB numpy array from camera
            timestamp_ms: Monotonic timestamp in milliseconds

        Returns:
            List of alert dicts: [{'detector': name, 'message': str, 'severity': str}]
        """
        alerts = []

        # Run MediaPipe detection
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        face_landmarks = None
        hand_landmarks = None

        try:
            face_result = self.face_detector.detect_for_video(mp_image, timestamp_ms)
            if face_result.face_landmarks:
                face_landmarks = face_result.face_landmarks[0]
        except Exception:
            pass

        try:
            hand_result = self.hand_detector.detect_for_video(mp_image, timestamp_ms)
            if hand_result.hand_landmarks:
                hand_landmarks = hand_result.hand_landmarks  # List of all detected hands
        except Exception:
            pass

        # Store for preview overlay
        self.last_face_landmarks = face_landmarks
        self.last_hand_landmarks = hand_landmarks

        frame_h, frame_w = rgb_frame.shape[:2]

        # Run each enabled detector
        for name, detector in self.detectors.items():
            if not detector.is_enabled():
                continue

            try:
                detected = detector.detect(
                    face_landmarks, hand_landmarks, frame_w, frame_h
                )
                if detected:
                    alerts.append({
                        'detector': name,
                        'message': detector.get_alert_message(),
                        'severity': detector.get_severity(),
                    })
            except Exception:
                pass

        return alerts

    def get_all_statuses(self):
        """Return dict of detector name -> status string."""
        return {name: d.get_status() for name, d in self.detectors.items()}

    def get_mouth_counter(self):
        """Return (current_open_frames, frames_threshold) for the mouth breathing detector."""
        d = self.detectors.get('mouth_breathing')
        if d:
            return d._counter, d.frames_threshold
        return 0, 450

    def get_nail_counter(self):
        """Return (current_proximity_frames, frames_threshold) for the nail biting detector."""
        d = self.detectors.get('nail_biting')
        if d:
            return d._counter, d.frames_threshold
        return 0, 60

    def cleanup(self):
        """Release MediaPipe resources."""
        if self.face_detector:
            self.face_detector.close()
            self.face_detector = None
        if self.hand_detector:
            self.hand_detector.close()
            self.hand_detector = None
