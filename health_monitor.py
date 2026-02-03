#!/usr/bin/env python3
"""
Comprehensive Health Monitor
Monitors multiple health indicators using computer vision:
- Mouth breathing
- Blink rate (eye strain)
- Posture
- Fatigue (yawning)
- Hydration reminders
- Eye rubbing
- Face touching
- Phone distraction
- Screen focus
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys

# Import all detectors
from base_models.mouth_breathing_detector import MouthBreathingDetector
from base_models.blink_detector import BlinkDetector
from base_models.posture_detector import PostureDetector
from base_models.fatigue_detector import FatigueDetector
from base_models.hydration_detector import HydrationDetector
from base_models.eye_rubbing_detector import EyeRubbingDetector
from base_models.face_touching_detector import FaceTouchingDetector
from base_models.focus_detector import FocusDetector
from base_models.phone_detector import PhoneDetector


class HealthMonitor:
    """Main health monitoring system."""

    def __init__(self, camera_index=0):
        """Initialize health monitor with all detectors."""
        print("="*60)
        print("COMPREHENSIVE HEALTH MONITOR")
        print("="*60)
        print("Initializing...")

        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Initialize MediaPipe Face Landmarker
        print("Loading face detection model...")
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        face_options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            running_mode=vision.RunningMode.VIDEO
        )
        self.face_detector = vision.FaceLandmarker.create_from_options(face_options)

        # Initialize MediaPipe Hand Detector (for hand-based detections)
        print("Loading hand detection model...")
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            print(f"Hand detection not available: {e}")
            self.hands = None

        # Initialize all detectors
        print("Initializing detectors...")
        self.detectors = {
            'mouth_breathing': MouthBreathingDetector(),
            'blink': BlinkDetector(low_blink_threshold=6),
            'posture': PostureDetector(),
            'fatigue': FatigueDetector(),
            'hydration': HydrationDetector(interval_minutes=45),
            'eye_rubbing': EyeRubbingDetector(),
            'face_touching': FaceTouchingDetector(),
            'focus': FocusDetector(),
            'phone': PhoneDetector()
        }

        # Enable/disable detectors (user can configure)
        self.detector_enabled = {key: True for key in self.detectors.keys()}

        self.frame_timestamp_ms = 0

        print("\nDetectors loaded:")
        for name, detector in self.detectors.items():
            status = "ENABLED" if self.detector_enabled[name] else "DISABLED"
            print(f"  - {detector.name}: {status}")

        print("\nControls:")
        print("  q: Quit")
        print("  h: Toggle hydration timer reset")
        print("  1-9: Toggle specific detectors")
        print("  SPACE: Toggle all overlays")
        print("="*60)

        self.show_overlays = True

    def draw_warning_banner(self, frame, messages):
        """Draw warning banner at top of screen."""
        if not messages:
            return

        frame_height, frame_width = frame.shape[:2]

        # Calculate banner height based on number of warnings
        banner_height = min(60 + (len(messages) * 30), 200)

        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame_width, banner_height), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Draw warning icon
        cv2.putText(frame, "!!", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        # Draw messages
        y_offset = 40
        for msg in messages:
            lines = msg.split('\n')
            for line in lines:
                cv2.putText(frame, line, (60, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_offset += 30

    def draw_status_panel(self, frame):
        """Draw status panel showing all detector states."""
        frame_height, frame_width = frame.shape[:2]

        # Panel dimensions
        panel_width = 350
        panel_height = min(400, frame_height - 50)
        panel_x = frame_width - panel_width - 10
        panel_y = 10

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Title
        cv2.putText(frame, "HEALTH MONITOR", (panel_x + 10, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw line
        cv2.line(frame, (panel_x + 10, panel_y + 40),
                (panel_x + panel_width - 10, panel_y + 40), (255, 255, 255), 1)

        # Draw each detector status
        y_offset = panel_y + 60
        for idx, (name, detector) in enumerate(self.detectors.items(), 1):
            if not self.detector_enabled[name]:
                continue

            # Status color
            if detector.is_warning_active():
                color = (0, 0, 255)  # Red
            elif detector.counter > detector.warning_threshold // 2:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 255, 0)  # Green

            # Draw status dot
            cv2.circle(frame, (panel_x + 20, y_offset - 5), 5, color, -1)

            # Draw detector name and status
            cv2.putText(frame, f"{detector.name}:", (panel_x + 35, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            status_text = detector.get_status()
            # Truncate if too long
            if len(status_text) > 35:
                status_text = status_text[:32] + "..."

            cv2.putText(frame, status_text, (panel_x + 35, y_offset + 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            y_offset += 35

            if y_offset > panel_y + panel_height - 30:
                break

    def run(self):
        """Main monitoring loop."""
        print("\nStarting health monitor...\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect face landmarks
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                self.frame_timestamp_ms += 33
                face_result = self.face_detector.detect_for_video(mp_image, self.frame_timestamp_ms)
                face_landmarks = face_result.face_landmarks[0] if face_result.face_landmarks else None
            except Exception as e:
                face_landmarks = None

            # Detect hands
            hand_landmarks = None
            if self.hands:
                try:
                    hand_result = self.hands.process(rgb_frame)
                    hand_landmarks = hand_result.multi_hand_landmarks if hand_result.multi_hand_landmarks else None
                except Exception:
                    hand_landmarks = None

            # Run all enabled detectors
            active_warnings = []

            for name, detector in self.detectors.items():
                if not self.detector_enabled[name]:
                    continue

                try:
                    # Detectors that need hand landmarks
                    if name in ['eye_rubbing', 'face_touching', 'phone']:
                        if hasattr(detector, 'detect') and 'hand_landmarks' in detector.detect.__code__.co_varnames:
                            detection_result = detector.detect(face_landmarks, frame_width, frame_height, hand_landmarks)
                        else:
                            detection_result = detector.detect(face_landmarks, frame_width, frame_height)
                    else:
                        detection_result = detector.detect(face_landmarks, frame_width, frame_height)

                    # Update detector state
                    detector.update(detection_result)
                    detector.check_warning_timeout()

                    # Collect active warnings
                    if detector.is_warning_active():
                        active_warnings.append(detector.warning_message)

                    # Draw overlays
                    if self.show_overlays and face_landmarks:
                        if name in ['eye_rubbing', 'face_touching', 'phone']:
                            if hasattr(detector, 'draw_overlay') and 'hand_landmarks' in detector.draw_overlay.__code__.co_varnames:
                                detector.draw_overlay(frame, face_landmarks, frame_width, frame_height, hand_landmarks)
                            else:
                                detector.draw_overlay(frame, face_landmarks, frame_width, frame_height)
                        else:
                            detector.draw_overlay(frame, face_landmarks, frame_width, frame_height)

                except Exception as e:
                    print(f"Error in {name}: {e}")

            # Draw warning banner
            if active_warnings:
                self.draw_warning_banner(frame, active_warnings)

            # Draw status panel
            self.draw_status_panel(frame)

            # Show frame
            cv2.imshow('Health Monitor', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.show_overlays = not self.show_overlays
            elif key == ord('h'):
                # Reset hydration timer
                self.detectors['hydration'].reset_timer()
                print("Hydration timer reset!")
            elif ord('1') <= key <= ord('9'):
                # Toggle detector
                idx = key - ord('1')
                detector_names = list(self.detectors.keys())
                if idx < len(detector_names):
                    name = detector_names[idx]
                    self.detector_enabled[name] = not self.detector_enabled[name]
                    status = "ENABLED" if self.detector_enabled[name] else "DISABLED"
                    print(f"{self.detectors[name].name}: {status}")

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nHealth monitor stopped.")


def main():
    """Main entry point."""
    try:
        monitor = HealthMonitor()
        monitor.run()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
