#!/usr/bin/env python3
"""
Test Blink Detector
Debug version to find correct thresholds and landmarks.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time


class BlinkDetectorDebug:
    """Debug version of blink detector with output."""

    def __init__(self):
        # Initialize MediaPipe Face Landmarker
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.frame_timestamp_ms = 0

        # Blink tracking
        self.blink_count = 0
        self.last_blink_time = time.time()
        self.was_eyes_closed = False

        # Track EAR values to find optimal threshold
        self.ear_history = []
        self.min_ear_seen = 1.0
        self.max_ear_seen = 0.0

        # Eye landmark indices - using MediaPipe Face Mesh standard indices
        # Left eye landmarks (around the eye)
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.LEFT_EYE_TOP = [159, 145]
        self.LEFT_EYE_BOTTOM = [23, 130]

        # Right eye landmarks
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_TOP = [386, 374]
        self.RIGHT_EYE_BOTTOM = [253, 359]

    def calculate_eye_aspect_ratio_simple(self, landmarks, top_indices, bottom_indices):
        """
        Simplified EAR calculation.
        Measures vertical eye opening.
        """
        try:
            # Get top and bottom points
            top_points = [landmarks[idx] for idx in top_indices]
            bottom_points = [landmarks[idx] for idx in bottom_indices]

            # Calculate average vertical distance
            vertical_distances = []
            for top in top_points:
                for bottom in bottom_points:
                    dist = abs(top.y - bottom.y)
                    vertical_distances.append(dist)

            avg_vertical = np.mean(vertical_distances)

            return avg_vertical

        except Exception as e:
            print(f"Error calculating EAR: {e}")
            return 0

    def run(self):
        """Test loop with debug output."""
        print("Starting Blink Detector Debug...")
        print("Look at the EAR values and manually blink to see changes")
        print("Press 'q' to quit")
        print("-" * 60)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                self.frame_timestamp_ms += 33
                detection_result = self.detector.detect_for_video(mp_image, self.frame_timestamp_ms)

                if detection_result.face_landmarks:
                    face_landmarks = detection_result.face_landmarks[0]

                    # Calculate EAR for both eyes
                    left_ear = self.calculate_eye_aspect_ratio_simple(
                        face_landmarks,
                        self.LEFT_EYE_TOP,
                        self.LEFT_EYE_BOTTOM
                    )
                    right_ear = self.calculate_eye_aspect_ratio_simple(
                        face_landmarks,
                        self.RIGHT_EYE_TOP,
                        self.RIGHT_EYE_BOTTOM
                    )

                    avg_ear = (left_ear + right_ear) / 2.0

                    # Track EAR range
                    self.ear_history.append(avg_ear)
                    if len(self.ear_history) > 100:
                        self.ear_history.pop(0)

                    self.min_ear_seen = min(self.min_ear_seen, avg_ear)
                    self.max_ear_seen = max(self.max_ear_seen, avg_ear)

                    # Draw eye landmarks to verify they're correct
                    # Left eye
                    for idx in self.LEFT_EYE_TOP + self.LEFT_EYE_BOTTOM:
                        landmark = face_landmarks[idx]
                        x = int(landmark.x * frame_width)
                        y = int(landmark.y * frame_height)
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                    # Right eye
                    for idx in self.RIGHT_EYE_TOP + self.RIGHT_EYE_BOTTOM:
                        landmark = face_landmarks[idx]
                        x = int(landmark.x * frame_width)
                        y = int(landmark.y * frame_height)
                        cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

                    # Test different thresholds
                    threshold_tests = [0.008, 0.012, 0.015, 0.018, 0.022, 0.025]

                    # Detect blink using single threshold (configurable)
                    # TRY ADJUSTING THIS VALUE:
                    # - Too many false blinks? DECREASE threshold
                    # - Missing real blinks? INCREASE threshold
                    BLINK_THRESHOLD = 0.012  # Balanced sensitivity
                    eyes_closed = avg_ear < BLINK_THRESHOLD

                    # Calculate suggested threshold (midpoint between min and max)
                    if self.min_ear_seen < self.max_ear_seen:
                        suggested_threshold = (self.min_ear_seen + self.max_ear_seen) / 2.0
                    else:
                        suggested_threshold = BLINK_THRESHOLD

                    # Display EAR value prominently
                    cv2.putText(frame, f"LEFT EAR: {left_ear:.4f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"RIGHT EAR: {right_ear:.4f}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"AVG EAR: {avg_ear:.4f}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    # Show current threshold and eye state
                    eye_state = "CLOSED" if eyes_closed else "OPEN"
                    state_color = (0, 0, 255) if eyes_closed else (0, 255, 0)
                    cv2.putText(frame, f"THRESHOLD: {BLINK_THRESHOLD:.3f}", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"EYES: {eye_state}", (10, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)

                    # Show EAR range analysis
                    cv2.putText(frame, f"MIN EAR seen: {self.min_ear_seen:.4f} (closed)",
                               (frame_width - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(frame, f"MAX EAR seen: {self.max_ear_seen:.4f} (open)",
                               (frame_width - 350, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"SUGGESTED: {suggested_threshold:.4f}",
                               (frame_width - 350, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    # Show instructions
                    cv2.putText(frame, "Do some normal blinks, then check SUGGESTED threshold",
                               (frame_width - 500, frame_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    cv2.putText(frame, "Suggested = midpoint between your open and closed eyes",
                               (frame_width - 500, frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    cv2.putText(frame, "Edit BLINK_THRESHOLD in code to adjust",
                               (frame_width - 500, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                    # Show threshold test reference (for calibration only)
                    y_offset = 180
                    cv2.putText(frame, "Threshold Reference:", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                    y_offset += 20
                    for threshold in threshold_tests:
                        would_detect = "CLOSED" if avg_ear < threshold else "OPEN"
                        color = (0, 0, 255) if avg_ear < threshold else (0, 255, 0)
                        cv2.putText(frame, f"  {threshold:.3f}: {would_detect}",
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        y_offset += 20

                    # Detect blink (transition from open to closed)
                    if eyes_closed and not self.was_eyes_closed:
                        print(f"BLINK DETECTED! (EAR: {avg_ear:.4f})")
                        self.blink_count += 1
                        self.last_blink_time = time.time()

                    self.was_eyes_closed = eyes_closed

                    # Show blink count
                    cv2.putText(frame, f"Total Blinks: {self.blink_count}",
                               (10, frame_height - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                else:
                    cv2.putText(frame, "No face detected", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()

            cv2.imshow('Blink Detector Debug', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    detector = BlinkDetectorDebug()
    detector.run()


if __name__ == "__main__":
    main()
