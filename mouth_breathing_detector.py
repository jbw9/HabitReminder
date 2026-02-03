#!/usr/bin/env python3
"""
Mouth Breathing Detection Program
Monitors camera feed and alerts when mouth breathing is detected for 3-5 seconds.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time

# Configuration parameters
MAR_THRESHOLD = 0.05  # Mouth aspect ratio threshold (higher = more open)
# NOTE: You may need to adjust this threshold based on your face
# Watch the MAR value on screen and set threshold accordingly
MOUTH_OPEN_FRAMES_THRESHOLD = 120  # ~4 seconds at 30fps
WARNING_DISPLAY_TIME = 3  # seconds
CAMERA_INDEX = 0

# MediaPipe Face Mesh landmark indices for mouth
# Using standard mouth landmarks
UPPER_LIP_TOP = 13  # Top of upper lip
LOWER_LIP_BOTTOM = 14  # Bottom of lower lip
UPPER_INNER_LIP = 13  # Upper inner lip
LOWER_INNER_LIP = 14  # Lower inner lip
MOUTH_LEFT = 61  # Left corner of mouth
MOUTH_RIGHT = 291  # Right corner of mouth


class MouthBreathingDetector:
    def __init__(self):
        # Initialize MediaPipe Face Landmarker with new API
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.frame_timestamp_ms = 0

        # Camera
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # State tracking
        self.mouth_open_counter = 0
        self.warning_active = False
        self.warning_start_time = 0
        self.show_landmarks = True  # Show facial feature indicators

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def calculate_mouth_aspect_ratio(self, landmarks, frame_width, frame_height):
        """
        Calculate Mouth Aspect Ratio (MAR).
        Uses vertical mouth opening divided by horizontal mouth width.
        Higher values indicate mouth is more open.
        """
        try:
            # Get key mouth landmarks
            # Landmark 13: Upper lip inner top
            # Landmark 14: Lower lip inner bottom
            # Landmarks 61, 291: Left and right mouth corners

            # Get vertical distance (lip separation)
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            vertical = abs(upper_lip.y - lower_lip.y)

            # Get horizontal distance (mouth width)
            left_corner = landmarks[61]
            right_corner = landmarks[291]
            horizontal = abs(right_corner.x - left_corner.x)

            # Avoid division by zero
            if horizontal < 0.001:
                return 0

            # Calculate MAR
            mar = vertical / horizontal

            return mar
        except Exception as e:
            print(f"Error calculating MAR: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def trigger_warning(self):
        """Trigger warning (sets flag for on-screen display)."""
        if not self.warning_active:
            self.warning_active = True
            self.warning_start_time = time.time()

    def draw_mouth_landmarks(self, frame, landmarks, frame_width, frame_height):
        """Draw visual indicators around mouth landmarks."""
        try:
            # Key mouth landmarks to visualize
            mouth_landmarks = [
                13, 14,  # Upper and lower lip center
                61, 291,  # Left and right corners
                78, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,  # Outer lip
                78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308  # Inner lip
            ]

            # Draw circles on key mouth points
            for idx in [13, 14, 61, 291]:  # Main points
                landmark = landmarks[idx]
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # Draw rectangle around mouth region
            mouth_points = []
            for idx in [61, 291, 13, 14, 78, 308, 95, 88]:
                landmark = landmarks[idx]
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                mouth_points.append((x, y))

            if mouth_points:
                # Get bounding box
                xs = [p[0] for p in mouth_points]
                ys = [p[1] for p in mouth_points]
                x_min, x_max = min(xs) - 10, max(xs) + 10
                y_min, y_max = min(ys) - 10, max(ys) + 10

                # Draw rectangle (green if closed, red if open)
                mar = self.calculate_mouth_aspect_ratio(landmarks, frame_width, frame_height)
                color = (0, 255, 0) if mar < MAR_THRESHOLD else (0, 0, 255)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                # Add label
                label = "CLOSED" if mar < MAR_THRESHOLD else "OPEN"
                cv2.putText(frame, label, (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        except Exception as e:
            print(f"Error drawing landmarks: {e}")

    def run(self):
        """Main detection loop."""
        print("Starting Mouth Breathing Detector...")
        print("Press 'q' to quit")
        print(f"Warning will trigger after {MOUTH_OPEN_FRAMES_THRESHOLD / 30:.1f} seconds of mouth breathing")
        print("-" * 60)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                # Create MediaPipe Image
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Detect face landmarks (use detect_for_video for video streams)
                self.frame_timestamp_ms += 33  # Approximate 30fps
                detection_result = self.detector.detect_for_video(mp_image, self.frame_timestamp_ms)
            except Exception as e:
                print(f"Error during detection: {e}")
                continue

            # Default status
            mouth_status = "No face detected"
            mar = 0

            if detection_result.face_landmarks:
                face_landmarks = detection_result.face_landmarks[0]

                # Debug: print number of landmarks on first frame
                if self.frame_timestamp_ms == 33:
                    print(f"Detected {len(face_landmarks)} landmarks")

                # Calculate Mouth Aspect Ratio
                mar = self.calculate_mouth_aspect_ratio(
                    face_landmarks,
                    frame_width,
                    frame_height
                )

                # Check if mouth is open
                if mar > MAR_THRESHOLD:
                    self.mouth_open_counter += 1
                    mouth_status = f"MOUTH OPEN ({self.mouth_open_counter} frames)"

                    # Trigger warning if threshold exceeded
                    if self.mouth_open_counter >= MOUTH_OPEN_FRAMES_THRESHOLD:
                        self.trigger_warning()
                        mouth_status = "WARNING TRIGGERED!"
                else:
                    # Reset counter when mouth closes
                    if self.mouth_open_counter > 0:
                        self.mouth_open_counter = 0
                    mouth_status = "Mouth closed - Good!"

                # Draw visual indicators around mouth
                if self.show_landmarks:
                    self.draw_mouth_landmarks(frame, face_landmarks, frame_width, frame_height)

            # Check if warning should be dismissed
            if self.warning_active and (time.time() - self.warning_start_time) > WARNING_DISPLAY_TIME:
                self.warning_active = False

            # Draw warning overlay if active
            if self.warning_active:
                # Create semi-transparent red overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame_width, 150), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

                # Warning text
                warning_text = "CLOSE YOUR MOUTH!"
                warning_text2 = "Breathe through your nose"
                cv2.putText(frame, warning_text, (frame_width//2 - 250, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                cv2.putText(frame, warning_text2, (frame_width//2 - 200, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # Display info on frame
            status_color = (0, 255, 0) if "closed" in mouth_status.lower() else (0, 165, 255)
            cv2.putText(frame, f"Status: {mouth_status}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, f"MAR: {mar:.4f} (Threshold: {MAR_THRESHOLD})", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "Close mouth: MAR should be < 0.03", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.putText(frame, "Open mouth: MAR should be > 0.05", (10, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.putText(frame, "Press 'q' to quit", (10, frame_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Show frame
            cv2.imshow('Mouth Breathing Detector', frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Detector stopped.")


def main():
    detector = MouthBreathingDetector()
    detector.run()


if __name__ == "__main__":
    main()
