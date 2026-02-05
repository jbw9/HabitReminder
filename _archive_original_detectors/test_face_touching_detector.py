#!/usr/bin/env python3
"""
Test Face Touching Detector
Debug version to test hand-to-face proximity detection.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time


class FaceTouchingDetectorDebug:
    """Debug version of face touching detector."""

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
        self.face_detector = vision.FaceLandmarker.create_from_options(options)

        # Initialize MediaPipe Hand Landmarker
        hand_base_options = python.BaseOptions(
            model_asset_path='hand_landmarker.task'
        )
        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base_options,
            num_hands=2,
            running_mode=vision.RunningMode.VIDEO
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)

        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.frame_timestamp_ms = 0

        # Detection tracking
        self.horizontal_threshold = 0.12  # Horizontal radius (face width)
        self.vertical_threshold = 0.35    # Vertical radius (face height - includes forehead)
        self.touches_in_period = []
        self.was_touching = False
        self.check_period = 120  # 2 minutes

    def run(self):
        """Test loop with debug output."""
        print("Starting Face Touching Detector Debug...")
        print("Touch your face to test detection")
        print("Press 'q' to quit")
        print("-" * 60)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect face landmarks
            face_landmarks = None
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                self.frame_timestamp_ms += 33
                face_result = self.face_detector.detect_for_video(mp_image, self.frame_timestamp_ms)
                if face_result.face_landmarks:
                    face_landmarks = face_result.face_landmarks[0]
            except Exception as e:
                print(f"Face detection error: {e}")

            # Detect hands
            hand_landmarks = None
            try:
                hand_result = self.hand_detector.detect_for_video(mp_image, self.frame_timestamp_ms)
                if hand_result.hand_landmarks:
                    hand_landmarks = hand_result.hand_landmarks
            except Exception as e:
                print(f"Hand detection error: {e}")

            # Process detection
            if face_landmarks:
                # Get face center (nose)
                nose = face_landmarks[1]
                face_center_x = nose.x
                face_center_y = nose.y
                face_center_pos = (int(face_center_x * frame_width),
                                  int(face_center_y * frame_height))

                # Draw face center
                cv2.circle(frame, face_center_pos, 10, (0, 255, 255), -1)
                cv2.putText(frame, "FACE CENTER", (face_center_pos[0] - 50, face_center_pos[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Draw detection oval (ellipse)
                horizontal_radius = int(self.horizontal_threshold * frame_width)
                vertical_radius = int(self.vertical_threshold * frame_height)
                cv2.ellipse(frame, face_center_pos, (horizontal_radius, vertical_radius),
                           0, 0, 360, (0, 255, 0), 2)

                # Check hands
                hand_near_face = False
                min_distance = 999
                closest_hand_point = None

                if hand_landmarks:
                    # Draw all detected hands
                    for hand in hand_landmarks:
                        # Draw hand skeleton
                        for landmark in hand:
                            x = int(landmark.x * frame_width)
                            y = int(landmark.y * frame_height)
                            cv2.circle(frame, (x, y), 5, (255, 0, 255), -1)

                        # Draw connections between landmarks
                        connections = [
                            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                            (5, 9), (9, 13), (13, 17)  # Palm
                        ]
                        for connection in connections:
                            start_idx, end_idx = connection
                            start = hand[start_idx]
                            end = hand[end_idx]
                            start_pos = (int(start.x * frame_width), int(start.y * frame_height))
                            end_pos = (int(end.x * frame_width), int(end.y * frame_height))
                            cv2.line(frame, start_pos, end_pos, (255, 0, 255), 2)

                        # Check key hand points (wrist, fingertips)
                        for point_idx in [0, 4, 8, 12, 16, 20]:
                            point = hand[point_idx]
                            hand_x = point.x
                            hand_y = point.y

                            # Calculate distance to face center
                            dist = np.sqrt((hand_x - face_center_x)**2 +
                                         (hand_y - face_center_y)**2)

                            if dist < min_distance:
                                min_distance = dist
                                closest_hand_point = (int(hand_x * frame_width),
                                                     int(hand_y * frame_height))

                            # Check if hand is inside oval (ellipse formula)
                            # (x - cx)^2 / rx^2 + (y - cy)^2 / ry^2 < 1
                            dx = (hand_x - face_center_x) / self.horizontal_threshold
                            dy = (hand_y - face_center_y) / self.vertical_threshold
                            inside_oval = (dx * dx + dy * dy) < 1.0

                            if inside_oval:
                                hand_near_face = True

                    # Draw line from closest hand point to face center
                    if closest_hand_point:
                        color = (0, 0, 255) if hand_near_face else (0, 255, 0)
                        cv2.line(frame, closest_hand_point, face_center_pos, color, 2)

                        # Show distance
                        mid_x = (closest_hand_point[0] + face_center_pos[0]) // 2
                        mid_y = (closest_hand_point[1] + face_center_pos[1]) // 2
                        cv2.putText(frame, f"{min_distance:.3f}", (mid_x, mid_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Detect touch event (transition from not touching to touching)
                    if hand_near_face and not self.was_touching:
                        # New touch detected
                        self.touches_in_period.append(time.time())
                        print(f"TOUCH DETECTED! (Distance: {min_distance:.3f})")

                    self.was_touching = hand_near_face

                    # Remove old touches (older than check_period)
                    current_time = time.time()
                    self.touches_in_period = [t for t in self.touches_in_period
                                            if current_time - t < self.check_period]

                    touch_count = len(self.touches_in_period)

                    # Display detection status
                    if hand_near_face:
                        status = f"TOUCHING FACE! (Distance: {min_distance:.3f})"
                        color = (0, 0, 255)

                        # Change oval color to red
                        cv2.ellipse(frame, face_center_pos,
                                   (int(self.horizontal_threshold * frame_width),
                                    int(self.vertical_threshold * frame_height)),
                                   0, 0, 360, (0, 0, 255), 3)
                    else:
                        status = f"Hands away - Good! (Min dist: {min_distance:.3f})"
                        color = (0, 255, 0)

                    cv2.putText(frame, status, (10, frame_height - 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # Show touch count
                    cv2.putText(frame, f"Touches in last 2min: {touch_count}/5",
                               (10, frame_height - 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Warning if too many touches
                    if touch_count >= 5:
                        cv2.rectangle(frame, (0, 0), (frame_width, 80), (0, 0, 255), -1)
                        cv2.putText(frame, "STOP TOUCHING YOUR FACE!", (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

                    cv2.putText(frame, f"Oval size: H={self.horizontal_threshold:.2f} V={self.vertical_threshold:.2f}",
                               (10, frame_height - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                else:
                    cv2.putText(frame, "No hands detected - Show your hands to camera",
                               (10, frame_height - 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    self.was_touching = False
            else:
                cv2.putText(frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show info
            cv2.putText(frame, "Face Touching Detector - Test Mode", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Yellow dot = face center | Green oval = detection zone",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow('Face Touching Detector Debug', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    detector = FaceTouchingDetectorDebug()
    detector.run()


if __name__ == "__main__":
    main()
