#!/usr/bin/env python3
"""
Test Eye Rubbing Detector
Debug version to test hand-to-eye proximity detection.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


class EyeRubbingDetectorDebug:
    """Debug version of eye rubbing detector."""

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

        # Initialize MediaPipe Hand Landmarker (new API)
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
        self.rubbing_counter = 0
        self.proximity_threshold = 0.10  # Distance threshold (smaller = need to be closer)

    def calculate_distance(self, point1, point2):
        """Calculate 2D distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def run(self):
        """Test loop with debug output."""
        print("Starting Eye Rubbing Detector Debug...")
        print("Move your hand near your eyes to test detection")
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
                # Get eye positions
                left_eye = face_landmarks[33]  # Left eye
                right_eye = face_landmarks[263]  # Right eye

                left_eye_pos = (int(left_eye.x * frame_width), int(left_eye.y * frame_height))
                right_eye_pos = (int(right_eye.x * frame_width), int(right_eye.y * frame_height))

                # Draw eyes
                cv2.circle(frame, left_eye_pos, 30, (0, 255, 0), 2)
                cv2.circle(frame, right_eye_pos, 30, (0, 255, 0), 2)
                cv2.putText(frame, "LEFT EYE", (left_eye_pos[0] - 40, left_eye_pos[1] - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, "RIGHT EYE", (right_eye_pos[0] - 40, right_eye_pos[1] - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Check hands
                hand_near_eye = False
                min_distance = 999

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
                        for point_idx in [0, 4, 8, 12, 16, 20]:  # Wrist and fingertips
                            point = hand[point_idx]
                            hand_x = point.x
                            hand_y = point.y

                            # Calculate distance to both eyes
                            dist_left = np.sqrt((hand_x - left_eye.x)**2 + (hand_y - left_eye.y)**2)
                            dist_right = np.sqrt((hand_x - right_eye.x)**2 + (hand_y - right_eye.y)**2)

                            min_dist = min(dist_left, dist_right)
                            if min_dist < min_distance:
                                min_distance = min_dist

                            if min_dist < self.proximity_threshold:
                                hand_near_eye = True

                                # Draw line from hand to nearest eye
                                hand_pos = (int(hand_x * frame_width), int(hand_y * frame_height))
                                eye_pos = left_eye_pos if dist_left < dist_right else right_eye_pos
                                cv2.line(frame, hand_pos, eye_pos, (0, 0, 255), 2)

                    # Display detection status
                    if hand_near_eye:
                        self.rubbing_counter += 1
                        status = f"HAND NEAR EYES! ({self.rubbing_counter} frames)"
                        color = (0, 0, 255)

                        # Draw warning
                        cv2.rectangle(frame, (0, 0), (frame_width, 80), (0, 0, 255), -1)
                        cv2.putText(frame, "STOP RUBBING YOUR EYES!", (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                    else:
                        status = "Hands away from eyes - Good!"
                        color = (0, 255, 0)
                        self.rubbing_counter = 0

                    cv2.putText(frame, status, (10, frame_height - 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(frame, f"Min distance: {min_distance:.3f} (Threshold: {self.proximity_threshold})",
                               (10, frame_height - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                else:
                    cv2.putText(frame, "No hands detected - Show your hands to camera",
                               (10, frame_height - 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                cv2.putText(frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show info
            cv2.putText(frame, "Eye Rubbing Detector - Test Mode", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Green circles = eye zones | Red line = hand too close",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow('Eye Rubbing Detector Debug', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    detector = EyeRubbingDetectorDebug()
    detector.run()


if __name__ == "__main__":
    main()
