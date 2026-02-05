#!/usr/bin/env python3
"""
Test Hydration Detector
Detects drinking by identifying cups/bottles near the mouth using object detection.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import urllib.request
import os


class HydrationDetectorDebug:
    """Debug version of hydration detector using object detection."""

    def __init__(self, reminder_interval=20):
        """
        Args:
            reminder_interval: Seconds between reminders (20 for testing, normally 2700 = 45 min)
        """
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

        # Initialize object detection model (MobileNet-SSD trained on COCO)
        print("Loading object detection model...")
        self.load_object_detection_model()

        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.frame_timestamp_ms = 0

        # Hydration tracking
        self.reminder_interval = reminder_interval
        self.last_drink_time = time.time()
        self.drinking_frames = 0
        self.drinking_threshold = 30  # Frames needed to confirm drinking (1 second)

        # Detection thresholds
        self.mouth_proximity_threshold = 0.20  # Distance for cup near mouth
        self.detection_confidence = 0.3  # Confidence threshold for object detection

        # COCO class IDs for drinkware
        self.drinkware_classes = {
            41: "cup",
            39: "bottle",
            46: "wine glass"
        }

    def load_object_detection_model(self):
        """Load MobileNet-SSD model for object detection."""
        # Download model files if not present
        prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        model_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"

        # Using MobileNet-SSD COCO model
        model_file = "MobileNetSSD_deploy.caffemodel"
        config_file = "MobileNetSSD_deploy.prototxt"

        # For now, let's use a simpler approach with pre-downloaded files
        # We'll use opencv's pre-trained model
        try:
            # Try to load from current directory
            if not os.path.exists(model_file):
                print("Downloading object detection model... (this may take a moment)")
                # Using a publicly available MobileNet-SSD model
                model_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"
                urllib.request.urlretrieve(model_url, model_file)

            if not os.path.exists(config_file):
                prototxt_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
                urllib.request.urlretrieve(prototxt_url, config_file)

            self.net = cv2.dnn.readNetFromCaffe(config_file, model_file)
            print("Object detection model loaded successfully!")

        except Exception as e:
            print(f"Could not load model: {e}")
            print("Will use simplified detection without object recognition")
            self.net = None

        # COCO-SSD class names (MobileNet-SSD uses these)
        self.class_names = ["background", "aeroplane", "bicycle", "bird", "boat",
                           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                           "sofa", "train", "tvmonitor"]

    def detect_drinkware(self, frame):
        """Detect cups, bottles, glasses in the frame."""
        if self.net is None:
            return []

        try:
            h, w = frame.shape[:2]

            # Prepare image for detection
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()

            drinkware = []

            # Process detections
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > self.detection_confidence:
                    class_id = int(detections[0, 0, i, 1])

                    # Check if it's a bottle (class 5 in MobileNet-SSD)
                    if class_id == 5:  # bottle class
                        # Get bounding box
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (x1, y1, x2, y2) = box.astype("int")

                        # Calculate center in normalized coordinates
                        center_x = (x1 + x2) / (2 * w)
                        center_y = (y1 + y2) / (2 * h)

                        drinkware.append({
                            'type': 'bottle',
                            'confidence': confidence,
                            'bbox': (x1, y1, x2, y2),
                            'center': (center_x, center_y)
                        })

            return drinkware

        except Exception as e:
            print(f"Detection error: {e}")
            return []

    def run(self):
        """Test loop with debug output."""
        print("Starting Hydration Detector Debug...")
        print("Hold a cup/bottle near your mouth to simulate drinking")
        print(f"Reminder every {self.reminder_interval} seconds")
        print("Press 'q' to quit")
        print("-" * 60)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect face
            face_landmarks = None
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                self.frame_timestamp_ms += 33
                face_result = self.face_detector.detect_for_video(mp_image, self.frame_timestamp_ms)
                if face_result.face_landmarks:
                    face_landmarks = face_result.face_landmarks[0]
            except Exception as e:
                print(f"Face detection error: {e}")

            # Detect drinkware (cups, bottles, glasses)
            drinkware = self.detect_drinkware(frame)

            try:
                if face_landmarks:
                    # Get mouth position
                    upper_lip = face_landmarks[13]
                    lower_lip = face_landmarks[14]
                    mouth_x = (upper_lip.x + lower_lip.x) / 2
                    mouth_y = (upper_lip.y + lower_lip.y) / 2
                    mouth_pos = (int(mouth_x * frame_width), int(mouth_y * frame_height))

                    # Draw mouth
                    cv2.circle(frame, mouth_pos, 15, (0, 255, 255), 2)
                    cv2.putText(frame, "MOUTH", (mouth_pos[0] - 30, mouth_pos[1] - 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    # Check for drinking
                    is_drinking = False
                    closest_drink = None
                    min_distance = 999

                    for drink in drinkware:
                        # Draw detected drinkware
                        x1, y1, x2, y2 = drink['bbox']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{drink['type']}: {drink['confidence']:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Calculate distance to mouth
                        drink_center_x, drink_center_y = drink['center']
                        distance = np.sqrt((drink_center_x - mouth_x)**2 +
                                         (drink_center_y - mouth_y)**2)

                        if distance < min_distance:
                            min_distance = distance
                            closest_drink = drink

                        # Check if near mouth
                        if distance < self.mouth_proximity_threshold:
                            is_drinking = True
                            # Draw line from drink to mouth
                            drink_pos = (int(drink_center_x * frame_width),
                                       int(drink_center_y * frame_height))
                            cv2.line(frame, drink_pos, mouth_pos, (0, 255, 0), 2)

                    # Track drinking
                    if is_drinking:
                        self.drinking_frames += 1
                        status_color = (0, 255, 255)
                        status = "DRINKING DETECTED!"

                        if self.drinking_frames >= self.drinking_threshold:
                            # Confirmed drinking!
                            self.last_drink_time = time.time()
                            print(f"DRINK CONFIRMED! Timer reset.")

                    else:
                        self.drinking_frames = 0
                        status_color = (100, 100, 100)
                        if len(drinkware) > 0:
                            status = f"Cup/bottle detected - bring closer to mouth"
                        else:
                            status = "No cup/bottle detected"

                    # Show status
                    cv2.putText(frame, status, (10, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

                    # Show detection info
                    cv2.putText(frame, f"Drinkware detected: {len(drinkware)}",
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    if closest_drink:
                        cv2.putText(frame, f"Closest: {min_distance:.3f} (need < {self.mouth_proximity_threshold})",
                                   (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                    # Show drinking progress
                    if self.drinking_frames > 0:
                        progress = min(self.drinking_frames / self.drinking_threshold, 1.0)
                        bar_width = int(400 * progress)
                        cv2.rectangle(frame, (10, 110), (10 + bar_width, 130), (0, 255, 255), -1)
                        cv2.rectangle(frame, (10, 110), (410, 130), (100, 100, 100), 2)
                        cv2.putText(frame, f"Drinking: {self.drinking_frames}/{self.drinking_threshold} frames",
                                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    # Calculate time since last drink
                    current_time = time.time()
                    time_since_drink = current_time - self.last_drink_time
                    time_until_reminder = self.reminder_interval - time_since_drink

                    # Show countdown
                    if time_until_reminder > 0:
                        minutes = int(time_until_reminder / 60)
                        seconds = int(time_until_reminder % 60)
                        cv2.putText(frame, f"Next reminder: {minutes}m {seconds}s",
                                   (10, frame_height - 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
                    else:
                        # Overdue - show warning!
                        cv2.rectangle(frame, (0, 0), (frame_width, 100), (0, 0, 200), -1)
                        cv2.putText(frame, "TIME TO HYDRATE!", (200, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

                    # Show last drink time
                    cv2.putText(frame, f"Last drink: {int(time_since_drink)}s ago",
                               (10, frame_height - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)

                else:
                    cv2.putText(frame, "No face detected", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()

            cv2.imshow('Hydration Detector Debug', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    # Use 20 seconds for testing (default would be 2700 = 45 minutes)
    detector = HydrationDetectorDebug(reminder_interval=20)
    detector.run()


if __name__ == "__main__":
    main()
