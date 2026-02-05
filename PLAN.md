# Habit Tracker - Mac Menu Bar App Implementation Plan

## Project Overview

**Goal:** Transform standalone health detector scripts into a native Mac menu bar application that monitors computer habits in the background and provides smart alerts.

**Current State:** Individual detector test scripts with working implementations for:
- ✅ Mouth breathing detection
- ✅ Blink rate monitoring
- ✅ Eye rubbing detection
- ✅ Face touching detection
- ⚠️ Hydration reminder (needs object detection refinement)

**Target State:** Professional Mac menu bar app with:
- Menu bar icon with dropdown menu
- Toggle switches for each detector
- Background camera processing
- Visual (notifications) and audio alerts
- Optional camera preview window
- Preferences/settings management
- Low CPU usage, non-intrusive

---

## Architecture Overview

### Component Hierarchy

```
HealthMonitorApp (Main)
├── MenuBarUI (rumps)
│   ├── Dropdown Menu
│   ├── Checkboxes for detectors
│   └── Preferences/Preview/Quit options
│
├── DetectorManager
│   ├── Enabled detectors list
│   ├── Detector instances
│   └── State management
│
├── CameraThread (Background Worker)
│   ├── Camera capture loop
│   ├── Frame processing
│   └── Detector execution
│
├── AlertSystem
│   ├── Visual notifications (macOS)
│   ├── Audio alerts
│   └── Cooldown management
│
└── PreferencesManager
    ├── Load/save config
    ├── Detector thresholds
    └── Alert settings
```

### Threading Model

```
Main Thread (UI Thread)
├── Menu bar event loop (rumps)
├── Handle user interactions
└── Display notifications

Background Thread (Camera Thread)
├── Capture frames from camera
├── Run enabled detectors
├── Send alerts to queue
└── Continuous loop

Communication
├── Thread-safe Queue for alerts
├── Thread Lock for shared state
└── Event for shutdown signal
```

---

## Project File Structure

```
habit-tracker/
├── PLAN.md                          # This file
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
│
├── main.py                          # Entry point - launch menu bar app
│
├── app/
│   ├── __init__.py
│   ├── menu_bar_app.py             # Main application class (rumps)
│   ├── detector_manager.py         # Manages detector lifecycle
│   ├── camera_thread.py            # Background camera worker
│   ├── alert_system.py             # Notification & audio alerts
│   ├── preferences_manager.py      # Settings load/save
│   └── preview_window.py           # Optional camera preview
│
├── detectors/
│   ├── __init__.py
│   ├── base_detector.py            # Abstract base class
│   ├── mouth_breathing_detector.py
│   ├── blink_detector.py
│   ├── eye_rubbing_detector.py
│   ├── face_touching_detector.py
│   └── hydration_detector.py
│
├── assets/
│   ├── icons/
│   │   ├── menubar_icon.png        # Menu bar icon (16x16, 32x32)
│   │   └── app_icon.png            # App icon (512x512)
│   └── sounds/
│       ├── alert_gentle.wav        # Gentle notification sound
│       └── alert_urgent.wav        # Urgent warning sound
│
├── config/
│   ├── default_config.json         # Default settings
│   └── user_config.json            # User preferences (created at runtime)
│
└── _archive_original_detectors/    # Original test scripts (reference)
    ├── base_models/
    └── test_*.py
```

---

## Technology Stack

### Required Libraries

```txt
# Core dependencies
opencv-python>=4.8.0              # Camera and image processing
mediapipe>=0.10.30                # Face and hand landmark detection
numpy>=1.24.0                     # Numerical operations

# Menu bar app
rumps>=0.4.0                      # Mac menu bar framework

# Notifications
pync>=2.0.3                       # macOS notifications (optional)
# OR use subprocess with osascript (built-in)

# Audio alerts
pygame>=2.5.0                     # Audio playback
# OR playsound>=1.3.0              # Simpler alternative

# Threading
# (built-in: threading, queue)
```

### MediaPipe Models (Already Downloaded)
- face_landmarker.task (3.6MB)
- hand_landmarker.task (7.6MB)

---

## Phase-by-Phase Implementation

---

## PHASE 1: Project Setup & Refactoring

**Goal:** Organize project structure and prepare detector classes for integration

### Task 1.1: Project Restructuring
- [x] Create `_archive_original_detectors/` folder
- [x] Move test files and base_models to archive
- [ ] Create new folder structure (app/, detectors/, assets/, config/)
- [ ] Create `__init__.py` files in Python packages
- [ ] Create `requirements.txt` with all dependencies
- [ ] Create `README.md` with project overview

### Task 1.2: Install Dependencies
```bash
pip install rumps opencv-python mediapipe numpy pygame
```
- [ ] Install rumps for menu bar
- [ ] Verify existing packages (opencv, mediapipe, numpy)
- [ ] Install pygame for audio alerts
- [ ] Test imports to ensure all packages work

### Task 1.3: Create Base Detector Class
**File:** `detectors/base_detector.py`

**Requirements:**
- Abstract base class that all detectors inherit from
- Thread-safe design
- Consistent interface for enable/disable
- Alert generation

**Methods:**
```python
class BaseDetector(ABC):
    def __init__(self, name, enabled=False)
    def enable()
    def disable()
    def is_enabled()
    @abstractmethod
    def detect(face_landmarks, hand_landmarks, frame)
    @abstractmethod
    def get_status()
    def should_alert()
    def reset_alert_cooldown()
```

**Implementation Steps:**
1. [ ] Create base class with __init__
2. [ ] Add enabled/disabled state management
3. [ ] Add abstract methods for detect() and get_status()
4. [ ] Add alert cooldown tracking (prevent spam)
5. [ ] Add configuration dict for thresholds
6. [ ] Add thread lock for state changes
7. [ ] Write docstrings for all methods

### Task 1.4: Refactor Mouth Breathing Detector
**File:** `detectors/mouth_breathing_detector.py`

**Reference:** `_archive_original_detectors/base_models/mouth_breathing_detector.py`

**Changes needed:**
1. [ ] Inherit from new BaseDetector
2. [ ] Remove run() method (no longer needed)
3. [ ] Update detect() to match new signature
4. [ ] Remove OpenCV drawing code (will be in preview window)
5. [ ] Make thread-safe (use locks if modifying state)
6. [ ] Add get_status() method returning dict
7. [ ] Keep threshold: 0.05 (already calibrated)
8. [ ] Test in isolation

**Detection Logic to Keep:**
- MAR calculation (working correctly)
- Threshold: 0.05
- Frame counter: 120 frames before alert

### Task 1.5: Refactor Blink Detector
**File:** `detectors/blink_detector.py`

**Reference:** `_archive_original_detectors/base_models/blink_detector.py`

**Changes needed:**
1. [ ] Inherit from new BaseDetector
2. [ ] Remove run() method
3. [ ] Update detect() to match signature
4. [ ] Remove drawing code
5. [ ] Make thread-safe
6. [ ] Keep calibrated threshold: 0.012
7. [ ] Keep blink counting logic
8. [ ] Test in isolation

**Detection Logic to Keep:**
- EAR calculation (fixed version)
- Threshold: 0.012
- Blink counter with 60-second window
- Alert if < 6 blinks per minute

### Task 1.6: Refactor Eye Rubbing Detector
**File:** `detectors/eye_rubbing_detector.py`

**Reference:** `_archive_original_detectors/base_models/eye_rubbing_detector.py`

**Changes needed:**
1. [ ] Inherit from new BaseDetector
2. [ ] Update to use new BaseDetector methods
3. [ ] Remove drawing code
4. [ ] Update threshold to 0.02 (from testing)
5. [ ] Make thread-safe
6. [ ] Test with hand detection

**Detection Logic to Keep:**
- Hand-to-eye proximity detection
- Threshold: 0.02 (calibrated)
- Hand landmarks: [0, 4, 8, 12, 16, 20]
- Eye landmarks: [33, 263]

### Task 1.7: Refactor Face Touching Detector
**File:** `detectors/face_touching_detector.py`

**Reference:** `_archive_original_detectors/test_face_touching_detector.py`

**Changes needed:**
1. [ ] Inherit from new BaseDetector
2. [ ] Use oval detection logic (tested and working)
3. [ ] Remove drawing code
4. [ ] Thresholds: horizontal=0.12, vertical=0.35
5. [ ] Track touches over 2-minute period
6. [ ] Alert if >= 5 touches in 2 minutes
7. [ ] Make thread-safe

**Detection Logic to Keep:**
- Oval shape detection (ellipse formula)
- Touch counting over time window
- Horizontal threshold: 0.12
- Vertical threshold: 0.35

### Task 1.8: Refactor Hydration Detector
**File:** `detectors/hydration_detector.py`

**Reference:** `_archive_original_detectors/test_hydration_detector.py`

**Decision Point:**
- Object detection approach was complex (SSL errors, model download)
- **Simplified approach:** Time-based with optional manual reset
- User can reset timer from menu when they drink water
- OR: Use hand-near-mouth as proxy (simpler than object detection)

**Implementation:**
1. [ ] Inherit from new BaseDetector
2. [ ] Time-based tracking (45-minute intervals)
3. [ ] Track last drink time
4. [ ] Alert when interval exceeded
5. [ ] Add reset_timer() method (called from menu)
6. [ ] Optional: Detect hand-to-mouth gesture as drinking proxy

**Alternative (if object detection desired):**
- Implement later as enhancement
- For now: manual timer reset is simpler and works

---

## PHASE 2: Menu Bar Application

**Goal:** Create working menu bar app with toggle switches

### Task 2.1: Create Basic Menu Bar App
**File:** `app/menu_bar_app.py`

**Steps:**
1. [ ] Import rumps
2. [ ] Create HealthMonitorApp class inheriting rumps.App
3. [ ] Set app name: "Habit Tracker"
4. [ ] Create placeholder icon (text-based for now)
5. [ ] Add quit button
6. [ ] Test: App appears in menu bar and can quit

**Minimal Code Structure:**
```python
import rumps

class HealthMonitorApp(rumps.App):
    def __init__(self):
        super().__init__("Habit Tracker", icon=None, quit_button=None)

    @rumps.clicked("Quit")
    def quit_app(self, _):
        rumps.quit_application()
```

### Task 2.2: Add Detector Toggle Checkboxes
**File:** `app/menu_bar_app.py`

**Steps:**
1. [ ] Create menu items for each detector
2. [ ] Make them checkable (rumps.MenuItem)
3. [ ] Add click handlers
4. [ ] Print state changes to console (for testing)
5. [ ] Test: Clicking toggles checkmark on/off

**Menu Structure:**
```python
menu = [
    rumps.MenuItem("Mouth Breathing", callback=self.toggle_mouth_breathing),
    rumps.MenuItem("Blink Rate", callback=self.toggle_blink),
    rumps.MenuItem("Eye Rubbing", callback=self.toggle_eye_rubbing),
    rumps.MenuItem("Face Touching", callback=self.toggle_face_touching),
    rumps.MenuItem("Hydration", callback=self.toggle_hydration),
    None,  # Separator
    rumps.MenuItem("Show Camera Preview", callback=self.toggle_preview),
    rumps.MenuItem("Preferences...", callback=self.show_preferences),
    None,
    rumps.MenuItem("Quit", callback=self.quit_app)
]
```

### Task 2.3: Add Menu Bar Icon
**File:** `assets/icons/menubar_icon.png`

**Steps:**
1. [ ] Find/create icon (health/eye symbol)
   - Size: 16x16 and 32x32 (for retina)
   - Transparent background
   - Simple, recognizable design
2. [ ] Save to assets/icons/
3. [ ] Load icon in app:
   ```python
   icon = "assets/icons/menubar_icon.png"
   super().__init__("Habit Tracker", icon=icon)
   ```
4. [ ] Test: Icon appears in menu bar

**Icon Options:**
- Eye symbol (👁)
- Health cross (➕)
- Custom designed icon

### Task 2.4: Create Main Entry Point
**File:** `main.py`

**Steps:**
1. [ ] Import HealthMonitorApp
2. [ ] Create if __name__ == '__main__' block
3. [ ] Initialize app
4. [ ] Run app.run()
5. [ ] Test: Launch app from command line

```python
#!/usr/bin/env python3
from app.menu_bar_app import HealthMonitorApp

if __name__ == '__main__':
    app = HealthMonitorApp()
    app.run()
```

### Task 2.5: Test Menu Bar App
**Testing Checklist:**
- [ ] App launches without errors
- [ ] Icon appears in menu bar
- [ ] Clicking icon shows dropdown menu
- [ ] All detector names visible
- [ ] Clicking toggles checkmark
- [ ] Quit button works

---

## PHASE 3: Detector Manager

**Goal:** Central management of all detectors and their state

### Task 3.1: Create DetectorManager Class
**File:** `app/detector_manager.py`

**Purpose:**
- Initialize all detector instances
- Track which detectors are enabled
- Provide interface to enable/disable detectors
- Run detection on a frame for enabled detectors only

**Class Structure:**
```python
class DetectorManager:
    def __init__(self):
        self.detectors = {}  # name -> detector instance
        self.face_detector = None  # MediaPipe face
        self.hand_detector = None  # MediaPipe hand

    def initialize_detectors(self):
        # Create all detector instances

    def initialize_mediapipe(self):
        # Load MediaPipe models

    def enable_detector(self, name):
        # Enable specific detector

    def disable_detector(self, name):
        # Disable specific detector

    def is_enabled(self, name):
        # Check if detector enabled

    def process_frame(self, frame):
        # Run all enabled detectors
        # Return list of alerts

    def get_all_statuses(self):
        # Get status dict for all detectors

    def cleanup(self):
        # Release resources
```

### Task 3.2: Initialize MediaPipe Models
**File:** `app/detector_manager.py`

**Steps:**
1. [ ] Load face_landmarker.task
2. [ ] Create FaceLandmarker (VIDEO mode)
3. [ ] Load hand_landmarker.task
4. [ ] Create HandLandmarker (VIDEO mode)
5. [ ] Store as instance variables
6. [ ] Add error handling
7. [ ] Test: Models load without errors

**Implementation:**
```python
def initialize_mediapipe(self):
    # Face detector
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    face_options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        running_mode=vision.RunningMode.VIDEO
    )
    self.face_detector = vision.FaceLandmarker.create_from_options(face_options)

    # Hand detector
    hand_base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    hand_options = vision.HandLandmarkerOptions(
        base_options=hand_base_options,
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO
    )
    self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)
```

### Task 3.3: Create Detector Instances
**File:** `app/detector_manager.py`

**Steps:**
1. [ ] Import all detector classes
2. [ ] Create instance of each detector
3. [ ] Store in self.detectors dict
4. [ ] Set initial enabled state (all disabled)
5. [ ] Test: Can create all detectors without errors

```python
def initialize_detectors(self):
    from detectors.mouth_breathing_detector import MouthBreathingDetector
    from detectors.blink_detector import BlinkDetector
    from detectors.eye_rubbing_detector import EyeRubbingDetector
    from detectors.face_touching_detector import FaceTouchingDetector
    from detectors.hydration_detector import HydrationDetector

    self.detectors = {
        'mouth_breathing': MouthBreathingDetector(),
        'blink_rate': BlinkDetector(),
        'eye_rubbing': EyeRubbingDetector(),
        'face_touching': FaceTouchingDetector(),
        'hydration': HydrationDetector()
    }
```

### Task 3.4: Implement Enable/Disable
**File:** `app/detector_manager.py`

**Steps:**
1. [ ] Implement enable_detector(name)
2. [ ] Implement disable_detector(name)
3. [ ] Implement is_enabled(name)
4. [ ] Add validation (name exists)
5. [ ] Test: Can toggle detectors on/off

```python
def enable_detector(self, name):
    if name in self.detectors:
        self.detectors[name].enable()
        print(f"Enabled: {name}")

def disable_detector(self, name):
    if name in self.detectors:
        self.detectors[name].disable()
        print(f"Disabled: {name}")
```

### Task 3.5: Implement Frame Processing
**File:** `app/detector_manager.py`

**Steps:**
1. [ ] Accept frame as input
2. [ ] Convert to RGB for MediaPipe
3. [ ] Run face detection
4. [ ] Run hand detection
5. [ ] For each enabled detector:
   - [ ] Call detect() with landmarks
   - [ ] Check if should_alert()
   - [ ] Collect alerts
6. [ ] Return list of alert dicts
7. [ ] Test: Process frame without errors

```python
def process_frame(self, frame, timestamp_ms):
    alerts = []

    # Detect face and hands
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    face_result = self.face_detector.detect_for_video(mp_image, timestamp_ms)
    hand_result = self.hand_detector.detect_for_video(mp_image, timestamp_ms)

    face_landmarks = face_result.face_landmarks[0] if face_result.face_landmarks else None
    hand_landmarks = hand_result.hand_landmarks if hand_result.hand_landmarks else None

    # Run enabled detectors
    for name, detector in self.detectors.items():
        if detector.is_enabled():
            detector.detect(face_landmarks, hand_landmarks, frame)
            if detector.should_alert():
                alerts.append({
                    'detector': name,
                    'message': detector.get_alert_message(),
                    'severity': detector.get_severity()
                })

    return alerts
```

---

## PHASE 4: Camera Background Thread

**Goal:** Run camera capture and detection in background thread

### Task 4.1: Create CameraThread Class
**File:** `app/camera_thread.py`

**Purpose:**
- Run in separate thread
- Capture frames from camera
- Process frames through DetectorManager
- Send alerts to main thread via queue
- Handle start/stop gracefully

**Class Structure:**
```python
import threading
import queue
import cv2

class CameraThread:
    def __init__(self, detector_manager, alert_queue):
        self.detector_manager = detector_manager
        self.alert_queue = alert_queue
        self.thread = None
        self.running = False
        self.camera = None

    def start(self):
        # Start background thread

    def stop(self):
        # Stop thread and cleanup

    def run(self):
        # Main camera loop

    def initialize_camera(self):
        # Open camera

    def cleanup_camera(self):
        # Release camera
```

### Task 4.2: Implement Camera Initialization
**File:** `app/camera_thread.py`

**Steps:**
1. [ ] Open camera with cv2.VideoCapture(0)
2. [ ] Set resolution (1280x720)
3. [ ] Add error handling (camera not available)
4. [ ] Test: Camera opens successfully

```python
def initialize_camera(self):
    self.camera = cv2.VideoCapture(0)
    if not self.camera.isOpened():
        raise RuntimeError("Could not open camera")

    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Camera initialized")
```

### Task 4.3: Implement Main Loop
**File:** `app/camera_thread.py`

**Steps:**
1. [ ] Implement run() method
2. [ ] Capture frames continuously
3. [ ] Convert BGR to RGB
4. [ ] Call detector_manager.process_frame()
5. [ ] Put alerts into queue
6. [ ] Add frame rate control (30 FPS)
7. [ ] Check self.running flag for exit
8. [ ] Test: Loop runs without errors

```python
def run(self):
    frame_timestamp_ms = 0
    frame_delay = 33  # ~30 FPS

    while self.running:
        ret, frame = self.camera.read()
        if not ret:
            continue

        frame_timestamp_ms += frame_delay

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        alerts = self.detector_manager.process_frame(rgb_frame, frame_timestamp_ms)

        # Send alerts to main thread
        for alert in alerts:
            self.alert_queue.put(alert)

        # Frame rate control
        cv2.waitKey(frame_delay)
```

### Task 4.4: Implement Start/Stop
**File:** `app/camera_thread.py`

**Steps:**
1. [ ] Implement start() method
2. [ ] Create and start thread
3. [ ] Set running flag to True
4. [ ] Implement stop() method
5. [ ] Set running flag to False
6. [ ] Wait for thread to finish (join)
7. [ ] Cleanup camera
8. [ ] Test: Can start and stop cleanly

```python
def start(self):
    if self.thread is not None:
        return  # Already running

    self.initialize_camera()
    self.running = True
    self.thread = threading.Thread(target=self.run, daemon=True)
    self.thread.start()
    print("Camera thread started")

def stop(self):
    if self.thread is None:
        return

    self.running = False
    self.thread.join(timeout=2.0)
    self.cleanup_camera()
    self.thread = None
    print("Camera thread stopped")
```

---

## PHASE 5: Alert System

**Goal:** Display visual and audio alerts when violations detected

### Task 5.1: Create AlertSystem Class
**File:** `app/alert_system.py`

**Purpose:**
- Show macOS notifications
- Play audio alerts
- Manage alert cooldowns (prevent spam)
- Track alert history

**Class Structure:**
```python
class AlertSystem:
    def __init__(self):
        self.cooldowns = {}  # detector_name -> last_alert_time
        self.cooldown_duration = 60  # seconds
        self.notification_enabled = True
        self.audio_enabled = True

    def send_alert(self, alert):
        # Process and send alert

    def show_notification(self, title, message):
        # Display macOS notification

    def play_sound(self, severity):
        # Play audio alert

    def is_on_cooldown(self, detector_name):
        # Check if alert is on cooldown
```

### Task 5.2: Implement macOS Notifications
**File:** `app/alert_system.py`

**Option 1: Using osascript (built-in)**
```python
import subprocess

def show_notification(self, title, message):
    script = f'''
    display notification "{message}" with title "{title}"
    '''
    subprocess.run(['osascript', '-e', script])
```

**Option 2: Using pync library**
```python
import pync

def show_notification(self, title, message):
    pync.notify(message, title=title, appIcon="assets/icons/app_icon.png")
```

**Steps:**
1. [ ] Choose notification method (osascript is simpler)
2. [ ] Implement show_notification()
3. [ ] Test: Notification appears in macOS
4. [ ] Handle errors gracefully

### Task 5.3: Implement Audio Alerts
**File:** `app/alert_system.py`

**Using pygame:**
```python
import pygame

def __init__(self):
    pygame.mixer.init()
    self.sounds = {
        'gentle': pygame.mixer.Sound('assets/sounds/alert_gentle.wav'),
        'urgent': pygame.mixer.Sound('assets/sounds/alert_urgent.wav')
    }

def play_sound(self, severity):
    sound_key = 'urgent' if severity == 'high' else 'gentle'
    self.sounds[sound_key].play()
```

**Steps:**
1. [ ] Find/create sound files (.wav format)
2. [ ] Save to assets/sounds/
3. [ ] Initialize pygame.mixer
4. [ ] Load sound files
5. [ ] Implement play_sound()
6. [ ] Test: Sound plays correctly
7. [ ] Handle missing files gracefully

### Task 5.4: Implement Cooldown Logic
**File:** `app/alert_system.py`

**Steps:**
1. [ ] Track last alert time per detector
2. [ ] Check cooldown before sending alert
3. [ ] Configurable cooldown duration (default 60s)
4. [ ] Test: Repeated violations don't spam

```python
import time

def is_on_cooldown(self, detector_name):
    if detector_name not in self.cooldowns:
        return False

    elapsed = time.time() - self.cooldowns[detector_name]
    return elapsed < self.cooldown_duration

def send_alert(self, alert):
    detector_name = alert['detector']

    # Check cooldown
    if self.is_on_cooldown(detector_name):
        return

    # Send alerts
    if self.notification_enabled:
        self.show_notification("Habit Tracker Alert", alert['message'])

    if self.audio_enabled:
        self.play_sound(alert['severity'])

    # Update cooldown
    self.cooldowns[detector_name] = time.time()
```

---

## PHASE 6: Integration

**Goal:** Connect all components together

### Task 6.1: Integrate DetectorManager with MenuBar
**File:** `app/menu_bar_app.py`

**Steps:**
1. [ ] Create DetectorManager instance in __init__
2. [ ] Initialize detectors on startup
3. [ ] In toggle callbacks, call detector_manager.enable/disable
4. [ ] Update checkmark state based on is_enabled()
5. [ ] Test: Toggling menu items enables/disables detectors

```python
def __init__(self):
    super().__init__("Habit Tracker", icon="assets/icons/menubar_icon.png")

    self.detector_manager = DetectorManager()
    self.detector_manager.initialize_mediapipe()
    self.detector_manager.initialize_detectors()

    # ... menu setup

def toggle_mouth_breathing(self, sender):
    detector_name = 'mouth_breathing'
    if sender.state:
        self.detector_manager.disable_detector(detector_name)
        sender.state = 0
    else:
        self.detector_manager.enable_detector(detector_name)
        sender.state = 1
```

### Task 6.2: Integrate CameraThread
**File:** `app/menu_bar_app.py`

**Steps:**
1. [ ] Create alert_queue (threading.Queue)
2. [ ] Create AlertSystem instance
3. [ ] Create CameraThread instance
4. [ ] Start camera thread on first detector enable
5. [ ] Stop camera thread when all detectors disabled
6. [ ] Test: Camera starts when detector enabled

```python
def __init__(self):
    # ...previous code...

    self.alert_queue = queue.Queue()
    self.alert_system = AlertSystem()
    self.camera_thread = CameraThread(self.detector_manager, self.alert_queue)

    # Start alert checker timer
    self.timer = rumps.Timer(self.check_alerts, 0.5)  # Check twice per second
    self.timer.start()
```

### Task 6.3: Process Alert Queue
**File:** `app/menu_bar_app.py`

**Steps:**
1. [ ] Create timer callback (rumps.Timer)
2. [ ] Check alert_queue for new alerts
3. [ ] Pass alerts to AlertSystem
4. [ ] Run timer continuously
5. [ ] Test: Alerts appear when violations detected

```python
def check_alerts(self, _):
    # Process all pending alerts
    while not self.alert_queue.empty():
        try:
            alert = self.alert_queue.get_nowait()
            self.alert_system.send_alert(alert)
        except queue.Empty:
            break
```

### Task 6.4: Camera Lifecycle Management
**File:** `app/menu_bar_app.py`

**Steps:**
1. [ ] Track if any detector is enabled
2. [ ] Start camera when first detector enabled
3. [ ] Stop camera when last detector disabled
4. [ ] Stop camera on app quit
5. [ ] Test: Camera starts/stops correctly

```python
def any_detector_enabled(self):
    for name in ['mouth_breathing', 'blink_rate', 'eye_rubbing', 'face_touching', 'hydration']:
        if self.detector_manager.is_enabled(name):
            return True
    return False

def update_camera_state(self):
    if self.any_detector_enabled():
        if not self.camera_thread.running:
            self.camera_thread.start()
    else:
        if self.camera_thread.running:
            self.camera_thread.stop()

def quit_app(self, _):
    self.camera_thread.stop()
    rumps.quit_application()
```

### Task 6.5: End-to-End Testing
**Testing Scenarios:**

**Test 1: Enable Single Detector**
- [ ] Enable mouth breathing detector
- [ ] Verify camera starts
- [ ] Breathe through mouth
- [ ] Verify alert appears
- [ ] Verify sound plays
- [ ] Disable detector
- [ ] Verify camera stops

**Test 2: Multiple Detectors**
- [ ] Enable face touching
- [ ] Enable eye rubbing
- [ ] Verify both detectors work
- [ ] Touch face -> alert
- [ ] Rub eyes -> alert
- [ ] Verify cooldowns work (no spam)

**Test 3: Toggle While Running**
- [ ] Enable detector
- [ ] Disable detector while running
- [ ] Enable different detector
- [ ] Verify state transitions cleanly

**Test 4: Alert Cooldown**
- [ ] Trigger alert
- [ ] Trigger same alert within 60s
- [ ] Verify second alert blocked
- [ ] Wait 60s
- [ ] Verify alert works again

**Test 5: App Lifecycle**
- [ ] Launch app
- [ ] Enable detectors
- [ ] Use for 5 minutes
- [ ] Quit app
- [ ] Verify clean shutdown
- [ ] Verify no zombie processes

---

## PHASE 7: Camera Preview Window (Optional)

**Goal:** Add toggleable preview window for debugging

### Task 7.1: Create PreviewWindow Class
**File:** `app/preview_window.py`

**Purpose:**
- Show live camera feed
- Overlay detection visualizations
- Display detector status text
- Toggleable from menu

**Design:**
- Separate OpenCV window
- Runs in camera thread or separate thread
- Shows same frame being processed
- Draws landmarks and detection zones

### Task 7.2: Implement Preview
**File:** `app/preview_window.py`

**Steps:**
1. [ ] Create window with cv2.imshow()
2. [ ] Draw detection overlays
3. [ ] Show enabled detectors
4. [ ] Display status for each
5. [ ] Handle window close
6. [ ] Test: Preview shows camera

### Task 7.3: Integrate with CameraThread
**File:** `app/camera_thread.py`

**Steps:**
1. [ ] Add preview_enabled flag
2. [ ] If enabled, draw overlays on frame
3. [ ] Show frame in window
4. [ ] Toggle from menu
5. [ ] Test: Can show/hide preview

---

## PHASE 8: Preferences & Configuration

**Goal:** Persistent settings and customization

### Task 8.1: Create Configuration System
**File:** `app/preferences_manager.py`

**Purpose:**
- Load/save user preferences
- Manage detector thresholds
- Alert settings
- Default values

**Config Structure (JSON):**
```json
{
  "detectors": {
    "mouth_breathing": {
      "enabled": false,
      "threshold": 0.05,
      "warning_frames": 120
    },
    "blink_rate": {
      "enabled": false,
      "threshold": 0.012,
      "min_blinks_per_minute": 6
    }
    // ... other detectors
  },
  "alerts": {
    "visual_enabled": true,
    "audio_enabled": true,
    "cooldown_seconds": 60
  },
  "camera": {
    "width": 1280,
    "height": 720,
    "fps": 30
  }
}
```

### Task 8.2: Implement Load/Save
**File:** `app/preferences_manager.py`

**Steps:**
1. [ ] Load default config
2. [ ] Load user config (if exists)
3. [ ] Merge configs
4. [ ] Save user changes
5. [ ] Apply to detectors on load

### Task 8.3: Create Preferences UI
**File:** `app/menu_bar_app.py`

**Option 1: Simple Dialog**
- Use rumps.Window for input

**Option 2: Tkinter Window**
- More features, sliders for thresholds

**Steps:**
1. [ ] Create preferences window
2. [ ] Show current settings
3. [ ] Allow editing
4. [ ] Save on confirm
5. [ ] Apply changes to running detectors

---

## PHASE 9: Polish & Optimization

### Task 9.1: Performance Optimization

**CPU Usage:**
- [ ] Profile detector performance
- [ ] Reduce frame rate if needed (20 FPS instead of 30)
- [ ] Skip frames when no detectors enabled
- [ ] Optimize MediaPipe calls

**Memory:**
- [ ] Check for memory leaks
- [ ] Release frames properly
- [ ] Limit alert queue size

### Task 9.2: Error Handling

**Add Error Handling For:**
- [ ] Camera not available
- [ ] MediaPipe models not found
- [ ] Permissions denied (camera)
- [ ] Detector exceptions
- [ ] Thread errors

**Implementation:**
- Try/except blocks
- Graceful degradation
- User-friendly error messages
- Logging for debugging

### Task 9.3: User Experience

**Improvements:**
- [ ] Status indicator in menu bar icon (change color?)
- [ ] Notification sound customization
- [ ] Alert message customization
- [ ] Statistics tracking (optional)
- [ ] Help/documentation

### Task 9.4: App Icon & Branding

**Steps:**
1. [ ] Design/find app icon (512x512)
2. [ ] Create menu bar icon variants
3. [ ] Update app name/branding
4. [ ] Add about dialog

---

## PHASE 10: Packaging & Distribution

### Task 10.1: Create Standalone App

**Using py2app:**
```bash
pip install py2app
python setup.py py2app
```

**Steps:**
1. [ ] Create setup.py for py2app
2. [ ] Include all assets
3. [ ] Include MediaPipe models
4. [ ] Build .app bundle
5. [ ] Test standalone app

### Task 10.2: Handle Permissions

**macOS Permissions Needed:**
- Camera access
- Notifications (optional)

**Steps:**
1. [ ] Add Info.plist entries
2. [ ] Request permissions on first launch
3. [ ] Handle permission denials gracefully

### Task 10.3: Documentation

**Create:**
1. [ ] README.md with features
2. [ ] Installation instructions
3. [ ] Usage guide
4. [ ] Troubleshooting section
5. [ ] License file

---

## Testing Strategy

### Unit Tests
- [ ] Test each detector independently
- [ ] Test DetectorManager
- [ ] Test AlertSystem cooldowns
- [ ] Test PreferencesManager load/save

### Integration Tests
- [ ] Test detector enable/disable flow
- [ ] Test alert flow (violation -> queue -> notification)
- [ ] Test camera lifecycle
- [ ] Test preferences persistence

### System Tests
- [ ] Long-running stability (1 hour+)
- [ ] CPU usage monitoring
- [ ] Memory usage monitoring
- [ ] Multiple violations in sequence
- [ ] All detectors enabled simultaneously

### User Acceptance Tests
- [ ] Install on fresh Mac
- [ ] Test with real usage
- [ ] Verify alerts are helpful, not annoying
- [ ] Check cooldown times are appropriate
- [ ] Verify performance is acceptable

---

## Success Criteria

### Phase 1-3: Foundation
- ✅ All detectors refactored and working
- ✅ Menu bar app appears and responds
- ✅ DetectorManager can enable/disable detectors

### Phase 4-6: Core Functionality
- ✅ Camera runs in background
- ✅ Detectors process frames correctly
- ✅ Alerts appear when violations occur
- ✅ No crashes or hangs

### Phase 7-9: Polish
- ✅ Preview window works (optional)
- ✅ Preferences persist across launches
- ✅ CPU usage < 10% on average
- ✅ No memory leaks

### Phase 10: Deployment
- ✅ Standalone app works
- ✅ Permissions handled correctly
- ✅ Documentation complete

---

## Known Challenges & Mitigation

### Challenge 1: Threading Complexity
**Risk:** Deadlocks, race conditions
**Mitigation:**
- Use thread-safe queue for communication
- Minimize shared state
- Use locks only where necessary
- Test thoroughly

### Challenge 2: Camera Performance
**Risk:** High CPU usage, battery drain
**Mitigation:**
- Reduce frame rate (20-30 FPS)
- Lower resolution if needed (640x480)
- Only process when detectors enabled
- Profile and optimize

### Challenge 3: macOS Permissions
**Risk:** User denies camera access
**Mitigation:**
- Clear permission request message
- Graceful handling if denied
- Instructions to enable in System Preferences
- Test on fresh macOS install

### Challenge 4: Alert Fatigue
**Risk:** Too many notifications annoy user
**Mitigation:**
- 60-second cooldown per detector
- Make cooldown configurable
- User can disable audio/visual separately
- Smart threshold tuning

---

## Future Enhancements (Post-MVP)

### Advanced Features
- [ ] Statistics dashboard (touches per day, blink rate trends)
- [ ] Machine learning to adapt thresholds to user
- [ ] Integration with health apps (Apple Health)
- [ ] Reminders during specific times (work hours)
- [ ] Focus mode (only certain detectors)

### Additional Detectors
- [ ] Posture detection (head/shoulder alignment)
- [ ] Screen distance (face size relative to frame)
- [ ] Break reminders (time-based)
- [ ] Fatigue detection (yawning)

### Platform Expansion
- [ ] Windows support (system tray)
- [ ] Linux support

---

## Development Timeline Estimate

**Assuming 2-3 hours per day:**

- **Week 1:** Phase 1-2 (Setup, refactoring, basic menu bar)
- **Week 2:** Phase 3-4 (DetectorManager, CameraThread)
- **Week 3:** Phase 5-6 (Alerts, integration)
- **Week 4:** Phase 7-9 (Preview, preferences, polish)
- **Week 5:** Phase 10 (Packaging, testing, documentation)

**Total: ~4-5 weeks to MVP**

---

## Next Steps

1. Install dependencies (`rumps`, `pygame`)
2. Create folder structure
3. Start with Phase 1, Task 1.3 (Base Detector class)
4. Proceed sequentially through phases
5. Test after each task
6. Commit frequently to version control

---

## References

### Documentation
- rumps: https://github.com/jaredks/rumps
- MediaPipe: https://developers.google.com/mediapipe
- OpenCV: https://docs.opencv.org/
- pygame: https://www.pygame.org/docs/

### Calibrated Thresholds (from testing)
- Mouth breathing: 0.05 MAR
- Blink rate: 0.012 EAR, < 6 blinks/min
- Eye rubbing: 0.02 proximity
- Face touching: 0.12 horizontal, 0.35 vertical oval

---

**Last Updated:** 2026-02-05
**Status:** Ready to implement
**Next Task:** Phase 1, Task 1.1 - Create folder structure
