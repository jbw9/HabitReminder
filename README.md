# Comprehensive Health Monitor

An AI-powered health monitoring system that uses your computer's camera to track various health indicators and remind you to maintain healthy computing habits.

## Features

### 9 Health Detectors

1. **Mouth Breathing Detector**
   - Detects when you breathe through your mouth instead of your nose
   - Helps maintain proper breathing habits
   - Alert after 4 seconds of mouth breathing

2. **Blink Rate Monitor**
   - Tracks your blink rate to prevent eye strain
   - Normal: 15-20 blinks/minute
   - Alerts when rate drops below 6 blinks/minute (common during computer use)

3. **Posture Monitor**
   - Detects when you're sitting too close to the screen
   - Alerts when your head is tilted at an unhealthy angle
   - Helps prevent "tech neck" and back problems

4. **Fatigue Detector**
   - Detects yawning patterns
   - Alerts after multiple yawns in 5 minutes
   - Suggests taking a break when tired

5. **Hydration Reminder**
   - Time-based reminders to drink water
   - Default: every 45 minutes
   - Press 'h' to reset timer after drinking

6. **Eye Rubbing Detector**
   - Detects when you rub your eyes
   - Sign of fatigue or eye strain
   - Can cause irritation

7. **Face Touching Monitor**
   - Tracks frequent face touching
   - Hygiene concern and stress indicator
   - Alerts after 5 touches in 2 minutes

8. **Screen Focus Monitor**
   - Detects when you're looking away from screen
   - Tracks attention and focus
   - Alerts after extended distraction

9. **Phone Distraction Detector**
   - Detects when you're likely looking at your phone (head down)
   - Helps maintain work focus
   - Alerts after 3 seconds

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure model file exists:**
   - The `face_landmarker.task` file should be in the project directory
   - It was downloaded during initial setup

## Usage

### Run the Health Monitor

```bash
python3 health_monitor.py
```

### Controls

- **q**: Quit the application
- **h**: Reset hydration timer (press after drinking water)
- **SPACE**: Toggle visual overlays on/off
- **1-9**: Toggle individual detectors on/off

### On-Screen Display

- **Status Panel (right side)**: Shows all active detectors and their current status
- **Warning Banner (top)**: Displays active warnings in red
- **Visual Indicators**: Colored boxes and lines around detected features
  - Green: Normal/Good
  - Orange: Warning threshold approaching
  - Red: Issue detected/Warning active

## Project Structure

```
woi/
├── health_monitor.py              # Main application
├── mouth_breathing_detector.py    # Original standalone detector
├── face_landmarker.task          # MediaPipe model file
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── base_models/                  # Detector modules
    ├── __init__.py
    ├── base_detector.py          # Base class for all detectors
    ├── mouth_breathing_detector.py
    ├── blink_detector.py
    ├── posture_detector.py
    ├── fatigue_detector.py
    ├── hydration_detector.py
    ├── eye_rubbing_detector.py
    ├── face_touching_detector.py
    ├── focus_detector.py
    └── phone_detector.py
```

## Configuration

Each detector can be customized by modifying its initialization parameters in `health_monitor.py`:

```python
self.detectors = {
    'mouth_breathing': MouthBreathingDetector(threshold=0.05),
    'blink': BlinkDetector(low_blink_threshold=6),
    'posture': PostureDetector(min_face_width_ratio=0.35),
    'fatigue': FatigueDetector(max_yawns=3),
    'hydration': HydrationDetector(interval_minutes=45),
    # ... etc
}
```

## How It Works

### Technology Stack

- **OpenCV**: Camera access and image processing
- **MediaPipe Face Landmarker**: Real-time facial landmark detection (478 landmarks)
- **MediaPipe Hands**: Hand detection for touch/rubbing detection
- **NumPy**: Mathematical calculations

### Detection Methods

- **Face-based**: Mouth breathing, blinking, posture, fatigue, focus
- **Hand-based**: Eye rubbing, face touching
- **Hybrid**: Phone detection (head angle + hand position)
- **Time-based**: Hydration reminders

### Architecture

- **Modular Design**: Each detector is an independent class
- **Base Class Pattern**: All detectors inherit from `BaseDetector`
- **Shared Resources**: Single camera and MediaPipe instance
- **Real-time Processing**: ~30 FPS with all detectors active

## Tips for Best Results

1. **Lighting**: Ensure good lighting for accurate face detection
2. **Camera Position**: Position camera at eye level, arm's length away
3. **Calibration**: First run will help you understand your normal ranges
4. **Thresholds**: Adjust sensitivity in detector initialization if needed
5. **Hand Detection**: Works best with hands clearly visible to camera

## Troubleshooting

### No face detected
- Ensure adequate lighting
- Check camera is not blocked
- Move closer to camera

### Too many false positives
- Adjust thresholds in detector initialization
- Disable overly sensitive detectors using number keys

### Low performance
- Close other applications
- Reduce camera resolution in `health_monitor.py`
- Disable some detectors

### Hand detection not working
- Ensure hands are visible to camera
- Check MediaPipe Hands is properly installed
- Verify lighting conditions

## Health Benefits

- **Reduced Eye Strain**: Through blink monitoring and break reminders
- **Better Posture**: Prevents neck and back problems
- **Improved Focus**: Reduces distractions
- **Better Hydration**: Regular water intake reminders
- **Fatigue Management**: Encourages breaks when tired
- **Better Breathing**: Promotes nasal breathing
- **Hygiene**: Reduces face touching

## Future Enhancements

Possible additions:
- Audio alerts (currently visual only)
- Statistics tracking and logging
- Machine learning-based personalization
- Multiple user profiles
- Integration with productivity apps
- Mobile companion app

## Privacy Note

- All processing happens locally on your machine
- No data is sent to external servers
- Camera feed is not recorded or saved
- You have full control over when monitoring is active

## License

This project is for personal use and health monitoring.

## Acknowledgments

- MediaPipe by Google for face and hand detection
- OpenCV for computer vision capabilities
- The Python community for excellent libraries
