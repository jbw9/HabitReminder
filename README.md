# Habit Tracker

A Mac menu bar application that monitors computer usage habits and provides intelligent alerts to promote healthier computing.

## Overview

Habit Tracker runs silently in your Mac's menu bar, using your camera to detect unhealthy habits like mouth breathing, insufficient blinking, eye rubbing, and excessive face touching. When a habit is detected, you'll receive gentle notifications to help you correct the behavior.

## Features

- 🌬️ **Mouth Breathing Detection** - Alerts when breathing through mouth instead of nose
- 👁️ **Blink Rate Monitoring** - Tracks blink frequency (should be 6+ per minute)
- 🤚 **Eye Rubbing Detection** - Warns when rubbing eyes too frequently
- 👆 **Face Touching Detection** - Monitors excessive face touching (hygiene & stress indicator)
- 💧 **Hydration Reminders** - Reminds you to drink water at regular intervals

## Technology

- **Computer Vision**: MediaPipe for face and hand landmark detection
- **UI**: rumps (Mac menu bar framework)
- **Alerts**: macOS notifications + audio alerts
- **Threading**: Background camera processing for non-intrusive operation

## Project Status

**Current Phase:** Planning & Setup
- ✅ Individual detectors tested and calibrated
- ✅ Comprehensive implementation plan created
- 🚧 Menu bar app architecture in progress

See [PLAN.md](PLAN.md) for detailed implementation roadmap.

## Original Research

The `_archive_original_detectors/` folder contains the original test scripts and detector implementations that were used to validate each detection algorithm and calibrate thresholds for accuracy.

## Requirements

```bash
opencv-python>=4.8.0
mediapipe>=0.10.30
numpy>=1.24.0
rumps>=0.4.0
pygame>=2.5.0
```

## Installation (Future)

_Coming soon - will be available as standalone Mac app_

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run (once implemented)
python main.py
```

## Privacy

All processing happens locally on your Mac. No images or data are sent to any external servers. The camera is only active when detectors are enabled, and you have full control over which features to use.

## License

_To be determined_

---

**Note:** This project is currently in development. See PLAN.md for implementation progress.
