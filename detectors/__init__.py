"""
Detectors Package
Contains all health monitoring detector classes.
"""

from .base_detector import BaseDetector
from .mouth_breathing_detector import MouthBreathingDetector
from .blink_detector import BlinkDetector
from .eye_rubbing_detector import EyeRubbingDetector
from .face_touching_detector import FaceTouchingDetector
from .hydration_detector import HydrationDetector

__all__ = [
    'BaseDetector',
    'MouthBreathingDetector',
    'BlinkDetector',
    'EyeRubbingDetector',
    'FaceTouchingDetector',
    'HydrationDetector',
]
