"""
Base Models Package
Contains all health monitoring detector classes.
"""

from .base_detector import BaseDetector
from .mouth_breathing_detector import MouthBreathingDetector
from .blink_detector import BlinkDetector
from .posture_detector import PostureDetector
from .fatigue_detector import FatigueDetector
from .hydration_detector import HydrationDetector
from .eye_rubbing_detector import EyeRubbingDetector
from .face_touching_detector import FaceTouchingDetector
from .focus_detector import FocusDetector
from .phone_detector import PhoneDetector

__all__ = [
    'BaseDetector',
    'MouthBreathingDetector',
    'BlinkDetector',
    'PostureDetector',
    'FatigueDetector',
    'HydrationDetector',
    'EyeRubbingDetector',
    'FaceTouchingDetector',
    'FocusDetector',
    'PhoneDetector'
]
