"""
Detectors Package
Contains all health monitoring detector classes.
"""

from .base_detector import BaseDetector
from .mouth_breathing_detector import MouthBreathingDetector

__all__ = [
    'BaseDetector',
    'MouthBreathingDetector',
]
