"""
License plate detection and blurring module for RedactAI.

This module provides functionality to detect license plates in images and videos
using YOLOv8 and apply blurring to protect privacy.
"""

from .plate_detector import PlateDetector
from .plate_blurrer import PlateBlurrer

__all__ = ['PlateDetector', 'PlateBlurrer']
