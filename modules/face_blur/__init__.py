"""
Face detection and blurring module for RedactAI.

This module provides functionality to detect faces in images and videos
using Haar Cascades and apply blurring to protect privacy.
"""

from .face_detector import FaceDetector
from .face_blurrer import FaceBlurrer

__all__ = ['FaceDetector', 'FaceBlurrer']
