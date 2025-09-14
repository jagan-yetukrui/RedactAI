"""
Text detection and name redaction module for RedactAI.

This module provides functionality to detect text in images and videos
using OCR and redact personal names using Named Entity Recognition.
"""

from .text_detector import TextDetector
from .name_redactor import NameRedactor
from .text_redactor import TextRedactor

__all__ = ['TextDetector', 'NameRedactor', 'TextRedactor']
