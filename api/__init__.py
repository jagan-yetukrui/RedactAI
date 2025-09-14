"""
API module for RedactAI.

This module provides REST API endpoints for media processing
using FastAPI framework.
"""

from .main import app
from .models import ProcessingRequest, ProcessingResponse, HealthResponse

__all__ = ['app', 'ProcessingRequest', 'ProcessingResponse', 'HealthResponse']
