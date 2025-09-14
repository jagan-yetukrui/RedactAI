"""
Geotagging module for RedactAI.

This module provides functionality to add geospatial metadata to processed
images and videos, including GPS coordinates and timestamps.
"""

from .geotagger import Geotagger
from .metadata_handler import MetadataHandler

__all__ = ['Geotagger', 'MetadataHandler']
