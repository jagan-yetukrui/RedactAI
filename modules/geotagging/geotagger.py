"""
Geotagging functionality for RedactAI.

This module provides functionality to add geospatial metadata to processed
images and videos, including GPS coordinates and timestamps.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from datetime import datetime, timezone, timedelta
import random
import json
import os

logger = logging.getLogger(__name__)

# Try to import geospatial libraries
try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available. Install Pillow for EXIF metadata handling.")

try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    logger.warning("Folium not available. Install folium for map generation.")


class Geotagger:
    """
    Geotagger that adds geospatial metadata to images and videos.
    
    This class provides functionality to add GPS coordinates, timestamps,
    and other metadata to processed media files.
    """
    
    def __init__(self, mock_gps: bool = True, gps_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = None):
        """
        Initialize the geotagger.
        
        Args:
            mock_gps: If True, generates mock GPS coordinates. If False, requires real GPS data.
            gps_bounds: GPS bounds as ((min_lat, min_lon), (max_lat, max_lon)) for mock data generation.
        """
        self.mock_gps = mock_gps
        self.gps_bounds = gps_bounds or ((37.7749, -122.4194), (37.7849, -122.4094))  # San Francisco area
        self.pil_available = PIL_AVAILABLE
        self.folium_available = FOLIUM_AVAILABLE
        
        logger.info(f"Geotagger initialized with mock_gps={mock_gps}")
    
    def generate_mock_gps(self) -> Tuple[float, float]:
        """
        Generate mock GPS coordinates within the specified bounds.
        
        Returns:
            Tuple of (latitude, longitude)
        """
        min_lat, min_lon = self.gps_bounds[0]
        max_lat, max_lon = self.gps_bounds[1]
        
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
        
        return lat, lon
    
    def add_geotag_to_image(self, image: np.ndarray, gps_coords: Tuple[float, float] = None,
                           timestamp: datetime = None, additional_metadata: Dict[str, Any] = None) -> np.ndarray:
        """
        Add geotag information to an image.
        
        Args:
            image: Input image as numpy array
            gps_coords: GPS coordinates as (latitude, longitude). If None, generates mock data.
            timestamp: Timestamp for the image. If None, uses current time.
            additional_metadata: Additional metadata to include
            
        Returns:
            Image with geotag overlay (visual representation)
        """
        if gps_coords is None:
            gps_coords = self.generate_mock_gps()
        
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        if additional_metadata is None:
            additional_metadata = {}
        
        # Create a copy of the image
        result_image = image.copy()
        
        # Add text overlay with geotag information
        overlay_text = self._create_geotag_overlay(gps_coords, timestamp, additional_metadata)
        result_image = self._add_text_overlay(result_image, overlay_text)
        
        logger.info(f"Added geotag to image: {gps_coords}, {timestamp}")
        return result_image
    
    def _create_geotag_overlay(self, gps_coords: Tuple[float, float], timestamp: datetime,
                              additional_metadata: Dict[str, Any]) -> List[str]:
        """
        Create text overlay for geotag information.
        
        Args:
            gps_coords: GPS coordinates as (latitude, longitude)
            timestamp: Timestamp for the image
            additional_metadata: Additional metadata to include
            
        Returns:
            List of text lines for overlay
        """
        lat, lon = gps_coords
        
        overlay_lines = [
            f"GPS: {lat:.6f}, {lon:.6f}",
            f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        ]
        
        # Add additional metadata
        for key, value in additional_metadata.items():
            overlay_lines.append(f"{key}: {value}")
        
        return overlay_lines
    
    def _add_text_overlay(self, image: np.ndarray, text_lines: List[str]) -> np.ndarray:
        """
        Add text overlay to an image.
        
        Args:
            image: Input image as numpy array
            text_lines: List of text lines to overlay
            
        Returns:
            Image with text overlay
        """
        result_image = image.copy()
        h, w = image.shape[:2]
        
        # Set font properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_color = (255, 255, 255)  # White text
        bg_color = (0, 0, 0)  # Black background
        
        # Calculate text size and position
        line_height = 25
        start_y = 30
        
        for i, line in enumerate(text_lines):
            y_pos = start_y + (i * line_height)
            
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            
            # Draw background rectangle
            cv2.rectangle(result_image, (10, y_pos - text_height - 5), 
                         (10 + text_width + 10, y_pos + 5), bg_color, -1)
            
            # Draw text
            cv2.putText(result_image, line, (15, y_pos), font, font_scale, text_color, font_thickness)
        
        return result_image
    
    def add_geotag_to_video_frame(self, frame: np.ndarray, frame_number: int,
                                 gps_coords: Tuple[float, float] = None,
                                 timestamp: datetime = None,
                                 additional_metadata: Dict[str, Any] = None) -> np.ndarray:
        """
        Add geotag information to a video frame.
        
        Args:
            frame: Input video frame as numpy array
            frame_number: Frame number in the video
            gps_coords: GPS coordinates as (latitude, longitude). If None, generates mock data.
            timestamp: Timestamp for the frame. If None, uses current time.
            additional_metadata: Additional metadata to include
            
        Returns:
            Frame with geotag overlay
        """
        if additional_metadata is None:
            additional_metadata = {}
        
        # Add frame number to metadata
        additional_metadata['frame'] = frame_number
        
        return self.add_geotag_to_image(frame, gps_coords, timestamp, additional_metadata)
    
    def process_video_with_geotags(self, video_path: str, output_path: str,
                                  gps_coords: Tuple[float, float] = None,
                                  start_timestamp: datetime = None,
                                  additional_metadata: Dict[str, Any] = None) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Process a video file and add geotags to each frame.
        
        Args:
            video_path: Path to input video file
            output_path: Path to output video file
            gps_coords: GPS coordinates as (latitude, longitude). If None, generates mock data.
            start_timestamp: Starting timestamp for the video. If None, uses current time.
            additional_metadata: Additional metadata to include
            
        Returns:
            Tuple of (total_frames_processed, list_of_metadata_per_frame)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        all_metadata = []
        
        if start_timestamp is None:
            start_timestamp = datetime.now(timezone.utc)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate timestamp for this frame
                frame_timestamp = start_timestamp + timedelta(seconds=frame_count / fps)
                
                # Add geotag to frame
                processed_frame = self.add_geotag_to_video_frame(
                    frame, frame_count, gps_coords, frame_timestamp, additional_metadata
                )
                
                # Store metadata for this frame
                frame_metadata = {
                    'frame_number': frame_count,
                    'timestamp': frame_timestamp.isoformat(),
                    'gps_coords': gps_coords or self.generate_mock_gps(),
                    'additional_metadata': additional_metadata or {}
                }
                all_metadata.append(frame_metadata)
                
                # Write processed frame
                out.write(processed_frame)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames with geotags")
        
        finally:
            cap.release()
            out.release()
        
        logger.info(f"Video geotagging complete. Processed {frame_count} frames")
        return frame_count, all_metadata
    
    def save_metadata_to_file(self, metadata: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save metadata to a JSON file.
        
        Args:
            metadata: List of metadata dictionaries
            output_path: Path to output JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Metadata saved to {output_path}")
    
    def create_geotag_summary(self, metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a summary of geotag metadata.
        
        Args:
            metadata: List of metadata dictionaries
            
        Returns:
            Dictionary containing geotag summary
        """
        if not metadata:
            return {
                'total_frames': 0,
                'gps_coords': None,
                'time_range': None,
                'duration_seconds': 0
            }
        
        # Extract GPS coordinates (assuming they're consistent)
        gps_coords = metadata[0]['gps_coords']
        
        # Extract timestamps
        timestamps = [datetime.fromisoformat(meta['timestamp']) for meta in metadata]
        start_time = min(timestamps)
        end_time = max(timestamps)
        duration = (end_time - start_time).total_seconds()
        
        return {
            'total_frames': len(metadata),
            'gps_coords': gps_coords,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'time_range': f"{start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}"
        }
    
    def generate_heatmap_data(self, metadata: List[Dict[str, Any]]) -> List[Tuple[float, float, Dict[str, Any]]]:
        """
        Generate heatmap data from geotag metadata.
        
        Args:
            metadata: List of metadata dictionaries
            
        Returns:
            List of tuples containing (latitude, longitude, additional_data)
        """
        heatmap_data = []
        
        for meta in metadata:
            lat, lon = meta['gps_coords']
            additional_data = {
                'frame_number': meta['frame_number'],
                'timestamp': meta['timestamp'],
                **meta['additional_metadata']
            }
            heatmap_data.append((lat, lon, additional_data))
        
        return heatmap_data
