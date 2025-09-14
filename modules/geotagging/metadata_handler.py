"""
Metadata handling functionality for RedactAI.

This module provides functionality to handle and process metadata
from images and videos, including EXIF data and custom metadata.
"""

import json
import os
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime, timezone
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Try to import EXIF handling libraries
try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available. Install Pillow for EXIF metadata handling.")


class MetadataHandler:
    """
    Metadata handler for processing and managing metadata from media files.
    
    This class provides functionality to extract, modify, and save metadata
    from images and videos, including EXIF data and custom metadata.
    """
    
    def __init__(self):
        """Initialize the metadata handler."""
        self.pil_available = PIL_AVAILABLE
        logger.info("MetadataHandler initialized")
    
    def extract_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """
        Extract metadata from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing extracted metadata
        """
        metadata = {
            'file_path': image_path,
            'file_size': os.path.getsize(image_path),
            'extraction_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Extract basic file information
        try:
            # Get file modification time
            mod_time = os.path.getmtime(image_path)
            metadata['file_modified'] = datetime.fromtimestamp(mod_time, timezone.utc).isoformat()
        except Exception as e:
            logger.warning(f"Could not get file modification time: {e}")
        
        # Extract EXIF data if PIL is available
        if self.pil_available:
            try:
                exif_data = self._extract_exif_data(image_path)
                metadata['exif'] = exif_data
            except Exception as e:
                logger.warning(f"Could not extract EXIF data: {e}")
        
        # Extract OpenCV image properties
        try:
            image = cv2.imread(image_path)
            if image is not None:
                h, w, c = image.shape
                metadata['image_properties'] = {
                    'width': w,
                    'height': h,
                    'channels': c,
                    'dtype': str(image.dtype)
                }
        except Exception as e:
            logger.warning(f"Could not extract image properties: {e}")
        
        return metadata
    
    def _extract_exif_data(self, image_path: str) -> Dict[str, Any]:
        """
        Extract EXIF data from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing EXIF data
        """
        exif_data = {}
        
        try:
            with Image.open(image_path) as img:
                exif = img._getexif()
                
                if exif is not None:
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        exif_data[tag] = value
        except Exception as e:
            logger.error(f"Error extracting EXIF data: {e}")
        
        return exif_data
    
    def extract_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing extracted metadata
        """
        metadata = {
            'file_path': video_path,
            'file_size': os.path.getsize(video_path),
            'extraction_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Extract basic file information
        try:
            mod_time = os.path.getmtime(video_path)
            metadata['file_modified'] = datetime.fromtimestamp(mod_time, timezone.utc).isoformat()
        except Exception as e:
            logger.warning(f"Could not get file modification time: {e}")
        
        # Extract video properties using OpenCV
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                metadata['video_properties'] = {
                    'fps': cap.get(cv2.CAP_PROP_FPS),
                    'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC))
                }
                
                # Calculate duration
                fps = metadata['video_properties']['fps']
                frame_count = metadata['video_properties']['frame_count']
                if fps > 0:
                    metadata['video_properties']['duration_seconds'] = frame_count / fps
                
                cap.release()
        except Exception as e:
            logger.warning(f"Could not extract video properties: {e}")
        
        return metadata
    
    def add_custom_metadata(self, metadata: Dict[str, Any], custom_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add custom metadata to existing metadata.
        
        Args:
            metadata: Existing metadata dictionary
            custom_data: Custom data to add
            
        Returns:
            Updated metadata dictionary
        """
        updated_metadata = metadata.copy()
        updated_metadata['custom'] = custom_data
        updated_metadata['last_modified'] = datetime.now(timezone.utc).isoformat()
        
        return updated_metadata
    
    def save_metadata(self, metadata: Dict[str, Any], output_path: str) -> None:
        """
        Save metadata to a JSON file.
        
        Args:
            metadata: Metadata dictionary to save
            output_path: Path to output JSON file
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Metadata saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            raise
    
    def load_metadata(self, metadata_path: str) -> Dict[str, Any]:
        """
        Load metadata from a JSON file.
        
        Args:
            metadata_path: Path to metadata JSON file
            
        Returns:
            Loaded metadata dictionary
        """
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Metadata loaded from {metadata_path}")
            return metadata
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise
    
    def merge_metadata(self, metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple metadata dictionaries.
        
        Args:
            metadata_list: List of metadata dictionaries to merge
            
        Returns:
            Merged metadata dictionary
        """
        if not metadata_list:
            return {}
        
        merged = {
            'merge_timestamp': datetime.now(timezone.utc).isoformat(),
            'source_count': len(metadata_list),
            'sources': []
        }
        
        # Collect all unique keys
        all_keys = set()
        for metadata in metadata_list:
            all_keys.update(metadata.keys())
        
        # Merge data for each key
        for key in all_keys:
            values = [meta.get(key) for meta in metadata_list if key in meta]
            if len(values) == 1:
                merged[key] = values[0]
            else:
                merged[key] = values
        
        # Add source information
        for i, metadata in enumerate(metadata_list):
            source_info = {
                'index': i,
                'file_path': metadata.get('file_path', f'source_{i}'),
                'extraction_timestamp': metadata.get('extraction_timestamp', 'unknown')
            }
            merged['sources'].append(source_info)
        
        return merged
    
    def create_processing_summary(self, metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a summary of processing metadata.
        
        Args:
            metadata_list: List of metadata dictionaries
            
        Returns:
            Dictionary containing processing summary
        """
        if not metadata_list:
            return {
                'total_files': 0,
                'total_size_bytes': 0,
                'file_types': {},
                'processing_timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        total_size = sum(meta.get('file_size', 0) for meta in metadata_list)
        
        # Count file types
        file_types = {}
        for metadata in metadata_list:
            file_path = metadata.get('file_path', '')
            ext = os.path.splitext(file_path)[1].lower()
            file_types[ext] = file_types.get(ext, 0) + 1
        
        # Calculate average file size
        avg_size = total_size / len(metadata_list) if metadata_list else 0
        
        return {
            'total_files': len(metadata_list),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'average_size_bytes': avg_size,
            'file_types': file_types,
            'processing_timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def filter_metadata_by_key(self, metadata: Dict[str, Any], key: str) -> Any:
        """
        Filter metadata by a specific key.
        
        Args:
            metadata: Metadata dictionary
            key: Key to filter by
            
        Returns:
            Value for the specified key, or None if not found
        """
        return metadata.get(key)
    
    def search_metadata(self, metadata_list: List[Dict[str, Any]], search_key: str, search_value: Any) -> List[Dict[str, Any]]:
        """
        Search metadata list for entries matching a key-value pair.
        
        Args:
            metadata_list: List of metadata dictionaries
            search_key: Key to search for
            search_value: Value to match
            
        Returns:
            List of matching metadata dictionaries
        """
        matches = []
        for metadata in metadata_list:
            if metadata.get(search_key) == search_value:
                matches.append(metadata)
        
        return matches
