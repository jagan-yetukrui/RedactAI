"""
Text redaction functionality for RedactAI.

This module provides comprehensive text redaction capabilities combining
OCR text detection with Named Entity Recognition for name redaction.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from .text_detector import TextDetector
from .name_redactor import NameRedactor

logger = logging.getLogger(__name__)


class TextRedactor:
    """
    Text redactor that detects and redacts text in images and videos.
    
    This class combines OCR text detection with Named Entity Recognition
    to provide comprehensive text redaction capabilities.
    """
    
    def __init__(self, ocr_engine: str = 'tesseract', blur_type: str = 'gaussian',
                 blur_strength: int = 15, redact_names_only: bool = True):
        """
        Initialize the text redactor.
        
        Args:
            ocr_engine: OCR engine to use ('tesseract', 'easyocr', or 'both')
            blur_type: Type of blur to apply ('gaussian', 'pixelate', 'blackout', 'mosaic')
            blur_strength: Strength of the blur effect
            redact_names_only: If True, only redact names. If False, redact all text.
        """
        self.text_detector = TextDetector(ocr_engine=ocr_engine)
        self.name_redactor = NameRedactor()
        self.blur_type = blur_type
        self.blur_strength = blur_strength
        self.redact_names_only = redact_names_only
        
        # Validate blur type
        valid_types = ['gaussian', 'pixelate', 'blackout', 'mosaic']
        if blur_type not in valid_types:
            raise ValueError(f"Invalid blur type. Must be one of: {valid_types}")
        
        logger.info(f"TextRedactor initialized with {blur_type} blur, strength: {blur_strength}")
    
    def redact_text_region(self, image: np.ndarray, text_box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Apply redaction to a specific text region.
        
        Args:
            image: Input image as numpy array
            text_box: Text bounding box as (x, y, width, height)
            
        Returns:
            Image with redacted text region
        """
        x, y, w, h = text_box
        result_image = image.copy()
        
        # Extract text region
        text_region = image[y:y+h, x:x+w]
        
        # Apply blur based on type
        if self.blur_type == 'gaussian':
            redacted_text = self._apply_gaussian_blur(text_region)
        elif self.blur_type == 'pixelate':
            redacted_text = self._apply_pixelate_blur(text_region)
        elif self.blur_type == 'blackout':
            redacted_text = self._apply_blackout_blur(text_region)
        elif self.blur_type == 'mosaic':
            redacted_text = self._apply_mosaic_blur(text_region)
        else:
            raise ValueError(f"Unknown blur type: {self.blur_type}")
        
        # Replace text region with redacted version
        result_image[y:y+h, x:x+w] = redacted_text
        
        return result_image
    
    def _apply_gaussian_blur(self, text_region: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to text region."""
        # Ensure kernel size is odd
        kernel_size = self.blur_strength if self.blur_strength % 2 == 1 else self.blur_strength + 1
        return cv2.GaussianBlur(text_region, (kernel_size, kernel_size), 0)
    
    def _apply_pixelate_blur(self, text_region: np.ndarray) -> np.ndarray:
        """Apply pixelate blur to text region."""
        h, w = text_region.shape[:2]
        
        # Calculate pixel size based on blur strength
        pixel_size = max(1, min(h, w) // self.blur_strength)
        
        # Resize down
        small = cv2.resize(text_region, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
        
        # Resize back up
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return pixelated
    
    def _apply_blackout_blur(self, text_region: np.ndarray) -> np.ndarray:
        """Apply blackout to text region."""
        blacked_out = np.zeros_like(text_region)
        return blacked_out
    
    def _apply_mosaic_blur(self, text_region: np.ndarray) -> np.ndarray:
        """Apply mosaic blur to text region."""
        h, w = text_region.shape[:2]
        
        # Calculate block size
        block_size = max(1, min(h, w) // self.blur_strength)
        
        # Create mosaic effect
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                # Get block region
                block = text_region[i:i+block_size, j:j+block_size]
                
                # Calculate average color
                avg_color = np.mean(block, axis=(0, 1))
                
                # Fill block with average color
                text_region[i:i+block_size, j:j+block_size] = avg_color
        
        return text_region
    
    def process_image(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process an image to detect and redact text.
        
        Args:
            image: Input image as numpy array
            confidence_threshold: Minimum confidence for text detection
            
        Returns:
            Tuple of (processed_image, processing_stats)
        """
        # Detect text regions
        text_detections = self.text_detector.detect_text_regions(image, confidence_threshold)
        
        # Filter detections if only redacting names
        if self.redact_names_only:
            text_detections = self._filter_name_detections(image, text_detections)
        
        # Apply redaction to each detected text region
        result_image = image.copy()
        redacted_count = 0
        
        for detection in text_detections:
            text_box = detection['bbox']
            result_image = self.redact_text_region(result_image, text_box)
            redacted_count += 1
        
        # Prepare processing stats
        stats = {
            'total_text_regions': len(text_detections),
            'redacted_regions': redacted_count,
            'redact_names_only': self.redact_names_only,
            'blur_type': self.blur_type,
            'blur_strength': self.blur_strength
        }
        
        logger.info(f"Processed {redacted_count} text regions in image")
        return result_image, stats
    
    def _filter_name_detections(self, image: np.ndarray, text_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter text detections to only include those containing names.
        
        Args:
            image: Input image as numpy array
            text_detections: List of text detection dictionaries
            
        Returns:
            Filtered list of detections containing names
        """
        name_detections = []
        
        for detection in text_detections:
            text = detection['text']
            
            # Check if text contains names
            if self.name_redactor.is_name(text):
                name_detections.append(detection)
        
        logger.info(f"Filtered {len(text_detections)} text regions to {len(name_detections)} name regions")
        return name_detections
    
    def process_video_frame(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single video frame to detect and redact text.
        
        Args:
            frame: Input video frame as numpy array
            confidence_threshold: Minimum confidence for text detection
            
        Returns:
            Tuple of (processed_frame, processing_stats)
        """
        return self.process_image(frame, confidence_threshold)
    
    def process_video(self, video_path: str, output_path: str, 
                     confidence_threshold: float = 0.5) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Process an entire video file to detect and redact text.
        
        Args:
            video_path: Path to input video file
            output_path: Path to output video file
            confidence_threshold: Minimum confidence for text detection
            
        Returns:
            Tuple of (total_frames_processed, list_of_stats_per_frame)
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
        all_stats = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, stats = self.process_video_frame(frame, confidence_threshold)
                all_stats.append(stats)
                
                # Write processed frame
                out.write(processed_frame)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")
        
        finally:
            cap.release()
            out.release()
        
        logger.info(f"Video processing complete. Processed {frame_count} frames")
        return frame_count, all_stats
    
    def get_processing_stats(self, stats_per_frame: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get processing statistics for video or batch processing.
        
        Args:
            stats_per_frame: List of stats dictionaries per frame
            
        Returns:
            Dictionary containing processing statistics
        """
        total_frames = len(stats_per_frame)
        total_text_regions = sum(stats['total_text_regions'] for stats in stats_per_frame)
        total_redacted = sum(stats['redacted_regions'] for stats in stats_per_frame)
        
        frames_with_text = sum(1 for stats in stats_per_frame if stats['total_text_regions'] > 0)
        text_detection_rate = frames_with_text / total_frames if total_frames > 0 else 0
        
        return {
            'total_frames': total_frames,
            'total_text_regions_detected': total_text_regions,
            'total_redacted_regions': total_redacted,
            'average_text_regions_per_frame': total_text_regions / total_frames if total_frames > 0 else 0,
            'frames_with_text': frames_with_text,
            'text_detection_rate': text_detection_rate,
            'redact_names_only': self.redact_names_only,
            'blur_type': self.blur_type,
            'blur_strength': self.blur_strength
        }
    
    def extract_text_from_image(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Extract all text from an image without redacting.
        
        Args:
            image: Input image as numpy array
            confidence_threshold: Minimum confidence for text detection
            
        Returns:
            List of text detection dictionaries
        """
        return self.text_detector.detect_text_regions(image, confidence_threshold)
    
    def redact_specific_text(self, image: np.ndarray, text_to_redact: List[str], 
                           confidence_threshold: float = 0.5) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Redact specific text strings from an image.
        
        Args:
            image: Input image as numpy array
            text_to_redact: List of text strings to redact
            confidence_threshold: Minimum confidence for text detection
            
        Returns:
            Tuple of (processed_image, list_of_redacted_text)
        """
        # Detect all text
        text_detections = self.text_detector.detect_text_regions(image, confidence_threshold)
        
        # Filter to only include specified text
        filtered_detections = []
        for detection in text_detections:
            if any(text.lower() in detection['text'].lower() for text in text_to_redact):
                filtered_detections.append(detection)
        
        # Apply redaction
        result_image = image.copy()
        redacted_text = []
        
        for detection in filtered_detections:
            text_box = detection['bbox']
            result_image = self.redact_text_region(result_image, text_box)
            redacted_text.append(detection)
        
        logger.info(f"Redacted {len(redacted_text)} specific text regions")
        return result_image, redacted_text
