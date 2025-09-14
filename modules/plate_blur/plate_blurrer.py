"""
License plate blurring functionality for RedactAI.

This module provides various blurring techniques to redact license plates
while maintaining visual quality of the processed media.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from .plate_detector import PlateDetector

logger = logging.getLogger(__name__)


class PlateBlurrer:
    """
    License plate blurrer that detects and blurs license plates in images and videos.
    
    This class combines license plate detection with various blurring techniques
    to provide privacy protection for detected license plates.
    """
    
    def __init__(self, blur_type: str = 'gaussian', blur_strength: int = 15, 
                 confidence_threshold: float = 0.5):
        """
        Initialize the license plate blurrer.
        
        Args:
            blur_type: Type of blur to apply ('gaussian', 'pixelate', 'blackout', 'mosaic')
            blur_strength: Strength of the blur effect
            confidence_threshold: Minimum confidence for plate detection
        """
        self.detector = PlateDetector(confidence_threshold=confidence_threshold)
        self.blur_type = blur_type
        self.blur_strength = blur_strength
        
        # Validate blur type
        valid_types = ['gaussian', 'pixelate', 'blackout', 'mosaic']
        if blur_type not in valid_types:
            raise ValueError(f"Invalid blur type. Must be one of: {valid_types}")
        
        logger.info(f"PlateBlurrer initialized with {blur_type} blur, strength: {blur_strength}")
    
    def blur_plate_region(self, image: np.ndarray, plate_box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Apply blur to a specific license plate region.
        
        Args:
            image: Input image as numpy array
            plate_box: License plate bounding box as (x, y, width, height)
            
        Returns:
            Image with blurred license plate region
        """
        x, y, w, h = plate_box
        result_image = image.copy()
        
        # Extract plate region
        plate_region = image[y:y+h, x:x+w]
        
        # Apply blur based on type
        if self.blur_type == 'gaussian':
            blurred_plate = self._apply_gaussian_blur(plate_region)
        elif self.blur_type == 'pixelate':
            blurred_plate = self._apply_pixelate_blur(plate_region)
        elif self.blur_type == 'blackout':
            blurred_plate = self._apply_blackout_blur(plate_region)
        elif self.blur_type == 'mosaic':
            blurred_plate = self._apply_mosaic_blur(plate_region)
        else:
            raise ValueError(f"Unknown blur type: {self.blur_type}")
        
        # Replace plate region with blurred version
        result_image[y:y+h, x:x+w] = blurred_plate
        
        return result_image
    
    def _apply_gaussian_blur(self, plate_region: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to license plate region."""
        # Ensure kernel size is odd
        kernel_size = self.blur_strength if self.blur_strength % 2 == 1 else self.blur_strength + 1
        return cv2.GaussianBlur(plate_region, (kernel_size, kernel_size), 0)
    
    def _apply_pixelate_blur(self, plate_region: np.ndarray) -> np.ndarray:
        """Apply pixelate blur to license plate region."""
        h, w = plate_region.shape[:2]
        
        # Calculate pixel size based on blur strength
        pixel_size = max(1, min(h, w) // self.blur_strength)
        
        # Resize down
        small = cv2.resize(plate_region, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
        
        # Resize back up
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return pixelated
    
    def _apply_blackout_blur(self, plate_region: np.ndarray) -> np.ndarray:
        """Apply blackout to license plate region."""
        blacked_out = np.zeros_like(plate_region)
        return blacked_out
    
    def _apply_mosaic_blur(self, plate_region: np.ndarray) -> np.ndarray:
        """Apply mosaic blur to license plate region."""
        h, w = plate_region.shape[:2]
        
        # Calculate block size
        block_size = max(1, min(h, w) // self.blur_strength)
        
        # Create mosaic effect
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                # Get block region
                block = plate_region[i:i+block_size, j:j+block_size]
                
                # Calculate average color
                avg_color = np.mean(block, axis=(0, 1))
                
                # Fill block with average color
                plate_region[i:i+block_size, j:j+block_size] = avg_color
        
        return plate_region
    
    def process_image(self, image: np.ndarray, min_confidence: float = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process an image to detect and blur all license plates.
        
        Args:
            image: Input image as numpy array
            min_confidence: Minimum confidence for plate detection
            
        Returns:
            Tuple of (processed_image, list_of_detected_plates)
        """
        # Detect license plates
        detections = self.detector.detect_plates(image)
        
        # Filter by confidence if specified
        if min_confidence is not None:
            detections = self.detector.filter_detections_by_confidence(detections, min_confidence)
        
        # Apply blur to each detected plate
        result_image = image.copy()
        for detection in detections:
            plate_box = detection['bbox']
            result_image = self.blur_plate_region(result_image, plate_box)
        
        logger.info(f"Processed {len(detections)} license plates in image")
        return result_image, detections
    
    def process_video_frame(self, frame: np.ndarray, min_confidence: float = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process a single video frame to detect and blur license plates.
        
        Args:
            frame: Input video frame as numpy array
            min_confidence: Minimum confidence for plate detection
            
        Returns:
            Tuple of (processed_frame, list_of_detected_plates)
        """
        return self.process_image(frame, min_confidence)
    
    def process_video(self, video_path: str, output_path: str, 
                     min_confidence: float = None) -> Tuple[int, List[List[Dict[str, Any]]]]:
        """
        Process an entire video file to detect and blur license plates.
        
        Args:
            video_path: Path to input video file
            output_path: Path to output video file
            min_confidence: Minimum confidence for plate detection
            
        Returns:
            Tuple of (total_frames_processed, list_of_plates_per_frame)
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
        all_plates = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, plates = self.process_video_frame(frame, min_confidence)
                all_plates.append(plates)
                
                # Write processed frame
                out.write(processed_frame)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")
        
        finally:
            cap.release()
            out.release()
        
        logger.info(f"Video processing complete. Processed {frame_count} frames")
        return frame_count, all_plates
    
    def get_processing_stats(self, plates_per_frame: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Get processing statistics for video or batch processing.
        
        Args:
            plates_per_frame: List of plates detected per frame
            
        Returns:
            Dictionary containing processing statistics
        """
        total_frames = len(plates_per_frame)
        total_plates = sum(len(plates) for plates in plates_per_frame)
        avg_plates_per_frame = total_plates / total_frames if total_frames > 0 else 0
        
        frames_with_plates = sum(1 for plates in plates_per_frame if len(plates) > 0)
        plate_detection_rate = frames_with_plates / total_frames if total_frames > 0 else 0
        
        # Calculate confidence statistics
        all_confidences = []
        for plates in plates_per_frame:
            all_confidences.extend([plate['confidence'] for plate in plates])
        
        confidence_stats = {}
        if all_confidences:
            confidence_stats = {
                'average_confidence': np.mean(all_confidences),
                'max_confidence': np.max(all_confidences),
                'min_confidence': np.min(all_confidences),
                'confidence_std': np.std(all_confidences)
            }
        
        return {
            'total_frames': total_frames,
            'total_plates_detected': total_plates,
            'average_plates_per_frame': avg_plates_per_frame,
            'frames_with_plates': frames_with_plates,
            'plate_detection_rate': plate_detection_rate,
            'blur_type': self.blur_type,
            'blur_strength': self.blur_strength,
            **confidence_stats
        }
