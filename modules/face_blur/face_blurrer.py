"""
Face blurring functionality for RedactAI.

This module provides various blurring techniques to redact faces
while maintaining visual quality of the processed media.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
import logging
from .face_detector import FaceDetector

logger = logging.getLogger(__name__)


class FaceBlurrer:
    """
    Face blurrer that detects and blurs faces in images and videos.
    
    This class combines face detection with various blurring techniques
    to provide privacy protection for detected faces.
    """
    
    def __init__(self, blur_type: str = 'gaussian', blur_strength: int = 15):
        """
        Initialize the face blurrer.
        
        Args:
            blur_type: Type of blur to apply ('gaussian', 'pixelate', 'blackout')
            blur_strength: Strength of the blur effect
        """
        self.detector = FaceDetector()
        self.blur_type = blur_type
        self.blur_strength = blur_strength
        
        # Validate blur type
        valid_types = ['gaussian', 'pixelate', 'blackout', 'mosaic']
        if blur_type not in valid_types:
            raise ValueError(f"Invalid blur type. Must be one of: {valid_types}")
        
        logger.info(f"FaceBlurrer initialized with {blur_type} blur, strength: {blur_strength}")
    
    def blur_face_region(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Apply blur to a specific face region.
        
        Args:
            image: Input image as numpy array
            face_box: Face bounding box as (x, y, width, height)
            
        Returns:
            Image with blurred face region
        """
        x, y, w, h = face_box
        result_image = image.copy()
        
        # Extract face region
        face_region = image[y:y+h, x:x+w]
        
        # Apply blur based on type
        if self.blur_type == 'gaussian':
            blurred_face = self._apply_gaussian_blur(face_region)
        elif self.blur_type == 'pixelate':
            blurred_face = self._apply_pixelate_blur(face_region)
        elif self.blur_type == 'blackout':
            blurred_face = self._apply_blackout_blur(face_region)
        elif self.blur_type == 'mosaic':
            blurred_face = self._apply_mosaic_blur(face_region)
        else:
            raise ValueError(f"Unknown blur type: {self.blur_type}")
        
        # Replace face region with blurred version
        result_image[y:y+h, x:x+w] = blurred_face
        
        return result_image
    
    def _apply_gaussian_blur(self, face_region: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to face region."""
        # Ensure kernel size is odd
        kernel_size = self.blur_strength if self.blur_strength % 2 == 1 else self.blur_strength + 1
        return cv2.GaussianBlur(face_region, (kernel_size, kernel_size), 0)
    
    def _apply_pixelate_blur(self, face_region: np.ndarray) -> np.ndarray:
        """Apply pixelate blur to face region."""
        h, w = face_region.shape[:2]
        
        # Calculate pixel size based on blur strength
        pixel_size = max(1, min(h, w) // self.blur_strength)
        
        # Resize down
        small = cv2.resize(face_region, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
        
        # Resize back up
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return pixelated
    
    def _apply_blackout_blur(self, face_region: np.ndarray) -> np.ndarray:
        """Apply blackout to face region."""
        blacked_out = np.zeros_like(face_region)
        return blacked_out
    
    def _apply_mosaic_blur(self, face_region: np.ndarray) -> np.ndarray:
        """Apply mosaic blur to face region."""
        h, w = face_region.shape[:2]
        
        # Calculate block size
        block_size = max(1, min(h, w) // self.blur_strength)
        
        # Create mosaic effect
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                # Get block region
                block = face_region[i:i+block_size, j:j+block_size]
                
                # Calculate average color
                avg_color = np.mean(block, axis=(0, 1))
                
                # Fill block with average color
                face_region[i:i+block_size, j:j+block_size] = avg_color
        
        return face_region
    
    def process_image(self, image: np.ndarray, confidence_threshold: float = 0.1) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Process an image to detect and blur all faces.
        
        Args:
            image: Input image as numpy array
            confidence_threshold: Minimum confidence for face detection
            
        Returns:
            Tuple of (processed_image, list_of_detected_faces)
        """
        # Detect faces
        faces = self.detector.detect_faces(image)
        
        # Filter faces by confidence
        valid_faces = []
        for face_box in faces:
            if self.detector.is_valid_face(image, face_box, confidence_threshold):
                valid_faces.append(face_box)
        
        # Apply blur to each detected face
        result_image = image.copy()
        for face_box in valid_faces:
            result_image = self.blur_face_region(result_image, face_box)
        
        logger.info(f"Processed {len(valid_faces)} faces in image")
        return result_image, valid_faces
    
    def process_video_frame(self, frame: np.ndarray, confidence_threshold: float = 0.1) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Process a single video frame to detect and blur faces.
        
        Args:
            frame: Input video frame as numpy array
            confidence_threshold: Minimum confidence for face detection
            
        Returns:
            Tuple of (processed_frame, list_of_detected_faces)
        """
        return self.process_image(frame, confidence_threshold)
    
    def process_video(self, video_path: str, output_path: str, 
                     confidence_threshold: float = 0.1) -> Tuple[int, List[List[Tuple[int, int, int, int]]]]:
        """
        Process an entire video file to detect and blur faces.
        
        Args:
            video_path: Path to input video file
            output_path: Path to output video file
            confidence_threshold: Minimum confidence for face detection
            
        Returns:
            Tuple of (total_frames_processed, list_of_faces_per_frame)
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
        all_faces = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, faces = self.process_video_frame(frame, confidence_threshold)
                all_faces.append(faces)
                
                # Write processed frame
                out.write(processed_frame)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")
        
        finally:
            cap.release()
            out.release()
        
        logger.info(f"Video processing complete. Processed {frame_count} frames")
        return frame_count, all_faces
    
    def get_processing_stats(self, faces_per_frame: List[List[Tuple[int, int, int, int]]]) -> dict:
        """
        Get processing statistics for video or batch processing.
        
        Args:
            faces_per_frame: List of faces detected per frame
            
        Returns:
            Dictionary containing processing statistics
        """
        total_frames = len(faces_per_frame)
        total_faces = sum(len(faces) for faces in faces_per_frame)
        avg_faces_per_frame = total_faces / total_frames if total_frames > 0 else 0
        
        frames_with_faces = sum(1 for faces in faces_per_frame if len(faces) > 0)
        face_detection_rate = frames_with_faces / total_frames if total_frames > 0 else 0
        
        return {
            'total_frames': total_frames,
            'total_faces_detected': total_faces,
            'average_faces_per_frame': avg_faces_per_frame,
            'frames_with_faces': frames_with_faces,
            'face_detection_rate': face_detection_rate,
            'blur_type': self.blur_type,
            'blur_strength': self.blur_strength
        }
