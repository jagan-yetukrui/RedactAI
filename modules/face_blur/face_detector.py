"""
Face detection using Haar Cascades.

This module implements face detection using OpenCV's Haar Cascade classifier
for detecting faces in images and video frames.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Face detector using Haar Cascades.
    
    This class provides face detection functionality using OpenCV's
    pre-trained Haar Cascade classifier for frontal face detection.
    """
    
    def __init__(self, cascade_path: Optional[str] = None):
        """
        Initialize the face detector.
        
        Args:
            cascade_path: Path to Haar cascade XML file. If None, uses default.
        """
        self.cascade_path = cascade_path or cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = None
        self._load_cascade()
    
    def _load_cascade(self) -> None:
        """Load the Haar cascade classifier."""
        try:
            self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
            if self.face_cascade.empty():
                raise ValueError(f"Failed to load cascade from {self.cascade_path}")
            logger.info("Face cascade loaded successfully")
        except Exception as e:
            logger.error(f"Error loading face cascade: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray, scale_factor: float = 1.1, 
                    min_neighbors: int = 5, min_size: Tuple[int, int] = (30, 30)) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array
            scale_factor: Parameter specifying how much the image size is reduced at each image scale
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it
            min_size: Minimum possible object size. Objects smaller than this are ignored
            
        Returns:
            List of bounding boxes as (x, y, width, height) tuples
        """
        if self.face_cascade is None:
            raise RuntimeError("Face cascade not loaded")
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert to list of tuples
        face_boxes = [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
        
        logger.info(f"Detected {len(face_boxes)} faces")
        return face_boxes
    
    def detect_faces_in_roi(self, image: np.ndarray, roi: Tuple[int, int, int, int],
                           scale_factor: float = 1.1, min_neighbors: int = 5,
                           min_size: Tuple[int, int] = (30, 30)) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a specific region of interest.
        
        Args:
            image: Input image as numpy array
            roi: Region of interest as (x, y, width, height)
            scale_factor: Parameter specifying how much the image size is reduced at each image scale
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it
            min_size: Minimum possible object size. Objects smaller than this are ignored
            
        Returns:
            List of bounding boxes as (x, y, width, height) tuples in original image coordinates
        """
        x, y, w, h = roi
        roi_image = image[y:y+h, x:x+w]
        
        # Detect faces in ROI
        roi_faces = self.detect_faces(roi_image, scale_factor, min_neighbors, min_size)
        
        # Convert coordinates back to original image
        global_faces = [(fx + x, fy + y, fw, fh) for fx, fy, fw, fh in roi_faces]
        
        return global_faces
    
    def get_face_confidence(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> float:
        """
        Get confidence score for a detected face.
        
        Args:
            image: Input image as numpy array
            face_box: Face bounding box as (x, y, width, height)
            
        Returns:
            Confidence score between 0 and 1
        """
        x, y, w, h = face_box
        face_region = image[y:y+h, x:x+w]
        
        # Convert to grayscale
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate variance as a simple confidence measure
        variance = np.var(gray_face)
        
        # Normalize to 0-1 range (heuristic)
        confidence = min(variance / 1000.0, 1.0)
        
        return confidence
    
    def is_valid_face(self, image: np.ndarray, face_box: Tuple[int, int, int, int],
                     min_confidence: float = 0.1) -> bool:
        """
        Check if a detected face is valid based on confidence threshold.
        
        Args:
            image: Input image as numpy array
            face_box: Face bounding box as (x, y, width, height)
            min_confidence: Minimum confidence threshold
            
        Returns:
            True if face is valid, False otherwise
        """
        confidence = self.get_face_confidence(image, face_box)
        return confidence >= min_confidence
