"""
License plate detection using YOLOv8.

This module implements license plate detection using YOLOv8 object detection
model for detecting license plates in images and video frames.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLO not available. Install ultralytics for license plate detection.")


class PlateDetector:
    """
    License plate detector using YOLOv8.
    
    This class provides license plate detection functionality using YOLOv8
    object detection model. Falls back to OpenCV-based detection if YOLO is not available.
    """
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        """
        Initialize the license plate detector.
        
        Args:
            model_path: Path to YOLOv8 model file. If None, uses default model.
            confidence_threshold: Minimum confidence for detection
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.use_yolo = YOLO_AVAILABLE
        
        if self.use_yolo:
            self._load_yolo_model(model_path)
        else:
            self._setup_opencv_detector()
    
    def _load_yolo_model(self, model_path: Optional[str] = None) -> None:
        """Load YOLOv8 model for license plate detection."""
        try:
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                # Use pre-trained YOLOv8 model
                self.model = YOLO('yolov8n.pt')
                logger.info("Using pre-trained YOLOv8 model for license plate detection")
            
            logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            self.use_yolo = False
            self._setup_opencv_detector()
    
    def _setup_opencv_detector(self) -> None:
        """Setup OpenCV-based license plate detection as fallback."""
        logger.info("Using OpenCV-based license plate detection")
        # This would implement a basic OpenCV-based detector
        # For now, we'll use a simple contour-based approach
    
    def detect_plates(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect license plates in an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detection dictionaries with 'bbox', 'confidence', and 'class' keys
        """
        if self.use_yolo and self.model is not None:
            return self._detect_plates_yolo(image)
        else:
            return self._detect_plates_opencv(image)
    
    def _detect_plates_yolo(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect license plates using YOLOv8."""
        try:
            # Run YOLO inference
            results = self.model(image, conf=self.confidence_threshold)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Convert to (x, y, width, height) format
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        
                        detections.append({
                            'bbox': (x, y, w, h),
                            'confidence': float(confidence),
                            'class': class_id,
                            'class_name': 'license_plate'
                        })
            
            logger.info(f"YOLO detected {len(detections)} license plates")
            return detections
            
        except Exception as e:
            logger.error(f"Error in YOLO detection: {e}")
            return []
    
    def _detect_plates_opencv(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect license plates using OpenCV-based approach.
        
        This is a simplified implementation that looks for rectangular regions
        that might contain license plates.
        """
        detections = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if contour is roughly rectangular
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter by aspect ratio (license plates are typically wider than tall)
                    aspect_ratio = w / h
                    if 2.0 <= aspect_ratio <= 6.0 and w > 50 and h > 15:
                        # Calculate confidence based on contour area and aspect ratio
                        area = cv2.contourArea(contour)
                        confidence = min(area / (w * h), 1.0) * 0.8  # Simple confidence metric
                        
                        if confidence >= self.confidence_threshold:
                            detections.append({
                                'bbox': (x, y, w, h),
                                'confidence': confidence,
                                'class': 0,
                                'class_name': 'license_plate'
                            })
            
            logger.info(f"OpenCV detected {len(detections)} potential license plates")
            return detections
            
        except Exception as e:
            logger.error(f"Error in OpenCV detection: {e}")
            return []
    
    def detect_plates_in_roi(self, image: np.ndarray, roi: Tuple[int, int, int, int]) -> List[Dict[str, Any]]:
        """
        Detect license plates in a specific region of interest.
        
        Args:
            image: Input image as numpy array
            roi: Region of interest as (x, y, width, height)
            
        Returns:
            List of detection dictionaries with global coordinates
        """
        x, y, w, h = roi
        roi_image = image[y:y+h, x:x+w]
        
        # Detect plates in ROI
        roi_detections = self.detect_plates(roi_image)
        
        # Convert coordinates back to original image
        global_detections = []
        for detection in roi_detections:
            bbox = detection['bbox']
            global_bbox = (bbox[0] + x, bbox[1] + y, bbox[2], bbox[3])
            
            global_detection = detection.copy()
            global_detection['bbox'] = global_bbox
            global_detections.append(global_detection)
        
        return global_detections
    
    def filter_detections_by_confidence(self, detections: List[Dict[str, Any]], 
                                      min_confidence: float = None) -> List[Dict[str, Any]]:
        """
        Filter detections by confidence threshold.
        
        Args:
            detections: List of detection dictionaries
            min_confidence: Minimum confidence threshold. If None, uses instance threshold
            
        Returns:
            Filtered list of detections
        """
        if min_confidence is None:
            min_confidence = self.confidence_threshold
        
        filtered = [det for det in detections if det['confidence'] >= min_confidence]
        logger.info(f"Filtered {len(detections)} detections to {len(filtered)} by confidence")
        
        return filtered
    
    def get_detection_statistics(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Dictionary containing detection statistics
        """
        if not detections:
            return {
                'total_detections': 0,
                'average_confidence': 0.0,
                'max_confidence': 0.0,
                'min_confidence': 0.0
            }
        
        confidences = [det['confidence'] for det in detections]
        
        return {
            'total_detections': len(detections),
            'average_confidence': np.mean(confidences),
            'max_confidence': np.max(confidences),
            'min_confidence': np.min(confidences),
            'confidence_std': np.std(confidences)
        }
