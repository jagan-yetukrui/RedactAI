"""
Text detection using OCR (Tesseract and EasyOCR).

This module implements text detection using multiple OCR engines
for detecting text regions in images and video frames.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Try to import OCR libraries
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("Tesseract not available. Install pytesseract for OCR functionality.")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available. Install easyocr for OCR functionality.")


class TextDetector:
    """
    Text detector using multiple OCR engines.
    
    This class provides text detection functionality using Tesseract and EasyOCR
    for detecting text regions in images and video frames.
    """
    
    def __init__(self, ocr_engine: str = 'tesseract', languages: List[str] = None):
        """
        Initialize the text detector.
        
        Args:
            ocr_engine: OCR engine to use ('tesseract', 'easyocr', or 'both')
            languages: List of languages for OCR. Defaults to ['en']
        """
        self.ocr_engine = ocr_engine.lower()
        self.languages = languages or ['en']
        
        # Initialize OCR engines
        self.tesseract_available = TESSERACT_AVAILABLE
        self.easyocr_available = EASYOCR_AVAILABLE
        self.easyocr_reader = None
        
        if self.ocr_engine in ['easyocr', 'both'] and self.easyocr_available:
            self._setup_easyocr()
        
        if self.ocr_engine in ['tesseract', 'both'] and not self.tesseract_available:
            logger.warning("Tesseract not available, falling back to EasyOCR")
            self.ocr_engine = 'easyocr'
        
        if not self.tesseract_available and not self.easyocr_available:
            raise RuntimeError("No OCR engines available. Install pytesseract or easyocr.")
        
        logger.info(f"TextDetector initialized with {self.ocr_engine} engine")
    
    def _setup_easyocr(self) -> None:
        """Setup EasyOCR reader."""
        try:
            self.easyocr_reader = easyocr.Reader(self.languages)
            logger.info("EasyOCR reader initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing EasyOCR: {e}")
            self.easyocr_available = False
    
    def detect_text_regions(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect text regions in an image.
        
        Args:
            image: Input image as numpy array
            confidence_threshold: Minimum confidence for text detection
            
        Returns:
            List of text detection dictionaries with 'bbox', 'text', and 'confidence' keys
        """
        if self.ocr_engine == 'tesseract':
            return self._detect_text_tesseract(image, confidence_threshold)
        elif self.ocr_engine == 'easyocr':
            return self._detect_text_easyocr(image, confidence_threshold)
        elif self.ocr_engine == 'both':
            # Use both engines and combine results
            tesseract_results = self._detect_text_tesseract(image, confidence_threshold) if self.tesseract_available else []
            easyocr_results = self._detect_text_easyocr(image, confidence_threshold) if self.easyocr_available else []
            return self._combine_ocr_results(tesseract_results, easyocr_results)
        else:
            raise ValueError(f"Unknown OCR engine: {self.ocr_engine}")
    
    def _detect_text_tesseract(self, image: np.ndarray, confidence_threshold: float) -> List[Dict[str, Any]]:
        """Detect text using Tesseract OCR."""
        if not self.tesseract_available:
            return []
        
        try:
            # Convert to RGB for Tesseract
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get detailed data from Tesseract
            data = pytesseract.image_to_data(rgb_image, output_type=pytesseract.Output.DICT)
            
            detections = []
            n_boxes = len(data['level'])
            
            for i in range(n_boxes):
                confidence = float(data['conf'][i])
                if confidence > confidence_threshold * 100:  # Tesseract returns confidence as percentage
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    text = data['text'][i].strip()
                    
                    if text:  # Only include non-empty text
                        detections.append({
                            'bbox': (x, y, w, h),
                            'text': text,
                            'confidence': confidence / 100.0,  # Convert to 0-1 range
                            'engine': 'tesseract'
                        })
            
            logger.info(f"Tesseract detected {len(detections)} text regions")
            return detections
            
        except Exception as e:
            logger.error(f"Error in Tesseract detection: {e}")
            return []
    
    def _detect_text_easyocr(self, image: np.ndarray, confidence_threshold: float) -> List[Dict[str, Any]]:
        """Detect text using EasyOCR."""
        if not self.easyocr_available or self.easyocr_reader is None:
            return []
        
        try:
            # EasyOCR expects RGB images
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run EasyOCR detection
            results = self.easyocr_reader.readtext(rgb_image)
            
            detections = []
            for (bbox, text, confidence) in results:
                if confidence >= confidence_threshold:
                    # Convert bbox format from EasyOCR to (x, y, w, h)
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x, y = int(min(x_coords)), int(min(y_coords))
                    w, h = int(max(x_coords) - x), int(max(y_coords) - y)
                    
                    detections.append({
                        'bbox': (x, y, w, h),
                        'text': text,
                        'confidence': confidence,
                        'engine': 'easyocr'
                    })
            
            logger.info(f"EasyOCR detected {len(detections)} text regions")
            return detections
            
        except Exception as e:
            logger.error(f"Error in EasyOCR detection: {e}")
            return []
    
    def _combine_ocr_results(self, tesseract_results: List[Dict[str, Any]], 
                           easyocr_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Combine results from multiple OCR engines.
        
        Args:
            tesseract_results: Results from Tesseract
            easyocr_results: Results from EasyOCR
            
        Returns:
            Combined list of detections
        """
        # Simple combination: use all results from both engines
        # In a more sophisticated implementation, we could deduplicate overlapping regions
        combined = tesseract_results + easyocr_results
        
        # Sort by confidence
        combined.sort(key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"Combined OCR results: {len(combined)} total detections")
        return combined
    
    def detect_text_in_roi(self, image: np.ndarray, roi: Tuple[int, int, int, int],
                          confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect text in a specific region of interest.
        
        Args:
            image: Input image as numpy array
            roi: Region of interest as (x, y, width, height)
            confidence_threshold: Minimum confidence for text detection
            
        Returns:
            List of text detection dictionaries with global coordinates
        """
        x, y, w, h = roi
        roi_image = image[y:y+h, x:x+w]
        
        # Detect text in ROI
        roi_detections = self.detect_text_regions(roi_image, confidence_threshold)
        
        # Convert coordinates back to original image
        global_detections = []
        for detection in roi_detections:
            bbox = detection['bbox']
            global_bbox = (bbox[0] + x, bbox[1] + y, bbox[2], bbox[3])
            
            global_detection = detection.copy()
            global_detection['bbox'] = global_bbox
            global_detections.append(global_detection)
        
        return global_detections
    
    def preprocess_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image to improve OCR accuracy.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def get_text_statistics(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about text detections.
        
        Args:
            detections: List of text detection dictionaries
            
        Returns:
            Dictionary containing text detection statistics
        """
        if not detections:
            return {
                'total_detections': 0,
                'average_confidence': 0.0,
                'total_characters': 0,
                'average_text_length': 0.0
            }
        
        confidences = [det['confidence'] for det in detections]
        text_lengths = [len(det['text']) for det in detections]
        total_chars = sum(text_lengths)
        
        return {
            'total_detections': len(detections),
            'average_confidence': np.mean(confidences),
            'max_confidence': np.max(confidences),
            'min_confidence': np.min(confidences),
            'total_characters': total_chars,
            'average_text_length': np.mean(text_lengths),
            'max_text_length': np.max(text_lengths),
            'min_text_length': np.min(text_lengths)
        }
