"""
Unit tests for plate blur module.

This module contains comprehensive tests for license plate detection and blurring functionality.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch
import tempfile
import os

from modules.plate_blur import PlateDetector, PlateBlurrer


class TestPlateDetector:
    """Test cases for PlateDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = PlateDetector()
        # Create a test image with a simple pattern
        self.test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    def test_initialization(self):
        """Test detector initialization."""
        assert self.detector.confidence_threshold is not None
        assert isinstance(self.detector.confidence_threshold, float)
        assert 0 <= self.detector.confidence_threshold <= 1
    
    def test_detect_plates_empty_image(self):
        """Test plate detection on empty image."""
        empty_image = np.zeros((50, 50, 3), dtype=np.uint8)
        plates = self.detector.detect_plates(empty_image)
        assert isinstance(plates, list)
    
    def test_detect_plates_no_plates(self):
        """Test plate detection on image with no plates."""
        plates = self.detector.detect_plates(self.test_image)
        assert isinstance(plates, list)
        # Should return empty list for simple test image
    
    def test_detect_plates_in_roi(self):
        """Test plate detection in region of interest."""
        roi = (10, 10, 50, 50)
        plates = self.detector.detect_plates_in_roi(self.test_image, roi)
        assert isinstance(plates, list)
    
    def test_filter_detections_by_confidence(self):
        """Test filtering detections by confidence."""
        detections = [
            {'bbox': (10, 10, 20, 20), 'confidence': 0.8, 'class': 0, 'class_name': 'license_plate'},
            {'bbox': (30, 30, 20, 20), 'confidence': 0.3, 'class': 0, 'class_name': 'license_plate'},
            {'bbox': (50, 50, 20, 20), 'confidence': 0.9, 'class': 0, 'class_name': 'license_plate'}
        ]
        
        filtered = self.detector.filter_detections_by_confidence(detections, min_confidence=0.5)
        assert len(filtered) == 2
        assert all(det['confidence'] >= 0.5 for det in filtered)
    
    def test_get_detection_statistics(self):
        """Test detection statistics calculation."""
        detections = [
            {'bbox': (10, 10, 20, 20), 'confidence': 0.8, 'class': 0, 'class_name': 'license_plate'},
            {'bbox': (30, 30, 20, 20), 'confidence': 0.6, 'class': 0, 'class_name': 'license_plate'}
        ]
        
        stats = self.detector.get_detection_statistics(detections)
        
        assert 'total_detections' in stats
        assert 'average_confidence' in stats
        assert 'max_confidence' in stats
        assert 'min_confidence' in stats
        
        assert stats['total_detections'] == 2
        assert stats['average_confidence'] == 0.7
        assert stats['max_confidence'] == 0.8
        assert stats['min_confidence'] == 0.6
    
    def test_get_detection_statistics_empty(self):
        """Test detection statistics with empty list."""
        stats = self.detector.get_detection_statistics([])
        
        assert stats['total_detections'] == 0
        assert stats['average_confidence'] == 0.0
        assert stats['max_confidence'] == 0.0
        assert stats['min_confidence'] == 0.0


class TestPlateBlurrer:
    """Test cases for PlateBlurrer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.blurrer = PlateBlurrer()
        self.test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    def test_initialization(self):
        """Test blurrer initialization."""
        assert self.blurrer.detector is not None
        assert self.blurrer.blur_type in ['gaussian', 'pixelate', 'blackout', 'mosaic']
        assert isinstance(self.blurrer.blur_strength, int)
    
    def test_initialization_invalid_blur_type(self):
        """Test initialization with invalid blur type."""
        with pytest.raises(ValueError):
            PlateBlurrer(blur_type='invalid')
    
    def test_blur_plate_region_gaussian(self):
        """Test Gaussian blur plate region."""
        plate_box = (10, 10, 30, 30)
        result = self.blurrer.blur_plate_region(self.test_image, plate_box)
        assert result.shape == self.test_image.shape
        assert result.dtype == self.test_image.dtype
    
    def test_blur_plate_region_pixelate(self):
        """Test pixelate blur plate region."""
        self.blurrer.blur_type = 'pixelate'
        plate_box = (10, 10, 30, 30)
        result = self.blurrer.blur_plate_region(self.test_image, plate_box)
        assert result.shape == self.test_image.shape
        assert result.dtype == self.test_image.dtype
    
    def test_blur_plate_region_blackout(self):
        """Test blackout blur plate region."""
        self.blurrer.blur_type = 'blackout'
        plate_box = (10, 10, 30, 30)
        result = self.blurrer.blur_plate_region(self.test_image, plate_box)
        assert result.shape == self.test_image.shape
        assert result.dtype == self.test_image.dtype
    
    def test_blur_plate_region_mosaic(self):
        """Test mosaic blur plate region."""
        self.blurrer.blur_type = 'mosaic'
        plate_box = (10, 10, 30, 30)
        result = self.blurrer.blur_plate_region(self.test_image, plate_box)
        assert result.shape == self.test_image.shape
        assert result.dtype == self.test_image.dtype
    
    def test_process_image(self):
        """Test image processing."""
        result_image, plates = self.blurrer.process_image(self.test_image)
        assert result_image.shape == self.test_image.shape
        assert isinstance(plates, list)
    
    def test_process_image_with_confidence(self):
        """Test image processing with confidence threshold."""
        result_image, plates = self.blurrer.process_image(self.test_image, min_confidence=0.5)
        assert result_image.shape == self.test_image.shape
        assert isinstance(plates, list)
    
    def test_process_video_frame(self):
        """Test video frame processing."""
        result_frame, plates = self.blurrer.process_video_frame(self.test_image)
        assert result_frame.shape == self.test_image.shape
        assert isinstance(plates, list)
    
    @patch('cv2.VideoCapture')
    @patch('cv2.VideoWriter')
    def test_process_video(self, mock_writer, mock_capture):
        """Test video processing."""
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: {cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480}[x]
        mock_cap.read.side_effect = [(True, self.test_image), (False, None)]
        mock_capture.return_value = mock_cap
        
        # Mock video writer
        mock_out = Mock()
        mock_writer.return_value = mock_out
        
        # Test video processing
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_input:
            temp_input.write(b'dummy video data')
            temp_input.flush()
            
            try:
                frame_count, plates_per_frame = self.blurrer.process_video(
                    temp_input.name, 'test_output.mp4'
                )
                assert isinstance(frame_count, int)
                assert isinstance(plates_per_frame, list)
            finally:
                os.unlink(temp_input.name)
    
    def test_get_processing_stats(self):
        """Test processing statistics calculation."""
        plates_per_frame = [
            [],
            [{'bbox': (10, 10, 20, 20), 'confidence': 0.8, 'class': 0, 'class_name': 'license_plate'}],
            [{'bbox': (15, 15, 25, 25), 'confidence': 0.7, 'class': 0, 'class_name': 'license_plate'}, 
             {'bbox': (30, 30, 40, 40), 'confidence': 0.9, 'class': 0, 'class_name': 'license_plate'}]
        ]
        stats = self.blurrer.get_processing_stats(plates_per_frame)
        
        assert 'total_frames' in stats
        assert 'total_plates_detected' in stats
        assert 'average_plates_per_frame' in stats
        assert 'frames_with_plates' in stats
        assert 'plate_detection_rate' in stats
        assert 'blur_type' in stats
        assert 'blur_strength' in stats
        
        assert stats['total_frames'] == 3
        assert stats['total_plates_detected'] == 3
        assert stats['frames_with_plates'] == 2
        assert stats['plate_detection_rate'] == 2/3


class TestPlateBlurIntegration:
    """Integration tests for plate blur module."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.blurrer = PlateBlurrer()
    
    def test_end_to_end_processing(self):
        """Test end-to-end image processing."""
        # Create a test image
        test_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        
        # Process the image
        result_image, plates = self.blurrer.process_image(test_image)
        
        # Verify results
        assert result_image.shape == test_image.shape
        assert isinstance(plates, list)
        assert result_image.dtype == test_image.dtype
    
    def test_different_blur_types(self):
        """Test different blur types on the same image."""
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        plate_box = (20, 20, 40, 40)
        
        blur_types = ['gaussian', 'pixelate', 'blackout', 'mosaic']
        
        for blur_type in blur_types:
            self.blurrer.blur_type = blur_type
            result = self.blurrer.blur_plate_region(test_image, plate_box)
            
            assert result.shape == test_image.shape
            assert result.dtype == test_image.dtype
    
    def test_blur_strength_variations(self):
        """Test different blur strengths."""
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        plate_box = (20, 20, 40, 40)
        
        blur_strengths = [5, 15, 25, 35]
        
        for strength in blur_strengths:
            self.blurrer.blur_strength = strength
            result = self.blurrer.blur_plate_region(test_image, plate_box)
            
            assert result.shape == test_image.shape
            assert result.dtype == test_image.dtype


if __name__ == '__main__':
    pytest.main([__file__])
