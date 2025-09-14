"""
Unit tests for face blur module.

This module contains comprehensive tests for face detection and blurring functionality.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch
import tempfile
import os

from modules.face_blur import FaceDetector, FaceBlurrer


class TestFaceDetector:
    """Test cases for FaceDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = FaceDetector()
        # Create a test image with a simple pattern
        self.test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    def test_initialization(self):
        """Test detector initialization."""
        assert self.detector.face_cascade is not None
        assert self.detector.cascade_path is not None
    
    def test_detect_faces_empty_image(self):
        """Test face detection on empty image."""
        empty_image = np.zeros((50, 50, 3), dtype=np.uint8)
        faces = self.detector.detect_faces(empty_image)
        assert isinstance(faces, list)
        assert len(faces) == 0
    
    def test_detect_faces_no_faces(self):
        """Test face detection on image with no faces."""
        faces = self.detector.detect_faces(self.test_image)
        assert isinstance(faces, list)
        # Should return empty list for simple test image
    
    def test_detect_faces_parameters(self):
        """Test face detection with different parameters."""
        faces = self.detector.detect_faces(
            self.test_image,
            scale_factor=1.2,
            min_neighbors=3,
            min_size=(20, 20)
        )
        assert isinstance(faces, list)
    
    def test_detect_faces_in_roi(self):
        """Test face detection in region of interest."""
        roi = (10, 10, 50, 50)
        faces = self.detector.detect_faces_in_roi(self.test_image, roi)
        assert isinstance(faces, list)
    
    def test_get_face_confidence(self):
        """Test face confidence calculation."""
        face_box = (10, 10, 30, 30)
        confidence = self.detector.get_face_confidence(self.test_image, face_box)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
    
    def test_is_valid_face(self):
        """Test face validation."""
        face_box = (10, 10, 30, 30)
        is_valid = self.detector.is_valid_face(self.test_image, face_box)
        assert isinstance(is_valid, bool)
    
    def test_is_valid_face_with_threshold(self):
        """Test face validation with confidence threshold."""
        face_box = (10, 10, 30, 30)
        is_valid = self.detector.is_valid_face(self.test_image, face_box, min_confidence=0.5)
        assert isinstance(is_valid, bool)


class TestFaceBlurrer:
    """Test cases for FaceBlurrer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.blurrer = FaceBlurrer()
        self.test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    def test_initialization(self):
        """Test blurrer initialization."""
        assert self.blurrer.detector is not None
        assert self.blurrer.blur_type in ['gaussian', 'pixelate', 'blackout', 'mosaic']
        assert isinstance(self.blurrer.blur_strength, int)
    
    def test_initialization_invalid_blur_type(self):
        """Test initialization with invalid blur type."""
        with pytest.raises(ValueError):
            FaceBlurrer(blur_type='invalid')
    
    def test_blur_face_region_gaussian(self):
        """Test Gaussian blur face region."""
        face_box = (10, 10, 30, 30)
        result = self.blurrer.blur_face_region(self.test_image, face_box)
        assert result.shape == self.test_image.shape
        assert result.dtype == self.test_image.dtype
    
    def test_blur_face_region_pixelate(self):
        """Test pixelate blur face region."""
        self.blurrer.blur_type = 'pixelate'
        face_box = (10, 10, 30, 30)
        result = self.blurrer.blur_face_region(self.test_image, face_box)
        assert result.shape == self.test_image.shape
        assert result.dtype == self.test_image.dtype
    
    def test_blur_face_region_blackout(self):
        """Test blackout blur face region."""
        self.blurrer.blur_type = 'blackout'
        face_box = (10, 10, 30, 30)
        result = self.blurrer.blur_face_region(self.test_image, face_box)
        assert result.shape == self.test_image.shape
        assert result.dtype == self.test_image.dtype
    
    def test_blur_face_region_mosaic(self):
        """Test mosaic blur face region."""
        self.blurrer.blur_type = 'mosaic'
        face_box = (10, 10, 30, 30)
        result = self.blurrer.blur_face_region(self.test_image, face_box)
        assert result.shape == self.test_image.shape
        assert result.dtype == self.test_image.dtype
    
    def test_process_image(self):
        """Test image processing."""
        result_image, faces = self.blurrer.process_image(self.test_image)
        assert result_image.shape == self.test_image.shape
        assert isinstance(faces, list)
    
    def test_process_image_with_confidence(self):
        """Test image processing with confidence threshold."""
        result_image, faces = self.blurrer.process_image(self.test_image, confidence_threshold=0.5)
        assert result_image.shape == self.test_image.shape
        assert isinstance(faces, list)
    
    def test_process_video_frame(self):
        """Test video frame processing."""
        result_frame, faces = self.blurrer.process_video_frame(self.test_image)
        assert result_frame.shape == self.test_image.shape
        assert isinstance(faces, list)
    
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
                frame_count, faces_per_frame = self.blurrer.process_video(
                    temp_input.name, 'test_output.mp4'
                )
                assert isinstance(frame_count, int)
                assert isinstance(faces_per_frame, list)
            finally:
                os.unlink(temp_input.name)
    
    def test_get_processing_stats(self):
        """Test processing statistics calculation."""
        faces_per_frame = [[], [(10, 10, 20, 20)], [(15, 15, 25, 25), (30, 30, 40, 40)]]
        stats = self.blurrer.get_processing_stats(faces_per_frame)
        
        assert 'total_frames' in stats
        assert 'total_faces_detected' in stats
        assert 'average_faces_per_frame' in stats
        assert 'frames_with_faces' in stats
        assert 'face_detection_rate' in stats
        assert 'blur_type' in stats
        assert 'blur_strength' in stats
        
        assert stats['total_frames'] == 3
        assert stats['total_faces_detected'] == 3
        assert stats['frames_with_faces'] == 2
        assert stats['face_detection_rate'] == 2/3


class TestFaceBlurIntegration:
    """Integration tests for face blur module."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.blurrer = FaceBlurrer()
    
    def test_end_to_end_processing(self):
        """Test end-to-end image processing."""
        # Create a test image
        test_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        
        # Process the image
        result_image, faces = self.blurrer.process_image(test_image)
        
        # Verify results
        assert result_image.shape == test_image.shape
        assert isinstance(faces, list)
        assert result_image.dtype == test_image.dtype
    
    def test_different_blur_types(self):
        """Test different blur types on the same image."""
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        face_box = (20, 20, 40, 40)
        
        blur_types = ['gaussian', 'pixelate', 'blackout', 'mosaic']
        
        for blur_type in blur_types:
            self.blurrer.blur_type = blur_type
            result = self.blurrer.blur_face_region(test_image, face_box)
            
            assert result.shape == test_image.shape
            assert result.dtype == test_image.dtype
    
    def test_blur_strength_variations(self):
        """Test different blur strengths."""
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        face_box = (20, 20, 40, 40)
        
        blur_strengths = [5, 15, 25, 35]
        
        for strength in blur_strengths:
            self.blurrer.blur_strength = strength
            result = self.blurrer.blur_face_region(test_image, face_box)
            
            assert result.shape == test_image.shape
            assert result.dtype == test_image.dtype


if __name__ == '__main__':
    pytest.main([__file__])
