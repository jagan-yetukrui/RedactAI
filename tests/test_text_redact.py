"""
Unit tests for text redact module.

This module contains comprehensive tests for text detection and name redaction functionality.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch
import tempfile
import os

from modules.text_redact import TextDetector, NameRedactor, TextRedactor


class TestTextDetector:
    """Test cases for TextDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = TextDetector()
        # Create a test image with a simple pattern
        self.test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    def test_initialization(self):
        """Test detector initialization."""
        assert self.detector.ocr_engine in ['tesseract', 'easyocr', 'both']
        assert isinstance(self.detector.languages, list)
        assert 'en' in self.detector.languages
    
    def test_detect_text_regions_empty_image(self):
        """Test text detection on empty image."""
        empty_image = np.zeros((50, 50, 3), dtype=np.uint8)
        text_regions = self.detector.detect_text_regions(empty_image)
        assert isinstance(text_regions, list)
    
    def test_detect_text_regions_no_text(self):
        """Test text detection on image with no text."""
        text_regions = self.detector.detect_text_regions(self.test_image)
        assert isinstance(text_regions, list)
        # Should return empty list for simple test image
    
    def test_detect_text_regions_with_confidence(self):
        """Test text detection with confidence threshold."""
        text_regions = self.detector.detect_text_regions(self.test_image, confidence_threshold=0.5)
        assert isinstance(text_regions, list)
    
    def test_detect_text_in_roi(self):
        """Test text detection in region of interest."""
        roi = (10, 10, 50, 50)
        text_regions = self.detector.detect_text_in_roi(self.test_image, roi)
        assert isinstance(text_regions, list)
    
    def test_preprocess_image_for_ocr(self):
        """Test image preprocessing for OCR."""
        preprocessed = self.detector.preprocess_image_for_ocr(self.test_image)
        assert preprocessed.shape[:2] == self.test_image.shape[:2]  # Same height and width
        assert len(preprocessed.shape) == 2  # Grayscale
    
    def test_get_text_statistics(self):
        """Test text statistics calculation."""
        detections = [
            {'bbox': (10, 10, 20, 20), 'text': 'Hello', 'confidence': 0.8, 'engine': 'tesseract'},
            {'bbox': (30, 30, 20, 20), 'text': 'World', 'confidence': 0.6, 'engine': 'tesseract'}
        ]
        
        stats = self.detector.get_text_statistics(detections)
        
        assert 'total_detections' in stats
        assert 'average_confidence' in stats
        assert 'total_characters' in stats
        assert 'average_text_length' in stats
        
        assert stats['total_detections'] == 2
        assert stats['total_characters'] == 10  # 'Hello' + 'World'
        assert stats['average_text_length'] == 5.0
    
    def test_get_text_statistics_empty(self):
        """Test text statistics with empty list."""
        stats = self.detector.get_text_statistics([])
        
        assert stats['total_detections'] == 0
        assert stats['average_confidence'] == 0.0
        assert stats['total_characters'] == 0
        assert stats['average_text_length'] == 0.0


class TestNameRedactor:
    """Test cases for NameRedactor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.redactor = NameRedactor()
    
    def test_initialization(self):
        """Test redactor initialization."""
        assert self.redactor.model_name is not None
        assert isinstance(self.redactor.custom_patterns, list)
        assert isinstance(self.redactor.compiled_patterns, list)
    
    def test_detect_names_empty_text(self):
        """Test name detection on empty text."""
        names = self.redactor.detect_names("")
        assert isinstance(names, list)
        assert len(names) == 0
    
    def test_detect_names_no_names(self):
        """Test name detection on text with no names."""
        text = "This is a simple sentence without any names."
        names = self.redactor.detect_names(text)
        assert isinstance(names, list)
        # Should return empty list for text without names
    
    def test_detect_names_with_names(self):
        """Test name detection on text with names."""
        text = "John Smith went to the store with Mary Johnson."
        names = self.redactor.detect_names(text)
        assert isinstance(names, list)
        # May or may not detect names depending on SpaCy availability
    
    def test_redact_names_empty_text(self):
        """Test name redaction on empty text."""
        redacted_text, redacted_names = self.redactor.redact_names("")
        assert redacted_text == ""
        assert isinstance(redacted_names, list)
        assert len(redacted_names) == 0
    
    def test_redact_names_no_names(self):
        """Test name redaction on text with no names."""
        text = "This is a simple sentence without any names."
        redacted_text, redacted_names = self.redactor.redact_names(text)
        assert redacted_text == text
        assert isinstance(redacted_names, list)
        assert len(redacted_names) == 0
    
    def test_redact_names_with_custom_char(self):
        """Test name redaction with custom redaction character."""
        text = "John Smith went to the store."
        redacted_text, redacted_names = self.redactor.redact_names(text, redaction_char='#')
        assert isinstance(redacted_text, str)
        assert isinstance(redacted_names, list)
    
    def test_redact_names_with_custom_length(self):
        """Test name redaction with custom redaction length."""
        text = "John Smith went to the store."
        redacted_text, redacted_names = self.redactor.redact_names(text, redaction_length=5)
        assert isinstance(redacted_text, str)
        assert isinstance(redacted_names, list)
    
    def test_get_name_statistics(self):
        """Test name statistics calculation."""
        detections = [
            {'text': 'John Smith', 'start': 0, 'end': 10, 'label': 'PERSON', 'confidence': 0.9, 'method': 'spacy'},
            {'text': 'Mary Johnson', 'start': 20, 'end': 32, 'label': 'PERSON', 'confidence': 0.8, 'method': 'spacy'}
        ]
        
        stats = self.redactor.get_name_statistics(detections)
        
        assert 'total_names' in stats
        assert 'unique_names' in stats
        assert 'average_confidence' in stats
        assert 'label_distribution' in stats
        
        assert stats['total_names'] == 2
        assert stats['unique_names'] == 2
        assert stats['average_confidence'] == 0.85
    
    def test_get_name_statistics_empty(self):
        """Test name statistics with empty list."""
        stats = self.redactor.get_name_statistics([])
        
        assert stats['total_names'] == 0
        assert stats['unique_names'] == 0
        assert stats['average_confidence'] == 0.0
        assert stats['label_distribution'] == {}
    
    def test_add_custom_pattern(self):
        """Test adding custom pattern."""
        pattern = r'\\b[A-Z][a-z]+ [A-Z][a-z]+\\b'
        self.redactor.add_custom_pattern(pattern)
        assert pattern in self.redactor.custom_patterns
        assert len(self.redactor.compiled_patterns) > 0
    
    def test_add_custom_pattern_invalid(self):
        """Test adding invalid custom pattern."""
        with patch('logging.Logger.warning') as mock_warning:
            self.redactor.add_custom_pattern('invalid[pattern')
            mock_warning.assert_called_once()
    
    def test_is_name(self):
        """Test name checking."""
        # Test with text that might be a name
        is_name = self.redactor.is_name("John Smith")
        assert isinstance(is_name, bool)
        
        # Test with text that is not a name
        is_name = self.redactor.is_name("the store")
        assert isinstance(is_name, bool)


class TestTextRedactor:
    """Test cases for TextRedactor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.redactor = TextRedactor()
        self.test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    def test_initialization(self):
        """Test redactor initialization."""
        assert self.redactor.text_detector is not None
        assert self.redactor.name_redactor is not None
        assert self.redactor.blur_type in ['gaussian', 'pixelate', 'blackout', 'mosaic']
        assert isinstance(self.redactor.blur_strength, int)
        assert isinstance(self.redactor.redact_names_only, bool)
    
    def test_initialization_invalid_blur_type(self):
        """Test initialization with invalid blur type."""
        with pytest.raises(ValueError):
            TextRedactor(blur_type='invalid')
    
    def test_redact_text_region_gaussian(self):
        """Test Gaussian blur text region."""
        text_box = (10, 10, 30, 30)
        result = self.redactor.redact_text_region(self.test_image, text_box)
        assert result.shape == self.test_image.shape
        assert result.dtype == self.test_image.dtype
    
    def test_redact_text_region_pixelate(self):
        """Test pixelate blur text region."""
        self.redactor.blur_type = 'pixelate'
        text_box = (10, 10, 30, 30)
        result = self.redactor.redact_text_region(self.test_image, text_box)
        assert result.shape == self.test_image.shape
        assert result.dtype == self.test_image.dtype
    
    def test_redact_text_region_blackout(self):
        """Test blackout blur text region."""
        self.redactor.blur_type = 'blackout'
        text_box = (10, 10, 30, 30)
        result = self.redactor.redact_text_region(self.test_image, text_box)
        assert result.shape == self.test_image.shape
        assert result.dtype == self.test_image.dtype
    
    def test_redact_text_region_mosaic(self):
        """Test mosaic blur text region."""
        self.redactor.blur_type = 'mosaic'
        text_box = (10, 10, 30, 30)
        result = self.redactor.redact_text_region(self.test_image, text_box)
        assert result.shape == self.test_image.shape
        assert result.dtype == self.test_image.dtype
    
    def test_process_image(self):
        """Test image processing."""
        result_image, stats = self.redactor.process_image(self.test_image)
        assert result_image.shape == self.test_image.shape
        assert isinstance(stats, dict)
        assert 'total_text_regions' in stats
        assert 'redacted_regions' in stats
    
    def test_process_image_with_confidence(self):
        """Test image processing with confidence threshold."""
        result_image, stats = self.redactor.process_image(self.test_image, confidence_threshold=0.5)
        assert result_image.shape == self.test_image.shape
        assert isinstance(stats, dict)
    
    def test_process_video_frame(self):
        """Test video frame processing."""
        result_frame, stats = self.redactor.process_video_frame(self.test_image)
        assert result_frame.shape == self.test_image.shape
        assert isinstance(stats, dict)
    
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
                frame_count, stats_per_frame = self.redactor.process_video(
                    temp_input.name, 'test_output.mp4'
                )
                assert isinstance(frame_count, int)
                assert isinstance(stats_per_frame, list)
            finally:
                os.unlink(temp_input.name)
    
    def test_get_processing_stats(self):
        """Test processing statistics calculation."""
        stats_per_frame = [
            {'total_text_regions': 0, 'redacted_regions': 0},
            {'total_text_regions': 2, 'redacted_regions': 1},
            {'total_text_regions': 3, 'redacted_regions': 2}
        ]
        stats = self.redactor.get_processing_stats(stats_per_frame)
        
        assert 'total_frames' in stats
        assert 'total_text_regions_detected' in stats
        assert 'total_redacted_regions' in stats
        assert 'average_text_regions_per_frame' in stats
        assert 'frames_with_text' in stats
        assert 'text_detection_rate' in stats
        
        assert stats['total_frames'] == 3
        assert stats['total_text_regions_detected'] == 5
        assert stats['total_redacted_regions'] == 3
        assert stats['frames_with_text'] == 2
        assert stats['text_detection_rate'] == 2/3
    
    def test_extract_text_from_image(self):
        """Test text extraction from image."""
        text_regions = self.redactor.extract_text_from_image(self.test_image)
        assert isinstance(text_regions, list)
    
    def test_redact_specific_text(self):
        """Test redacting specific text strings."""
        text_to_redact = ['test', 'sample']
        result_image, redacted_text = self.redactor.redact_specific_text(
            self.test_image, text_to_redact
        )
        assert result_image.shape == self.test_image.shape
        assert isinstance(redacted_text, list)


class TestTextRedactIntegration:
    """Integration tests for text redact module."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.redactor = TextRedactor()
    
    def test_end_to_end_processing(self):
        """Test end-to-end image processing."""
        # Create a test image
        test_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        
        # Process the image
        result_image, stats = self.redactor.process_image(test_image)
        
        # Verify results
        assert result_image.shape == test_image.shape
        assert isinstance(stats, dict)
        assert result_image.dtype == test_image.dtype
    
    def test_different_blur_types(self):
        """Test different blur types on the same image."""
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        text_box = (20, 20, 40, 40)
        
        blur_types = ['gaussian', 'pixelate', 'blackout', 'mosaic']
        
        for blur_type in blur_types:
            self.redactor.blur_type = blur_type
            result = self.redactor.redact_text_region(test_image, text_box)
            
            assert result.shape == test_image.shape
            assert result.dtype == test_image.dtype
    
    def test_blur_strength_variations(self):
        """Test different blur strengths."""
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        text_box = (20, 20, 40, 40)
        
        blur_strengths = [5, 15, 25, 35]
        
        for strength in blur_strengths:
            self.redactor.blur_strength = strength
            result = self.redactor.redact_text_region(test_image, text_box)
            
            assert result.shape == test_image.shape
            assert result.dtype == test_image.dtype


if __name__ == '__main__':
    pytest.main([__file__])
