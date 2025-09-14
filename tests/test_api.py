"""
Unit tests for API module.

This module contains comprehensive tests for the FastAPI application.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
import numpy as np
import cv2

from api.main import app
from api.models import ProcessingRequest, ProcessingResponse, HealthResponse


class TestAPI:
    """Test cases for API endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "RedactAI API"
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "face_detection_available" in data
        assert "plate_detection_available" in data
        assert "text_detection_available" in data
        assert "geotagging_available" in data
    
    def test_statistics_endpoint(self):
        """Test statistics endpoint."""
        response = self.client.get("/statistics")
        assert response.status_code == 200
        data = response.json()
        assert "total_files_processed" in data
        assert "total_faces_detected" in data
        assert "total_plates_detected" in data
        assert "total_text_regions_detected" in data
        assert "total_names_redacted" in data
    
    @patch('api.main.process_image')
    def test_process_image_endpoint(self, mock_process_image):
        """Test image processing endpoint."""
        # Mock the process_image function
        mock_process_image.return_value = {
            'faces_detected': 2,
            'plates_detected': 1,
            'text_regions_detected': 3,
            'names_redacted': 1
        }
        
        # Create a test image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            # Encode test image
            _, buffer = cv2.imencode('.jpg', self.test_image)
            temp_file.write(buffer.tobytes())
            temp_file.flush()
            
            try:
                # Test image processing
                with open(temp_file.name, 'rb') as f:
                    response = self.client.post(
                        "/process",
                        files={"file": ("test.jpg", f, "image/jpeg")},
                        data={
                            "process_faces": True,
                            "process_plates": True,
                            "process_text": True,
                            "redact_names_only": True
                        }
                    )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert "output_path" in data
                assert "processing_time_seconds" in data
                assert "faces_detected" in data
                assert "plates_detected" in data
                assert "text_regions_detected" in data
                assert "names_redacted" in data
                
            finally:
                os.unlink(temp_file.name)
    
    def test_process_invalid_file_type(self):
        """Test processing with invalid file type."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_file.write(b'This is not an image')
            temp_file.flush()
            
            try:
                with open(temp_file.name, 'rb') as f:
                    response = self.client.post(
                        "/process",
                        files={"file": ("test.txt", f, "text/plain")},
                        data={"process_faces": True}
                    )
                
                assert response.status_code == 400
                data = response.json()
                assert "detail" in data
                
            finally:
                os.unlink(temp_file.name)
    
    def test_process_no_file(self):
        """Test processing with no file."""
        response = self.client.post(
            "/process",
            data={"process_faces": True}
        )
        assert response.status_code == 422  # Validation error
    
    @patch('api.main.process_video')
    def test_process_video_endpoint(self, mock_process_video):
        """Test video processing endpoint."""
        # Mock the process_video function
        mock_process_video.return_value = {
            'faces_detected': 5,
            'plates_detected': 2,
            'text_regions_detected': 8,
            'names_redacted': 3,
            'frames_processed': 100
        }
        
        # Create a test video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_file.write(b'dummy video data')
            temp_file.flush()
            
            try:
                with open(temp_file.name, 'rb') as f:
                    response = self.client.post(
                        "/process",
                        files={"file": ("test.mp4", f, "video/mp4")},
                        data={
                            "process_faces": True,
                            "process_plates": True,
                            "process_text": True
                        }
                    )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert "frames_processed" in data
                
            finally:
                os.unlink(temp_file.name)
    
    def test_download_file_endpoint(self):
        """Test file download endpoint."""
        # Create a test output file
        test_output_path = "data/output_media/test_file.jpg"
        os.makedirs("data/output_media", exist_ok=True)
        
        with open(test_output_path, 'w') as f:
            f.write("test content")
        
        try:
            response = self.client.get("/download/test_file.jpg")
            assert response.status_code == 200
            assert response.headers["content-type"] == "image/jpeg"
        finally:
            if os.path.exists(test_output_path):
                os.unlink(test_output_path)
    
    def test_download_nonexistent_file(self):
        """Test downloading nonexistent file."""
        response = self.client.get("/download/nonexistent.jpg")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
    
    def test_delete_file_endpoint(self):
        """Test file deletion endpoint."""
        # Create a test output file
        test_output_path = "data/output_media/test_delete.jpg"
        os.makedirs("data/output_media", exist_ok=True)
        
        with open(test_output_path, 'w') as f:
            f.write("test content")
        
        try:
            response = self.client.delete("/files/test_delete.jpg")
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert not os.path.exists(test_output_path)
        finally:
            if os.path.exists(test_output_path):
                os.unlink(test_output_path)
    
    def test_delete_nonexistent_file(self):
        """Test deleting nonexistent file."""
        response = self.client.delete("/files/nonexistent.jpg")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data


class TestAPIModels:
    """Test cases for API models."""
    
    def test_processing_request_defaults(self):
        """Test ProcessingRequest with default values."""
        request = ProcessingRequest()
        assert request.process_faces is True
        assert request.process_plates is True
        assert request.process_text is True
        assert request.redact_names_only is True
        assert request.face_blur_type == "gaussian"
        assert request.plate_blur_type == "gaussian"
        assert request.text_blur_type == "gaussian"
        assert request.face_blur_strength == 15
        assert request.plate_blur_strength == 15
        assert request.text_blur_strength == 15
        assert request.face_confidence == 0.5
        assert request.plate_confidence == 0.5
        assert request.text_confidence == 0.5
        assert request.add_geotags is False
        assert request.ocr_engine == "tesseract"
    
    def test_processing_request_custom_values(self):
        """Test ProcessingRequest with custom values."""
        request = ProcessingRequest(
            process_faces=False,
            face_blur_type="pixelate",
            face_blur_strength=25,
            face_confidence=0.8,
            add_geotags=True,
            gps_latitude=37.7749,
            gps_longitude=-122.4194,
            custom_metadata={"test": "value"}
        )
        assert request.process_faces is False
        assert request.face_blur_type == "pixelate"
        assert request.face_blur_strength == 25
        assert request.face_confidence == 0.8
        assert request.add_geotags is True
        assert request.gps_latitude == 37.7749
        assert request.gps_longitude == -122.4194
        assert request.custom_metadata == {"test": "value"}
    
    def test_processing_response(self):
        """Test ProcessingResponse model."""
        response = ProcessingResponse(
            success=True,
            message="Processing completed",
            processing_time_seconds=1.5,
            faces_detected=2,
            plates_detected=1,
            text_regions_detected=3,
            names_redacted=1,
            input_file_size=1024,
            output_file_size=2048,
            processing_started="2024-01-01T00:00:00Z",
            processing_completed="2024-01-01T00:00:01Z"
        )
        assert response.success is True
        assert response.message == "Processing completed"
        assert response.processing_time_seconds == 1.5
        assert response.faces_detected == 2
        assert response.plates_detected == 1
        assert response.text_regions_detected == 3
        assert response.names_redacted == 1
        assert response.input_file_size == 1024
        assert response.output_file_size == 2048
    
    def test_health_response(self):
        """Test HealthResponse model."""
        response = HealthResponse(
            status="healthy",
            timestamp="2024-01-01T00:00:00Z",
            version="1.0.0",
            face_detection_available=True,
            plate_detection_available=True,
            text_detection_available=True,
            geotagging_available=True,
            opencv_available=True,
            tesseract_available=True,
            easyocr_available=False,
            spacy_available=True,
            yolo_available=False
        )
        assert response.status == "healthy"
        assert response.timestamp == "2024-01-01T00:00:00Z"
        assert response.version == "1.0.0"
        assert response.face_detection_available is True
        assert response.plate_detection_available is True
        assert response.text_detection_available is True
        assert response.geotagging_available is True
        assert response.opencv_available is True
        assert response.tesseract_available is True
        assert response.easyocr_available is False
        assert response.spacy_available is True
        assert response.yolo_available is False


class TestAPIValidation:
    """Test cases for API validation."""
    
    def test_processing_request_validation(self):
        """Test ProcessingRequest validation."""
        # Test valid request
        request = ProcessingRequest(
            face_blur_strength=20,
            plate_blur_strength=25,
            text_blur_strength=30,
            face_confidence=0.7,
            plate_confidence=0.8,
            text_confidence=0.9
        )
        assert request.face_blur_strength == 20
        assert request.plate_blur_strength == 25
        assert request.text_blur_strength == 30
        assert request.face_confidence == 0.7
        assert request.plate_confidence == 0.8
        assert request.text_confidence == 0.9
    
    def test_processing_request_invalid_values(self):
        """Test ProcessingRequest with invalid values."""
        # Test invalid blur strength
        with pytest.raises(ValueError):
            ProcessingRequest(face_blur_strength=100)  # Too high
        
        with pytest.raises(ValueError):
            ProcessingRequest(face_blur_strength=0)  # Too low
        
        # Test invalid confidence
        with pytest.raises(ValueError):
            ProcessingRequest(face_confidence=1.5)  # Too high
        
        with pytest.raises(ValueError):
            ProcessingRequest(face_confidence=-0.1)  # Too low
        
        # Test invalid GPS coordinates
        with pytest.raises(ValueError):
            ProcessingRequest(gps_latitude=100.0)  # Too high
        
        with pytest.raises(ValueError):
            ProcessingRequest(gps_latitude=-100.0)  # Too low
        
        with pytest.raises(ValueError):
            ProcessingRequest(gps_longitude=200.0)  # Too high
        
        with pytest.raises(ValueError):
            ProcessingRequest(gps_longitude=-200.0)  # Too low


if __name__ == '__main__':
    pytest.main([__file__])
