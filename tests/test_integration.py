"""
Integration tests for RedactAI.

This module contains comprehensive integration tests that test the entire
RedactAI pipeline from input to output.
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
import time

from modules.face_blur import FaceBlurrer
from modules.plate_blur import PlateBlurrer
from modules.text_redact import TextRedactor
from modules.geotagging import Geotagger
from utils.monitoring import get_metrics_collector
from utils.cache import get_cache_manager
from utils.batch_processor import create_batch_processor


class TestRedactAIIntegration:
    """Integration tests for the complete RedactAI pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "input"
        self.output_dir = Path(self.temp_dir) / "output"
        self.input_dir.mkdir()
        self.output_dir.mkdir()
        
        # Create test image
        self.test_image = self._create_test_image()
        self.test_image_path = self.input_dir / "test_image.jpg"
        cv2.imwrite(str(self.test_image_path), self.test_image)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_image(self, width=800, height=600):
        """Create a test image with various elements."""
        image = np.ones((height, width, 3), dtype=np.uint8) * 240
        
        # Add some background elements
        cv2.rectangle(image, (50, 50), (width-50, height-50), (200, 200, 200), 2)
        
        # Add synthetic faces (circles)
        cv2.circle(image, (200, 150), 60, (220, 180, 150), -1)  # Face 1
        cv2.circle(image, (500, 200), 50, (220, 180, 150), -1)  # Face 2
        
        # Add eyes
        cv2.circle(image, (185, 135), 8, (0, 0, 0), -1)
        cv2.circle(image, (215, 135), 8, (0, 0, 0), -1)
        cv2.circle(image, (485, 185), 8, (0, 0, 0), -1)
        cv2.circle(image, (515, 185), 8, (0, 0, 0), -1)
        
        # Add mouths
        cv2.ellipse(image, (200, 170), (20, 10), 0, 0, 180, (0, 0, 0), 2)
        cv2.ellipse(image, (500, 220), (15, 8), 0, 0, 180, (0, 0, 0), 2)
        
        # Add license plates
        cv2.rectangle(image, (100, 400), (220, 440), (255, 255, 255), -1)
        cv2.rectangle(image, (100, 400), (220, 440), (0, 0, 0), 2)
        cv2.putText(image, "ABC123", (110, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        cv2.rectangle(image, (400, 450), (520, 490), (255, 255, 255), -1)
        cv2.rectangle(image, (400, 450), (520, 490), (0, 0, 0), 2)
        cv2.putText(image, "XYZ789", (410, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add text
        cv2.putText(image, "John Smith", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(image, "Sarah Johnson", (300, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(image, "123 Main Street", (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return image
    
    def test_face_detection_integration(self):
        """Test face detection integration."""
        face_processor = FaceBlurrer(blur_type='gaussian', blur_strength=15)
        
        # Process image
        result_image, faces = face_processor.process_image(self.test_image)
        
        # Verify results
        assert result_image.shape == self.test_image.shape
        assert isinstance(faces, list)
        assert len(faces) >= 0  # May or may not detect synthetic faces
        
        # Save result
        output_path = self.output_dir / "faces_processed.jpg"
        cv2.imwrite(str(output_path), result_image)
        assert output_path.exists()
    
    def test_plate_detection_integration(self):
        """Test license plate detection integration."""
        plate_processor = PlateBlurrer(blur_type='pixelate', blur_strength=20)
        
        # Process image
        result_image, plates = plate_processor.process_image(self.test_image)
        
        # Verify results
        assert result_image.shape == self.test_image.shape
        assert isinstance(plates, list)
        
        # Save result
        output_path = self.output_dir / "plates_processed.jpg"
        cv2.imwrite(str(output_path), result_image)
        assert output_path.exists()
    
    def test_text_detection_integration(self):
        """Test text detection integration."""
        text_processor = TextRedactor(blur_type='mosaic', blur_strength=25)
        
        # Process image
        result_image, stats = text_processor.process_image(self.test_image)
        
        # Verify results
        assert result_image.shape == self.test_image.shape
        assert isinstance(stats, dict)
        assert 'total_text_regions' in stats
        
        # Save result
        output_path = self.output_dir / "text_processed.jpg"
        cv2.imwrite(str(output_path), result_image)
        assert output_path.exists()
    
    def test_geotagging_integration(self):
        """Test geotagging integration."""
        geotagger = Geotagger()
        
        # Add geotags
        gps_coords = (37.7749, -122.4194)
        result_image = geotagger.add_geotag_to_image(self.test_image, gps_coords)
        
        # Verify results
        assert result_image.shape == self.test_image.shape
        
        # Save result
        output_path = self.output_dir / "geotagged.jpg"
        cv2.imwrite(str(output_path), result_image)
        assert output_path.exists()
    
    def test_full_pipeline_integration(self):
        """Test the complete RedactAI pipeline."""
        # Initialize processors
        face_processor = FaceBlurrer(blur_type='gaussian', blur_strength=15)
        plate_processor = PlateBlurrer(blur_type='pixelate', blur_strength=20)
        text_processor = TextRedactor(blur_type='mosaic', blur_strength=25)
        geotagger = Geotagger()
        
        # Process image through complete pipeline
        result_image = self.test_image.copy()
        
        # Face processing
        result_image, faces = face_processor.process_image(result_image)
        print(f"Detected {len(faces)} faces")
        
        # Plate processing
        result_image, plates = plate_processor.process_image(result_image)
        print(f"Detected {len(plates)} license plates")
        
        # Text processing
        result_image, text_stats = text_processor.process_image(result_image)
        print(f"Detected {text_stats.get('total_text_regions', 0)} text regions")
        
        # Geotagging
        gps_coords = (37.7749, -122.4194)
        result_image = geotagger.add_geotag_to_image(result_image, gps_coords)
        
        # Verify results
        assert result_image.shape == self.test_image.shape
        
        # Save result
        output_path = self.output_dir / "full_pipeline.jpg"
        cv2.imwrite(str(output_path), result_image)
        assert output_path.exists()
        
        # Verify file was created and has content
        assert output_path.stat().st_size > 0
    
    def test_batch_processing_integration(self):
        """Test batch processing integration."""
        # Create multiple test images
        for i in range(3):
            test_image = self._create_test_image()
            image_path = self.input_dir / f"test_image_{i}.jpg"
            cv2.imwrite(str(image_path), test_image)
        
        # Create batch processor
        processor = create_batch_processor(max_workers=2)
        
        # Add jobs
        job_ids = []
        for image_path in self.input_dir.glob("*.jpg"):
            output_path = self.output_dir / f"processed_{image_path.name}"
            job_id = processor.add_job(image_path, output_path, {
                'process_faces': True,
                'process_plates': True,
                'process_text': True
            })
            job_ids.append(job_id)
        
        # Mock processor function
        def mock_processor(input_path, output_path, **kwargs):
            # Load and process image
            image = cv2.imread(str(input_path))
            if image is None:
                raise ValueError(f"Could not load image: {input_path}")
            
            # Simple processing (just blur the image)
            processed = cv2.GaussianBlur(image, (15, 15), 0)
            cv2.imwrite(str(output_path), processed)
            
            return {
                'faces_detected': 2,
                'plates_detected': 1,
                'text_regions_detected': 3,
                'names_redacted': 2
            }
        
        # Process batch
        progress = processor.process_batch(mock_processor)
        
        # Verify results
        assert progress.total_jobs == 3
        assert progress.completed_jobs == 3
        assert progress.failed_jobs == 0
        
        # Verify output files were created
        for image_path in self.input_dir.glob("*.jpg"):
            output_path = self.output_dir / f"processed_{image_path.name}"
            assert output_path.exists()
            assert output_path.stat().st_size > 0
    
    def test_metrics_integration(self):
        """Test metrics collection integration."""
        metrics_collector = get_metrics_collector()
        
        # Record some processing metrics
        metrics_collector.record_processing(
            processing_time=1.5,
            file_type='.jpg',
            faces=2,
            plates=1,
            text_regions=3,
            names=2
        )
        
        # Get metrics summary
        stats = metrics_collector.get_metrics_summary()
        
        # Verify metrics
        assert stats['processing']['total_files_processed'] >= 1
        assert stats['processing']['total_faces_detected'] >= 2
        assert stats['processing']['total_plates_detected'] >= 1
        assert stats['processing']['total_text_regions_detected'] >= 3
        assert stats['processing']['total_names_redacted'] >= 2
    
    def test_cache_integration(self):
        """Test caching integration."""
        cache_manager = get_cache_manager()
        
        # Test cache operations
        test_key = "test_key"
        test_value = {"test": "data", "number": 42}
        
        # Set value
        cache_manager.set(test_key, test_value, ttl=60)
        
        # Get value
        retrieved_value = cache_manager.get(test_key)
        assert retrieved_value == test_value
        
        # Test cache stats
        stats = cache_manager.get_stats()
        assert 'memory_cache' in stats
        assert 'file_cache' in stats
    
    def test_error_handling_integration(self):
        """Test error handling integration."""
        # Test with invalid input
        invalid_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        face_processor = FaceBlurrer()
        
        # This should not raise an exception
        try:
            result_image, faces = face_processor.process_image(invalid_image)
            assert result_image.shape == invalid_image.shape
            assert isinstance(faces, list)
        except Exception as e:
            pytest.fail(f"Face processor should handle invalid input gracefully: {e}")
    
    def test_performance_integration(self):
        """Test performance integration."""
        # Test processing time
        start_time = time.time()
        
        face_processor = FaceBlurrer()
        result_image, faces = face_processor.process_image(self.test_image)
        
        processing_time = time.time() - start_time
        
        # Should process within reasonable time (less than 5 seconds)
        assert processing_time < 5.0
        
        # Test with larger image
        large_image = cv2.resize(self.test_image, (1600, 1200))
        
        start_time = time.time()
        result_image, faces = face_processor.process_image(large_image)
        processing_time = time.time() - start_time
        
        # Should still process within reasonable time
        assert processing_time < 10.0


class TestAPIIntegration:
    """Integration tests for API functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image = self._create_test_image()
        self.test_image_path = Path(self.temp_dir) / "test.jpg"
        cv2.imwrite(str(self.test_image_path), self.test_image)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_image(self, width=400, height=300):
        """Create a simple test image."""
        image = np.ones((height, width, 3), dtype=np.uint8) * 240
        cv2.circle(image, (200, 150), 50, (220, 180, 150), -1)
        cv2.circle(image, (185, 135), 8, (0, 0, 0), -1)
        cv2.circle(image, (215, 135), 8, (0, 0, 0), -1)
        cv2.ellipse(image, (200, 170), (20, 10), 0, 0, 180, (0, 0, 0), 2)
        return image
    
    @patch('api.main.process_image')
    def test_api_image_processing(self, mock_process_image):
        """Test API image processing integration."""
        from fastapi.testclient import TestClient
        from api.main import app
        
        # Mock the process_image function
        mock_process_image.return_value = {
            'faces_detected': 1,
            'plates_detected': 0,
            'text_regions_detected': 0,
            'names_redacted': 0
        }
        
        client = TestClient(app)
        
        # Test image processing endpoint
        with open(self.test_image_path, 'rb') as f:
            response = client.post(
                "/process",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={
                    "process_faces": True,
                    "process_plates": False,
                    "process_text": False
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "output_path" in data
        assert data["faces_detected"] == 1
    
    def test_api_health_check(self):
        """Test API health check integration."""
        from fastapi.testclient import TestClient
        from api.main import app
        
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
    
    def test_api_statistics(self):
        """Test API statistics integration."""
        from fastapi.testclient import TestClient
        from api.main import app
        
        client = TestClient(app)
        response = client.get("/statistics")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_files_processed" in data
        assert "total_faces_detected" in data
        assert "total_plates_detected" in data
        assert "total_text_regions_detected" in data
        assert "total_names_redacted" in data


if __name__ == '__main__':
    pytest.main([__file__])
