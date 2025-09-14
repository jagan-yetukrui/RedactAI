"""
Test script to verify RedactAI installation.

This script tests all major components to ensure they work correctly.
"""

import sys
import logging
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test importing all major modules."""
    logger.info("Testing imports...")
    
    try:
        import cv2
        logger.info("✓ OpenCV imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        logger.info("✓ NumPy imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import NumPy: {e}")
        return False
    
    try:
        import spacy
        logger.info("✓ SpaCy imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import SpaCy: {e}")
        return False
    
    try:
        import fastapi
        logger.info("✓ FastAPI imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import FastAPI: {e}")
        return False
    
    try:
        import streamlit
        logger.info("✓ Streamlit imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import Streamlit: {e}")
        return False
    
    return True


def test_modules():
    """Test importing RedactAI modules."""
    logger.info("Testing RedactAI modules...")
    
    try:
        from modules.face_blur import FaceDetector, FaceBlurrer
        logger.info("✓ Face blur module imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import face blur module: {e}")
        return False
    
    try:
        from modules.plate_blur import PlateDetector, PlateBlurrer
        logger.info("✓ Plate blur module imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import plate blur module: {e}")
        return False
    
    try:
        from modules.text_redact import TextDetector, NameRedactor, TextRedactor
        logger.info("✓ Text redact module imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import text redact module: {e}")
        return False
    
    try:
        from modules.geotagging import Geotagger, MetadataHandler
        logger.info("✓ Geotagging module imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import geotagging module: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality of modules."""
    logger.info("Testing basic functionality...")
    
    try:
        from modules.face_blur import FaceBlurrer
        import numpy as np
        
        # Create a test image
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # Test face blurrer
        blurrer = FaceBlurrer()
        result_image, faces = blurrer.process_image(test_image)
        
        assert result_image.shape == test_image.shape
        assert isinstance(faces, list)
        logger.info("✓ Face blurrer basic functionality works")
        
    except Exception as e:
        logger.error(f"✗ Face blurrer test failed: {e}")
        return False
    
    try:
        from modules.plate_blur import PlateBlurrer
        import numpy as np
        
        # Create a test image
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # Test plate blurrer
        blurrer = PlateBlurrer()
        result_image, plates = blurrer.process_image(test_image)
        
        assert result_image.shape == test_image.shape
        assert isinstance(plates, list)
        logger.info("✓ Plate blurrer basic functionality works")
        
    except Exception as e:
        logger.error(f"✗ Plate blurrer test failed: {e}")
        return False
    
    try:
        from modules.text_redact import TextRedactor
        import numpy as np
        
        # Create a test image
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # Test text redactor
        redactor = TextRedactor()
        result_image, stats = redactor.process_image(test_image)
        
        assert result_image.shape == test_image.shape
        assert isinstance(stats, dict)
        logger.info("✓ Text redactor basic functionality works")
        
    except Exception as e:
        logger.error(f"✗ Text redactor test failed: {e}")
        return False
    
    return True


def test_api_models():
    """Test API models."""
    logger.info("Testing API models...")
    
    try:
        from api.models import ProcessingRequest, ProcessingResponse, HealthResponse
        
        # Test ProcessingRequest
        request = ProcessingRequest()
        assert request.process_faces is True
        assert request.face_blur_type == "gaussian"
        
        # Test ProcessingResponse
        response = ProcessingResponse(
            success=True,
            message="Test",
            processing_time_seconds=1.0,
            faces_detected=0,
            plates_detected=0,
            text_regions_detected=0,
            names_redacted=0,
            input_file_size=1024,
            processing_started="2024-01-01T00:00:00Z",
            processing_completed="2024-01-01T00:00:01Z"
        )
        assert response.success is True
        
        logger.info("✓ API models work correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ API models test failed: {e}")
        return False


def main():
    """Main test function."""
    logger.info("RedactAI Installation Test")
    logger.info("=========================")
    
    tests = [
        ("Import Dependencies", test_imports),
        ("Import Modules", test_modules),
        ("Basic Functionality", test_basic_functionality),
        ("API Models", test_api_models)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning test: {test_name}")
        try:
            if test_func():
                logger.info(f"✓ {test_name} passed")
                passed += 1
            else:
                logger.error(f"✗ {test_name} failed")
        except Exception as e:
            logger.error(f"✗ {test_name} failed with exception: {e}")
            logger.error(traceback.format_exc())
    
    logger.info(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! RedactAI is ready to use.")
        logger.info("Run 'python app.py' to start the application.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
