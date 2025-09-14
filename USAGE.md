# RedactAI Usage Guide

This guide provides comprehensive instructions for using RedactAI, including setup, API usage, and dashboard features.

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd RedactAI

# Install dependencies
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_sm

# Run setup script
python setup.py
```

### 2. Start the Application

```bash
# Run both API and dashboard
python app.py

# Or run only the API
python app.py api

# Or run only the dashboard
python app.py dashboard
```

### 3. Access the Services

- **API Documentation**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501
- **API Health Check**: http://localhost:8000/health

## API Usage

### Basic Image Processing

```python
import requests

# Process an image
with open('input.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/process',
        files={'file': f},
        data={
            'process_faces': True,
            'process_plates': True,
            'process_text': True,
            'redact_names_only': True
        }
    )

result = response.json()
print(f"Faces detected: {result['faces_detected']}")
print(f"Plates detected: {result['plates_detected']}")
print(f"Text regions detected: {result['text_regions_detected']}")
```

### Advanced Processing Options

```python
# Advanced processing with custom settings
response = requests.post(
    'http://localhost:8000/process',
    files={'file': open('input.jpg', 'rb')},
    data={
        'process_faces': True,
        'process_plates': True,
        'process_text': True,
        'redact_names_only': True,
        'face_blur_type': 'gaussian',
        'plate_blur_type': 'pixelate',
        'text_blur_type': 'mosaic',
        'face_blur_strength': 20,
        'plate_blur_strength': 15,
        'text_blur_strength': 25,
        'face_confidence': 0.7,
        'plate_confidence': 0.8,
        'text_confidence': 0.6,
        'add_geotags': True,
        'gps_latitude': 37.7749,
        'gps_longitude': -122.4194,
        'ocr_engine': 'tesseract'
    }
)
```

### Video Processing

```python
# Process a video file
with open('input.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/process',
        files={'file': f},
        data={
            'process_faces': True,
            'process_plates': True,
            'process_text': True
        }
    )

result = response.json()
print(f"Frames processed: {result['frames_processed']}")
print(f"Total faces detected: {result['faces_detected']}")
```

### Download Processed Files

```python
# Download processed file
response = requests.get('http://localhost:8000/download/processed_filename.jpg')
with open('output.jpg', 'wb') as f:
    f.write(response.content)
```

### Get Processing Statistics

```python
# Get processing statistics
response = requests.get('http://localhost:8000/statistics')
stats = response.json()
print(f"Total files processed: {stats['total_files_processed']}")
print(f"Total faces detected: {stats['total_faces_detected']}")
```

## Dashboard Usage

### 1. Home Page

- View system status and health
- Quick statistics overview
- Feature descriptions

### 2. Statistics Page

- Detailed processing statistics
- Detection breakdown charts
- File type distribution
- Processing timeline

### 3. Process Media Page

- Upload images and videos
- Configure processing options
- Real-time processing
- Download results

### 4. Geospatial View

- Interactive heatmaps
- Location-based visualization
- Redaction density maps

### 5. Settings Page

- API configuration
- Display preferences
- System information

## Programmatic Usage

### Using Modules Directly

```python
from modules.face_blur import FaceBlurrer
from modules.plate_blur import PlateBlurrer
from modules.text_redact import TextRedactor
import cv2

# Load image
image = cv2.imread('input.jpg')

# Process faces
face_processor = FaceBlurrer(blur_type='gaussian', blur_strength=15)
processed_image, faces = face_processor.process_image(image)

# Process license plates
plate_processor = PlateBlurrer(blur_type='pixelate', blur_strength=20)
processed_image, plates = plate_processor.process_image(processed_image)

# Process text
text_processor = TextRedactor(blur_type='mosaic', blur_strength=25)
processed_image, stats = text_processor.process_image(processed_image)

# Save result
cv2.imwrite('output.jpg', processed_image)
```

### Batch Processing

```python
import os
from pathlib import Path

def process_directory(input_dir, output_dir):
    """Process all images in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    face_processor = FaceBlurrer()
    plate_processor = PlateBlurrer()
    text_processor = TextRedactor()

    for image_file in input_path.glob('*.jpg'):
        # Load image
        image = cv2.imread(str(image_file))

        # Process
        processed_image, faces = face_processor.process_image(image)
        processed_image, plates = plate_processor.process_image(processed_image)
        processed_image, stats = text_processor.process_image(processed_image)

        # Save
        output_file = output_path / f"processed_{image_file.name}"
        cv2.imwrite(str(output_file), processed_image)

        print(f"Processed {image_file.name}: {len(faces)} faces, {len(plates)} plates")

# Usage
process_directory('input_images', 'output_images')
```

## Configuration

### Environment Variables

```bash
# API Configuration
export REDACT_AI_API_HOST=0.0.0.0
export REDACT_AI_API_PORT=8000

# Dashboard Configuration
export REDACT_AI_DASHBOARD_PORT=8501

# Processing Configuration
export REDACT_AI_DEFAULT_BLUR_STRENGTH=15
export REDACT_AI_DEFAULT_CONFIDENCE=0.5
```

### Configuration File

Create a `config.yaml` file:

```yaml
api:
  host: "0.0.0.0"
  port: 8000

dashboard:
  port: 8501

processing:
  default_blur_strength: 15
  default_confidence: 0.5
  max_file_size: 100MB

geotagging:
  mock_gps: true
  default_center: [37.7749, -122.4194]
```

## Docker Usage

### Build and Run

```bash
# Build the image
docker build -t redact-ai .

# Run the container
docker run -p 8000:8000 -p 8501:8501 redact-ai

# Run with Docker Compose
docker-compose up -d
```

### Docker Compose

```yaml
version: "3.8"
services:
  redact-ai:
    build: .
    ports:
      - "8000:8000"
      - "8501:8501"
    volumes:
      - ./data:/app/data
    environment:
      - PYTHONPATH=/app
```

## Testing

### Run Unit Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_face_blur.py -v

# Run with coverage
python -m pytest tests/ --cov=modules --cov-report=html
```

### Test Installation

```bash
# Test if everything is working
python test_installation.py
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed

   ```bash
   pip install -r requirements.txt
   ```

2. **SpaCy Model Missing**: Download the required model

   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **OpenCV Issues**: Install system dependencies

   ```bash
   # Ubuntu/Debian
   sudo apt-get install libgl1-mesa-glx libglib2.0-0

   # macOS
   brew install opencv
   ```

4. **Tesseract Not Found**: Install Tesseract OCR

   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr

   # macOS
   brew install tesseract
   ```

### Performance Optimization

1. **GPU Acceleration**: Install CUDA-enabled OpenCV
2. **Memory Management**: Process large videos in chunks
3. **Parallel Processing**: Use multiprocessing for batch operations

### Logging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## API Reference

### Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /process` - Process media file
- `GET /statistics` - Get processing statistics
- `GET /download/{filename}` - Download processed file
- `DELETE /files/{filename}` - Delete processed file

### Response Formats

All API responses follow a consistent format:

```json
{
  "success": true,
  "message": "Processing completed successfully",
  "data": { ... },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Support

For issues and questions:

1. Check the troubleshooting section
2. Run the test installation script
3. Check the logs for error messages
4. Review the API documentation at `/docs`

## License

This project is licensed under the MIT License - see the LICENSE file for details.
