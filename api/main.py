"""
FastAPI main application for RedactAI.

This module provides the main FastAPI application with all endpoints
for media processing and management.
"""

import os
import time
import logging
from datetime import datetime, timezone
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

# Import our modules
from .models import (
    ProcessingRequest, ProcessingResponse, HealthResponse, ErrorResponse,
    BatchProcessingRequest, BatchProcessingResponse, StatisticsResponse
)
from ..modules.face_blur import FaceBlurrer
from ..modules.plate_blur import PlateBlurrer
from ..modules.text_redact import TextRedactor
from ..modules.geotagging import Geotagger, MetadataHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RedactAI API",
    description="AI-powered privacy tool for redacting sensitive information from media files",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for processors
face_processor: Optional[FaceBlurrer] = None
plate_processor: Optional[PlateBlurrer] = None
text_processor: Optional[TextRedactor] = None
geotagger: Optional[Geotagger] = None
metadata_handler: Optional[MetadataHandler] = None

# Processing statistics
processing_stats = {
    'total_files_processed': 0,
    'total_faces_detected': 0,
    'total_plates_detected': 0,
    'total_text_regions_detected': 0,
    'total_names_redacted': 0,
    'total_processing_time': 0.0,
    'file_type_breakdown': {},
    'first_processing': None,
    'last_processing': None
}


@app.on_event("startup")
async def startup_event():
    """Initialize processors on startup."""
    global face_processor, plate_processor, text_processor, geotagger, metadata_handler
    
    try:
        # Initialize processors
        face_processor = FaceBlurrer()
        plate_processor = PlateBlurrer()
        text_processor = TextRedactor()
        geotagger = Geotagger()
        metadata_handler = MetadataHandler()
        
        # Create output directories
        os.makedirs("data/output_media", exist_ok=True)
        os.makedirs("data/metadata", exist_ok=True)
        
        logger.info("RedactAI API started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RedactAI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check component availability
        face_available = face_processor is not None
        plate_available = plate_processor is not None
        text_available = text_processor is not None
        geotag_available = geotagger is not None
        
        # Check dependencies
        opencv_available = True  # We're using it
        tesseract_available = hasattr(text_processor, 'text_detector') and text_processor.text_detector.tesseract_available
        easyocr_available = hasattr(text_processor, 'text_detector') and text_processor.text_detector.easyocr_available
        spacy_available = hasattr(text_processor, 'name_redactor') and text_processor.name_redactor.spacy_available
        yolo_available = hasattr(plate_processor, 'detector') and plate_processor.detector.use_yolo
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(timezone.utc),
            version="1.0.0",
            face_detection_available=face_available,
            plate_detection_available=plate_available,
            text_detection_available=text_available,
            geotagging_available=geotag_available,
            opencv_available=opencv_available,
            tesseract_available=tesseract_available,
            easyocr_available=easyocr_available,
            spacy_available=spacy_available,
            yolo_available=yolo_available
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/process", response_model=ProcessingResponse)
async def process_media(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    process_faces: bool = True,
    process_plates: bool = True,
    process_text: bool = True,
    redact_names_only: bool = True,
    face_blur_type: str = "gaussian",
    plate_blur_type: str = "gaussian",
    text_blur_type: str = "gaussian",
    face_blur_strength: int = 15,
    plate_blur_strength: int = 15,
    text_blur_strength: int = 15,
    face_confidence: float = 0.5,
    plate_confidence: float = 0.5,
    text_confidence: float = 0.5,
    add_geotags: bool = False,
    gps_latitude: Optional[float] = None,
    gps_longitude: Optional[float] = None,
    ocr_engine: str = "tesseract",
    custom_metadata: Optional[dict] = None
):
    """Process a single media file."""
    start_time = time.time()
    processing_started = datetime.now(timezone.utc)
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file type
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ['.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov']:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Generate output filename
        output_filename = f"processed_{int(time.time())}_{file.filename}"
        output_path = f"data/output_media/{output_filename}"
        
        # Process based on file type
        if file_extension in ['.jpg', '.jpeg', '.png']:
            # Process image
            result = await process_image(
                file_content, output_path, process_faces, process_plates, process_text,
                redact_names_only, face_blur_type, plate_blur_type, text_blur_type,
                face_blur_strength, plate_blur_strength, text_blur_strength,
                face_confidence, plate_confidence, text_confidence,
                add_geotags, gps_latitude, gps_longitude, ocr_engine, custom_metadata
            )
        else:
            # Process video
            result = await process_video(
                file_content, output_path, process_faces, process_plates, process_text,
                redact_names_only, face_blur_type, plate_blur_type, text_blur_type,
                face_blur_strength, plate_blur_strength, text_blur_strength,
                face_confidence, plate_confidence, text_confidence,
                add_geotags, gps_latitude, gps_longitude, ocr_engine, custom_metadata
            )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        processing_completed = datetime.now(timezone.utc)
        
        # Update statistics
        update_processing_stats(file_extension, processing_time, result)
        
        # Prepare response
        response = ProcessingResponse(
            success=True,
            message="Processing completed successfully",
            output_path=output_path,
            processing_time_seconds=processing_time,
            frames_processed=result.get('frames_processed'),
            faces_detected=result.get('faces_detected', 0),
            plates_detected=result.get('plates_detected', 0),
            text_regions_detected=result.get('text_regions_detected', 0),
            names_redacted=result.get('names_redacted', 0),
            input_file_size=file_size,
            output_file_size=os.path.getsize(output_path) if os.path.exists(output_path) else None,
            metadata_path=result.get('metadata_path'),
            geotag_data=result.get('geotag_data'),
            processing_started=processing_started,
            processing_completed=processing_completed
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


async def process_image(
    file_content: bytes, output_path: str, process_faces: bool, process_plates: bool,
    process_text: bool, redact_names_only: bool, face_blur_type: str, plate_blur_type: str,
    text_blur_type: str, face_blur_strength: int, plate_blur_strength: int,
    text_blur_strength: int, face_confidence: float, plate_confidence: float,
    text_confidence: float, add_geotags: bool, gps_latitude: Optional[float],
    gps_longitude: Optional[float], ocr_engine: str, custom_metadata: Optional[dict]
) -> dict:
    """Process an image file."""
    # Decode image
    nparr = np.frombuffer(file_content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Could not decode image")
    
    result = {
        'faces_detected': 0,
        'plates_detected': 0,
        'text_regions_detected': 0,
        'names_redacted': 0
    }
    
    # Process faces
    if process_faces and face_processor:
        face_processor.blur_type = face_blur_type
        face_processor.blur_strength = face_blur_strength
        processed_image, faces = face_processor.process_image(image, face_confidence)
        result['faces_detected'] = len(faces)
        image = processed_image
    
    # Process license plates
    if process_plates and plate_processor:
        plate_processor.blur_type = plate_blur_type
        plate_processor.blur_strength = plate_blur_strength
        processed_image, plates = plate_processor.process_image(image, plate_confidence)
        result['plates_detected'] = len(plates)
        image = processed_image
    
    # Process text
    if process_text and text_processor:
        text_processor.blur_type = text_blur_type
        text_processor.blur_strength = text_blur_strength
        text_processor.redact_names_only = redact_names_only
        text_processor.text_detector.ocr_engine = ocr_engine
        processed_image, stats = text_processor.process_image(image, text_confidence)
        result['text_regions_detected'] = stats.get('total_text_regions', 0)
        result['names_redacted'] = stats.get('redacted_regions', 0)
        image = processed_image
    
    # Add geotags
    if add_geotags and geotagger:
        gps_coords = (gps_latitude, gps_longitude) if gps_latitude and gps_longitude else None
        image = geotagger.add_geotag_to_image(image, gps_coords, custom_metadata=custom_metadata)
        result['geotag_data'] = {
            'gps_coords': gps_coords or geotagger.generate_mock_gps(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    # Save processed image
    cv2.imwrite(output_path, image)
    
    return result


async def process_video(
    file_content: bytes, output_path: str, process_faces: bool, process_plates: bool,
    process_text: bool, redact_names_only: bool, face_blur_type: str, plate_blur_type: str,
    text_blur_type: str, face_blur_strength: int, plate_blur_strength: int,
    text_blur_strength: int, face_confidence: float, plate_confidence: float,
    text_confidence: float, add_geotags: bool, gps_latitude: Optional[float],
    gps_longitude: Optional[float], ocr_engine: str, custom_metadata: Optional[dict]
) -> dict:
    """Process a video file."""
    # Save temporary input file
    temp_input_path = f"data/temp_{int(time.time())}.mp4"
    with open(temp_input_path, 'wb') as f:
        f.write(file_content)
    
    try:
        result = {
            'faces_detected': 0,
            'plates_detected': 0,
            'text_regions_detected': 0,
            'names_redacted': 0,
            'frames_processed': 0
        }
        
        # Process faces
        if process_faces and face_processor:
            face_processor.blur_type = face_blur_type
            face_processor.blur_strength = face_blur_strength
            frames_processed, faces_per_frame = face_processor.process_video(
                temp_input_path, output_path, face_confidence
            )
            result['faces_detected'] = sum(len(faces) for faces in faces_per_frame)
            result['frames_processed'] = frames_processed
        
        # Process license plates
        if process_plates and plate_processor:
            plate_processor.blur_type = plate_blur_type
            plate_processor.blur_strength = plate_blur_strength
            frames_processed, plates_per_frame = plate_processor.process_video(
                temp_input_path, output_path, plate_confidence
            )
            result['plates_detected'] = sum(len(plates) for plates in plates_per_frame)
            result['frames_processed'] = frames_processed
        
        # Process text
        if process_text and text_processor:
            text_processor.blur_type = text_blur_type
            text_processor.blur_strength = text_blur_strength
            text_processor.redact_names_only = redact_names_only
            text_processor.text_detector.ocr_engine = ocr_engine
            frames_processed, stats_per_frame = text_processor.process_video(
                temp_input_path, output_path, text_confidence
            )
            result['text_regions_detected'] = sum(stats.get('total_text_regions', 0) for stats in stats_per_frame)
            result['names_redacted'] = sum(stats.get('redacted_regions', 0) for stats in stats_per_frame)
            result['frames_processed'] = frames_processed
        
        return result
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)


def update_processing_stats(file_extension: str, processing_time: float, result: dict):
    """Update global processing statistics."""
    global processing_stats
    
    processing_stats['total_files_processed'] += 1
    processing_stats['total_faces_detected'] += result.get('faces_detected', 0)
    processing_stats['total_plates_detected'] += result.get('plates_detected', 0)
    processing_stats['total_text_regions_detected'] += result.get('text_regions_detected', 0)
    processing_stats['total_names_redacted'] += result.get('names_redacted', 0)
    processing_stats['total_processing_time'] += processing_time
    
    # Update file type breakdown
    processing_stats['file_type_breakdown'][file_extension] = processing_stats['file_type_breakdown'].get(file_extension, 0) + 1
    
    # Update timestamps
    now = datetime.now(timezone.utc)
    if processing_stats['first_processing'] is None:
        processing_stats['first_processing'] = now
    processing_stats['last_processing'] = now


@app.get("/statistics", response_model=StatisticsResponse)
async def get_statistics():
    """Get processing statistics."""
    global processing_stats
    
    avg_time = 0.0
    if processing_stats['total_files_processed'] > 0:
        avg_time = processing_stats['total_processing_time'] / processing_stats['total_files_processed']
    
    return StatisticsResponse(
        total_files_processed=processing_stats['total_files_processed'],
        total_faces_detected=processing_stats['total_faces_detected'],
        total_plates_detected=processing_stats['total_plates_detected'],
        total_text_regions_detected=processing_stats['total_text_regions_detected'],
        total_names_redacted=processing_stats['total_names_redacted'],
        average_processing_time_seconds=avg_time,
        total_processing_time_seconds=processing_stats['total_processing_time'],
        file_type_breakdown=processing_stats['file_type_breakdown'],
        first_processing=processing_stats['first_processing'],
        last_processing=processing_stats['last_processing']
    )


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a processed file."""
    file_path = f"data/output_media/{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, filename=filename)


@app.delete("/files/{filename}")
async def delete_file(filename: str):
    """Delete a processed file."""
    file_path = f"data/output_media/{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        os.remove(file_path)
        return {"message": f"File {filename} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
