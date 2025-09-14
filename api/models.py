"""
Pydantic models for RedactAI API.

This module defines the data models used for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime


class ProcessingRequest(BaseModel):
    """Request model for media processing."""
    
    # Processing options
    process_faces: bool = Field(True, description="Whether to process faces")
    process_plates: bool = Field(True, description="Whether to process license plates")
    process_text: bool = Field(True, description="Whether to process text")
    redact_names_only: bool = Field(True, description="Whether to redact only names in text")
    
    # Blur settings
    face_blur_type: Literal["gaussian", "pixelate", "blackout", "mosaic"] = Field("gaussian", description="Face blur type")
    plate_blur_type: Literal["gaussian", "pixelate", "blackout", "mosaic"] = Field("gaussian", description="Plate blur type")
    text_blur_type: Literal["gaussian", "pixelate", "blackout", "mosaic"] = Field("gaussian", description="Text blur type")
    
    # Blur strength (1-50)
    face_blur_strength: int = Field(15, ge=1, le=50, description="Face blur strength")
    plate_blur_strength: int = Field(15, ge=1, le=50, description="Plate blur strength")
    text_blur_strength: int = Field(15, ge=1, le=50, description="Text blur strength")
    
    # Detection confidence thresholds (0.0-1.0)
    face_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Face detection confidence threshold")
    plate_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Plate detection confidence threshold")
    text_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Text detection confidence threshold")
    
    # Geotagging options
    add_geotags: bool = Field(False, description="Whether to add geotags")
    gps_latitude: Optional[float] = Field(None, ge=-90, le=90, description="GPS latitude")
    gps_longitude: Optional[float] = Field(None, ge=-180, le=180, description="GPS longitude")
    
    # OCR engine selection
    ocr_engine: Literal["tesseract", "easyocr", "both"] = Field("tesseract", description="OCR engine to use")
    
    # Additional metadata
    custom_metadata: Optional[Dict[str, Any]] = Field(None, description="Custom metadata to include")


class ProcessingResponse(BaseModel):
    """Response model for media processing."""
    
    success: bool = Field(description="Whether processing was successful")
    message: str = Field(description="Processing message")
    output_path: Optional[str] = Field(None, description="Path to processed output file")
    
    # Processing statistics
    processing_time_seconds: float = Field(description="Total processing time in seconds")
    frames_processed: Optional[int] = Field(None, description="Number of frames processed (for videos)")
    
    # Detection statistics
    faces_detected: int = Field(0, description="Number of faces detected")
    plates_detected: int = Field(0, description="Number of license plates detected")
    text_regions_detected: int = Field(0, description="Number of text regions detected")
    names_redacted: int = Field(0, description="Number of names redacted")
    
    # File information
    input_file_size: int = Field(description="Input file size in bytes")
    output_file_size: Optional[int] = Field(None, description="Output file size in bytes")
    
    # Metadata
    metadata_path: Optional[str] = Field(None, description="Path to metadata JSON file")
    geotag_data: Optional[Dict[str, Any]] = Field(None, description="Geotag information")
    
    # Timestamps
    processing_started: datetime = Field(description="Processing start timestamp")
    processing_completed: datetime = Field(description="Processing completion timestamp")


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(description="Service status")
    timestamp: datetime = Field(description="Health check timestamp")
    version: str = Field(description="API version")
    
    # Service components status
    face_detection_available: bool = Field(description="Whether face detection is available")
    plate_detection_available: bool = Field(description="Whether plate detection is available")
    text_detection_available: bool = Field(description="Whether text detection is available")
    geotagging_available: bool = Field(description="Whether geotagging is available")
    
    # Dependencies status
    opencv_available: bool = Field(description="Whether OpenCV is available")
    tesseract_available: bool = Field(description="Whether Tesseract is available")
    easyocr_available: bool = Field(description="Whether EasyOCR is available")
    spacy_available: bool = Field(description="Whether SpaCy is available")
    yolo_available: bool = Field(description="Whether YOLO is available")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(description="Error timestamp")


class BatchProcessingRequest(BaseModel):
    """Request model for batch processing."""
    
    file_paths: List[str] = Field(description="List of file paths to process")
    processing_options: ProcessingRequest = Field(description="Processing options to apply to all files")
    output_directory: str = Field(description="Directory to save processed files")
    parallel_processing: bool = Field(False, description="Whether to process files in parallel")


class BatchProcessingResponse(BaseModel):
    """Response model for batch processing."""
    
    success: bool = Field(description="Whether batch processing was successful")
    total_files: int = Field(description="Total number of files processed")
    successful_files: int = Field(description="Number of successfully processed files")
    failed_files: int = Field(description="Number of failed files")
    
    # Processing statistics
    total_processing_time_seconds: float = Field(description="Total processing time for all files")
    average_processing_time_seconds: float = Field(description="Average processing time per file")
    
    # Results
    results: List[ProcessingResponse] = Field(description="Individual processing results")
    errors: List[ErrorResponse] = Field(description="Processing errors")
    
    # Timestamps
    batch_started: datetime = Field(description="Batch processing start timestamp")
    batch_completed: datetime = Field(description="Batch processing completion timestamp")


class StatisticsResponse(BaseModel):
    """Response model for processing statistics."""
    
    total_files_processed: int = Field(description="Total number of files processed")
    total_faces_detected: int = Field(description="Total number of faces detected")
    total_plates_detected: int = Field(description="Total number of license plates detected")
    total_text_regions_detected: int = Field(description="Total number of text regions detected")
    total_names_redacted: int = Field(description="Total number of names redacted")
    
    # Performance metrics
    average_processing_time_seconds: float = Field(description="Average processing time per file")
    total_processing_time_seconds: float = Field(description="Total processing time")
    
    # File type breakdown
    file_type_breakdown: Dict[str, int] = Field(description="Breakdown by file type")
    
    # Time range
    first_processing: Optional[datetime] = Field(None, description="First processing timestamp")
    last_processing: Optional[datetime] = Field(None, description="Last processing timestamp")
