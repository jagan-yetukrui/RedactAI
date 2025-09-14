"""
Advanced real-time video processing system for RedactAI.

This module implements sophisticated real-time video processing with
frame-by-frame analysis, temporal consistency, and adaptive processing
for optimal performance and quality.
"""

import cv2
import numpy as np
import logging
import threading
import time
import queue
from typing import List, Dict, Any, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json

from ..utils.monitoring import get_metrics_collector, start_timer, end_timer
from ..utils.cache import cached
from ..utils.logger import get_logger
from .ensemble_detector import DetectionType, EnsembleDetection, get_ensemble_detector
from .adaptive_blur import BlurRegion, BlurType, get_adaptive_blur_processor

logger = get_logger(__name__)


class ProcessingMode(Enum):
    """Video processing modes."""
    REALTIME = "realtime"
    BATCH = "batch"
    ADAPTIVE = "adaptive"
    STREAMING = "streaming"


@dataclass
class FrameMetadata:
    """Metadata for a video frame."""
    
    frame_number: int
    timestamp: float
    width: int
    height: int
    fps: float
    detections: List[EnsembleDetection] = field(default_factory=list)
    processing_time: float = 0.0
    quality_score: float = 0.0
    motion_score: float = 0.0
    blur_regions: List[BlurRegion] = field(default_factory=list)


@dataclass
class VideoProcessingConfig:
    """Configuration for video processing."""
    
    # Processing parameters
    target_fps: float = 30.0
    max_processing_fps: float = 60.0
    frame_skip_threshold: float = 0.8  # Skip frames if processing is too slow
    
    # Detection parameters
    detection_types: List[DetectionType] = field(default_factory=lambda: [
        DetectionType.FACE, DetectionType.LICENSE_PLATE, DetectionType.TEXT
    ])
    detection_interval: int = 1  # Process every N frames
    temporal_consistency: bool = True
    motion_threshold: float = 0.1
    
    # Blur parameters
    blur_types: List[BlurType] = field(default_factory=lambda: [
        BlurType.ADAPTIVE_GAUSSIAN, BlurType.PIXELATE, BlurType.CONTENT_AWARE
    ])
    adaptive_blur: bool = True
    privacy_level: int = 4  # 1-5 scale
    
    # Performance parameters
    max_workers: int = 4
    use_multiprocessing: bool = False
    buffer_size: int = 100
    memory_limit_mb: int = 2048


class TemporalConsistencyTracker:
    """Tracks temporal consistency across video frames."""
    
    def __init__(self, config: VideoProcessingConfig):
        """Initialize temporal consistency tracker."""
        self.config = config
        self.previous_detections: Dict[DetectionType, List[EnsembleDetection]] = {}
        self.detection_history: List[FrameMetadata] = []
        self.motion_vectors: List[np.ndarray] = []
        self.max_history = 10
        
    def update_detections(self, frame_metadata: FrameMetadata) -> FrameMetadata:
        """Update detections with temporal consistency."""
        if not self.config.temporal_consistency:
            return frame_metadata
        
        # Calculate motion between frames
        motion_score = self._calculate_motion_score(frame_metadata)
        frame_metadata.motion_score = motion_score
        
        # Apply temporal smoothing to detections
        smoothed_detections = {}
        for detection_type in self.config.detection_types:
            current_detections = [d for d in frame_metadata.detections if d.detection_type == detection_type]
            smoothed = self._smooth_detections(detection_type, current_detections, motion_score)
            smoothed_detections[detection_type] = smoothed
        
        # Update frame metadata
        frame_metadata.detections = []
        for detections in smoothed_detections.values():
            frame_metadata.detections.extend(detections)
        
        # Update history
        self.detection_history.append(frame_metadata)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
        
        return frame_metadata
    
    def _calculate_motion_score(self, frame_metadata: FrameMetadata) -> float:
        """Calculate motion score between current and previous frame."""
        if len(self.detection_history) < 2:
            return 0.0
        
        prev_frame = self.detection_history[-1]
        current_detections = frame_metadata.detections
        prev_detections = prev_frame.detections
        
        if not current_detections or not prev_detections:
            return 0.0
        
        # Calculate average displacement of detections
        total_displacement = 0.0
        matched_detections = 0
        
        for curr_det in current_detections:
            best_match = None
            min_distance = float('inf')
            
            for prev_det in prev_detections:
                if curr_det.detection_type == prev_det.detection_type:
                    # Calculate center distance
                    curr_center = curr_det.bbox[0] + curr_det.bbox[2]//2, curr_det.bbox[1] + curr_det.bbox[3]//2
                    prev_center = prev_det.bbox[0] + prev_det.bbox[2]//2, prev_det.bbox[1] + prev_det.bbox[3]//2
                    
                    distance = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_match = prev_det
            
            if best_match and min_distance < 100:  # Reasonable movement threshold
                total_displacement += min_distance
                matched_detections += 1
        
        if matched_detections == 0:
            return 1.0  # High motion if no matches
        
        avg_displacement = total_displacement / matched_detections
        motion_score = min(avg_displacement / 50.0, 1.0)  # Normalize to 0-1
        
        return motion_score
    
    def _smooth_detections(self, detection_type: DetectionType, 
                          current_detections: List[EnsembleDetection],
                          motion_score: float) -> List[EnsembleDetection]:
        """Smooth detections using temporal information."""
        if not current_detections:
            return []
        
        # Get previous detections of same type
        prev_detections = self.previous_detections.get(detection_type, [])
        
        if not prev_detections or motion_score > self.config.motion_threshold:
            # High motion or no previous detections - use current as-is
            self.previous_detections[detection_type] = current_detections
            return current_detections
        
        # Apply temporal smoothing
        smoothed_detections = []
        
        for curr_det in current_detections:
            # Find best matching previous detection
            best_match = None
            min_distance = float('inf')
            
            for prev_det in prev_detections:
                distance = self._calculate_detection_distance(curr_det, prev_det)
                if distance < min_distance:
                    min_distance = distance
                    best_match = prev_det
            
            if best_match and min_distance < 50:  # Close enough to smooth
                # Interpolate position and confidence
                smoothed_det = self._interpolate_detection(curr_det, best_match, motion_score)
                smoothed_detections.append(smoothed_det)
            else:
                # No good match - use current detection
                smoothed_detections.append(curr_det)
        
        # Update previous detections
        self.previous_detections[detection_type] = current_detections
        
        return smoothed_detections
    
    def _calculate_detection_distance(self, det1: EnsembleDetection, 
                                    det2: EnsembleDetection) -> float:
        """Calculate distance between two detections."""
        center1 = det1.bbox[0] + det1.bbox[2]//2, det1.bbox[1] + det1.bbox[3]//2
        center2 = det2.bbox[0] + det2.bbox[2]//2, det2.bbox[1] + det2.bbox[3]//2
        
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _interpolate_detection(self, current: EnsembleDetection, 
                              previous: EnsembleDetection,
                              motion_score: float) -> EnsembleDetection:
        """Interpolate between current and previous detection."""
        # Interpolation factor based on motion
        alpha = 0.3 + motion_score * 0.4  # Range: 0.3-0.7
        
        # Interpolate bounding box
        curr_bbox = current.bbox
        prev_bbox = previous.bbox
        
        new_bbox = (
            int(curr_bbox[0] * alpha + prev_bbox[0] * (1 - alpha)),
            int(curr_bbox[1] * alpha + prev_bbox[1] * (1 - alpha)),
            int(curr_bbox[2] * alpha + prev_bbox[2] * (1 - alpha)),
            int(curr_bbox[3] * alpha + prev_bbox[3] * (1 - alpha))
        )
        
        # Interpolate confidence
        new_confidence = current.confidence * alpha + previous.confidence * (1 - alpha)
        
        # Create new detection
        smoothed_detection = EnsembleDetection(
            detection_type=current.detection_type,
            confidence=new_confidence,
            bbox=new_bbox,
            contributing_models=current.contributing_models,
            individual_results=current.individual_results,
            ensemble_score=current.ensemble_score,
            processing_time=current.processing_time,
            metadata=current.metadata
        )
        
        return smoothed_detection


class RealTimeVideoProcessor:
    """Advanced real-time video processor."""
    
    def __init__(self, config: VideoProcessingConfig = None):
        """Initialize real-time video processor."""
        self.config = config or VideoProcessingConfig()
        self.ensemble_detector = get_ensemble_detector()
        self.adaptive_blur_processor = get_adaptive_blur_processor()
        self.temporal_tracker = TemporalConsistencyTracker(config)
        self.metrics_collector = get_metrics_collector()
        
        # Processing state
        self.is_processing = False
        self.current_frame = 0
        self.start_time = 0.0
        self.total_frames = 0
        
        # Performance tracking
        self.frame_times = []
        self.detection_times = []
        self.blur_times = []
        
        # Threading
        self.processing_thread = None
        self.frame_queue = queue.Queue(maxsize=self.config.buffer_size)
        self.result_queue = queue.Queue(maxsize=self.config.buffer_size)
        
        logger.info("Real-time video processor initialized")
    
    def process_video_file(self, input_path: Path, output_path: Path,
                          progress_callback: Optional[Callable[[float], None]] = None) -> Dict[str, Any]:
        """Process a video file with real-time optimization."""
        start_timer("video_processing")
        
        # Open video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Processing statistics
        stats = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'skipped_frames': 0,
            'total_detections': 0,
            'processing_time': 0.0,
            'average_fps': 0.0,
            'detection_types': {dt.value: 0 for dt in self.config.detection_types}
        }
        
        self.is_processing = True
        self.start_time = time.time()
        
        try:
            frame_count = 0
            last_detection_frame = -1
            
            while self.is_processing and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start_time = time.time()
                
                # Determine if we should run detection on this frame
                should_detect = (
                    frame_count % self.config.detection_interval == 0 or
                    frame_count - last_detection_frame > 30  # Force detection every 30 frames
                )
                
                # Process frame
                processed_frame, frame_metadata = self._process_frame(
                    frame, frame_count, should_detect
                )
                
                # Write processed frame
                out.write(processed_frame)
                
                # Update statistics
                frame_time = time.time() - frame_start_time
                self.frame_times.append(frame_time)
                
                stats['processed_frames'] += 1
                stats['total_detections'] += len(frame_metadata.detections)
                
                for detection in frame_metadata.detections:
                    stats['detection_types'][detection.detection_type.value] += 1
                
                if should_detect:
                    last_detection_frame = frame_count
                
                # Update progress
                if progress_callback:
                    progress = frame_count / total_frames
                    progress_callback(progress)
                
                # Check if we're falling behind
                if frame_time > 1.0 / self.config.target_fps:
                    stats['skipped_frames'] += 1
                    if frame_count % 10 == 0:
                        logger.warning(f"Processing falling behind at frame {frame_count}")
                
                frame_count += 1
                
                # Memory management
                if len(self.frame_times) > 1000:
                    self.frame_times = self.frame_times[-500:]
        
        finally:
            self.is_processing = False
            cap.release()
            out.release()
            
            # Calculate final statistics
            total_time = time.time() - self.start_time
            stats['processing_time'] = total_time
            stats['average_fps'] = stats['processed_frames'] / total_time if total_time > 0 else 0
            
            end_timer("video_processing")
            
            # Record metrics
            self.metrics_collector.record_processing(
                processing_time=total_time,
                file_type='.mp4',
                frames_processed=stats['processed_frames'],
                total_detections=stats['total_detections']
            )
        
        logger.info(f"Video processing completed: {stats['processed_frames']} frames in {total_time:.2f}s")
        return stats
    
    def _process_frame(self, frame: np.ndarray, frame_number: int, 
                      should_detect: bool) -> Tuple[np.ndarray, FrameMetadata]:
        """Process a single frame."""
        start_time = time.time()
        
        # Create frame metadata
        frame_metadata = FrameMetadata(
            frame_number=frame_number,
            timestamp=frame_number / 30.0,  # Assuming 30 FPS
            width=frame.shape[1],
            height=frame.shape[0],
            fps=30.0
        )
        
        # Run detection if needed
        if should_detect:
            detection_start = time.time()
            detections = self.ensemble_detector.detect_all(
                frame, self.config.detection_types
            )
            detection_time = time.time() - detection_start
            self.detection_times.append(detection_time)
            
            # Flatten detections
            all_detections = []
            for detection_list in detections.values():
                all_detections.extend(detection_list)
            
            frame_metadata.detections = all_detections
        else:
            # Use previous frame's detections with temporal smoothing
            frame_metadata = self.temporal_tracker.update_detections(frame_metadata)
        
        # Convert detections to blur regions
        blur_regions = self._create_blur_regions(frame_metadata.detections)
        frame_metadata.blur_regions = blur_regions
        
        # Apply adaptive blurring
        if blur_regions:
            blur_start = time.time()
            processed_frame = self.adaptive_blur_processor.process_image(frame, blur_regions)[0]
            blur_time = time.time() - blur_start
            self.blur_times.append(blur_time)
        else:
            processed_frame = frame.copy()
        
        # Update frame metadata
        frame_metadata.processing_time = time.time() - start_time
        
        return processed_frame, frame_metadata
    
    def _create_blur_regions(self, detections: List[EnsembleDetection]) -> List[BlurRegion]:
        """Convert detections to blur regions."""
        blur_regions = []
        
        for detection in detections:
            # Determine blur type based on detection type
            if detection.detection_type == DetectionType.FACE:
                blur_type = BlurType.ADAPTIVE_GAUSSIAN
                privacy_level = 5
            elif detection.detection_type == DetectionType.LICENSE_PLATE:
                blur_type = BlurType.PIXELATE
                privacy_level = 4
            elif detection.detection_type == DetectionType.TEXT:
                blur_type = BlurType.CONTENT_AWARE
                privacy_level = 3
            else:
                blur_type = BlurType.GAUSSIAN
                privacy_level = 3
            
            # Calculate adaptive blur strength
            base_strength = 15.0
            confidence_multiplier = 0.5 + detection.confidence * 0.5
            blur_strength = base_strength * confidence_multiplier
            
            # Create blur region
            blur_region = BlurRegion(
                bbox=detection.bbox,
                blur_type=blur_type,
                blur_strength=blur_strength,
                confidence=detection.confidence,
                content_type=detection.detection_type.value,
                privacy_level=privacy_level,
                context={'ensemble_score': detection.ensemble_score}
            )
            
            blur_regions.append(blur_region)
        
        return blur_regions
    
    def process_live_stream(self, source: Union[int, str], 
                           output_callback: Callable[[np.ndarray], None]) -> None:
        """Process live video stream."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        self.is_processing = True
        frame_count = 0
        
        try:
            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, _ = self._process_frame(frame, frame_count, True)
                
                # Send to output callback
                output_callback(processed_frame)
                
                frame_count += 1
                
                # Check for exit condition
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            self.is_processing = False
    
    def stop_processing(self):
        """Stop video processing."""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            'frame_times': {
                'average': np.mean(self.frame_times) if self.frame_times else 0,
                'min': np.min(self.frame_times) if self.frame_times else 0,
                'max': np.max(self.frame_times) if self.frame_times else 0,
                'total_frames': len(self.frame_times)
            },
            'detection_times': {
                'average': np.mean(self.detection_times) if self.detection_times else 0,
                'total_detections': len(self.detection_times)
            },
            'blur_times': {
                'average': np.mean(self.blur_times) if self.blur_times else 0,
                'total_blurs': len(self.blur_times)
            },
            'processing_state': {
                'is_processing': self.is_processing,
                'current_frame': self.current_frame,
                'queue_size': self.frame_queue.qsize()
            }
        }
        
        return stats


# Global video processor instance
_video_processor: Optional[RealTimeVideoProcessor] = None


def get_video_processor() -> RealTimeVideoProcessor:
    """Get the global video processor instance."""
    global _video_processor
    if _video_processor is None:
        _video_processor = RealTimeVideoProcessor()
    return _video_processor


def process_video_file(input_path: Path, output_path: Path,
                      progress_callback: Optional[Callable[[float], None]] = None) -> Dict[str, Any]:
    """Process a video file with real-time optimization."""
    processor = get_video_processor()
    return processor.process_video_file(input_path, output_path, progress_callback)
