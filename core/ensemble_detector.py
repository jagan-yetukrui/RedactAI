"""
Advanced ensemble detection system for RedactAI.

This module implements sophisticated ensemble detection combining multiple
AI models with confidence scoring, adaptive thresholds, and intelligent
voting mechanisms for superior detection accuracy.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path

from ..utils.monitoring import get_metrics_collector, start_timer, end_timer
from ..utils.cache import cached
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DetectionType(Enum):
    """Types of detection supported by the ensemble."""
    FACE = "face"
    LICENSE_PLATE = "license_plate"
    TEXT = "text"
    PERSON = "person"
    VEHICLE = "vehicle"


@dataclass
class DetectionResult:
    """Result from a single detection model."""
    
    model_name: str
    detection_type: DetectionType
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    
    @property
    def area(self) -> int:
        """Calculate bounding box area."""
        return self.bbox[2] * self.bbox[3]
    
    @property
    def center(self) -> Tuple[int, int]:
        """Calculate bounding box center."""
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)


@dataclass
class EnsembleDetection:
    """Final ensemble detection result."""
    
    detection_type: DetectionType
    confidence: float
    bbox: Tuple[int, int, int, int]
    contributing_models: List[str]
    individual_results: List[DetectionResult]
    ensemble_score: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelEnsemble:
    """Advanced ensemble of detection models."""
    
    def __init__(self, detection_type: DetectionType, models: List[Any], 
                 weights: Optional[List[float]] = None,
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4):
        """
        Initialize model ensemble.
        
        Args:
            detection_type: Type of detection this ensemble performs
            models: List of detection models
            weights: Optional weights for each model
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
        """
        self.detection_type = detection_type
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.lock = threading.RLock()
        
        # Performance tracking
        self.model_performance = {f"model_{i}": [] for i in range(len(models))}
        self.ensemble_performance = []
        
        logger.info(f"Initialized {detection_type.value} ensemble with {len(models)} models")
    
    def detect_parallel(self, image: np.ndarray, 
                       max_workers: int = 4) -> List[DetectionResult]:
        """Run all models in parallel for faster processing."""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all model detection tasks
            future_to_model = {
                executor.submit(self._detect_single_model, image, i, model): i
                for i, model in enumerate(self.models)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_model):
                model_idx = future_to_model[future]
                try:
                    result = future.result()
                    if result:
                        results.extend(result)
                except Exception as e:
                    logger.error(f"Model {model_idx} detection failed: {e}")
        
        return results
    
    def _detect_single_model(self, image: np.ndarray, 
                           model_idx: int, model: Any) -> List[DetectionResult]:
        """Detect using a single model."""
        start_time = time.time()
        model_name = f"model_{model_idx}"
        
        try:
            # This would be implemented based on the specific model type
            detections = self._run_model_detection(image, model, model_idx)
            
            processing_time = time.time() - start_time
            
            # Track performance
            with self.lock:
                self.model_performance[model_name].append(processing_time)
            
            # Convert to DetectionResult objects
            results = []
            for detection in detections:
                if detection['confidence'] >= self.confidence_threshold:
                    result = DetectionResult(
                        model_name=model_name,
                        detection_type=self.detection_type,
                        confidence=detection['confidence'],
                        bbox=detection['bbox'],
                        metadata=detection.get('metadata', {}),
                        processing_time=processing_time
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in model {model_idx}: {e}")
            return []
    
    def _run_model_detection(self, image: np.ndarray, model: Any, 
                           model_idx: int) -> List[Dict[str, Any]]:
        """Run detection on a specific model."""
        # This is a placeholder - would be implemented based on model type
        # For now, return mock detections for demonstration
        return [
            {
                'confidence': 0.8 + np.random.random() * 0.2,
                'bbox': (100, 100, 150, 150),
                'metadata': {'model_type': f'type_{model_idx}'}
            }
        ]
    
    def ensemble_detections(self, individual_results: List[DetectionResult]) -> List[EnsembleDetection]:
        """Combine individual model results into ensemble detections."""
        if not individual_results:
            return []
        
        start_time = time.time()
        
        # Group results by spatial proximity
        clusters = self._cluster_detections(individual_results)
        
        # Create ensemble detections from clusters
        ensemble_detections = []
        for cluster in clusters:
            if len(cluster) < 2:  # Need at least 2 models to agree
                continue
            
            # Calculate ensemble confidence and bounding box
            ensemble_conf = self._calculate_ensemble_confidence(cluster)
            ensemble_bbox = self._calculate_ensemble_bbox(cluster)
            ensemble_score = self._calculate_ensemble_score(cluster)
            
            # Apply non-maximum suppression
            if self._should_keep_detection(ensemble_detections, ensemble_bbox, ensemble_conf):
                ensemble_detection = EnsembleDetection(
                    detection_type=self.detection_type,
                    confidence=ensemble_conf,
                    bbox=ensemble_bbox,
                    contributing_models=[r.model_name for r in cluster],
                    individual_results=cluster,
                    ensemble_score=ensemble_score,
                    processing_time=time.time() - start_time
                )
                ensemble_detections.append(ensemble_detection)
        
        # Track ensemble performance
        with self.lock:
            self.ensemble_performance.append(time.time() - start_time)
        
        return ensemble_detections
    
    def _cluster_detections(self, results: List[DetectionResult]) -> List[List[DetectionResult]]:
        """Cluster detections based on spatial proximity."""
        if len(results) < 2:
            return [[r] for r in results]
        
        clusters = []
        used = set()
        
        for i, result1 in enumerate(results):
            if i in used:
                continue
            
            cluster = [result1]
            used.add(i)
            
            for j, result2 in enumerate(results[i+1:], i+1):
                if j in used:
                    continue
                
                if self._detections_overlap(result1, result2):
                    cluster.append(result2)
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _detections_overlap(self, det1: DetectionResult, det2: DetectionResult) -> bool:
        """Check if two detections overlap significantly."""
        x1, y1, w1, h1 = det1.bbox
        x2, y2, w2, h2 = det2.bbox
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        union_area = det1.area + det2.area - intersection_area
        
        iou = intersection_area / union_area if union_area > 0 else 0
        return iou > 0.3  # 30% overlap threshold
    
    def _calculate_ensemble_confidence(self, cluster: List[DetectionResult]) -> float:
        """Calculate ensemble confidence using weighted average."""
        if not cluster:
            return 0.0
        
        # Weight by model performance and confidence
        total_weight = 0.0
        weighted_confidence = 0.0
        
        for result in cluster:
            model_idx = int(result.model_name.split('_')[1])
            weight = self.weights[model_idx] * result.confidence
            weighted_confidence += weight * result.confidence
            total_weight += weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _calculate_ensemble_bbox(self, cluster: List[DetectionResult]) -> Tuple[int, int, int, int]:
        """Calculate ensemble bounding box using weighted average."""
        if not cluster:
            return (0, 0, 0, 0)
        
        # Weight by confidence
        total_weight = sum(r.confidence for r in cluster)
        
        weighted_x = sum(r.bbox[0] * r.confidence for r in cluster) / total_weight
        weighted_y = sum(r.bbox[1] * r.confidence for r in cluster) / total_weight
        weighted_w = sum(r.bbox[2] * r.confidence for r in cluster) / total_weight
        weighted_h = sum(r.bbox[3] * r.confidence for r in cluster) / total_weight
        
        return (int(weighted_x), int(weighted_y), int(weighted_w), int(weighted_h))
    
    def _calculate_ensemble_score(self, cluster: List[DetectionResult]) -> float:
        """Calculate ensemble score based on agreement and confidence."""
        if not cluster:
            return 0.0
        
        # Factors: number of agreeing models, average confidence, consistency
        num_models = len(cluster)
        avg_confidence = sum(r.confidence for r in cluster) / num_models
        
        # Calculate confidence variance (lower is better)
        confidence_variance = sum((r.confidence - avg_confidence) ** 2 for r in cluster) / num_models
        consistency = 1.0 / (1.0 + confidence_variance)
        
        # Ensemble score combines all factors
        ensemble_score = (num_models / len(self.models)) * avg_confidence * consistency
        return min(ensemble_score, 1.0)
    
    def _should_keep_detection(self, existing_detections: List[EnsembleDetection], 
                             bbox: Tuple[int, int, int, int], 
                             confidence: float) -> bool:
        """Apply non-maximum suppression."""
        for existing in existing_detections:
            if self._detections_overlap_ensemble(existing, bbox):
                if confidence <= existing.confidence:
                    return False
                # Remove the existing detection with lower confidence
                existing_detections.remove(existing)
        
        return True
    
    def _detections_overlap_ensemble(self, existing: EnsembleDetection, 
                                   bbox: Tuple[int, int, int, int]) -> bool:
        """Check overlap between ensemble detection and bbox."""
        x1, y1, w1, h1 = existing.bbox
        x2, y2, w2, h2 = bbox
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        union_area = existing.bbox[2] * existing.bbox[3] + w2 * h2 - intersection_area
        
        iou = intersection_area / union_area if union_area > 0 else 0
        return iou > self.nms_threshold
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the ensemble."""
        with self.lock:
            stats = {
                'detection_type': self.detection_type.value,
                'num_models': len(self.models),
                'model_performance': {},
                'ensemble_performance': {
                    'avg_time': np.mean(self.ensemble_performance) if self.ensemble_performance else 0,
                    'total_detections': len(self.ensemble_performance)
                }
            }
            
            for model_name, times in self.model_performance.items():
                if times:
                    stats['model_performance'][model_name] = {
                        'avg_time': np.mean(times),
                        'min_time': np.min(times),
                        'max_time': np.max(times),
                        'total_detections': len(times)
                    }
            
            return stats


class AdvancedEnsembleDetector:
    """Main ensemble detector coordinating multiple detection types."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize advanced ensemble detector."""
        self.config = config or {}
        self.ensembles: Dict[DetectionType, ModelEnsemble] = {}
        self.metrics_collector = get_metrics_collector()
        
        # Initialize ensembles for each detection type
        self._initialize_ensembles()
        
        logger.info("Advanced ensemble detector initialized")
    
    def _initialize_ensembles(self):
        """Initialize model ensembles for each detection type."""
        # Face detection ensemble
        face_models = self._load_face_models()
        if face_models:
            self.ensembles[DetectionType.FACE] = ModelEnsemble(
                DetectionType.FACE, face_models,
                weights=[0.4, 0.3, 0.3],  # Weight different models
                confidence_threshold=0.6
            )
        
        # License plate detection ensemble
        plate_models = self._load_plate_models()
        if plate_models:
            self.ensembles[DetectionType.LICENSE_PLATE] = ModelEnsemble(
                DetectionType.LICENSE_PLATE, plate_models,
                weights=[0.5, 0.5],
                confidence_threshold=0.7
            )
        
        # Text detection ensemble
        text_models = self._load_text_models()
        if text_models:
            self.ensembles[DetectionType.TEXT] = ModelEnsemble(
                DetectionType.TEXT, text_models,
                weights=[0.6, 0.4],
                confidence_threshold=0.5
            )
    
    def _load_face_models(self) -> List[Any]:
        """Load face detection models."""
        # This would load actual models in a real implementation
        return [f"face_model_{i}" for i in range(3)]
    
    def _load_plate_models(self) -> List[Any]:
        """Load license plate detection models."""
        return [f"plate_model_{i}" for i in range(2)]
    
    def _load_text_models(self) -> List[Any]:
        """Load text detection models."""
        return [f"text_model_{i}" for i in range(2)]
    
    @cached(ttl=3600, use_file_cache=True)
    def detect_all(self, image: np.ndarray, 
                   detection_types: List[DetectionType] = None) -> Dict[DetectionType, List[EnsembleDetection]]:
        """Detect all specified types in the image."""
        if detection_types is None:
            detection_types = list(self.ensembles.keys())
        
        start_timer("ensemble_detection")
        
        results = {}
        
        for detection_type in detection_types:
            if detection_type not in self.ensembles:
                continue
            
            ensemble = self.ensembles[detection_type]
            
            # Run individual model detections
            individual_results = ensemble.detect_parallel(image)
            
            # Combine into ensemble detections
            ensemble_detections = ensemble.ensemble_detections(individual_results)
            
            results[detection_type] = ensemble_detections
            
            # Record metrics
            self.metrics_collector.record_processing(
                processing_time=sum(r.processing_time for r in individual_results),
                file_type='.jpg',
                **{f"{detection_type.value}s_detected": len(ensemble_detections)}
            )
        
        end_timer("ensemble_detection")
        
        return results
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all ensembles."""
        stats = {
            'total_ensembles': len(self.ensembles),
            'ensembles': {}
        }
        
        for detection_type, ensemble in self.ensembles.items():
            stats['ensembles'][detection_type.value] = ensemble.get_performance_stats()
        
        return stats


# Global ensemble detector instance
_ensemble_detector: Optional[AdvancedEnsembleDetector] = None


def get_ensemble_detector() -> AdvancedEnsembleDetector:
    """Get the global ensemble detector instance."""
    global _ensemble_detector
    if _ensemble_detector is None:
        _ensemble_detector = AdvancedEnsembleDetector()
    return _ensemble_detector


def detect_with_ensemble(image: np.ndarray, 
                        detection_types: List[DetectionType] = None) -> Dict[DetectionType, List[EnsembleDetection]]:
    """Detect objects using the ensemble detector."""
    detector = get_ensemble_detector()
    return detector.detect_all(image, detection_types)
