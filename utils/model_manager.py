"""
Advanced model management system for RedactAI.

This module provides intelligent model loading, caching, and management
for all AI models used in the RedactAI system.
"""

import os
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import threading
import time

from .cache import get_cache_manager, cached
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    
    name: str
    type: str
    version: str
    path: Path
    size_bytes: int
    loaded_at: datetime
    last_used: datetime
    use_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'type': self.type,
            'version': self.version,
            'path': str(self.path),
            'size_bytes': self.size_bytes,
            'loaded_at': self.loaded_at.isoformat(),
            'last_used': self.last_used.isoformat(),
            'use_count': self.use_count,
            'metadata': self.metadata
        }


class ModelManager:
    """Advanced model manager for RedactAI."""
    
    def __init__(self, models_dir: Path = None, cache_models: bool = True):
        """
        Initialize model manager.
        
        Args:
            models_dir: Directory to store models
            cache_models: Whether to cache loaded models
        """
        self.models_dir = models_dir or Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_models = cache_models
        
        # Model registry
        self.loaded_models: Dict[str, Any] = {}
        self.model_info: Dict[str, ModelInfo] = {}
        self.model_loaders: Dict[str, Callable] = {}
        
        # Threading
        self.lock = threading.RLock()
        
        # Cache manager
        self.cache_manager = get_cache_manager() if cache_models else None
        
        # Initialize default model loaders
        self._setup_default_loaders()
    
    def _setup_default_loaders(self) -> None:
        """Setup default model loaders."""
        # Face detection models
        self.register_model_loader("haar_face", self._load_haar_face_model)
        self.register_model_loader("dnn_face", self._load_dnn_face_model)
        
        # License plate detection models
        self.register_model_loader("yolo_plate", self._load_yolo_plate_model)
        self.register_model_loader("opencv_plate", self._load_opencv_plate_model)
        
        # Text detection models
        self.register_model_loader("tesseract", self._load_tesseract_model)
        self.register_model_loader("easyocr", self._load_easyocr_model)
        self.register_model_loader("spacy_ner", self._load_spacy_ner_model)
    
    def register_model_loader(self, model_type: str, loader_func: Callable) -> None:
        """Register a model loader function."""
        self.model_loaders[model_type] = loader_func
        logger.info(f"Registered model loader for type: {model_type}")
    
    def _load_haar_face_model(self, model_path: Path) -> Any:
        """Load Haar cascade face detection model."""
        import cv2
        
        if not model_path.exists():
            # Use default OpenCV cascade
            model_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        
        cascade = cv2.CascadeClassifier(str(model_path))
        if cascade.empty():
            raise ValueError(f"Failed to load Haar cascade from {model_path}")
        
        return cascade
    
    def _load_dnn_face_model(self, model_path: Path) -> Any:
        """Load DNN face detection model."""
        import cv2
        
        if not model_path.exists():
            raise FileNotFoundError(f"DNN model not found: {model_path}")
        
        # Load DNN model
        net = cv2.dnn.readNetFromTensorflow(str(model_path))
        return net
    
    def _load_yolo_plate_model(self, model_path: Path) -> Any:
        """Load YOLO license plate detection model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics not available for YOLO model loading")
        
        if not model_path.exists():
            # Use default YOLOv8 model
            model = YOLO('yolov8n.pt')
        else:
            model = YOLO(str(model_path))
        
        return model
    
    def _load_opencv_plate_model(self, model_path: Path) -> Any:
        """Load OpenCV-based license plate detection model."""
        # This would implement OpenCV-based plate detection
        # For now, return a placeholder
        return {"type": "opencv_plate", "path": str(model_path)}
    
    def _load_tesseract_model(self, model_path: Path) -> Any:
        """Load Tesseract OCR model."""
        try:
            import pytesseract
        except ImportError:
            raise ImportError("pytesseract not available for Tesseract model loading")
        
        # Tesseract doesn't need explicit model loading
        # Just verify it's available
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise RuntimeError(f"Tesseract not available: {e}")
        
        return {"type": "tesseract", "available": True}
    
    def _load_easyocr_model(self, model_path: Path) -> Any:
        """Load EasyOCR model."""
        try:
            import easyocr
        except ImportError:
            raise ImportError("easyocr not available for EasyOCR model loading")
        
        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en'])
        return reader
    
    def _load_spacy_ner_model(self, model_path: Path) -> Any:
        """Load SpaCy NER model."""
        try:
            import spacy
        except ImportError:
            raise ImportError("spacy not available for NER model loading")
        
        if not model_path.exists():
            # Use default model
            model_name = "en_core_web_sm"
        else:
            model_name = str(model_path)
        
        try:
            nlp = spacy.load(model_name)
        except OSError:
            raise RuntimeError(f"SpaCy model not found: {model_name}")
        
        return nlp
    
    def load_model(self, model_name: str, model_type: str, 
                   model_path: Optional[Path] = None, 
                   force_reload: bool = False) -> Any:
        """
        Load a model with caching and management.
        
        Args:
            model_name: Name of the model
            model_type: Type of the model
            model_path: Path to model file
            force_reload: Force reload even if already loaded
            
        Returns:
            Loaded model object
        """
        with self.lock:
            # Check if model is already loaded
            if not force_reload and model_name in self.loaded_models:
                self.model_info[model_name].last_used = datetime.now(timezone.utc)
                self.model_info[model_name].use_count += 1
                return self.loaded_models[model_name]
            
            # Check cache if enabled
            if self.cache_models:
                cache_key = f"model_{model_name}_{model_type}"
                cached_model = self.cache_manager.get(cache_key)
                if cached_model is not None:
                    logger.info(f"Loaded model {model_name} from cache")
                    self.loaded_models[model_name] = cached_model
                    self.model_info[model_name] = ModelInfo(
                        name=model_name,
                        type=model_type,
                        version="cached",
                        path=model_path or Path("cached"),
                        size_bytes=0,
                        loaded_at=datetime.now(timezone.utc),
                        last_used=datetime.now(timezone.utc)
                    )
                    return cached_model
            
            # Load model using registered loader
            if model_type not in self.model_loaders:
                raise ValueError(f"No loader registered for model type: {model_type}")
            
            loader_func = self.model_loaders[model_type]
            
            try:
                start_time = time.time()
                model = loader_func(model_path)
                load_time = time.time() - start_time
                
                # Get model size
                model_size = 0
                if model_path and model_path.exists():
                    model_size = model_path.stat().st_size
                
                # Store model info
                model_info = ModelInfo(
                    name=model_name,
                    type=model_type,
                    version="1.0.0",  # Could be extracted from model metadata
                    path=model_path or Path("unknown"),
                    size_bytes=model_size,
                    loaded_at=datetime.now(timezone.utc),
                    last_used=datetime.now(timezone.utc),
                    metadata={"load_time": load_time}
                )
                
                # Store model
                self.loaded_models[model_name] = model
                self.model_info[model_name] = model_info
                
                # Cache model if enabled
                if self.cache_models:
                    self.cache_manager.set(cache_key, model, ttl=3600)  # Cache for 1 hour
                
                logger.info(f"Loaded model {model_name} ({model_type}) in {load_time:.2f}s")
                return model
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory."""
        with self.lock:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
                del self.model_info[model_name]
                logger.info(f"Unloaded model {model_name}")
                return True
            return False
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a loaded model."""
        with self.lock:
            if model_name in self.loaded_models:
                self.model_info[model_name].last_used = datetime.now(timezone.utc)
                self.model_info[model_name].use_count += 1
                return self.loaded_models[model_name]
            return None
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a loaded model."""
        return self.model_info.get(model_name)
    
    def list_loaded_models(self) -> List[str]:
        """List all loaded model names."""
        with self.lock:
            return list(self.loaded_models.keys())
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded models."""
        with self.lock:
            if not self.model_info:
                return {
                    'total_models': 0,
                    'total_size_bytes': 0,
                    'models': []
                }
            
            total_size = sum(info.size_bytes for info in self.model_info.values())
            total_use_count = sum(info.use_count for info in self.model_info.values())
            
            return {
                'total_models': len(self.model_info),
                'total_size_bytes': total_size,
                'total_use_count': total_use_count,
                'models': [info.to_dict() for info in self.model_info.values()]
            }
    
    def cleanup_unused_models(self, max_age_hours: int = 24) -> int:
        """Clean up unused models older than specified age."""
        with self.lock:
            current_time = datetime.now(timezone.utc)
            cutoff_time = current_time - timedelta(hours=max_age_hours)
            
            models_to_remove = []
            for name, info in self.model_info.items():
                if info.last_used < cutoff_time:
                    models_to_remove.append(name)
            
            for name in models_to_remove:
                self.unload_model(name)
            
            logger.info(f"Cleaned up {len(models_to_remove)} unused models")
            return len(models_to_remove)
    
    def export_model_info(self, output_path: Path) -> None:
        """Export model information to file."""
        with self.lock:
            model_data = {
                'exported_at': datetime.now(timezone.utc).isoformat(),
                'stats': self.get_model_stats(),
                'models': [info.to_dict() for info in self.model_info.values()]
            }
            
            with open(output_path, 'w') as f:
                json.dump(model_data, f, indent=2, default=str)
            
            logger.info(f"Model info exported to {output_path}")
    
    def clear_all_models(self) -> None:
        """Clear all loaded models."""
        with self.lock:
            self.loaded_models.clear()
            self.model_info.clear()
            logger.info("All models cleared")


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def load_model(model_name: str, model_type: str, 
               model_path: Optional[Path] = None, 
               force_reload: bool = False) -> Any:
    """Load a model using the global model manager."""
    return get_model_manager().load_model(model_name, model_type, model_path, force_reload)


def get_model(model_name: str) -> Optional[Any]:
    """Get a loaded model."""
    return get_model_manager().get_model(model_name)


def unload_model(model_name: str) -> bool:
    """Unload a model."""
    return get_model_manager().unload_model(model_name)


def get_model_stats() -> Dict[str, Any]:
    """Get model statistics."""
    return get_model_manager().get_model_stats()


def cleanup_unused_models(max_age_hours: int = 24) -> int:
    """Clean up unused models."""
    return get_model_manager().cleanup_unused_models(max_age_hours)


@cached(ttl=3600, use_file_cache=True)
def get_cached_model(model_name: str, model_type: str, model_path: str) -> Any:
    """Get a cached model with automatic loading."""
    return load_model(model_name, model_type, Path(model_path) if model_path else None)
