"""
Advanced GPU acceleration system for RedactAI.

This module provides sophisticated GPU acceleration for AI model inference,
image processing, and parallel computation to maximize performance.
"""

import numpy as np
import cv2
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from pathlib import Path
import json

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    cp_ndimage = None

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    transforms = None

from ..utils.monitoring import get_metrics_collector, start_timer, end_timer
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GPUDevice(Enum):
    """Available GPU devices."""
    CUDA = "cuda"
    OPENCL = "opencl"
    CPU = "cpu"


@dataclass
class GPUConfig:
    """GPU acceleration configuration."""
    
    # Device selection
    preferred_device: GPUDevice = GPUDevice.CUDA
    fallback_to_cpu: bool = True
    memory_fraction: float = 0.8  # Use 80% of GPU memory
    
    # Performance settings
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    
    # Model settings
    use_mixed_precision: bool = True
    optimize_for_inference: bool = True
    
    # Memory management
    clear_cache_interval: int = 100  # Clear cache every N operations
    max_memory_usage: int = 8 * 1024 * 1024 * 1024  # 8GB


class GPUManager:
    """Advanced GPU resource manager."""
    
    def __init__(self, config: GPUConfig = None):
        """Initialize GPU manager."""
        self.config = config or GPUConfig()
        self.device_info = self._detect_devices()
        self.current_device = self._select_device()
        self.memory_usage = 0
        self.operation_count = 0
        self.lock = threading.Lock()
        
        # Initialize device-specific resources
        self._initialize_device()
        
        logger.info(f"GPU Manager initialized with device: {self.current_device}")
    
    def _detect_devices(self) -> Dict[str, Any]:
        """Detect available GPU devices."""
        devices = {
            'cuda': False,
            'opencl': False,
            'cpu': True,
            'cuda_devices': [],
            'opencl_devices': []
        }
        
        # Check CUDA availability
        if CUPY_AVAILABLE:
            try:
                devices['cuda'] = True
                devices['cuda_devices'] = list(range(cp.cuda.runtime.getDeviceCount()))
                logger.info(f"CUDA devices detected: {devices['cuda_devices']}")
            except Exception as e:
                logger.warning(f"CUDA detection failed: {e}")
        
        # Check PyTorch CUDA
        if TORCH_AVAILABLE and torch.cuda.is_available():
            devices['cuda'] = True
            devices['cuda_devices'] = list(range(torch.cuda.device_count()))
            logger.info(f"PyTorch CUDA devices: {devices['cuda_devices']}")
        
        # Check OpenCL (simplified)
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            if platforms:
                devices['opencl'] = True
                for platform in platforms:
                    devices['opencl_devices'].extend(platform.get_devices())
                logger.info(f"OpenCL devices detected: {len(devices['opencl_devices'])}")
        except ImportError:
            logger.debug("OpenCL not available")
        
        return devices
    
    def _select_device(self) -> GPUDevice:
        """Select the best available device."""
        if self.config.preferred_device == GPUDevice.CUDA and self.device_info['cuda']:
            return GPUDevice.CUDA
        elif self.config.preferred_device == GPUDevice.OPENCL and self.device_info['opencl']:
            return GPUDevice.OPENCL
        elif self.device_info['cuda']:
            return GPUDevice.CUDA
        elif self.device_info['opencl']:
            return GPUDevice.OPENCL
        else:
            if self.config.fallback_to_cpu:
                return GPUDevice.CPU
            else:
                raise RuntimeError("No suitable GPU device available")
    
    def _initialize_device(self):
        """Initialize device-specific resources."""
        if self.current_device == GPUDevice.CUDA and CUPY_AVAILABLE:
            # Set CUDA device
            if self.device_info['cuda_devices']:
                cp.cuda.Device(self.device_info['cuda_devices'][0]).use()
                logger.info(f"Using CUDA device {self.device_info['cuda_devices'][0]}")
        
        elif self.current_device == GPUDevice.CPU:
            logger.info("Using CPU for computation")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get current device information."""
        info = {
            'current_device': self.current_device.value,
            'available_devices': self.device_info,
            'memory_usage': self.memory_usage,
            'operation_count': self.operation_count
        }
        
        if self.current_device == GPUDevice.CUDA and CUPY_AVAILABLE:
            try:
                mempool = cp.get_default_memory_pool()
                info['gpu_memory_total'] = cp.cuda.runtime.getDeviceProperties(0).totalGlobalMem
                info['gpu_memory_used'] = mempool.used_bytes()
                info['gpu_memory_available'] = info['gpu_memory_total'] - info['gpu_memory_used']
            except Exception as e:
                logger.warning(f"Failed to get GPU memory info: {e}")
        
        return info
    
    def clear_memory(self):
        """Clear GPU memory cache."""
        if self.current_device == GPUDevice.CUDA and CUPY_AVAILABLE:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                logger.debug("GPU memory cleared")
            except Exception as e:
                logger.warning(f"Failed to clear GPU memory: {e}")
        
        self.memory_usage = 0
    
    def _check_memory_usage(self):
        """Check and manage memory usage."""
        self.operation_count += 1
        
        if self.operation_count % self.config.clear_cache_interval == 0:
            self.clear_memory()


class GPUImageProcessor:
    """GPU-accelerated image processing."""
    
    def __init__(self, gpu_manager: GPUManager):
        """Initialize GPU image processor."""
        self.gpu_manager = gpu_manager
        self.logger = get_logger(__name__)
    
    def gaussian_blur_gpu(self, image: np.ndarray, kernel_size: int, 
                         sigma: float = 0) -> np.ndarray:
        """GPU-accelerated Gaussian blur."""
        if self.gpu_manager.current_device == GPUDevice.CPU:
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        start_timer("gpu_gaussian_blur")
        
        try:
            if CUPY_AVAILABLE and self.gpu_manager.current_device == GPUDevice.CUDA:
                # Convert to CuPy array
                gpu_image = cp.asarray(image)
                
                # Apply Gaussian blur using CuPy
                blurred = cp_ndimage.gaussian_filter(
                    gpu_image, 
                    sigma=sigma if sigma > 0 else kernel_size / 6.0,
                    mode='constant'
                )
                
                # Convert back to NumPy
                result = cp.asnumpy(blurred)
            else:
                # Fallback to CPU
                result = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
            
            self.gpu_manager._check_memory_usage()
            end_timer("gpu_gaussian_blur")
            return result
            
        except Exception as e:
            self.logger.warning(f"GPU blur failed, falling back to CPU: {e}")
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def pixelate_gpu(self, image: np.ndarray, pixel_size: int) -> np.ndarray:
        """GPU-accelerated pixelation."""
        if self.gpu_manager.current_device == GPUDevice.CPU:
            return self._pixelate_cpu(image, pixel_size)
        
        start_timer("gpu_pixelate")
        
        try:
            if CUPY_AVAILABLE and self.gpu_manager.current_device == GPUDevice.CUDA:
                # Convert to CuPy array
                gpu_image = cp.asarray(image)
                h, w = gpu_image.shape[:2]
                
                # Calculate new dimensions
                new_h = h // pixel_size
                new_w = w // pixel_size
                
                if new_h < 1 or new_w < 1:
                    return image
                
                # Resize down
                small_image = cp.zeros((new_h, new_w, gpu_image.shape[2]), dtype=gpu_image.dtype)
                
                for y in range(new_h):
                    for x in range(new_w):
                        y_start = y * pixel_size
                        y_end = min((y + 1) * pixel_size, h)
                        x_start = x * pixel_size
                        x_end = min((x + 1) * pixel_size, w)
                        
                        # Calculate average color
                        region = gpu_image[y_start:y_end, x_start:x_end]
                        avg_color = cp.mean(region, axis=(0, 1))
                        small_image[y, x] = avg_color
                
                # Resize back up
                result = cp.zeros_like(gpu_image)
                for y in range(h):
                    for x in range(w):
                        small_y = y // pixel_size
                        small_x = x // pixel_size
                        result[y, x] = small_image[small_y, small_x]
                
                # Convert back to NumPy
                result = cp.asnumpy(result)
            else:
                result = self._pixelate_cpu(image, pixel_size)
            
            self.gpu_manager._check_memory_usage()
            end_timer("gpu_pixelate")
            return result
            
        except Exception as e:
            self.logger.warning(f"GPU pixelate failed, falling back to CPU: {e}")
            return self._pixelate_cpu(image, pixel_size)
    
    def _pixelate_cpu(self, image: np.ndarray, pixel_size: int) -> np.ndarray:
        """CPU-based pixelation fallback."""
        h, w = image.shape[:2]
        
        # Resize down
        small_h = h // pixel_size
        small_w = w // pixel_size
        
        if small_h < 1 or small_w < 1:
            return image
        
        small_image = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        
        # Resize back up
        return cv2.resize(small_image, (w, h), interpolation=cv2.INTER_NEAREST)
    
    def batch_process_gpu(self, images: List[np.ndarray], 
                         operation: str, **kwargs) -> List[np.ndarray]:
        """Process multiple images in batch on GPU."""
        if not images:
            return []
        
        start_timer("gpu_batch_process")
        
        try:
            if CUPY_AVAILABLE and self.gpu_manager.current_device == GPUDevice.CUDA:
                # Convert all images to GPU
                gpu_images = [cp.asarray(img) for img in images]
                
                # Process in batches
                batch_size = self.gpu_manager.config.batch_size
                results = []
                
                for i in range(0, len(gpu_images), batch_size):
                    batch = gpu_images[i:i + batch_size]
                    
                    if operation == "gaussian_blur":
                        kernel_size = kwargs.get('kernel_size', 15)
                        sigma = kwargs.get('sigma', 0)
                        batch_results = [
                            cp_ndimage.gaussian_filter(img, sigma=sigma if sigma > 0 else kernel_size / 6.0)
                            for img in batch
                        ]
                    elif operation == "pixelate":
                        pixel_size = kwargs.get('pixel_size', 20)
                        batch_results = [self._pixelate_gpu_single(img, pixel_size) for img in batch]
                    else:
                        batch_results = batch
                    
                    # Convert back to CPU
                    results.extend([cp.asnumpy(img) for img in batch_results])
                
                self.gpu_manager._check_memory_usage()
                end_timer("gpu_batch_process")
                return results
            else:
                # Fallback to CPU processing
                if operation == "gaussian_blur":
                    return [cv2.GaussianBlur(img, (kwargs.get('kernel_size', 15), kwargs.get('kernel_size', 15)), 0) for img in images]
                elif operation == "pixelate":
                    return [self._pixelate_cpu(img, kwargs.get('pixel_size', 20)) for img in images]
                else:
                    return images
                    
        except Exception as e:
            self.logger.warning(f"GPU batch processing failed, falling back to CPU: {e}")
            # Fallback to CPU
            if operation == "gaussian_blur":
                return [cv2.GaussianBlur(img, (kwargs.get('kernel_size', 15), kwargs.get('kernel_size', 15)), 0) for img in images]
            elif operation == "pixelate":
                return [self._pixelate_cpu(img, kwargs.get('pixel_size', 20)) for img in images]
            else:
                return images
    
    def _pixelate_gpu_single(self, gpu_image: cp.ndarray, pixel_size: int) -> cp.ndarray:
        """GPU pixelation for single image."""
        h, w = gpu_image.shape[:2]
        new_h = h // pixel_size
        new_w = w // pixel_size
        
        if new_h < 1 or new_w < 1:
            return gpu_image
        
        # Create small image
        small_image = cp.zeros((new_h, new_w, gpu_image.shape[2]), dtype=gpu_image.dtype)
        
        for y in range(new_h):
            for x in range(new_w):
                y_start = y * pixel_size
                y_end = min((y + 1) * pixel_size, h)
                x_start = x * pixel_size
                x_end = min((x + 1) * pixel_size, w)
                
                region = gpu_image[y_start:y_end, x_start:x_end]
                avg_color = cp.mean(region, axis=(0, 1))
                small_image[y, x] = avg_color
        
        # Expand back to original size
        result = cp.zeros_like(gpu_image)
        for y in range(h):
            for x in range(w):
                small_y = y // pixel_size
                small_x = x // pixel_size
                result[y, x] = small_image[small_y, small_x]
        
        return result


class GPUModelInference:
    """GPU-accelerated model inference."""
    
    def __init__(self, gpu_manager: GPUManager):
        """Initialize GPU model inference."""
        self.gpu_manager = gpu_manager
        self.models = {}
        self.logger = get_logger(__name__)
    
    def load_model_gpu(self, model_name: str, model_path: str) -> bool:
        """Load model for GPU inference."""
        try:
            if TORCH_AVAILABLE and self.gpu_manager.current_device == GPUDevice.CUDA:
                # Load PyTorch model
                model = torch.load(model_path, map_location='cuda')
                model.eval()
                
                if self.gpu_manager.config.optimize_for_inference:
                    model = torch.jit.optimize_for_inference(model)
                
                self.models[model_name] = model
                self.logger.info(f"Model {model_name} loaded on GPU")
                return True
            else:
                self.logger.warning("PyTorch not available or not using CUDA")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def inference_gpu(self, model_name: str, input_data: np.ndarray) -> np.ndarray:
        """Run GPU inference."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        start_timer("gpu_inference")
        
        try:
            if TORCH_AVAILABLE and self.gpu_manager.current_device == GPUDevice.CUDA:
                model = self.models[model_name]
                
                # Convert to PyTorch tensor
                if len(input_data.shape) == 3:
                    input_tensor = torch.from_numpy(input_data).permute(2, 0, 1).unsqueeze(0)
                else:
                    input_tensor = torch.from_numpy(input_data)
                
                input_tensor = input_tensor.cuda()
                
                # Run inference
                with torch.no_grad():
                    if self.gpu_manager.config.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            output = model(input_tensor)
                    else:
                        output = model(input_tensor)
                
                # Convert back to NumPy
                result = output.cpu().numpy()
                
                self.gpu_manager._check_memory_usage()
                end_timer("gpu_inference")
                return result
            else:
                raise RuntimeError("GPU inference not available")
                
        except Exception as e:
            self.logger.error(f"GPU inference failed: {e}")
            raise
    
    def batch_inference_gpu(self, model_name: str, 
                           input_batch: List[np.ndarray]) -> List[np.ndarray]:
        """Run batch inference on GPU."""
        if not input_batch:
            return []
        
        start_timer("gpu_batch_inference")
        
        try:
            if TORCH_AVAILABLE and self.gpu_manager.current_device == GPUDevice.CUDA:
                model = self.models[model_name]
                
                # Prepare batch tensor
                batch_tensors = []
                for img in input_batch:
                    if len(img.shape) == 3:
                        tensor = torch.from_numpy(img).permute(2, 0, 1)
                    else:
                        tensor = torch.from_numpy(img)
                    batch_tensors.append(tensor)
                
                batch_tensor = torch.stack(batch_tensors).cuda()
                
                # Run batch inference
                with torch.no_grad():
                    if self.gpu_manager.config.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            outputs = model(batch_tensor)
                    else:
                        outputs = model(batch_tensor)
                
                # Convert back to NumPy
                results = [output.cpu().numpy() for output in outputs]
                
                self.gpu_manager._check_memory_usage()
                end_timer("gpu_batch_inference")
                return results
            else:
                raise RuntimeError("GPU batch inference not available")
                
        except Exception as e:
            self.logger.error(f"GPU batch inference failed: {e}")
            raise


class GPUAccelerationManager:
    """Main GPU acceleration manager."""
    
    def __init__(self, config: GPUConfig = None):
        """Initialize GPU acceleration manager."""
        self.config = config or GPUConfig()
        self.gpu_manager = GPUManager(self.config)
        self.image_processor = GPUImageProcessor(self.gpu_manager)
        self.model_inference = GPUModelInference(self.gpu_manager)
        self.metrics_collector = get_metrics_collector()
        
        logger.info("GPU Acceleration Manager initialized")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get GPU performance statistics."""
        device_info = self.gpu_manager.get_device_info()
        
        stats = {
            'device_info': device_info,
            'gpu_available': self.gpu_manager.current_device != GPUDevice.CPU,
            'cupy_available': CUPY_AVAILABLE,
            'torch_available': TORCH_AVAILABLE,
            'memory_usage_mb': device_info.get('gpu_memory_used', 0) / (1024 * 1024),
            'memory_total_mb': device_info.get('gpu_memory_total', 0) / (1024 * 1024)
        }
        
        return stats
    
    def optimize_for_inference(self):
        """Optimize GPU for inference."""
        if self.gpu_manager.current_device == GPUDevice.CUDA and CUPY_AVAILABLE:
            # Set memory pool for better performance
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=self.config.max_memory_usage)
            
            # Enable memory pool
            mempool.set_limit(size=None)
            
            logger.info("GPU optimized for inference")
    
    def clear_gpu_memory(self):
        """Clear GPU memory."""
        self.gpu_manager.clear_memory()
        logger.info("GPU memory cleared")


# Global GPU acceleration manager
_gpu_manager: Optional[GPUAccelerationManager] = None


def get_gpu_manager() -> GPUAccelerationManager:
    """Get the global GPU acceleration manager."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUAccelerationManager()
    return _gpu_manager


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return get_gpu_manager().gpu_manager.current_device != GPUDevice.CPU


def get_gpu_performance_stats() -> Dict[str, Any]:
    """Get GPU performance statistics."""
    return get_gpu_manager().get_performance_stats()
