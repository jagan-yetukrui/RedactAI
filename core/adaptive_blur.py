"""
Advanced adaptive blurring system for RedactAI.

This module implements sophisticated adaptive blurring techniques that
intelligently adjust blur parameters based on content analysis, context,
and privacy requirements for optimal redaction quality.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path

from ..utils.monitoring import get_metrics_collector, start_timer, end_timer
from ..utils.cache import cached
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BlurType(Enum):
    """Types of blurring algorithms available."""
    GAUSSIAN = "gaussian"
    PIXELATE = "pixelate"
    MOSAIC = "mosaic"
    BLACKOUT = "blackout"
    ADAPTIVE_GAUSSIAN = "adaptive_gaussian"
    CONTENT_AWARE = "content_aware"
    PRIVACY_PRESERVING = "privacy_preserving"


@dataclass
class BlurRegion:
    """A region to be blurred with associated metadata."""
    
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    blur_type: BlurType
    blur_strength: float
    confidence: float
    content_type: str
    privacy_level: int  # 1-5 scale
    context: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def area(self) -> int:
        """Calculate region area."""
        return self.bbox[2] * self.bbox[3]
    
    @property
    def center(self) -> Tuple[int, int]:
        """Calculate region center."""
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)


@dataclass
class AdaptiveBlurConfig:
    """Configuration for adaptive blurring."""
    
    # Base blur parameters
    base_gaussian_kernel: int = 15
    base_pixelate_size: int = 20
    base_mosaic_size: int = 25
    
    # Adaptive parameters
    enable_content_analysis: bool = True
    enable_privacy_scoring: bool = True
    enable_context_awareness: bool = True
    
    # Privacy levels (1-5 scale)
    privacy_levels: Dict[str, int] = field(default_factory=lambda: {
        'face': 5,
        'license_plate': 4,
        'text': 3,
        'person': 4,
        'vehicle': 2
    })
    
    # Content analysis weights
    content_weights: Dict[str, float] = field(default_factory=lambda: {
        'size': 0.3,
        'position': 0.2,
        'context': 0.3,
        'privacy_level': 0.2
    })


class ContentAnalyzer:
    """Analyzes image content to determine optimal blur parameters."""
    
    def __init__(self, config: AdaptiveBlurConfig = None):
        """Initialize content analyzer."""
        self.config = config or AdaptiveBlurConfig()
        self.logger = get_logger(__name__)
    
    def analyze_region(self, image: np.ndarray, region: BlurRegion) -> Dict[str, Any]:
        """Analyze a region to determine optimal blur parameters."""
        x, y, w, h = region.bbox
        roi = image[y:y+h, x:x+w]
        
        if roi.size == 0:
            return {}
        
        analysis = {
            'size_score': self._analyze_size(roi),
            'position_score': self._analyze_position(image, region),
            'context_score': self._analyze_context(image, region),
            'privacy_score': self._get_privacy_score(region),
            'texture_complexity': self._analyze_texture(roi),
            'edge_density': self._analyze_edges(roi),
            'color_variance': self._analyze_colors(roi)
        }
        
        return analysis
    
    def _analyze_size(self, roi: np.ndarray) -> float:
        """Analyze region size relative to image."""
        area = roi.shape[0] * roi.shape[1]
        
        # Normalize size score (larger regions need stronger blur)
        if area < 1000:
            return 0.3
        elif area < 5000:
            return 0.6
        else:
            return 1.0
    
    def _analyze_position(self, image: np.ndarray, region: BlurRegion) -> float:
        """Analyze region position in image."""
        img_h, img_w = image.shape[:2]
        x, y, w, h = region.bbox
        
        # Center regions are more important
        center_x = img_w // 2
        center_y = img_h // 2
        region_center_x = x + w // 2
        region_center_y = y + h // 2
        
        # Calculate distance from center
        distance = math.sqrt((region_center_x - center_x)**2 + (region_center_y - center_y)**2)
        max_distance = math.sqrt(center_x**2 + center_y**2)
        
        # Closer to center = higher score
        return 1.0 - (distance / max_distance)
    
    def _analyze_context(self, image: np.ndarray, region: BlurRegion) -> float:
        """Analyze surrounding context of the region."""
        x, y, w, h = region.bbox
        
        # Expand region for context analysis
        margin = 50
        context_x = max(0, x - margin)
        context_y = max(0, y - margin)
        context_w = min(image.shape[1] - context_x, w + 2 * margin)
        context_h = min(image.shape[0] - context_y, h + 2 * margin)
        
        context_roi = image[context_y:context_y+context_h, context_x:context_x+context_w]
        
        if context_roi.size == 0:
            return 0.5
        
        # Analyze context complexity
        gray_context = cv2.cvtColor(context_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_context, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Higher edge density = more complex context = higher score
        return min(edge_density * 2, 1.0)
    
    def _get_privacy_score(self, region: BlurRegion) -> float:
        """Get privacy score based on content type."""
        content_type = region.content_type.lower()
        privacy_level = self.config.privacy_levels.get(content_type, 3)
        return privacy_level / 5.0
    
    def _analyze_texture(self, roi: np.ndarray) -> float:
        """Analyze texture complexity of the region."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate local binary pattern variance
        lbp = self._local_binary_pattern(gray)
        texture_variance = np.var(lbp)
        
        # Normalize texture complexity
        return min(texture_variance / 1000, 1.0)
    
    def _local_binary_pattern(self, image: np.ndarray) -> np.ndarray:
        """Calculate local binary pattern for texture analysis."""
        # Simplified LBP implementation
        rows, cols = image.shape
        lbp = np.zeros_like(image)
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                center = image[i, j]
                binary_string = ""
                
                # 8-neighborhood
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                
                for neighbor in neighbors:
                    binary_string += "1" if neighbor >= center else "0"
                
                lbp[i, j] = int(binary_string, 2)
        
        return lbp
    
    def _analyze_edges(self, roi: np.ndarray) -> float:
        """Analyze edge density in the region."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        return edge_density
    
    def _analyze_colors(self, roi: np.ndarray) -> float:
        """Analyze color variance in the region."""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate variance for each channel
        h_var = np.var(hsv[:, :, 0])
        s_var = np.var(hsv[:, :, 1])
        v_var = np.var(hsv[:, :, 2])
        
        # Combined color variance
        color_variance = (h_var + s_var + v_var) / 3
        return min(color_variance / 1000, 1.0)


class AdaptiveBlurProcessor:
    """Advanced adaptive blur processor."""
    
    def __init__(self, config: AdaptiveBlurConfig = None):
        """Initialize adaptive blur processor."""
        self.config = config or AdaptiveBlurConfig()
        self.content_analyzer = ContentAnalyzer(config)
        self.metrics_collector = get_metrics_collector()
        self.logger = get_logger(__name__)
    
    def process_image(self, image: np.ndarray, 
                     regions: List[BlurRegion]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process image with adaptive blurring."""
        start_timer("adaptive_blur_processing")
        
        result_image = image.copy()
        processing_stats = {
            'total_regions': len(regions),
            'processed_regions': 0,
            'blur_types_used': {},
            'average_blur_strength': 0.0,
            'processing_time': 0.0
        }
        
        total_blur_strength = 0.0
        
        for region in regions:
            try:
                # Analyze region content
                if self.config.enable_content_analysis:
                    analysis = self.content_analyzer.analyze_region(image, region)
                    region = self._adapt_blur_parameters(region, analysis)
                
                # Apply adaptive blur
                result_image = self._apply_adaptive_blur(result_image, region)
                
                # Update statistics
                processing_stats['processed_regions'] += 1
                blur_type = region.blur_type.value
                processing_stats['blur_types_used'][blur_type] = \
                    processing_stats['blur_types_used'].get(blur_type, 0) + 1
                total_blur_strength += region.blur_strength
                
            except Exception as e:
                self.logger.error(f"Error processing region {region.bbox}: {e}")
        
        # Calculate final statistics
        if processing_stats['processed_regions'] > 0:
            processing_stats['average_blur_strength'] = \
                total_blur_strength / processing_stats['processed_regions']
        
        processing_stats['processing_time'] = end_timer("adaptive_blur_processing")
        
        # Record metrics
        self.metrics_collector.record_processing(
            processing_time=processing_stats['processing_time'],
            file_type='.jpg',
            regions_processed=processing_stats['processed_regions']
        )
        
        return result_image, processing_stats
    
    def _adapt_blur_parameters(self, region: BlurRegion, 
                              analysis: Dict[str, Any]) -> BlurRegion:
        """Adapt blur parameters based on content analysis."""
        # Calculate adaptive score
        adaptive_score = 0.0
        weights = self.config.content_weights
        
        for factor, weight in weights.items():
            if factor in analysis:
                adaptive_score += analysis[factor] * weight
        
        # Adjust blur strength based on adaptive score
        base_strength = region.blur_strength
        adaptive_multiplier = 0.5 + adaptive_score * 1.5  # Range: 0.5-2.0
        new_strength = base_strength * adaptive_multiplier
        
        # Adjust blur type based on content analysis
        new_blur_type = self._select_optimal_blur_type(region, analysis)
        
        # Create new region with adapted parameters
        adapted_region = BlurRegion(
            bbox=region.bbox,
            blur_type=new_blur_type,
            blur_strength=new_strength,
            confidence=region.confidence,
            content_type=region.content_type,
            privacy_level=region.privacy_level,
            context=region.context
        )
        
        return adapted_region
    
    def _select_optimal_blur_type(self, region: BlurRegion, 
                                 analysis: Dict[str, Any]) -> BlurType:
        """Select optimal blur type based on content analysis."""
        content_type = region.content_type.lower()
        privacy_level = region.privacy_level
        
        # High privacy content gets stronger blur
        if privacy_level >= 4:
            if analysis.get('texture_complexity', 0) > 0.7:
                return BlurType.CONTENT_AWARE
            else:
                return BlurType.PRIVACY_PRESERVING
        
        # Face detection - use adaptive gaussian
        if content_type == 'face':
            return BlurType.ADAPTIVE_GAUSSIAN
        
        # License plates - use pixelate for better privacy
        if content_type == 'license_plate':
            return BlurType.PIXELATE
        
        # Text - use mosaic for readability while maintaining privacy
        if content_type == 'text':
            return BlurType.MOSAIC
        
        # Default to gaussian
        return BlurType.GAUSSIAN
    
    def _apply_adaptive_blur(self, image: np.ndarray, region: BlurRegion) -> np.ndarray:
        """Apply adaptive blur to a specific region."""
        x, y, w, h = region.bbox
        
        # Ensure coordinates are within image bounds
        x = max(0, min(x, image.shape[1] - 1))
        y = max(0, min(y, image.shape[0] - 1))
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return image
        
        # Extract region
        roi = image[y:y+h, x:x+w]
        
        # Apply blur based on type
        if region.blur_type == BlurType.GAUSSIAN:
            blurred_roi = self._apply_gaussian_blur(roi, region.blur_strength)
        elif region.blur_type == BlurType.PIXELATE:
            blurred_roi = self._apply_pixelate_blur(roi, region.blur_strength)
        elif region.blur_type == BlurType.MOSAIC:
            blurred_roi = self._apply_mosaic_blur(roi, region.blur_strength)
        elif region.blur_type == BlurType.BLACKOUT:
            blurred_roi = self._apply_blackout_blur(roi)
        elif region.blur_type == BlurType.ADAPTIVE_GAUSSIAN:
            blurred_roi = self._apply_adaptive_gaussian_blur(roi, region.blur_strength)
        elif region.blur_type == BlurType.CONTENT_AWARE:
            blurred_roi = self._apply_content_aware_blur(roi, region.blur_strength)
        elif region.blur_type == BlurType.PRIVACY_PRESERVING:
            blurred_roi = self._apply_privacy_preserving_blur(roi, region.blur_strength)
        else:
            blurred_roi = self._apply_gaussian_blur(roi, region.blur_strength)
        
        # Replace region in image
        image[y:y+h, x:x+w] = blurred_roi
        
        return image
    
    def _apply_gaussian_blur(self, roi: np.ndarray, strength: float) -> np.ndarray:
        """Apply Gaussian blur."""
        kernel_size = int(strength)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        return cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
    
    def _apply_pixelate_blur(self, roi: np.ndarray, strength: float) -> np.ndarray:
        """Apply pixelate blur."""
        pixel_size = int(strength)
        if pixel_size < 2:
            pixel_size = 2
        
        h, w = roi.shape[:2]
        
        # Resize down
        small_h = h // pixel_size
        small_w = w // pixel_size
        
        if small_h < 1 or small_w < 1:
            return roi
        
        small_roi = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        
        # Resize back up
        return cv2.resize(small_roi, (w, h), interpolation=cv2.INTER_NEAREST)
    
    def _apply_mosaic_blur(self, roi: np.ndarray, strength: float) -> np.ndarray:
        """Apply mosaic blur."""
        mosaic_size = int(strength)
        if mosaic_size < 2:
            mosaic_size = 2
        
        h, w = roi.shape[:2]
        
        # Create mosaic effect
        for y in range(0, h, mosaic_size):
            for x in range(0, w, mosaic_size):
                # Get the average color of the mosaic block
                end_y = min(y + mosaic_size, h)
                end_x = min(x + mosaic_size, w)
                
                block = roi[y:end_y, x:end_x]
                avg_color = np.mean(block, axis=(0, 1))
                
                # Apply the average color to the entire block
                roi[y:end_y, x:end_x] = avg_color
        
        return roi
    
    def _apply_blackout_blur(self, roi: np.ndarray) -> np.ndarray:
        """Apply blackout blur."""
        return np.zeros_like(roi)
    
    def _apply_adaptive_gaussian_blur(self, roi: np.ndarray, strength: float) -> np.ndarray:
        """Apply adaptive Gaussian blur with varying kernel sizes."""
        h, w = roi.shape[:2]
        
        # Create gradient blur - stronger in center, weaker at edges
        center_y, center_x = h // 2, w // 2
        max_distance = math.sqrt(center_x**2 + center_y**2)
        
        result = roi.copy()
        
        # Apply blur with varying strength
        for y in range(h):
            for x in range(w):
                distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                normalized_distance = distance / max_distance
                
                # Calculate adaptive kernel size
                adaptive_strength = strength * (0.5 + normalized_distance * 0.5)
                kernel_size = int(adaptive_strength)
                
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                if kernel_size >= 3:
                    # Apply local blur
                    y1 = max(0, y - kernel_size//2)
                    y2 = min(h, y + kernel_size//2 + 1)
                    x1 = max(0, x - kernel_size//2)
                    x2 = min(w, x + kernel_size//2 + 1)
                    
                    local_roi = roi[y1:y2, x1:x2]
                    if local_roi.size > 0:
                        blurred = cv2.GaussianBlur(local_roi, (kernel_size, kernel_size), 0)
                        result[y, x] = blurred[y-y1, x-x1]
        
        return result
    
    def _apply_content_aware_blur(self, roi: np.ndarray, strength: float) -> np.ndarray:
        """Apply content-aware blur that preserves important features."""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Detect edges and important features
        edges = cv2.Canny(gray, 50, 150)
        
        # Create mask for important regions
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Apply different blur strengths based on content
        result = roi.copy()
        
        # Strong blur for non-edge regions
        strong_blur = cv2.GaussianBlur(roi, (int(strength * 2), int(strength * 2)), 0)
        
        # Light blur for edge regions
        light_blur = cv2.GaussianBlur(roi, (int(strength * 0.5), int(strength * 0.5)), 0)
        
        # Combine based on edge mask
        mask = edges_dilated > 0
        result[mask] = light_blur[mask]
        result[~mask] = strong_blur[~mask]
        
        return result
    
    def _apply_privacy_preserving_blur(self, roi: np.ndarray, strength: float) -> np.ndarray:
        """Apply privacy-preserving blur with multiple techniques."""
        # Combine multiple blur techniques for maximum privacy
        
        # First pass: Strong Gaussian blur
        blurred = cv2.GaussianBlur(roi, (int(strength * 1.5), int(strength * 1.5)), 0)
        
        # Second pass: Pixelate for additional privacy
        pixelated = self._apply_pixelate_blur(blurred, strength * 0.8)
        
        # Third pass: Add noise for extra obfuscation
        noise = np.random.normal(0, 10, pixelated.shape).astype(np.uint8)
        result = cv2.add(pixelated, noise)
        
        return result


# Global adaptive blur processor instance
_adaptive_blur_processor: Optional[AdaptiveBlurProcessor] = None


def get_adaptive_blur_processor() -> AdaptiveBlurProcessor:
    """Get the global adaptive blur processor instance."""
    global _adaptive_blur_processor
    if _adaptive_blur_processor is None:
        _adaptive_blur_processor = AdaptiveBlurProcessor()
    return _adaptive_blur_processor


def process_with_adaptive_blur(image: np.ndarray, 
                              regions: List[BlurRegion]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Process image with adaptive blurring."""
    processor = get_adaptive_blur_processor()
    return processor.process_image(image, regions)
