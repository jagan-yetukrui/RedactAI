"""
Monitoring and metrics system for RedactAI.

This module provides comprehensive monitoring capabilities including
performance metrics, system health monitoring, and processing statistics.
"""

import time
import psutil
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict, deque
import threading
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetrics:
    """Metrics for processing operations."""
    
    # Counters
    total_files_processed: int = 0
    total_faces_detected: int = 0
    total_plates_detected: int = 0
    total_text_regions_detected: int = 0
    total_names_redacted: int = 0
    total_frames_processed: int = 0
    
    # Timing metrics
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    min_processing_time: float = float('inf')
    max_processing_time: float = 0.0
    
    # Error metrics
    total_errors: int = 0
    error_rate: float = 0.0
    
    # File type breakdown
    file_type_counts: Dict[str, int] = field(default_factory=dict)
    
    # Timestamps
    first_processing: Optional[datetime] = None
    last_processing: Optional[datetime] = None
    
    def update_processing(self, processing_time: float, file_type: str, 
                         faces: int = 0, plates: int = 0, text_regions: int = 0, 
                         names: int = 0, frames: int = 0, error: bool = False):
        """Update metrics with new processing data."""
        self.total_files_processed += 1
        self.total_faces_detected += faces
        self.total_plates_detected += plates
        self.total_text_regions_detected += text_regions
        self.total_names_redacted += names
        self.total_frames_processed += frames
        
        # Update timing metrics
        self.total_processing_time += processing_time
        self.average_processing_time = self.total_processing_time / self.total_files_processed
        self.min_processing_time = min(self.min_processing_time, processing_time)
        self.max_processing_time = max(self.max_processing_time, processing_time)
        
        # Update error metrics
        if error:
            self.total_errors += 1
        self.error_rate = self.total_errors / self.total_files_processed
        
        # Update file type counts
        self.file_type_counts[file_type] = self.file_type_counts.get(file_type, 0) + 1
        
        # Update timestamps
        now = datetime.now(timezone.utc)
        if self.first_processing is None:
            self.first_processing = now
        self.last_processing = now


@dataclass
class SystemMetrics:
    """System resource metrics."""
    
    # CPU metrics
    cpu_percent: float = 0.0
    cpu_count: int = 0
    
    # Memory metrics
    memory_percent: float = 0.0
    memory_used: int = 0
    memory_available: int = 0
    memory_total: int = 0
    
    # Disk metrics
    disk_percent: float = 0.0
    disk_used: int = 0
    disk_free: int = 0
    disk_total: int = 0
    
    # Network metrics
    network_sent: int = 0
    network_received: int = 0
    
    # Process metrics
    process_count: int = 0
    thread_count: int = 0
    
    def update(self):
        """Update system metrics."""
        # CPU metrics
        self.cpu_percent = psutil.cpu_percent(interval=1)
        self.cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.memory_percent = memory.percent
        self.memory_used = memory.used
        self.memory_available = memory.available
        self.memory_total = memory.total
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.disk_percent = (disk.used / disk.total) * 100
        self.disk_used = disk.used
        self.disk_free = disk.free
        self.disk_total = disk.total
        
        # Network metrics
        network = psutil.net_io_counters()
        self.network_sent = network.bytes_sent
        self.network_received = network.bytes_recv
        
        # Process metrics
        self.process_count = len(psutil.pids())
        self.thread_count = psutil.Process().num_threads()


class PerformanceProfiler:
    """Performance profiler for timing operations."""
    
    def __init__(self):
        """Initialize profiler."""
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.active_timers: Dict[str, float] = {}
        self.lock = threading.Lock()
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        with self.lock:
            self.active_timers[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        with self.lock:
            if operation in self.active_timers:
                duration = time.time() - self.active_timers[operation]
                self.timings[operation].append(duration)
                del self.active_timers[operation]
                return duration
            return 0.0
    
    def get_timing_stats(self, operation: str) -> Dict[str, float]:
        """Get timing statistics for an operation."""
        if operation not in self.timings or not self.timings[operation]:
            return {}
        
        timings = self.timings[operation]
        return {
            'count': len(timings),
            'total': sum(timings),
            'average': sum(timings) / len(timings),
            'min': min(timings),
            'max': max(timings),
            'median': sorted(timings)[len(timings) // 2]
        }
    
    def get_all_timings(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics for all operations."""
        return {op: self.get_timing_stats(op) for op in self.timings}


class HealthChecker:
    """System health checker."""
    
    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, Callable[[], bool]] = {}
        self.last_check: Dict[str, datetime] = {}
        self.check_interval = 30  # seconds
    
    def register_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """Register a health check function."""
        self.checks[name] = check_func
    
    def run_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all health checks."""
        results = {}
        
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                is_healthy = check_func()
                duration = time.time() - start_time
                
                results[name] = {
                    'healthy': is_healthy,
                    'duration': duration,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'error': None
                }
                
                self.last_check[name] = datetime.now(timezone.utc)
                
            except Exception as e:
                results[name] = {
                    'healthy': False,
                    'duration': 0.0,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'error': str(e)
                }
        
        return results
    
    def is_healthy(self) -> bool:
        """Check if all systems are healthy."""
        results = self.run_checks()
        return all(result['healthy'] for result in results.values())


class MetricsCollector:
    """Main metrics collector and manager."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize metrics collector."""
        self.config = config or {}
        self.processing_metrics = ProcessingMetrics()
        self.system_metrics = SystemMetrics()
        self.profiler = PerformanceProfiler()
        self.health_checker = HealthChecker()
        
        # Initialize health checks
        self._setup_health_checks()
        
        # Start background monitoring
        self.monitoring_thread = None
        self.monitoring_active = False
        self.start_monitoring()
    
    def _setup_health_checks(self) -> None:
        """Setup default health checks."""
        def check_disk_space():
            """Check if there's enough disk space."""
            disk = psutil.disk_usage('/')
            return disk.free > 1024 * 1024 * 1024  # 1GB free
        
        def check_memory():
            """Check if there's enough memory."""
            memory = psutil.virtual_memory()
            return memory.available > 512 * 1024 * 1024  # 512MB available
        
        def check_cpu():
            """Check if CPU usage is not too high."""
            return psutil.cpu_percent(interval=1) < 90.0
        
        self.health_checker.register_check('disk_space', check_disk_space)
        self.health_checker.register_check('memory', check_memory)
        self.health_checker.register_check('cpu', check_cpu)
    
    def start_monitoring(self) -> None:
        """Start background monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Started background monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Stopped background monitoring")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                self.system_metrics.update()
                time.sleep(10)  # Update every 10 seconds
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def record_processing(self, processing_time: float, file_type: str, 
                         faces: int = 0, plates: int = 0, text_regions: int = 0, 
                         names: int = 0, frames: int = 0, error: bool = False) -> None:
        """Record a processing operation."""
        self.processing_metrics.update_processing(
            processing_time, file_type, faces, plates, text_regions, names, frames, error
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        return {
            'processing': {
                'total_files_processed': self.processing_metrics.total_files_processed,
                'total_faces_detected': self.processing_metrics.total_faces_detected,
                'total_plates_detected': self.processing_metrics.total_plates_detected,
                'total_text_regions_detected': self.processing_metrics.total_text_regions_detected,
                'total_names_redacted': self.processing_metrics.total_names_redacted,
                'total_frames_processed': self.processing_metrics.total_frames_processed,
                'average_processing_time': self.processing_metrics.average_processing_time,
                'error_rate': self.processing_metrics.error_rate,
                'file_type_breakdown': self.processing_metrics.file_type_counts,
                'first_processing': self.processing_metrics.first_processing.isoformat() if self.processing_metrics.first_processing else None,
                'last_processing': self.processing_metrics.last_processing.isoformat() if self.processing_metrics.last_processing else None
            },
            'system': {
                'cpu_percent': self.system_metrics.cpu_percent,
                'memory_percent': self.system_metrics.memory_percent,
                'disk_percent': self.system_metrics.disk_percent,
                'process_count': self.system_metrics.process_count,
                'thread_count': self.system_metrics.thread_count
            },
            'performance': self.profiler.get_all_timings(),
            'health': self.health_checker.run_checks()
        }
    
    def export_metrics(self, file_path: Path) -> None:
        """Export metrics to file."""
        try:
            metrics_data = self.get_metrics_summary()
            with open(file_path, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            logger.info(f"Metrics exported to {file_path}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.processing_metrics = ProcessingMetrics()
        self.profiler = PerformanceProfiler()
        logger.info("Metrics reset")


# Global metrics collector instance
metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return metrics_collector


def record_processing(processing_time: float, file_type: str, **kwargs) -> None:
    """Record a processing operation."""
    metrics_collector.record_processing(processing_time, file_type, **kwargs)


def start_timer(operation: str) -> None:
    """Start timing an operation."""
    metrics_collector.profiler.start_timer(operation)


def end_timer(operation: str) -> float:
    """End timing an operation."""
    return metrics_collector.profiler.end_timer(operation)


def get_metrics_summary() -> Dict[str, Any]:
    """Get metrics summary."""
    return metrics_collector.get_metrics_summary()


def is_system_healthy() -> bool:
    """Check if system is healthy."""
    return metrics_collector.health_checker.is_healthy()
