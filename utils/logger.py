"""
Advanced logging system for RedactAI.

This module provides comprehensive logging capabilities with structured logging,
performance monitoring, and error tracking.
"""

import logging
import logging.handlers
import json
import sys
import traceback
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, asdict
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Structured log entry."""
    
    timestamp: str
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    extra_data: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def __init__(self, include_extra: bool = True):
        """Initialize formatter."""
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record."""
        # Create structured log entry
        log_entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            level=record.levelname,
            logger_name=record.name,
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line_number=record.lineno,
            thread_id=record.thread,
            process_id=record.process,
            extra_data=record.__dict__.get('extra_data', {})
        )
        
        # Add exception info if present
        if record.exc_info:
            log_entry.extra_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add any additional fields from the record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 'msecs',
                          'relativeCreated', 'thread', 'threadName', 'processName', 'process',
                          'getMessage', 'exc_info', 'exc_text', 'stack_info', 'extra_data']:
                log_entry.extra_data[key] = value
        
        return log_entry.to_json()


class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self, logger_name: str = "performance"):
        """Initialize performance logger."""
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
    
    def log_processing_time(self, operation: str, duration: float, 
                           file_type: str = None, file_size: int = None, 
                           **kwargs) -> None:
        """Log processing time metrics."""
        extra_data = {
            'operation': operation,
            'duration_seconds': duration,
            'file_type': file_type,
            'file_size_bytes': file_size,
            **kwargs
        }
        
        self.logger.info(f"Processing completed: {operation}", extra={'extra_data': extra_data})
    
    def log_detection_results(self, operation: str, faces: int = 0, plates: int = 0, 
                             text_regions: int = 0, names: int = 0, **kwargs) -> None:
        """Log detection results."""
        extra_data = {
            'operation': operation,
            'faces_detected': faces,
            'plates_detected': plates,
            'text_regions_detected': text_regions,
            'names_redacted': names,
            **kwargs
        }
        
        self.logger.info(f"Detection completed: {operation}", extra={'extra_data': extra_data})
    
    def log_error(self, operation: str, error: Exception, **kwargs) -> None:
        """Log error with context."""
        extra_data = {
            'operation': operation,
            'error_type': type(error).__name__,
            'error_message': str(error),
            **kwargs
        }
        
        self.logger.error(f"Error in {operation}: {error}", extra={'extra_data': extra_data})


class ErrorTracker:
    """Error tracking and reporting system."""
    
    def __init__(self, max_errors: int = 1000):
        """Initialize error tracker."""
        self.max_errors = max_errors
        self.errors: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self.logger = logging.getLogger("error_tracker")
    
    def track_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """Track an error with context."""
        with self.lock:
            error_entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error_type': type(error).__name__,
                'error_message': str(error),
                'traceback': traceback.format_exc(),
                'context': context or {},
                'thread_id': threading.get_ident(),
                'process_id': os.getpid()
            }
            
            self.errors.append(error_entry)
            
            # Keep only recent errors
            if len(self.errors) > self.max_errors:
                self.errors = self.errors[-self.max_errors:]
            
            # Log the error
            self.logger.error(f"Error tracked: {error}", extra={'extra_data': error_entry})
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics."""
        with self.lock:
            if not self.errors:
                return {
                    'total_errors': 0,
                    'error_types': {},
                    'recent_errors': []
                }
            
            # Count error types
            error_types = {}
            for error in self.errors:
                error_type = error['error_type']
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # Get recent errors (last 10)
            recent_errors = self.errors[-10:]
            
            return {
                'total_errors': len(self.errors),
                'error_types': error_types,
                'recent_errors': recent_errors
            }
    
    def clear_errors(self) -> None:
        """Clear all tracked errors."""
        with self.lock:
            self.errors.clear()


class LogManager:
    """Centralized log manager for RedactAI."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize log manager."""
        self.config = config or {}
        self.loggers: Dict[str, logging.Logger] = {}
        self.performance_logger = PerformanceLogger()
        self.error_tracker = ErrorTracker()
        self.setup_logging()
    
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler for all logs
        file_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "redact_ai.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = StructuredFormatter()
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "errors.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
        
        # Performance log handler
        perf_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "performance.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(file_formatter)
        
        perf_logger = logging.getLogger("performance")
        perf_logger.addHandler(perf_handler)
        perf_logger.propagate = False
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance."""
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        return self.loggers[name]
    
    def log_processing_start(self, operation: str, **kwargs) -> None:
        """Log processing start."""
        logger = self.get_logger("processing")
        extra_data = {
            'operation': operation,
            'status': 'started',
            **kwargs
        }
        logger.info(f"Processing started: {operation}", extra={'extra_data': extra_data})
    
    def log_processing_end(self, operation: str, duration: float, **kwargs) -> None:
        """Log processing end."""
        logger = self.get_logger("processing")
        extra_data = {
            'operation': operation,
            'status': 'completed',
            'duration_seconds': duration,
            **kwargs
        }
        logger.info(f"Processing completed: {operation}", extra={'extra_data': extra_data})
    
    def log_processing_error(self, operation: str, error: Exception, **kwargs) -> None:
        """Log processing error."""
        logger = self.get_logger("processing")
        extra_data = {
            'operation': operation,
            'status': 'error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            **kwargs
        }
        logger.error(f"Processing error: {operation}", extra={'extra_data': extra_data})
        
        # Track error
        self.error_tracker.track_error(error, extra_data)
    
    def log_detection_results(self, operation: str, results: Dict[str, Any]) -> None:
        """Log detection results."""
        logger = self.get_logger("detection")
        extra_data = {
            'operation': operation,
            **results
        }
        logger.info(f"Detection results: {operation}", extra={'extra_data': extra_data})
    
    def log_api_request(self, method: str, endpoint: str, status_code: int, 
                       duration: float, **kwargs) -> None:
        """Log API request."""
        logger = self.get_logger("api")
        extra_data = {
            'method': method,
            'endpoint': endpoint,
            'status_code': status_code,
            'duration_seconds': duration,
            **kwargs
        }
        logger.info(f"API request: {method} {endpoint}", extra={'extra_data': extra_data})
    
    def log_system_event(self, event: str, **kwargs) -> None:
        """Log system event."""
        logger = self.get_logger("system")
        extra_data = {
            'event': event,
            **kwargs
        }
        logger.info(f"System event: {event}", extra={'extra_data': extra_data})
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary."""
        return self.error_tracker.get_error_summary()
    
    def clear_errors(self) -> None:
        """Clear error tracking."""
        self.error_tracker.clear_errors()


# Global log manager instance
_log_manager: Optional[LogManager] = None


def get_log_manager() -> LogManager:
    """Get the global log manager instance."""
    global _log_manager
    if _log_manager is None:
        _log_manager = LogManager()
    return _log_manager


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return get_log_manager().get_logger(name)


@contextmanager
def log_processing(operation: str, **kwargs):
    """Context manager for logging processing operations."""
    log_manager = get_log_manager()
    start_time = time.time()
    
    try:
        log_manager.log_processing_start(operation, **kwargs)
        yield
        duration = time.time() - start_time
        log_manager.log_processing_end(operation, duration, **kwargs)
    except Exception as e:
        duration = time.time() - start_time
        log_manager.log_processing_error(operation, e, duration=duration, **kwargs)
        raise


def log_detection_results(operation: str, results: Dict[str, Any]) -> None:
    """Log detection results."""
    get_log_manager().log_detection_results(operation, results)


def log_api_request(method: str, endpoint: str, status_code: int, 
                   duration: float, **kwargs) -> None:
    """Log API request."""
    get_log_manager().log_api_request(method, endpoint, status_code, duration, **kwargs)


def log_system_event(event: str, **kwargs) -> None:
    """Log system event."""
    get_log_manager().log_system_event(event, **kwargs)


def get_error_summary() -> Dict[str, Any]:
    """Get error summary."""
    return get_log_manager().get_error_summary()


def clear_errors() -> None:
    """Clear error tracking."""
    get_log_manager().clear_errors()
