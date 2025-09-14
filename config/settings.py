"""
Configuration settings for RedactAI.

This module provides centralized configuration management for all components
of the RedactAI system, including processing parameters, API settings, and
deployment configurations.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for media processing parameters."""
    
    # Face detection settings
    face_detection: Dict[str, Any] = field(default_factory=lambda: {
        'cascade_path': 'haarcascade_frontalface_default.xml',
        'scale_factor': 1.1,
        'min_neighbors': 5,
        'min_size': (30, 30),
        'confidence_threshold': 0.5,
        'blur_type': 'gaussian',
        'blur_strength': 15
    })
    
    # License plate detection settings
    plate_detection: Dict[str, Any] = field(default_factory=lambda: {
        'model_path': None,  # Will use default YOLOv8 model
        'confidence_threshold': 0.5,
        'blur_type': 'gaussian',
        'blur_strength': 15,
        'use_yolo': True,
        'fallback_to_opencv': True
    })
    
    # Text detection and redaction settings
    text_detection: Dict[str, Any] = field(default_factory=lambda: {
        'ocr_engine': 'tesseract',  # 'tesseract', 'easyocr', or 'both'
        'languages': ['en'],
        'confidence_threshold': 0.5,
        'redact_names_only': True,
        'blur_type': 'gaussian',
        'blur_strength': 15,
        'spacy_model': 'en_core_web_sm'
    })
    
    # Geotagging settings
    geotagging: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'mock_gps': True,
        'gps_bounds': {
            'min_lat': 37.7749,
            'min_lon': -122.4194,
            'max_lat': 37.7849,
            'max_lon': -122.4094
        },
        'overlay_timestamp': True,
        'overlay_coordinates': True
    })
    
    # Video processing settings
    video_processing: Dict[str, Any] = field(default_factory=lambda: {
        'max_frames': 10000,
        'batch_size': 10,
        'fps': 30,
        'codec': 'mp4v',
        'quality': 90
    })
    
    # Performance settings
    performance: Dict[str, Any] = field(default_factory=lambda: {
        'max_memory_usage': '2GB',
        'parallel_processing': True,
        'num_workers': 4,
        'cache_size': 1000,
        'enable_gpu': False
    })


@dataclass
class APIConfig:
    """Configuration for API server settings."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    
    # CORS settings
    cors: Dict[str, Any] = field(default_factory=lambda: {
        'allow_origins': ["*"],
        'allow_credentials': True,
        'allow_methods': ["*"],
        'allow_headers': ["*"]
    })
    
    # File upload settings
    upload: Dict[str, Any] = field(default_factory=lambda: {
        'max_file_size': 100 * 1024 * 1024,  # 100MB
        'allowed_extensions': ['.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov'],
        'upload_timeout': 300  # 5 minutes
    })
    
    # Rate limiting
    rate_limiting: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'requests_per_minute': 60,
        'burst_size': 10
    })


@dataclass
class DashboardConfig:
    """Configuration for Streamlit dashboard settings."""
    
    port: int = 8501
    host: str = "0.0.0.0"
    theme: str = "light"  # 'light' or 'dark'
    
    # Display settings
    display: Dict[str, Any] = field(default_factory=lambda: {
        'show_processing_details': True,
        'auto_refresh_interval': 30,  # seconds
        'max_preview_size': (800, 600),
        'show_confidence_scores': True
    })
    
    # Chart settings
    charts: Dict[str, Any] = field(default_factory=lambda: {
        'default_chart_type': 'bar',
        'color_scheme': 'viridis',
        'animation_enabled': True
    })


@dataclass
class DatabaseConfig:
    """Configuration for database settings."""
    
    enabled: bool = False
    type: str = "sqlite"  # 'sqlite', 'postgresql', 'mysql'
    
    # SQLite settings
    sqlite: Dict[str, Any] = field(default_factory=lambda: {
        'path': 'data/redact_ai.db',
        'timeout': 30
    })
    
    # PostgreSQL settings
    postgresql: Dict[str, Any] = field(default_factory=lambda: {
        'host': 'localhost',
        'port': 5432,
        'database': 'redact_ai',
        'username': 'redact_ai',
        'password': 'password'
    })
    
    # MySQL settings
    mysql: Dict[str, Any] = field(default_factory=lambda: {
        'host': 'localhost',
        'port': 3306,
        'database': 'redact_ai',
        'username': 'redact_ai',
        'password': 'password'
    })


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File logging
    file_logging: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'path': 'logs/redact_ai.log',
        'max_size': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5
    })
    
    # Console logging
    console_logging: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'colorize': True
    })


@dataclass
class SecurityConfig:
    """Configuration for security settings."""
    
    # API security
    api_security: Dict[str, Any] = field(default_factory=lambda: {
        'require_authentication': False,
        'api_key_required': False,
        'rate_limiting_enabled': True,
        'cors_enabled': True
    })
    
    # File security
    file_security: Dict[str, Any] = field(default_factory=lambda: {
        'scan_uploads': True,
        'max_file_size': 100 * 1024 * 1024,  # 100MB
        'allowed_mime_types': [
            'image/jpeg', 'image/png', 'video/mp4', 'video/avi', 'video/quicktime'
        ],
        'quarantine_suspicious': True
    })
    
    # Data privacy
    data_privacy: Dict[str, Any] = field(default_factory=lambda: {
        'auto_delete_processed_files': False,
        'retention_days': 30,
        'encrypt_metadata': False,
        'anonymize_logs': True
    })


@dataclass
class RedactAIConfig:
    """Main configuration class for RedactAI."""
    
    # Component configurations
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Global settings
    app_name: str = "RedactAI"
    version: str = "1.0.0"
    debug: bool = False
    environment: str = "production"  # 'development', 'staging', 'production'
    
    # Paths
    base_path: Path = field(default_factory=lambda: Path.cwd())
    data_path: Path = field(default_factory=lambda: Path.cwd() / "data")
    logs_path: Path = field(default_factory=lambda: Path.cwd() / "logs")
    models_path: Path = field(default_factory=lambda: Path.cwd() / "models")
    
    def __post_init__(self):
        """Initialize paths and create directories."""
        self.data_path = self.base_path / "data"
        self.logs_path = self.base_path / "logs"
        self.models_path = self.base_path / "models"
        
        # Create necessary directories
        self.data_path.mkdir(exist_ok=True)
        self.logs_path.mkdir(exist_ok=True)
        self.models_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_path / "input_media").mkdir(exist_ok=True)
        (self.data_path / "output_media").mkdir(exist_ok=True)
        (self.data_path / "metadata").mkdir(exist_ok=True)
        (self.data_path / "sample_data").mkdir(exist_ok=True)


class ConfigManager:
    """Configuration manager for loading and saving settings."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration manager."""
        self.config_path = config_path or Path("config/settings.yaml")
        self.config = RedactAIConfig()
    
    def load_config(self, config_path: Optional[Path] = None) -> RedactAIConfig:
        """Load configuration from file."""
        config_file = config_path or self.config_path
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Update configuration with loaded data
                self._update_config_from_dict(self.config, config_data)
                logger.info(f"Configuration loaded from {config_file}")
                
            except Exception as e:
                logger.warning(f"Failed to load configuration from {config_file}: {e}")
                logger.info("Using default configuration")
        else:
            logger.info(f"Configuration file {config_file} not found, using defaults")
        
        return self.config
    
    def save_config(self, config: RedactAIConfig, config_path: Optional[Path] = None) -> None:
        """Save configuration to file."""
        config_file = config_path or self.config_path
        
        try:
            # Ensure directory exists
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert config to dictionary
            config_dict = self._config_to_dict(config)
            
            with open(config_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_file}: {e}")
    
    def _update_config_from_dict(self, config: RedactAIConfig, data: Dict[str, Any]) -> None:
        """Update configuration object from dictionary."""
        for key, value in data.items():
            if hasattr(config, key):
                if isinstance(value, dict) and hasattr(getattr(config, key), '__dict__'):
                    # Recursively update nested objects
                    self._update_config_from_dict(getattr(config, key), value)
                else:
                    setattr(config, key, value)
    
    def _config_to_dict(self, config: RedactAIConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        result = {}
        for key, value in config.__dict__.items():
            if hasattr(value, '__dict__'):
                result[key] = self._config_to_dict(value)
            else:
                result[key] = value
        return result
    
    def get_processing_config(self) -> ProcessingConfig:
        """Get processing configuration."""
        return self.config.processing
    
    def get_api_config(self) -> APIConfig:
        """Get API configuration."""
        return self.config.api
    
    def get_dashboard_config(self) -> DashboardConfig:
        """Get dashboard configuration."""
        return self.config.dashboard
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        return self.config.database
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        return self.config.logging
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration."""
        return self.config.security


# Global configuration instance
config_manager = ConfigManager()
config = config_manager.load_config()


def get_config() -> RedactAIConfig:
    """Get the global configuration instance."""
    return config


def reload_config() -> RedactAIConfig:
    """Reload configuration from file."""
    global config
    config = config_manager.load_config()
    return config


def save_config() -> None:
    """Save current configuration to file."""
    config_manager.save_config(config)
