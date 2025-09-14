"""
Advanced installation script for RedactAI.

This script provides comprehensive installation, setup, and verification
for the RedactAI system with all dependencies and configurations.
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RedactAIInstaller:
    """Advanced installer for RedactAI."""
    
    def __init__(self):
        """Initialize installer."""
        self.system = platform.system().lower()
        self.python_version = sys.version_info
        self.project_root = Path(__file__).parent
        self.install_log = []
        
        # System requirements
        self.requirements = {
            'python': (3, 10),
            'pip': True,
            'git': True,
            'tesseract': True,
            'opencv_deps': True
        }
        
        # Python packages
        self.python_packages = [
            'opencv-python==4.8.1.78',
            'numpy==1.24.3',
            'Pillow==10.0.1',
            'spacy==3.7.2',
            'ultralytics==8.0.196',
            'easyocr==1.7.0',
            'pytesseract==0.3.10',
            'fastapi==0.104.1',
            'uvicorn[standard]==0.24.0',
            'streamlit==1.28.1',
            'streamlit-folium==0.13.0',
            'pandas==2.1.3',
            'plotly==5.17.0',
            'folium==0.14.0',
            'requests==2.31.0',
            'httpx==0.25.2',
            'pytest==7.4.3',
            'pytest-asyncio==0.21.1',
            'python-multipart==0.0.6',
            'pydantic==2.5.0',
            'python-dotenv==1.0.0',
            'psutil==5.9.6',
            'pyyaml==6.0.1'
        ]
    
    def log_step(self, step: str, success: bool = True, message: str = ""):
        """Log installation step."""
        status = "✓" if success else "✗"
        log_entry = {
            'step': step,
            'success': success,
            'message': message,
            'timestamp': time.time()
        }
        self.install_log.append(log_entry)
        
        if success:
            logger.info(f"{status} {step}")
        else:
            logger.error(f"{status} {step}: {message}")
    
    def check_python_version(self) -> bool:
        """Check Python version compatibility."""
        required_major, required_minor = self.requirements['python']
        
        if self.python_version.major < required_major or \
           (self.python_version.major == required_major and self.python_version.minor < required_minor):
            self.log_step(
                "Python version check",
                False,
                f"Python {required_major}.{required_minor}+ required, found {self.python_version.major}.{self.python_version.minor}"
            )
            return False
        
        self.log_step(f"Python version check ({self.python_version.major}.{self.python_version.minor})")
        return True
    
    def check_system_dependencies(self) -> bool:
        """Check system dependencies."""
        all_good = True
        
        # Check pip
        try:
            subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                         capture_output=True, check=True)
            self.log_step("pip availability")
        except subprocess.CalledProcessError:
            self.log_step("pip availability", False, "pip not available")
            all_good = False
        
        # Check git
        try:
            subprocess.run(['git', '--version'], capture_output=True, check=True)
            self.log_step("git availability")
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.log_step("git availability", False, "git not available")
            all_good = False
        
        # Check Tesseract
        try:
            if self.system == 'windows':
                subprocess.run(['tesseract', '--version'], capture_output=True, check=True)
            else:
                subprocess.run(['tesseract', '--version'], capture_output=True, check=True)
            self.log_step("Tesseract availability")
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.log_step("Tesseract availability", False, "Tesseract not found")
            all_good = False
        
        return all_good
    
    def install_system_dependencies(self) -> bool:
        """Install system dependencies."""
        if self.system == 'linux':
            return self._install_linux_dependencies()
        elif self.system == 'darwin':
            return self._install_macos_dependencies()
        elif self.system == 'windows':
            return self._install_windows_dependencies()
        else:
            self.log_step("System dependencies", False, f"Unsupported system: {self.system}")
            return False
    
    def _install_linux_dependencies(self) -> bool:
        """Install Linux dependencies."""
        try:
            # Update package list
            subprocess.run(['sudo', 'apt-get', 'update'], check=True)
            
            # Install dependencies
            packages = [
                'tesseract-ocr',
                'tesseract-ocr-eng',
                'libgl1-mesa-glx',
                'libglib2.0-0',
                'libsm6',
                'libxext6',
                'libxrender-dev',
                'libgomp1',
                'libgcc-s1'
            ]
            
            for package in packages:
                subprocess.run(['sudo', 'apt-get', 'install', '-y', package], check=True)
            
            self.log_step("Linux dependencies installed")
            return True
            
        except subprocess.CalledProcessError as e:
            self.log_step("Linux dependencies", False, str(e))
            return False
    
    def _install_macos_dependencies(self) -> bool:
        """Install macOS dependencies."""
        try:
            # Check if Homebrew is installed
            subprocess.run(['brew', '--version'], capture_output=True, check=True)
            
            # Install dependencies
            packages = ['tesseract', 'opencv']
            
            for package in packages:
                subprocess.run(['brew', 'install', package], check=True)
            
            self.log_step("macOS dependencies installed")
            return True
            
        except subprocess.CalledProcessError as e:
            self.log_step("macOS dependencies", False, str(e))
            return False
    
    def _install_windows_dependencies(self) -> bool:
        """Install Windows dependencies."""
        try:
            # Install Tesseract via chocolatey or direct download
            # This is a simplified version - in practice, you'd want more robust handling
            self.log_step("Windows dependencies", False, "Please install Tesseract manually from https://github.com/UB-Mannheim/tesseract/wiki")
            return False
            
        except Exception as e:
            self.log_step("Windows dependencies", False, str(e))
            return False
    
    def install_python_packages(self) -> bool:
        """Install Python packages."""
        try:
            # Upgrade pip first
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'
            ], check=True)
            
            # Install packages
            for package in self.python_packages:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package
                ], check=True)
            
            self.log_step("Python packages installed")
            return True
            
        except subprocess.CalledProcessError as e:
            self.log_step("Python packages", False, str(e))
            return False
    
    def download_spacy_model(self) -> bool:
        """Download SpaCy model."""
        try:
            subprocess.run([
                sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'
            ], check=True)
            
            self.log_step("SpaCy model downloaded")
            return True
            
        except subprocess.CalledProcessError as e:
            self.log_step("SpaCy model", False, str(e))
            return False
    
    def create_directories(self) -> bool:
        """Create necessary directories."""
        try:
            directories = [
                'data/input_media',
                'data/output_media',
                'data/metadata',
                'data/sample_data',
                'logs',
                'models',
                'config'
            ]
            
            for directory in directories:
                path = self.project_root / directory
                path.mkdir(parents=True, exist_ok=True)
            
            self.log_step("Directories created")
            return True
            
        except Exception as e:
            self.log_step("Directories", False, str(e))
            return False
    
    def generate_sample_data(self) -> bool:
        """Generate sample data for testing."""
        try:
            from data.sample_data.generate_sample_data import generate_all_sample_data
            generate_all_sample_data()
            
            self.log_step("Sample data generated")
            return True
            
        except Exception as e:
            self.log_step("Sample data", False, str(e))
            return False
    
    def run_tests(self) -> bool:
        """Run test suite."""
        try:
            # Run basic tests
            subprocess.run([
                sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'
            ], check=True, cwd=self.project_root)
            
            self.log_step("Tests passed")
            return True
            
        except subprocess.CalledProcessError as e:
            self.log_step("Tests", False, str(e))
            return False
    
    def create_config_file(self) -> bool:
        """Create default configuration file."""
        try:
            config_data = {
                'processing': {
                    'face_detection': {
                        'cascade_path': 'haarcascade_frontalface_default.xml',
                        'scale_factor': 1.1,
                        'min_neighbors': 5,
                        'min_size': [30, 30],
                        'confidence_threshold': 0.5,
                        'blur_type': 'gaussian',
                        'blur_strength': 15
                    },
                    'plate_detection': {
                        'model_path': None,
                        'confidence_threshold': 0.5,
                        'blur_type': 'gaussian',
                        'blur_strength': 15,
                        'use_yolo': True,
                        'fallback_to_opencv': True
                    },
                    'text_detection': {
                        'ocr_engine': 'tesseract',
                        'languages': ['en'],
                        'confidence_threshold': 0.5,
                        'redact_names_only': True,
                        'blur_type': 'gaussian',
                        'blur_strength': 15,
                        'spacy_model': 'en_core_web_sm'
                    },
                    'geotagging': {
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
                    }
                },
                'api': {
                    'host': '0.0.0.0',
                    'port': 8000,
                    'workers': 1,
                    'reload': False,
                    'log_level': 'info'
                },
                'dashboard': {
                    'port': 8501,
                    'host': '0.0.0.0',
                    'theme': 'light'
                },
                'logging': {
                    'level': 'INFO',
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                }
            }
            
            config_path = self.project_root / 'config' / 'settings.yaml'
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            self.log_step("Configuration file created")
            return True
            
        except Exception as e:
            self.log_step("Configuration", False, str(e))
            return False
    
    def verify_installation(self) -> bool:
        """Verify installation."""
        try:
            # Test imports
            import cv2
            import numpy as np
            import spacy
            import fastapi
            import streamlit
            
            # Test basic functionality
            from modules.face_blur import FaceBlurrer
            from modules.plate_blur import PlateBlurrer
            from modules.text_redact import TextRedactor
            
            # Test API
            from api.main import app
            
            self.log_step("Installation verification")
            return True
            
        except Exception as e:
            self.log_step("Installation verification", False, str(e))
            return False
    
    def install(self, skip_system_deps: bool = False, skip_tests: bool = False) -> bool:
        """Run complete installation."""
        logger.info("Starting RedactAI installation...")
        logger.info(f"System: {self.system}")
        logger.info(f"Python: {self.python_version.major}.{self.python_version.minor}")
        
        steps = [
            ("Python version check", self.check_python_version),
            ("System dependencies check", self.check_system_dependencies),
            ("System dependencies install", self.install_system_dependencies if not skip_system_deps else lambda: True),
            ("Python packages install", self.install_python_packages),
            ("SpaCy model download", self.download_spacy_model),
            ("Directories creation", self.create_directories),
            ("Configuration file creation", self.create_config_file),
            ("Sample data generation", self.generate_sample_data),
            ("Tests execution", self.run_tests if not skip_tests else lambda: True),
            ("Installation verification", self.verify_installation)
        ]
        
        success_count = 0
        total_steps = len(steps)
        
        for step_name, step_func in steps:
            try:
                if step_func():
                    success_count += 1
                else:
                    logger.warning(f"Step '{step_name}' failed, continuing...")
            except Exception as e:
                logger.error(f"Step '{step_name}' failed with exception: {e}")
        
        # Generate installation report
        self._generate_report()
        
        success_rate = success_count / total_steps
        if success_rate >= 0.8:  # 80% success rate
            logger.info(f"Installation completed with {success_count}/{total_steps} steps successful")
            logger.info("RedactAI is ready to use!")
            logger.info("Run 'python app.py' to start the application")
            return True
        else:
            logger.error(f"Installation failed with only {success_count}/{total_steps} steps successful")
            return False
    
    def _generate_report(self) -> None:
        """Generate installation report."""
        report = {
            'installation_time': time.time(),
            'system': self.system,
            'python_version': f"{self.python_version.major}.{self.python_version.minor}",
            'steps': self.install_log,
            'success_count': sum(1 for step in self.install_log if step['success']),
            'total_steps': len(self.install_log)
        }
        
        report_path = self.project_root / 'logs' / 'installation_report.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Installation report saved to {report_path}")


def main():
    """Main installation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RedactAI Installation Script")
    parser.add_argument("--skip-system-deps", action="store_true", 
                       help="Skip system dependencies installation")
    parser.add_argument("--skip-tests", action="store_true", 
                       help="Skip running tests")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    installer = RedactAIInstaller()
    success = installer.install(
        skip_system_deps=args.skip_system_deps,
        skip_tests=args.skip_tests
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
