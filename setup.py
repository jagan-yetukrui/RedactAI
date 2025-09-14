"""
Setup script for RedactAI.

This script helps with initial setup and dependency installation.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(command, description):
    """Run a command and handle errors."""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error: {description}")
        logger.error(f"Command: {command}")
        logger.error(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 10):
        logger.error("Python 3.10 or higher is required")
        return False
    logger.info(f"Python version: {sys.version}")
    return True


def install_dependencies():
    """Install Python dependencies."""
    logger.info("Installing Python dependencies...")
    
    # Upgrade pip first
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True


def download_spacy_model():
    """Download SpaCy model."""
    logger.info("Downloading SpaCy model...")
    return run_command("python -m spacy download en_core_web_sm", "Downloading SpaCy model")


def create_directories():
    """Create necessary directories."""
    logger.info("Creating directories...")
    
    directories = [
        "data/input_media",
        "data/output_media", 
        "data/metadata",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    return True


def run_tests():
    """Run unit tests."""
    logger.info("Running unit tests...")
    return run_command("python -m pytest tests/ -v", "Running unit tests")


def main():
    """Main setup function."""
    logger.info("RedactAI Setup")
    logger.info("=============")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        logger.error("Failed to create directories")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Failed to install dependencies")
        sys.exit(1)
    
    # Download SpaCy model
    if not download_spacy_model():
        logger.error("Failed to download SpaCy model")
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        logger.warning("Some tests failed, but setup continues")
    
    logger.info("Setup completed successfully!")
    logger.info("You can now run RedactAI with: python app.py")


if __name__ == "__main__":
    main()
