"""
Main application entry point for RedactAI.

This module provides the main application that can run both the API and dashboard.
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    
    try:
        import cv2
        logger.info("OpenCV available")
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import numpy
        logger.info("NumPy available")
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import spacy
        logger.info("SpaCy available")
    except ImportError:
        missing_deps.append("spacy")
    
    try:
        import fastapi
        logger.info("FastAPI available")
    except ImportError:
        missing_deps.append("fastapi")
    
    try:
        import streamlit
        logger.info("Streamlit available")
    except ImportError:
        missing_deps.append("streamlit")
    
    if missing_deps:
        logger.error(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.error("Please install missing dependencies with: pip install -r requirements.txt")
        return False
    
    return True


def run_api():
    """Run the FastAPI server."""
    logger.info("Starting RedactAI API server...")
    
    try:
        import uvicorn
        from api.main import app
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Error starting API server: {e}")
        sys.exit(1)


def run_dashboard():
    """Run the Streamlit dashboard."""
    logger.info("Starting RedactAI Dashboard...")
    
    try:
        import streamlit.web.cli as stcli
        
        # Set Streamlit configuration
        os.environ["STREAMLIT_SERVER_PORT"] = "8501"
        os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
        
        # Run Streamlit
        sys.argv = [
            "streamlit",
            "run",
            "dashboard_app/main.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true"
        ]
        
        stcli.main()
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        sys.exit(1)


def run_both():
    """Run both API and dashboard concurrently."""
    logger.info("Starting RedactAI with both API and Dashboard...")
    
    try:
        import multiprocessing
        import signal
        
        # Start API process
        api_process = multiprocessing.Process(target=run_api)
        api_process.start()
        
        # Wait a bit for API to start
        time.sleep(5)
        
        # Start Dashboard process
        dashboard_process = multiprocessing.Process(target=run_dashboard)
        dashboard_process.start()
        
        def signal_handler(sig, frame):
            logger.info("Shutting down RedactAI...")
            api_process.terminate()
            dashboard_process.terminate()
            api_process.join()
            dashboard_process.join()
            sys.exit(0)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Wait for processes
        api_process.join()
        dashboard_process.join()
        
    except Exception as e:
        logger.error(f"Error running both services: {e}")
        sys.exit(1)


def main():
    """Main application entry point."""
    logger.info("RedactAI - AI-powered privacy tool")
    logger.info("Version 1.0.0")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs("data/input_media", exist_ok=True)
    os.makedirs("data/output_media", exist_ok=True)
    os.makedirs("data/metadata", exist_ok=True)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "api":
            run_api()
        elif mode == "dashboard":
            run_dashboard()
        elif mode == "both":
            run_both()
        else:
            print("Usage: python app.py [api|dashboard|both]")
            print("  api       - Run only the FastAPI server")
            print("  dashboard - Run only the Streamlit dashboard")
            print("  both      - Run both API and dashboard (default)")
            sys.exit(1)
    else:
        # Default: run both
        run_both()


if __name__ == "__main__":
    main()
