"""
Command-line interface for RedactAI.

This module provides a comprehensive CLI for all RedactAI functionality
including processing, batch operations, and system management.
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from config import get_config
from utils.logger import get_logger, setup_logging
from utils.monitoring import get_metrics_collector
from utils.batch_processor import create_batch_processor, process_directory_batch
from utils.model_manager import get_model_manager
from modules.face_blur import FaceBlurrer
from modules.plate_blur import PlateBlurrer
from modules.text_redact import TextRedactor
from modules.geotagging import Geotagger

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="redact-ai",
        description="RedactAI - AI-powered privacy tool for redacting sensitive information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image
  redact-ai process image.jpg --output processed.jpg --faces --plates --text
  
  # Process all images in a directory
  redact-ai batch process input_dir/ --output output_dir/ --faces --plates
  
  # Start the web interface
  redact-ai serve --api --dashboard
  
  # Generate sample data
  redact-ai generate-sample-data --output data/sample_data/
  
  # Show system status
  redact-ai status
        """
    )
    
    # Global options
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    parser.add_argument("--config", "-c", type=Path, help="Configuration file path")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Set logging level")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process a single file")
    process_parser.add_argument("input", type=Path, help="Input file path")
    process_parser.add_argument("--output", "-o", type=Path, help="Output file path")
    process_parser.add_argument("--faces", action="store_true", help="Process faces")
    process_parser.add_argument("--plates", action="store_true", help="Process license plates")
    process_parser.add_argument("--text", action="store_true", help="Process text")
    process_parser.add_argument("--names-only", action="store_true", help="Redact names only")
    process_parser.add_argument("--blur-type", choices=["gaussian", "pixelate", "blackout", "mosaic"],
                               default="gaussian", help="Blur type")
    process_parser.add_argument("--blur-strength", type=int, default=15, help="Blur strength")
    process_parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence")
    process_parser.add_argument("--geotags", action="store_true", help="Add geotags")
    process_parser.add_argument("--gps-lat", type=float, help="GPS latitude")
    process_parser.add_argument("--gps-lon", type=float, help="GPS longitude")
    
    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch processing operations")
    batch_subparsers = batch_parser.add_subparsers(dest="batch_command", help="Batch operations")
    
    # Batch process
    batch_process_parser = batch_subparsers.add_parser("process", help="Process multiple files")
    batch_process_parser.add_argument("input_dir", type=Path, help="Input directory")
    batch_process_parser.add_argument("--output", "-o", type=Path, help="Output directory")
    batch_process_parser.add_argument("--faces", action="store_true", help="Process faces")
    batch_process_parser.add_argument("--plates", action="store_true", help="Process license plates")
    batch_process_parser.add_argument("--text", action="store_true", help="Process text")
    batch_process_parser.add_argument("--names-only", action="store_true", help="Redact names only")
    batch_process_parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    batch_process_parser.add_argument("--extensions", nargs="+", 
                                     default=[".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov"],
                                     help="File extensions to process")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start web services")
    serve_parser.add_argument("--api", action="store_true", help="Start API server")
    serve_parser.add_argument("--dashboard", action="store_true", help="Start dashboard")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host address")
    serve_parser.add_argument("--api-port", type=int, default=8000, help="API port")
    serve_parser.add_argument("--dashboard-port", type=int, default=8501, help="Dashboard port")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # Generate sample data command
    sample_parser = subparsers.add_parser("generate-sample-data", help="Generate sample data")
    sample_parser.add_argument("--output", "-o", type=Path, default=Path("data/sample_data"),
                              help="Output directory")
    
    # Model management commands
    model_parser = subparsers.add_parser("models", help="Model management")
    model_subparsers = model_parser.add_subparsers(dest="model_command", help="Model operations")
    
    # List models
    model_list_parser = model_subparsers.add_parser("list", help="List loaded models")
    
    # Load model
    model_load_parser = model_subparsers.add_parser("load", help="Load a model")
    model_load_parser.add_argument("name", help="Model name")
    model_load_parser.add_argument("type", help="Model type")
    model_load_parser.add_argument("--path", type=Path, help="Model path")
    
    # Unload model
    model_unload_parser = model_subparsers.add_parser("unload", help="Unload a model")
    model_unload_parser.add_argument("name", help="Model name")
    
    # Cleanup models
    model_cleanup_parser = model_subparsers.add_parser("cleanup", help="Cleanup unused models")
    model_cleanup_parser.add_argument("--age-hours", type=int, default=24, help="Max age in hours")
    
    return parser


def process_single_file(args) -> int:
    """Process a single file."""
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = args.input.parent / f"processed_{args.input.name}"
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing {args.input} -> {output_path}")
    
    try:
        import cv2
        import numpy as np
        
        # Load image
        image = cv2.imread(str(args.input))
        if image is None:
            logger.error(f"Could not load image: {args.input}")
            return 1
        
        # Process based on options
        if args.faces:
            logger.info("Processing faces...")
            face_processor = FaceBlurrer(blur_type=args.blur_type, blur_strength=args.blur_strength)
            image, faces = face_processor.process_image(image, args.confidence)
            logger.info(f"Detected {len(faces)} faces")
        
        if args.plates:
            logger.info("Processing license plates...")
            plate_processor = PlateBlurrer(blur_type=args.blur_type, blur_strength=args.blur_strength)
            image, plates = plate_processor.process_image(image, args.confidence)
            logger.info(f"Detected {len(plates)} license plates")
        
        if args.text:
            logger.info("Processing text...")
            text_processor = TextRedactor(blur_type=args.blur_type, blur_strength=args.blur_strength,
                                        redact_names_only=args.names_only)
            image, stats = text_processor.process_image(image, args.confidence)
            logger.info(f"Detected {stats.get('total_text_regions', 0)} text regions")
        
        if args.geotags:
            logger.info("Adding geotags...")
            geotagger = Geotagger()
            gps_coords = (args.gps_lat, args.gps_lon) if args.gps_lat and args.gps_lon else None
            image = geotagger.add_geotag_to_image(image, gps_coords)
        
        # Save result
        cv2.imwrite(str(output_path), image)
        logger.info(f"Saved processed image to {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return 1


def process_batch(args) -> int:
    """Process multiple files in batch."""
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1
    
    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = args.input_dir.parent / f"{args.input_dir.name}_processed"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Batch processing {args.input_dir} -> {output_dir}")
    
    # Create processing options
    processing_options = {
        'process_faces': args.faces,
        'process_plates': args.plates,
        'process_text': args.text,
        'redact_names_only': args.names_only,
        'blur_type': 'gaussian',
        'blur_strength': 15,
        'confidence': 0.5
    }
    
    try:
        # Process directory
        results = process_directory_batch(
            input_dir=args.input_dir,
            output_dir=output_dir,
            processor_func=process_single_file,  # This would need to be adapted
            processing_options=processing_options,
            max_workers=args.workers,
            file_extensions=args.extensions
        )
        
        logger.info(f"Batch processing completed: {results['completed_jobs']}/{results['total_jobs']} successful")
        return 0
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return 1


def serve_services(args) -> int:
    """Start web services."""
    if not args.api and not args.dashboard:
        # Start both by default
        args.api = True
        args.dashboard = True
    
    try:
        if args.api and args.dashboard:
            # Start both services
            from app import run_both
            run_both()
        elif args.api:
            # Start API only
            from app import run_api
            run_api()
        elif args.dashboard:
            # Start dashboard only
            from app import run_dashboard
            run_dashboard()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error starting services: {e}")
        return 1


def show_status(args) -> int:
    """Show system status."""
    try:
        metrics_collector = get_metrics_collector()
        stats = metrics_collector.get_metrics_summary()
        
        if args.json:
            print(json.dumps(stats, indent=2, default=str))
        else:
            print("RedactAI System Status")
            print("=" * 50)
            print(f"Total files processed: {stats['processing']['total_files_processed']}")
            print(f"Faces detected: {stats['processing']['total_faces_detected']}")
            print(f"Plates detected: {stats['processing']['total_plates_detected']}")
            print(f"Text regions detected: {stats['processing']['total_text_regions_detected']}")
            print(f"Names redacted: {stats['processing']['total_names_redacted']}")
            print(f"Average processing time: {stats['processing']['average_processing_time_seconds']:.2f}s")
            print(f"System healthy: {metrics_collector.health_checker.is_healthy()}")
            print(f"CPU usage: {stats['system']['cpu_percent']:.1f}%")
            print(f"Memory usage: {stats['system']['memory_percent']:.1f}%")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return 1


def generate_sample_data(args) -> int:
    """Generate sample data."""
    try:
        from data.sample_data.generate_sample_data import generate_all_sample_data
        
        # Update output directory
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Change to output directory
        original_cwd = Path.cwd()
        os.chdir(args.output.parent)
        
        generate_all_sample_data()
        
        os.chdir(original_cwd)
        
        logger.info(f"Sample data generated in {args.output}")
        return 0
        
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        return 1


def manage_models(args) -> int:
    """Manage models."""
    model_manager = get_model_manager()
    
    try:
        if args.model_command == "list":
            models = model_manager.list_loaded_models()
            if models:
                print("Loaded models:")
                for model in models:
                    info = model_manager.get_model_info(model)
                    print(f"  {model} ({info.type}) - {info.size_bytes} bytes")
            else:
                print("No models loaded")
        
        elif args.model_command == "load":
            model = model_manager.load_model(args.name, args.type, args.path)
            print(f"Loaded model {args.name}")
        
        elif args.model_command == "unload":
            if model_manager.unload_model(args.name):
                print(f"Unloaded model {args.name}")
            else:
                print(f"Model {args.name} not found")
        
        elif args.model_command == "cleanup":
            removed = model_manager.cleanup_unused_models(args.age_hours)
            print(f"Cleaned up {removed} unused models")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error managing models: {e}")
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.ERROR
    else:
        log_level = getattr(logging, args.log_level)
    
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load configuration if specified
    if args.config:
        from config import reload_config
        reload_config()
    
    # Handle commands
    if args.command == "process":
        return process_single_file(args)
    elif args.command == "batch":
        return process_batch(args)
    elif args.command == "serve":
        return serve_services(args)
    elif args.command == "status":
        return show_status(args)
    elif args.command == "generate-sample-data":
        return generate_sample_data(args)
    elif args.command == "models":
        return manage_models(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
