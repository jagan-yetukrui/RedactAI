"""
Advanced batch processing system for RedactAI.

This module provides efficient batch processing capabilities for handling
large datasets of images and videos with parallel processing and progress tracking.
"""

import os
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Iterator, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, Empty
import logging
from datetime import datetime, timezone, timedelta
import json

from .monitoring import get_metrics_collector, record_processing
from .logger import get_logger, log_processing

logger = get_logger(__name__)


@dataclass
class BatchJob:
    """A batch processing job."""
    
    job_id: str
    input_path: Path
    output_path: Path
    processing_options: Dict[str, Any]
    status: str = "pending"  # pending, processing, completed, failed
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'job_id': self.job_id,
            'input_path': str(self.input_path),
            'output_path': str(self.output_path),
            'processing_options': self.processing_options,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'result': self.result
        }


@dataclass
class BatchProgress:
    """Batch processing progress information."""
    
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    processing_jobs: int
    pending_jobs: int
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    current_job: Optional[str] = None
    
    @property
    def progress_percent(self) -> float:
        """Get progress percentage."""
        if self.total_jobs == 0:
            return 0.0
        return (self.completed_jobs + self.failed_jobs) / self.total_jobs * 100
    
    @property
    def success_rate(self) -> float:
        """Get success rate percentage."""
        completed = self.completed_jobs + self.failed_jobs
        if completed == 0:
            return 0.0
        return self.completed_jobs / completed * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_jobs': self.total_jobs,
            'completed_jobs': self.completed_jobs,
            'failed_jobs': self.failed_jobs,
            'processing_jobs': self.processing_jobs,
            'pending_jobs': self.pending_jobs,
            'progress_percent': self.progress_percent,
            'success_rate': self.success_rate,
            'start_time': self.start_time.isoformat(),
            'estimated_completion': self.estimated_completion.isoformat() if self.estimated_completion else None,
            'current_job': self.current_job
        }


class BatchProcessor:
    """Advanced batch processor for media files."""
    
    def __init__(self, max_workers: int = 4, use_multiprocessing: bool = False,
                 progress_callback: Optional[Callable[[BatchProgress], None]] = None):
        """
        Initialize batch processor.
        
        Args:
            max_workers: Maximum number of worker threads/processes
            use_multiprocessing: Whether to use multiprocessing instead of threading
            progress_callback: Callback function for progress updates
        """
        self.max_workers = max_workers
        self.use_multiprocessing = use_multiprocessing
        self.progress_callback = progress_callback
        
        # Job management
        self.jobs: Dict[str, BatchJob] = {}
        self.job_queue: Queue = Queue()
        self.completed_jobs: List[BatchJob] = []
        self.failed_jobs: List[BatchJob] = []
        
        # Progress tracking
        self.progress = BatchProgress(
            total_jobs=0,
            completed_jobs=0,
            failed_jobs=0,
            processing_jobs=0,
            pending_jobs=0,
            start_time=datetime.now(timezone.utc)
        )
        
        # Threading
        self.lock = threading.Lock()
        self.is_processing = False
        self.stop_event = threading.Event()
        
        # Metrics
        self.metrics_collector = get_metrics_collector()
    
    def add_job(self, input_path: Path, output_path: Path, 
                processing_options: Dict[str, Any], job_id: Optional[str] = None) -> str:
        """Add a job to the batch."""
        if job_id is None:
            job_id = f"job_{len(self.jobs)}_{int(time.time())}"
        
        job = BatchJob(
            job_id=job_id,
            input_path=input_path,
            output_path=output_path,
            processing_options=processing_options
        )
        
        with self.lock:
            self.jobs[job_id] = job
            self.job_queue.put(job)
            self.progress.total_jobs += 1
            self.progress.pending_jobs += 1
        
        logger.info(f"Added job {job_id}: {input_path} -> {output_path}")
        return job_id
    
    def add_jobs_from_directory(self, input_dir: Path, output_dir: Path,
                               processing_options: Dict[str, Any],
                               file_extensions: List[str] = None) -> List[str]:
        """Add jobs for all files in a directory."""
        if file_extensions is None:
            file_extensions = ['.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov']
        
        job_ids = []
        
        for file_path in input_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                # Create output path maintaining directory structure
                relative_path = file_path.relative_to(input_dir)
                output_path = output_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                job_id = self.add_job(file_path, output_path, processing_options)
                job_ids.append(job_id)
        
        logger.info(f"Added {len(job_ids)} jobs from directory {input_dir}")
        return job_ids
    
    def process_job(self, job: BatchJob, processor_func: Callable) -> BatchJob:
        """Process a single job."""
        job.status = "processing"
        job.started_at = datetime.now(timezone.utc)
        
        try:
            with log_processing(f"batch_job_{job.job_id}", job_id=job.job_id):
                # Process the file
                result = processor_func(
                    str(job.input_path),
                    str(job.output_path),
                    **job.processing_options
                )
                
                # Update job status
                job.status = "completed"
                job.completed_at = datetime.now(timezone.utc)
                job.result = result
                
                # Record metrics
                processing_time = (job.completed_at - job.started_at).total_seconds()
                file_type = job.input_path.suffix.lower()
                
                record_processing(
                    processing_time=processing_time,
                    file_type=file_type,
                    faces=result.get('faces_detected', 0),
                    plates=result.get('plates_detected', 0),
                    text_regions=result.get('text_regions_detected', 0),
                    names=result.get('names_redacted', 0),
                    frames=result.get('frames_processed', 0)
                )
                
                logger.info(f"Job {job.job_id} completed successfully")
                
        except Exception as e:
            job.status = "failed"
            job.completed_at = datetime.now(timezone.utc)
            job.error_message = str(e)
            
            # Record error metrics
            processing_time = (job.completed_at - job.started_at).total_seconds()
            record_processing(
                processing_time=processing_time,
                file_type=job.input_path.suffix.lower(),
                error=True
            )
            
            logger.error(f"Job {job.job_id} failed: {e}")
        
        return job
    
    def process_batch(self, processor_func: Callable, 
                     progress_interval: float = 1.0) -> BatchProgress:
        """
        Process all jobs in the batch.
        
        Args:
            processor_func: Function to process individual files
            progress_interval: Interval for progress updates in seconds
            
        Returns:
            Final progress information
        """
        if self.is_processing:
            raise RuntimeError("Batch processing already in progress")
        
        self.is_processing = True
        self.stop_event.clear()
        self.progress.start_time = datetime.now(timezone.utc)
        
        logger.info(f"Starting batch processing with {self.max_workers} workers")
        
        try:
            # Choose executor type
            executor_class = ProcessPoolExecutor if self.use_multiprocessing else ThreadPoolExecutor
            
            with executor_class(max_workers=self.max_workers) as executor:
                # Submit all jobs
                future_to_job = {}
                
                while not self.job_queue.empty() and not self.stop_event.is_set():
                    try:
                        job = self.job_queue.get_nowait()
                        future = executor.submit(self.process_job, job, processor_func)
                        future_to_job[future] = job
                    except Empty:
                        break
                
                # Process completed jobs
                start_time = time.time()
                last_progress_update = start_time
                
                for future in as_completed(future_to_job):
                    if self.stop_event.is_set():
                        break
                    
                    job = future_to_job[future]
                    
                    try:
                        completed_job = future.result()
                        
                        with self.lock:
                            if completed_job.status == "completed":
                                self.completed_jobs.append(completed_job)
                                self.progress.completed_jobs += 1
                            else:
                                self.failed_jobs.append(completed_job)
                                self.progress.failed_jobs += 1
                            
                            self.progress.processing_jobs -= 1
                            self.progress.pending_jobs = max(0, self.progress.pending_jobs - 1)
                            
                            # Update current job
                            if self.job_queue.empty():
                                self.progress.current_job = None
                            else:
                                self.progress.current_job = "processing..."
                            
                            # Estimate completion time
                            if self.progress.completed_jobs > 0:
                                elapsed = time.time() - start_time
                                rate = self.progress.completed_jobs / elapsed
                                remaining = self.progress.pending_jobs + self.progress.processing_jobs
                                if rate > 0:
                                    eta_seconds = remaining / rate
                                    self.progress.estimated_completion = datetime.now(timezone.utc) + timedelta(seconds=eta_seconds)
                    
                    except Exception as e:
                        logger.error(f"Error processing job {job.job_id}: {e}")
                        
                        with self.lock:
                            job.status = "failed"
                            job.error_message = str(e)
                            self.failed_jobs.append(job)
                            self.progress.failed_jobs += 1
                            self.progress.processing_jobs -= 1
                    
                    # Update progress callback
                    current_time = time.time()
                    if (self.progress_callback and 
                        current_time - last_progress_update >= progress_interval):
                        self.progress_callback(self.progress)
                        last_progress_update = current_time
        
        finally:
            self.is_processing = False
            logger.info("Batch processing completed")
        
        return self.progress
    
    def stop_processing(self) -> None:
        """Stop batch processing."""
        self.stop_event.set()
        logger.info("Stop signal sent for batch processing")
    
    def get_job_status(self, job_id: str) -> Optional[BatchJob]:
        """Get status of a specific job."""
        return self.jobs.get(job_id)
    
    def get_progress(self) -> BatchProgress:
        """Get current progress."""
        return self.progress
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of processing results."""
        total_processing_time = sum(
            (job.completed_at - job.started_at).total_seconds()
            for job in self.completed_jobs + self.failed_jobs
            if job.started_at and job.completed_at
        )
        
        total_faces = sum(
            job.result.get('faces_detected', 0) if job.result else 0
            for job in self.completed_jobs
        )
        
        total_plates = sum(
            job.result.get('plates_detected', 0) if job.result else 0
            for job in self.completed_jobs
        )
        
        total_text_regions = sum(
            job.result.get('text_regions_detected', 0) if job.result else 0
            for job in self.completed_jobs
        )
        
        total_names = sum(
            job.result.get('names_redacted', 0) if job.result else 0
            for job in self.completed_jobs
        )
        
        return {
            'total_jobs': self.progress.total_jobs,
            'completed_jobs': self.progress.completed_jobs,
            'failed_jobs': self.progress.failed_jobs,
            'success_rate': self.progress.success_rate,
            'total_processing_time': total_processing_time,
            'average_processing_time': total_processing_time / max(self.progress.completed_jobs, 1),
            'total_faces_detected': total_faces,
            'total_plates_detected': total_plates,
            'total_text_regions_detected': total_text_regions,
            'total_names_redacted': total_names,
            'start_time': self.progress.start_time.isoformat(),
            'end_time': datetime.now(timezone.utc).isoformat()
        }
    
    def export_results(self, output_path: Path) -> None:
        """Export processing results to file."""
        results = {
            'summary': self.get_results_summary(),
            'jobs': [job.to_dict() for job in self.completed_jobs + self.failed_jobs],
            'progress': self.progress.to_dict()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results exported to {output_path}")
    
    def clear_jobs(self) -> None:
        """Clear all jobs and reset state."""
        with self.lock:
            self.jobs.clear()
            self.completed_jobs.clear()
            self.failed_jobs.clear()
            
            # Clear queue
            while not self.job_queue.empty():
                try:
                    self.job_queue.get_nowait()
                except Empty:
                    break
            
            # Reset progress
            self.progress = BatchProgress(
                total_jobs=0,
                completed_jobs=0,
                failed_jobs=0,
                processing_jobs=0,
                pending_jobs=0,
                start_time=datetime.now(timezone.utc)
            )
        
        logger.info("All jobs cleared")


def create_batch_processor(max_workers: int = 4, use_multiprocessing: bool = False,
                          progress_callback: Optional[Callable[[BatchProgress], None]] = None) -> BatchProcessor:
    """Create a new batch processor instance."""
    return BatchProcessor(max_workers, use_multiprocessing, progress_callback)


def process_directory_batch(input_dir: Path, output_dir: Path,
                           processor_func: Callable, processing_options: Dict[str, Any],
                           max_workers: int = 4, file_extensions: List[str] = None) -> Dict[str, Any]:
    """
    Process all files in a directory using batch processing.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        processor_func: Function to process individual files
        processing_options: Processing options
        max_workers: Maximum number of workers
        file_extensions: List of file extensions to process
        
    Returns:
        Processing results summary
    """
    processor = create_batch_processor(max_workers)
    
    # Add jobs from directory
    processor.add_jobs_from_directory(input_dir, output_dir, processing_options, file_extensions)
    
    # Process batch
    progress = processor.process_batch(processor_func)
    
    # Get results
    results = processor.get_results_summary()
    
    logger.info(f"Batch processing completed: {results['completed_jobs']}/{results['total_jobs']} jobs successful")
    
    return results
