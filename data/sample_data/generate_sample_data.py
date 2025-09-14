"""
Sample data generator for RedactAI testing.

This script generates synthetic test data including images with faces,
license plates, and text for comprehensive testing of the RedactAI pipeline.
"""

import cv2
import numpy as np
import random
import os
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_face_image(width=800, height=600, num_faces=3):
    """Create a sample image with synthetic faces."""
    # Create base image
    image = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Add some background elements
    cv2.rectangle(image, (50, 50), (width-50, height-50), (200, 200, 200), 2)
    
    # Generate synthetic faces (simple circles with features)
    for i in range(num_faces):
        # Random position
        x = random.randint(100, width - 200)
        y = random.randint(100, height - 200)
        
        # Face (circle)
        face_radius = random.randint(40, 80)
        cv2.circle(image, (x, y), face_radius, (220, 180, 150), -1)  # Skin color
        
        # Eyes
        eye_y = y - 20
        cv2.circle(image, (x - 20, eye_y), 8, (0, 0, 0), -1)
        cv2.circle(image, (x + 20, eye_y), 8, (0, 0, 0), -1)
        
        # Nose
        cv2.circle(image, (x, y), 5, (200, 160, 130), -1)
        
        # Mouth
        cv2.ellipse(image, (x, y + 20), (15, 8), 0, 0, 180, (0, 0, 0), 2)
        
        # Add some text near the face
        cv2.putText(image, f"Person {i+1}", (x - 30, y + face_radius + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return image


def create_sample_license_plate_image(width=800, height=600, num_plates=2):
    """Create a sample image with synthetic license plates."""
    # Create base image
    image = np.ones((height, width, 3), dtype=np.uint8) * 200  # Gray background
    
    # Add road-like background
    cv2.rectangle(image, (0, height//2), (width, height), (100, 100, 100), -1)
    
    # Generate synthetic license plates
    for i in range(num_plates):
        # Random position
        x = random.randint(100, width - 200)
        y = random.randint(height//2 - 100, height//2 + 50)
        
        # License plate background
        plate_width = 120
        plate_height = 40
        cv2.rectangle(image, (x, y), (x + plate_width, y + plate_height), (255, 255, 255), -1)
        cv2.rectangle(image, (x, y), (x + plate_width, y + plate_height), (0, 0, 0), 2)
        
        # License plate text
        plate_text = f"ABC{random.randint(100, 999)}"
        cv2.putText(image, plate_text, (x + 10, y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add some context text
        cv2.putText(image, f"Vehicle {i+1}", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return image


def create_sample_text_image(width=800, height=600, num_text_regions=4):
    """Create a sample image with text and names."""
    # Create base image
    image = np.ones((height, width, 3), dtype=np.uint8) * 250  # White background
    
    # Add some background elements
    cv2.rectangle(image, (50, 50), (width-50, height-50), (0, 0, 0), 2)
    
    # Sample names and text
    names = ["John Smith", "Sarah Johnson", "Michael Brown", "Emily Davis"]
    addresses = ["123 Main St", "456 Oak Ave", "789 Pine Rd", "321 Elm St"]
    companies = ["TechCorp Inc", "DataSoft LLC", "CloudTech Co", "AI Solutions"]
    
    y_offset = 100
    for i in range(num_text_regions):
        # Name
        name = names[i % len(names)]
        cv2.putText(image, f"Name: {name}", (100, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        y_offset += 40
        
        # Address
        address = addresses[i % len(addresses)]
        cv2.putText(image, f"Address: {address}", (100, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_offset += 30
        
        # Company
        company = companies[i % len(companies)]
        cv2.putText(image, f"Company: {company}", (100, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_offset += 50
        
        # Add some decorative elements
        cv2.line(image, (80, y_offset - 20), (width - 80, y_offset - 20), (200, 200, 200), 1)
        y_offset += 20
    
    return image


def create_sample_mixed_image(width=800, height=600):
    """Create a complex sample image with faces, plates, and text."""
    # Create base image
    image = np.ones((height, width, 3), dtype=np.uint8) * 220
    
    # Add faces
    face_image = create_sample_face_image(width, height//2, 2)
    image[:height//2, :] = face_image[:height//2, :]
    
    # Add license plates
    plate_image = create_sample_license_plate_image(width, height//2, 1)
    image[height//2:, :] = plate_image[height//2:, :]
    
    # Add text overlay
    cv2.putText(image, "CONFIDENTIAL DOCUMENT", (width//2 - 150, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    return image


def create_sample_video_frames(width=640, height=480, num_frames=30):
    """Create sample video frames for testing."""
    frames = []
    
    for frame_num in range(num_frames):
        # Create base frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Add moving elements
        x_offset = int(50 + 20 * np.sin(frame_num * 0.2))
        y_offset = int(100 + 10 * np.cos(frame_num * 0.3))
        
        # Moving face
        cv2.circle(frame, (x_offset + 100, y_offset), 40, (220, 180, 150), -1)
        cv2.circle(frame, (x_offset + 85, y_offset - 15), 5, (0, 0, 0), -1)
        cv2.circle(frame, (x_offset + 115, y_offset - 15), 5, (0, 0, 0), -1)
        cv2.ellipse(frame, (x_offset + 100, y_offset + 10), (15, 8), 0, 0, 180, (0, 0, 0), 2)
        
        # Moving license plate
        plate_x = int(200 + 30 * np.cos(frame_num * 0.1))
        plate_y = int(300 + 20 * np.sin(frame_num * 0.15))
        cv2.rectangle(frame, (plate_x, plate_y), (plate_x + 100, plate_y + 30), (255, 255, 255), -1)
        cv2.rectangle(frame, (plate_x, plate_y), (plate_x + 100, plate_y + 30), (0, 0, 0), 2)
        cv2.putText(frame, f"XYZ{frame_num:03d}", (plate_x + 10, plate_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Frame number
        cv2.putText(frame, f"Frame {frame_num + 1}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        frames.append(frame)
    
    return frames


def save_video(frames, output_path, fps=30):
    """Save frames as video file."""
    if not frames:
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()


def generate_all_sample_data():
    """Generate all sample data files."""
    # Create output directory
    output_dir = Path("data/sample_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating sample data...")
    
    # Generate sample images
    logger.info("Creating sample face image...")
    face_image = create_sample_face_image()
    cv2.imwrite(str(output_dir / "sample_people.jpg"), face_image)
    
    logger.info("Creating sample license plate image...")
    plate_image = create_sample_license_plate_image()
    cv2.imwrite(str(output_dir / "sample_vehicles.jpg"), plate_image)
    
    logger.info("Creating sample text image...")
    text_image = create_sample_text_image()
    cv2.imwrite(str(output_dir / "sample_documents.jpg"), text_image)
    
    logger.info("Creating sample mixed image...")
    mixed_image = create_sample_mixed_image()
    cv2.imwrite(str(output_dir / "sample_mixed.jpg"), mixed_image)
    
    # Generate sample videos
    logger.info("Creating sample street scene video...")
    street_frames = create_sample_video_frames(640, 480, 60)
    save_video(street_frames, str(output_dir / "sample_street_scene.mp4"))
    
    logger.info("Creating sample office meeting video...")
    office_frames = create_sample_video_frames(640, 480, 45)
    save_video(office_frames, str(output_dir / "sample_office_meeting.mp4"))
    
    logger.info("Creating sample traffic video...")
    traffic_frames = create_sample_video_frames(640, 480, 90)
    save_video(traffic_frames, str(output_dir / "sample_traffic.mp4"))
    
    logger.info("Sample data generation completed!")
    logger.info(f"Generated files in: {output_dir}")


if __name__ == "__main__":
    generate_all_sample_data()
