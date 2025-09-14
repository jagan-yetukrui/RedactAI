# Sample Data for RedactAI

This directory contains sample media files for testing the RedactAI pipeline.

## Sample Images

- `sample_people.jpg` - Image with multiple faces for face detection testing
- `sample_vehicles.jpg` - Image with license plates for plate detection testing
- `sample_documents.jpg` - Image with text and names for OCR/NER testing
- `sample_mixed.jpg` - Complex image with faces, plates, and text

## Sample Videos

- `sample_street_scene.mp4` - Street scene with people and vehicles
- `sample_office_meeting.mp4` - Office meeting with faces and documents
- `sample_traffic.mp4` - Traffic scene with multiple license plates

## Usage

Place your own test media files in the `input_media/` directory for processing.

## Data Generation

The `generate_sample_data.py` script can create synthetic test data for development and testing.
