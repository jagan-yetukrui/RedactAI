"""
Streamlit dashboard for RedactAI.

This module provides a comprehensive dashboard for visualizing
processing results, managing media files, and monitoring system status.
"""

import streamlit as st
import requests
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import base64
import io

# Configure page
st.set_page_config(
    page_title="RedactAI Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    .status-unhealthy {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> Dict[str, Any]:
    """Check API health status."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def get_statistics() -> Dict[str, Any]:
    """Get processing statistics from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/statistics", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {}
    except Exception as e:
        st.error(f"Error fetching statistics: {e}")
        return {}


def process_file(file, processing_options: Dict[str, Any]) -> Dict[str, Any]:
    """Process a file using the API."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        
        response = requests.post(
            f"{API_BASE_URL}/process",
            files=files,
            data=processing_options,
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    """Main dashboard application."""
    # Header
    st.markdown('<h1 class="main-header">🔒 RedactAI Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "📊 Statistics", "🔧 Process Media", "🗺️ Geospatial View", "⚙️ Settings"]
    )
    
    # Check API health
    with st.spinner("Checking API status..."):
        health_status = check_api_health()
    
    # Display health status
    if health_status.get("status") == "healthy":
        st.sidebar.markdown('<p class="status-healthy">✅ API Online</p>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<p class="status-unhealthy">❌ API Offline</p>', unsafe_allow_html=True)
        st.error(f"API is not available: {health_status.get('error', 'Unknown error')}")
        return
    
    # Page routing
    if page == "🏠 Home":
        show_home_page(health_status)
    elif page == "📊 Statistics":
        show_statistics_page()
    elif page == "🔧 Process Media":
        show_process_media_page()
    elif page == "🗺️ Geospatial View":
        show_geospatial_page()
    elif page == "⚙️ Settings":
        show_settings_page()


def show_home_page(health_status: Dict[str, Any]):
    """Display the home page."""
    st.header("Welcome to RedactAI")
    st.markdown("""
    RedactAI is a comprehensive AI-powered privacy tool that automatically detects and redacts 
    sensitive information from images and videos, including faces, license plates, and personal names.
    """)
    
    # System status
    st.subheader("System Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "API Status",
            "Online" if health_status.get("status") == "healthy" else "Offline",
            delta=None
        )
    
    with col2:
        st.metric(
            "Face Detection",
            "Available" if health_status.get("face_detection_available") else "Unavailable",
            delta=None
        )
    
    with col3:
        st.metric(
            "Plate Detection",
            "Available" if health_status.get("plate_detection_available") else "Unavailable",
            delta=None
        )
    
    with col4:
        st.metric(
            "Text Detection",
            "Available" if health_status.get("text_detection_available") else "Unavailable",
            delta=None
        )
    
    # Quick stats
    st.subheader("Quick Statistics")
    stats = get_statistics()
    
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Files Processed", stats.get("total_files_processed", 0))
        
        with col2:
            st.metric("Faces Detected", stats.get("total_faces_detected", 0))
        
        with col3:
            st.metric("Plates Detected", stats.get("total_plates_detected", 0))
        
        with col4:
            st.metric("Names Redacted", stats.get("total_names_redacted", 0))
    
    # Features overview
    st.subheader("Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔍 Detection Capabilities:**
        - Face detection using Haar Cascades
        - License plate detection using YOLOv8
        - Text detection using OCR (Tesseract/EasyOCR)
        - Named Entity Recognition for personal names
        """)
    
    with col2:
        st.markdown("""
        **🎨 Redaction Options:**
        - Gaussian blur
        - Pixelation
        - Mosaic effect
        - Blackout
        - Geotagging and metadata overlay
        """)


def show_statistics_page():
    """Display the statistics page."""
    st.header("Processing Statistics")
    
    # Get statistics
    stats = get_statistics()
    
    if not stats:
        st.warning("No statistics available. Process some files to see statistics.")
        return
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Files", stats.get("total_files_processed", 0))
    
    with col2:
        st.metric("Total Processing Time", f"{stats.get('total_processing_time_seconds', 0):.2f}s")
    
    with col3:
        st.metric("Average Time per File", f"{stats.get('average_processing_time_seconds', 0):.2f}s")
    
    with col4:
        st.metric("Detection Rate", f"{stats.get('total_faces_detected', 0) + stats.get('total_plates_detected', 0) + stats.get('total_text_regions_detected', 0)}")
    
    # Detection breakdown
    st.subheader("Detection Breakdown")
    
    detection_data = {
        "Type": ["Faces", "License Plates", "Text Regions", "Names Redacted"],
        "Count": [
            stats.get("total_faces_detected", 0),
            stats.get("total_plates_detected", 0),
            stats.get("total_text_regions_detected", 0),
            stats.get("total_names_redacted", 0)
        ]
    }
    
    df = pd.DataFrame(detection_data)
    
    # Bar chart
    fig = px.bar(df, x="Type", y="Count", title="Detection Counts by Type")
    st.plotly_chart(fig, use_container_width=True)
    
    # File type breakdown
    if stats.get("file_type_breakdown"):
        st.subheader("File Type Breakdown")
        
        file_types = list(stats["file_type_breakdown"].keys())
        file_counts = list(stats["file_type_breakdown"].values())
        
        fig = px.pie(values=file_counts, names=file_types, title="Files Processed by Type")
        st.plotly_chart(fig, use_container_width=True)
    
    # Time range
    if stats.get("first_processing") and stats.get("last_processing"):
        st.subheader("Processing Timeline")
        
        first = datetime.fromisoformat(stats["first_processing"].replace('Z', '+00:00'))
        last = datetime.fromisoformat(stats["last_processing"].replace('Z', '+00:00'))
        
        st.info(f"First processing: {first.strftime('%Y-%m-%d %H:%M:%S')}")
        st.info(f"Last processing: {last.strftime('%Y-%m-%d %H:%M:%S')}")


def show_process_media_page():
    """Display the media processing page."""
    st.header("Process Media Files")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file to process",
        type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'],
        help="Supported formats: JPG, PNG, MP4, AVI, MOV"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.subheader("File Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("File Name", uploaded_file.name)
        
        with col2:
            st.metric("File Size", f"{uploaded_file.size / 1024:.2f} KB")
        
        with col3:
            st.metric("File Type", uploaded_file.type)
        
        # Processing options
        st.subheader("Processing Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Detection Options:**")
            process_faces = st.checkbox("Detect Faces", value=True)
            process_plates = st.checkbox("Detect License Plates", value=True)
            process_text = st.checkbox("Detect Text", value=True)
            redact_names_only = st.checkbox("Redact Names Only", value=True)
        
        with col2:
            st.markdown("**Blur Settings:**")
            face_blur_type = st.selectbox("Face Blur Type", ["gaussian", "pixelate", "blackout", "mosaic"])
            plate_blur_type = st.selectbox("Plate Blur Type", ["gaussian", "pixelate", "blackout", "mosaic"])
            text_blur_type = st.selectbox("Text Blur Type", ["gaussian", "pixelate", "blackout", "mosaic"])
        
        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                face_confidence = st.slider("Face Detection Confidence", 0.0, 1.0, 0.5, 0.1)
                plate_confidence = st.slider("Plate Detection Confidence", 0.0, 1.0, 0.5, 0.1)
                text_confidence = st.slider("Text Detection Confidence", 0.0, 1.0, 0.5, 0.1)
            
            with col2:
                face_blur_strength = st.slider("Face Blur Strength", 1, 50, 15)
                plate_blur_strength = st.slider("Plate Blur Strength", 1, 50, 15)
                text_blur_strength = st.slider("Text Blur Strength", 1, 50, 15)
        
        # Geotagging options
        with st.expander("Geotagging Options"):
            add_geotags = st.checkbox("Add Geotags", value=False)
            
            if add_geotags:
                col1, col2 = st.columns(2)
                with col1:
                    gps_latitude = st.number_input("GPS Latitude", value=37.7749, format="%.6f")
                with col2:
                    gps_longitude = st.number_input("GPS Longitude", value=-122.4194, format="%.6f")
            else:
                gps_latitude = None
                gps_longitude = None
        
        # Process button
        if st.button("Process File", type="primary"):
            with st.spinner("Processing file..."):
                # Prepare processing options
                processing_options = {
                    "process_faces": process_faces,
                    "process_plates": process_plates,
                    "process_text": process_text,
                    "redact_names_only": redact_names_only,
                    "face_blur_type": face_blur_type,
                    "plate_blur_type": plate_blur_type,
                    "text_blur_type": text_blur_type,
                    "face_blur_strength": face_blur_strength,
                    "plate_blur_strength": plate_blur_strength,
                    "text_blur_strength": text_blur_strength,
                    "face_confidence": face_confidence,
                    "plate_confidence": plate_confidence,
                    "text_confidence": text_confidence,
                    "add_geotags": add_geotags,
                    "gps_latitude": gps_latitude,
                    "gps_longitude": gps_longitude,
                    "ocr_engine": "tesseract"
                }
                
                # Process file
                result = process_file(uploaded_file, processing_options)
                
                if result.get("success"):
                    st.success("File processed successfully!")
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Faces Detected", result.get("faces_detected", 0))
                    
                    with col2:
                        st.metric("Plates Detected", result.get("plates_detected", 0))
                    
                    with col3:
                        st.metric("Text Regions", result.get("text_regions_detected", 0))
                    
                    with col4:
                        st.metric("Names Redacted", result.get("names_redacted", 0))
                    
                    # Processing time
                    st.metric("Processing Time", f"{result.get('processing_time_seconds', 0):.2f}s")
                    
                    # Download link
                    if result.get("output_path"):
                        st.subheader("Download Processed File")
                        st.download_button(
                            label="Download Processed File",
                            data=open(result["output_path"], "rb").read(),
                            file_name=f"processed_{uploaded_file.name}",
                            mime=uploaded_file.type
                        )
                else:
                    st.error(f"Processing failed: {result.get('error', 'Unknown error')}")


def show_geospatial_page():
    """Display the geospatial visualization page."""
    st.header("Geospatial View")
    
    # Check if folium is available
    try:
        import folium
        from streamlit_folium import st_folium
    except ImportError:
        st.error("Folium is not available. Please install it to view geospatial data.")
        return
    
    # Get statistics for geospatial data
    stats = get_statistics()
    
    if not stats or stats.get("total_files_processed", 0) == 0:
        st.warning("No processed files available for geospatial visualization.")
        return
    
    # Create a simple map centered on San Francisco
    m = folium.Map(
        location=[37.7749, -122.4194],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add some sample markers (in a real implementation, these would come from actual geotag data)
    sample_locations = [
        [37.7749, -122.4194, "Sample Location 1"],
        [37.7849, -122.4094, "Sample Location 2"],
        [37.7649, -122.4294, "Sample Location 3"]
    ]
    
    for lat, lon, name in sample_locations:
        folium.Marker(
            [lat, lon],
            popup=name,
            tooltip=name
        ).add_to(m)
    
    # Display the map
    st_folium(m, width=700, height=500)
    
    # Geospatial statistics
    st.subheader("Geospatial Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Locations", len(sample_locations))
    
    with col2:
        st.metric("Files with Geotags", stats.get("total_files_processed", 0))
    
    with col3:
        st.metric("Coverage Area", "San Francisco Bay Area")


def show_settings_page():
    """Display the settings page."""
    st.header("Settings")
    
    # API Configuration
    st.subheader("API Configuration")
    
    api_url = st.text_input("API Base URL", value=API_BASE_URL)
    
    if st.button("Test API Connection"):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                st.success("API connection successful!")
            else:
                st.error(f"API connection failed: HTTP {response.status_code}")
        except Exception as e:
            st.error(f"API connection failed: {e}")
    
    # Display Settings
    st.subheader("Display Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.checkbox("Dark Mode", value=False)
        st.checkbox("Show Processing Details", value=True)
    
    with col2:
        st.checkbox("Auto-refresh Statistics", value=True)
        st.checkbox("Show File Previews", value=True)
    
    # About
    st.subheader("About RedactAI")
    
    st.markdown("""
    **RedactAI v1.0.0**
    
    A comprehensive AI-powered privacy tool for automatically detecting and redacting 
    sensitive information from media files.
    
    **Features:**
    - Face detection and blurring
    - License plate detection and redaction
    - Text detection and name redaction
    - Geotagging and metadata overlay
    - REST API and web dashboard
    
    **Built with:**
    - Python 3.10+
    - OpenCV, SpaCy, Tesseract
    - FastAPI, Streamlit
    - YOLOv8, EasyOCR
    """)


if __name__ == "__main__":
    main()
