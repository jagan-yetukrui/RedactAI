"""
Geospatial heatmap generation for RedactAI.

This module provides functionality to generate interactive heatmaps
of redacted data using Folium and other geospatial visualization tools.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

# Try to import geospatial libraries
try:
    import folium
    from folium import plugins
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    logger.warning("Folium not available. Install folium for heatmap generation.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available. Install numpy for heatmap generation.")


class HeatmapGenerator:
    """
    Heatmap generator for visualizing redacted data geospatially.
    
    This class provides functionality to create interactive heatmaps
    showing the distribution of redacted entities across geographic locations.
    """
    
    def __init__(self, default_center: Tuple[float, float] = (37.7749, -122.4194),
                 default_zoom: int = 12):
        """
        Initialize the heatmap generator.
        
        Args:
            default_center: Default center coordinates (latitude, longitude)
            default_zoom: Default zoom level
        """
        self.default_center = default_center
        self.default_zoom = default_zoom
        self.folium_available = FOLIUM_AVAILABLE
        self.numpy_available = NUMPY_AVAILABLE
        
        if not self.folium_available:
            logger.warning("Folium not available. Heatmap generation will be limited.")
        
        logger.info("HeatmapGenerator initialized")
    
    def create_heatmap(self, data_points: List[Tuple[float, float, Dict[str, Any]]],
                      map_center: Tuple[float, float] = None,
                      zoom_level: int = None,
                      heatmap_type: str = 'heatmap') -> Optional[folium.Map]:
        """
        Create a heatmap from data points.
        
        Args:
            data_points: List of (latitude, longitude, metadata) tuples
            map_center: Center coordinates for the map. If None, uses default.
            zoom_level: Zoom level for the map. If None, uses default.
            heatmap_type: Type of heatmap ('heatmap', 'clustered', 'markers')
            
        Returns:
            Folium map object or None if Folium is not available
        """
        if not self.folium_available:
            logger.error("Folium not available. Cannot create heatmap.")
            return None
        
        if not data_points:
            logger.warning("No data points provided for heatmap")
            return None
        
        # Use provided or default center and zoom
        center = map_center or self.default_center
        zoom = zoom_level or self.default_zoom
        
        # Create base map
        m = folium.Map(
            location=center,
            zoom_start=zoom,
            tiles='OpenStreetMap'
        )
        
        # Add different types of visualizations based on heatmap_type
        if heatmap_type == 'heatmap':
            self._add_heatmap_layer(m, data_points)
        elif heatmap_type == 'clustered':
            self._add_clustered_markers(m, data_points)
        elif heatmap_type == 'markers':
            self._add_individual_markers(m, data_points)
        else:
            logger.warning(f"Unknown heatmap type: {heatmap_type}. Using markers.")
            self._add_individual_markers(m, data_points)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        logger.info(f"Created {heatmap_type} heatmap with {len(data_points)} data points")
        return m
    
    def _add_heatmap_layer(self, map_obj: folium.Map, data_points: List[Tuple[float, float, Dict[str, Any]]]) -> None:
        """Add a heatmap layer to the map."""
        if not self.numpy_available:
            logger.warning("NumPy not available. Using simple heatmap.")
            self._add_individual_markers(map_obj, data_points)
            return
        
        # Extract coordinates
        coordinates = [(lat, lon) for lat, lon, _ in data_points]
        
        # Create heatmap layer
        heat_data = [[coord[0], coord[1]] for coord in coordinates]
        
        plugins.HeatMap(
            heat_data,
            name='Redaction Heatmap',
            min_opacity=0.4,
            max_zoom=18,
            radius=25,
            blur=15,
            gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'orange', 1.0: 'red'}
        ).add_to(map_obj)
    
    def _add_clustered_markers(self, map_obj: folium.Map, data_points: List[Tuple[float, float, Dict[str, Any]]]) -> None:
        """Add clustered markers to the map."""
        # Create marker cluster
        marker_cluster = plugins.MarkerCluster(
            name='Redaction Clusters',
            overlay=True,
            control=True
        ).add_to(map_obj)
        
        # Add individual markers
        for lat, lon, metadata in data_points:
            # Create popup content
            popup_content = self._create_popup_content(metadata)
            
            # Create marker
            folium.Marker(
                [lat, lon],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"Lat: {lat:.4f}, Lon: {lon:.4f}",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(marker_cluster)
    
    def _add_individual_markers(self, map_obj: folium.Map, data_points: List[Tuple[float, float, Dict[str, Any]]]) -> None:
        """Add individual markers to the map."""
        for lat, lon, metadata in data_points:
            # Create popup content
            popup_content = self._create_popup_content(metadata)
            
            # Determine marker color based on metadata
            color = self._get_marker_color(metadata)
            
            # Create marker
            folium.Marker(
                [lat, lon],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"Lat: {lat:.4f}, Lon: {lon:.4f}",
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(map_obj)
    
    def _create_popup_content(self, metadata: Dict[str, Any]) -> str:
        """Create popup content for markers."""
        content = "<div style='font-family: Arial, sans-serif;'>"
        content += "<h4>Redaction Data</h4>"
        
        # Add metadata fields
        for key, value in metadata.items():
            if key not in ['latitude', 'longitude']:  # Skip coordinate fields
                content += f"<p><strong>{key}:</strong> {value}</p>"
        
        content += "</div>"
        return content
    
    def _get_marker_color(self, metadata: Dict[str, Any]) -> str:
        """Get marker color based on metadata."""
        # Determine color based on detection type or count
        if 'faces_detected' in metadata and metadata['faces_detected'] > 0:
            return 'red'
        elif 'plates_detected' in metadata and metadata['plates_detected'] > 0:
            return 'blue'
        elif 'text_regions_detected' in metadata and metadata['text_regions_detected'] > 0:
            return 'green'
        else:
            return 'gray'
    
    def create_density_heatmap(self, data_points: List[Tuple[float, float, Dict[str, Any]]],
                              map_center: Tuple[float, float] = None,
                              zoom_level: int = None) -> Optional[folium.Map]:
        """
        Create a density heatmap showing concentration of redacted data.
        
        Args:
            data_points: List of (latitude, longitude, metadata) tuples
            map_center: Center coordinates for the map
            zoom_level: Zoom level for the map
            
        Returns:
            Folium map object or None if Folium is not available
        """
        if not self.folium_available:
            logger.error("Folium not available. Cannot create density heatmap.")
            return None
        
        if not data_points:
            logger.warning("No data points provided for density heatmap")
            return None
        
        # Use provided or default center and zoom
        center = map_center or self.default_center
        zoom = zoom_level or self.default_zoom
        
        # Create base map
        m = folium.Map(
            location=center,
            zoom_start=zoom,
            tiles='OpenStreetMap'
        )
        
        # Add density heatmap layer
        self._add_density_heatmap_layer(m, data_points)
        
        # Add individual markers for reference
        self._add_individual_markers(m, data_points)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        logger.info(f"Created density heatmap with {len(data_points)} data points")
        return m
    
    def _add_density_heatmap_layer(self, map_obj: folium.Map, data_points: List[Tuple[float, float, Dict[str, Any]]]) -> None:
        """Add a density heatmap layer to the map."""
        if not self.numpy_available:
            logger.warning("NumPy not available. Using simple heatmap.")
            self._add_heatmap_layer(map_obj, data_points)
            return
        
        # Extract coordinates and weights
        coordinates = []
        weights = []
        
        for lat, lon, metadata in data_points:
            coordinates.append([lat, lon])
            
            # Calculate weight based on detection counts
            weight = 1
            if 'faces_detected' in metadata:
                weight += metadata['faces_detected'] * 2
            if 'plates_detected' in metadata:
                weight += metadata['plates_detected'] * 1.5
            if 'text_regions_detected' in metadata:
                weight += metadata['text_regions_detected'] * 1
            
            weights.append(weight)
        
        # Create weighted heatmap
        heat_data = [[coord[0], coord[1], weight] for coord, weight in zip(coordinates, weights)]
        
        plugins.HeatMap(
            heat_data,
            name='Density Heatmap',
            min_opacity=0.3,
            max_zoom=18,
            radius=30,
            blur=20,
            gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
        ).add_to(map_obj)
    
    def save_heatmap(self, map_obj: folium.Map, output_path: str) -> None:
        """
        Save heatmap to HTML file.
        
        Args:
            map_obj: Folium map object
            output_path: Path to save HTML file
        """
        if map_obj is None:
            logger.error("Cannot save None map object")
            return
        
        try:
            map_obj.save(output_path)
            logger.info(f"Heatmap saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving heatmap: {e}")
            raise
    
    def create_heatmap_from_metadata(self, metadata_file: str, output_path: str,
                                   heatmap_type: str = 'heatmap') -> bool:
        """
        Create heatmap from metadata file.
        
        Args:
            metadata_file: Path to metadata JSON file
            output_path: Path to save heatmap HTML file
            heatmap_type: Type of heatmap to create
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Extract data points
            data_points = self._extract_data_points_from_metadata(metadata)
            
            if not data_points:
                logger.warning("No valid data points found in metadata")
                return False
            
            # Create heatmap
            map_obj = self.create_heatmap(data_points, heatmap_type=heatmap_type)
            
            if map_obj is None:
                return False
            
            # Save heatmap
            self.save_heatmap(map_obj, output_path)
            return True
            
        except Exception as e:
            logger.error(f"Error creating heatmap from metadata: {e}")
            return False
    
    def _extract_data_points_from_metadata(self, metadata: Dict[str, Any]) -> List[Tuple[float, float, Dict[str, Any]]]:
        """
        Extract data points from metadata.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            List of (latitude, longitude, metadata) tuples
        """
        data_points = []
        
        # Handle different metadata formats
        if isinstance(metadata, list):
            # List of metadata objects
            for item in metadata:
                if 'gps_coords' in item:
                    lat, lon = item['gps_coords']
                    data_points.append((lat, lon, item))
        elif isinstance(metadata, dict):
            # Single metadata object
            if 'gps_coords' in metadata:
                lat, lon = metadata['gps_coords']
                data_points.append((lat, lon, metadata))
            elif 'sources' in metadata:
                # Merged metadata with sources
                for source in metadata.get('sources', []):
                    if 'gps_coords' in source:
                        lat, lon = source['gps_coords']
                        data_points.append((lat, lon, source))
        
        return data_points
    
    def get_heatmap_statistics(self, data_points: List[Tuple[float, float, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Get statistics about heatmap data.
        
        Args:
            data_points: List of (latitude, longitude, metadata) tuples
            
        Returns:
            Dictionary containing heatmap statistics
        """
        if not data_points:
            return {
                'total_points': 0,
                'bounding_box': None,
                'center_point': None,
                'detection_counts': {}
            }
        
        # Extract coordinates
        latitudes = [point[0] for point in data_points]
        longitudes = [point[1] for point in data_points]
        
        # Calculate bounding box
        min_lat, max_lat = min(latitudes), max(latitudes)
        min_lon, max_lon = min(longitudes), max(longitudes)
        
        # Calculate center point
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        
        # Count detections
        detection_counts = {
            'faces_detected': 0,
            'plates_detected': 0,
            'text_regions_detected': 0,
            'names_redacted': 0
        }
        
        for _, _, metadata in data_points:
            for key in detection_counts:
                if key in metadata:
                    detection_counts[key] += metadata[key]
        
        return {
            'total_points': len(data_points),
            'bounding_box': {
                'min_lat': min_lat,
                'max_lat': max_lat,
                'min_lon': min_lon,
                'max_lon': max_lon
            },
            'center_point': (center_lat, center_lon),
            'detection_counts': detection_counts
        }
