"""
Advanced analytics and 3D visualization system for RedactAI.

This module provides sophisticated analytics, real-time monitoring,
and interactive 3D visualizations for comprehensive data insights.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import folium
from folium import plugins
import json
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import time
import threading
from pathlib import Path

from ..utils.monitoring import get_metrics_collector
from ..utils.logger import get_logger
from ..security.audit_system import get_security_manager, AuditEventType

logger = get_logger(__name__)


@dataclass
class AnalyticsData:
    """Analytics data structure."""
    
    timestamp: datetime
    processing_metrics: Dict[str, Any]
    system_metrics: Dict[str, Any]
    security_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'processing_metrics': self.processing_metrics,
            'system_metrics': self.system_metrics,
            'security_metrics': self.security_metrics,
            'performance_metrics': self.performance_metrics
        }


class RealTimeAnalytics:
    """Real-time analytics engine."""
    
    def __init__(self, update_interval: int = 5):
        """Initialize real-time analytics."""
        self.update_interval = update_interval
        self.metrics_collector = get_metrics_collector()
        self.security_manager = get_security_manager()
        self.data_history: List[AnalyticsData] = []
        self.lock = threading.Lock()
        
        # Start data collection thread
        self.collection_thread = threading.Thread(target=self._collect_data, daemon=True)
        self.collection_thread.start()
        
        logger.info("Real-time analytics initialized")
    
    def _collect_data(self):
        """Collect analytics data continuously."""
        while True:
            try:
                # Collect metrics
                processing_metrics = self.metrics_collector.get_metrics_summary()
                
                # Collect security metrics
                security_metrics = self._get_security_metrics()
                
                # Collect performance metrics
                performance_metrics = self._get_performance_metrics()
                
                # Create analytics data
                analytics_data = AnalyticsData(
                    timestamp=datetime.now(timezone.utc),
                    processing_metrics=processing_metrics['processing'],
                    system_metrics=processing_metrics['system'],
                    security_metrics=security_metrics,
                    performance_metrics=performance_metrics
                )
                
                # Store data
                with self.lock:
                    self.data_history.append(analytics_data)
                    
                    # Keep only last 1000 records
                    if len(self.data_history) > 1000:
                        self.data_history = self.data_history[-1000:]
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error collecting analytics data: {e}")
                time.sleep(self.update_interval)
    
    def _get_security_metrics(self) -> Dict[str, Any]:
        """Get security-related metrics."""
        try:
            # Get recent security events
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(hours=24)
            
            security_events = self.security_manager.audit_logger.query_events(
                start_date, end_date, [AuditEventType.SECURITY_VIOLATION]
            )
            
            return {
                'security_violations_24h': len(security_events),
                'failed_logins': len([e for e in security_events if 'login' in e.action]),
                'unauthorized_access': len([e for e in security_events if 'unauthorized' in e.action]),
                'data_breaches': len([e for e in security_events if 'breach' in e.action])
            }
        except Exception as e:
            logger.error(f"Error getting security metrics: {e}")
            return {}
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        try:
            # Calculate performance trends
            if len(self.data_history) < 2:
                return {}
            
            recent_data = self.data_history[-10:]  # Last 10 data points
            
            # Calculate trends
            processing_times = [d.processing_metrics.get('average_processing_time', 0) for d in recent_data]
            cpu_usage = [d.system_metrics.get('cpu_percent', 0) for d in recent_data]
            memory_usage = [d.system_metrics.get('memory_percent', 0) for d in recent_data]
            
            return {
                'avg_processing_time': np.mean(processing_times),
                'processing_trend': self._calculate_trend(processing_times),
                'cpu_trend': self._calculate_trend(cpu_usage),
                'memory_trend': self._calculate_trend(memory_usage),
                'performance_score': self._calculate_performance_score(recent_data)
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 2:
            return "stable"
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_performance_score(self, data: List[AnalyticsData]) -> float:
        """Calculate overall performance score (0-100)."""
        if not data:
            return 0.0
        
        latest = data[-1]
        
        # Factors: processing speed, system resources, error rate
        processing_score = max(0, 100 - latest.processing_metrics.get('average_processing_time', 0) * 10)
        cpu_score = max(0, 100 - latest.system_metrics.get('cpu_percent', 0))
        memory_score = max(0, 100 - latest.system_metrics.get('memory_percent', 0))
        error_score = max(0, 100 - latest.processing_metrics.get('error_rate', 0) * 100)
        
        # Weighted average
        overall_score = (processing_score * 0.3 + cpu_score * 0.25 + 
                        memory_score * 0.25 + error_score * 0.2)
        
        return min(overall_score, 100.0)
    
    def get_current_metrics(self) -> AnalyticsData:
        """Get current metrics."""
        with self.lock:
            if self.data_history:
                return self.data_history[-1]
            else:
                # Return empty data if no history
                return AnalyticsData(
                    timestamp=datetime.now(timezone.utc),
                    processing_metrics={},
                    system_metrics={},
                    security_metrics={},
                    performance_metrics={}
                )
    
    def get_historical_data(self, hours: int = 24) -> List[AnalyticsData]:
        """Get historical data for the specified hours."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        with self.lock:
            return [d for d in self.data_history if d.timestamp >= cutoff_time]


class AdvancedVisualizations:
    """Advanced visualization components."""
    
    def __init__(self, analytics: RealTimeAnalytics):
        """Initialize advanced visualizations."""
        self.analytics = analytics
    
    def create_3d_performance_surface(self, hours: int = 24) -> go.Figure:
        """Create 3D performance surface visualization."""
        data = self.analytics.get_historical_data(hours)
        
        if len(data) < 2:
            return go.Figure()
        
        # Extract data
        timestamps = [d.timestamp for d in data]
        processing_times = [d.processing_metrics.get('average_processing_time', 0) for d in data]
        cpu_usage = [d.system_metrics.get('cpu_percent', 0) for d in data]
        memory_usage = [d.system_metrics.get('memory_percent', 0) for d in data]
        
        # Create 3D surface
        fig = go.Figure(data=[
            go.Surface(
                x=timestamps,
                y=cpu_usage,
                z=processing_times,
                colorscale='Viridis',
                name='Processing Time vs CPU',
                hovertemplate='<b>Time:</b> %{x}<br>' +
                             '<b>CPU:</b> %{y}%<br>' +
                             '<b>Processing Time:</b> %{z}s<br>' +
                             '<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='3D Performance Surface - Processing Time vs System Resources',
            scene=dict(
                xaxis_title='Time',
                yaxis_title='CPU Usage (%)',
                zaxis_title='Processing Time (s)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_animated_timeline(self, hours: int = 24) -> go.Figure:
        """Create animated timeline visualization."""
        data = self.analytics.get_historical_data(hours)
        
        if len(data) < 2:
            return go.Figure()
        
        # Prepare data
        timestamps = [d.timestamp for d in data]
        faces_detected = [d.processing_metrics.get('total_faces_detected', 0) for d in data]
        plates_detected = [d.processing_metrics.get('total_plates_detected', 0) for d in data]
        text_regions = [d.processing_metrics.get('total_text_regions_detected', 0) for d in data]
        
        # Create animated scatter plot
        fig = go.Figure()
        
        # Add traces for each detection type
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=faces_detected,
            mode='lines+markers',
            name='Faces Detected',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=plates_detected,
            mode='lines+markers',
            name='License Plates Detected',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=text_regions,
            mode='lines+markers',
            name='Text Regions Detected',
            line=dict(color='green', width=3),
            marker=dict(size=8)
        ))
        
        # Add animation
        fig.update_layout(
            title='Real-time Detection Timeline',
            xaxis_title='Time',
            yaxis_title='Detection Count',
            hovermode='x unified',
            width=1000,
            height=500
        )
        
        # Add range selector
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1h", step="hour", stepmode="backward"),
                        dict(count=6, label="6h", step="hour", stepmode="backward"),
                        dict(count=24, label="24h", step="hour", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        return fig
    
    def create_heatmap_visualization(self, hours: int = 24) -> go.Figure:
        """Create heatmap visualization of processing patterns."""
        data = self.analytics.get_historical_data(hours)
        
        if len(data) < 2:
            return go.Figure()
        
        # Create hourly heatmap data
        hourly_data = {}
        for d in data:
            hour = d.timestamp.hour
            if hour not in hourly_data:
                hourly_data[hour] = {
                    'processing_time': [],
                    'cpu_usage': [],
                    'memory_usage': [],
                    'detection_count': []
                }
            
            hourly_data[hour]['processing_time'].append(d.processing_metrics.get('average_processing_time', 0))
            hourly_data[hour]['cpu_usage'].append(d.system_metrics.get('cpu_percent', 0))
            hourly_data[hour]['memory_usage'].append(d.system_metrics.get('memory_percent', 0))
            hourly_data[hour]['detection_count'].append(
                d.processing_metrics.get('total_faces_detected', 0) +
                d.processing_metrics.get('total_plates_detected', 0) +
                d.processing_metrics.get('total_text_regions_detected', 0)
            )
        
        # Calculate averages
        hours_list = sorted(hourly_data.keys())
        metrics = ['processing_time', 'cpu_usage', 'memory_usage', 'detection_count']
        
        heatmap_data = []
        for metric in metrics:
            row = []
            for hour in hours_list:
                values = hourly_data[hour][metric]
                avg_value = np.mean(values) if values else 0
                row.append(avg_value)
            heatmap_data.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=hours_list,
            y=metrics,
            colorscale='RdYlBu_r',
            hoverongaps=False,
            hovertemplate='<b>Hour:</b> %{x}<br>' +
                         '<b>Metric:</b> %{y}<br>' +
                         '<b>Value:</b> %{z}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title='Processing Patterns Heatmap (24h)',
            xaxis_title='Hour of Day',
            yaxis_title='Metrics',
            width=800,
            height=400
        )
        
        return fig
    
    def create_geospatial_heatmap(self, data_points: List[Dict[str, Any]]) -> folium.Map:
        """Create geospatial heatmap of redaction activities."""
        if not data_points:
            # Default to San Francisco
            center_lat, center_lon = 37.7749, -122.4194
        else:
            # Calculate center from data points
            lats = [point.get('latitude', 37.7749) for point in data_points]
            lons = [point.get('longitude', -122.4194) for point in data_points]
            center_lat, center_lon = np.mean(lats), np.mean(lons)
        
        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add heatmap layer
        if data_points:
            heat_data = []
            for point in data_points:
                lat = point.get('latitude', center_lat)
                lon = point.get('longitude', center_lon)
                intensity = point.get('intensity', 1)
                heat_data.append([lat, lon, intensity])
            
            plugins.HeatMap(
                heat_data,
                name='Redaction Activity',
                min_opacity=0.2,
                max_zoom=18,
                radius=25,
                blur=15,
                gradient={0.4: 'blue', 0.65: 'lime', 0.85: 'orange', 1.0: 'red'}
            ).add_to(m)
        
        # Add markers for specific locations
        for i, point in enumerate(data_points[:10]):  # Limit to first 10 points
            lat = point.get('latitude', center_lat)
            lon = point.get('longitude', center_lon)
            count = point.get('count', 1)
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=count * 2,
                popup=f'Redactions: {count}',
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.6
            ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m
    
    def create_performance_dashboard(self) -> Dict[str, go.Figure]:
        """Create comprehensive performance dashboard."""
        current_metrics = self.analytics.get_current_metrics()
        
        # System metrics gauge
        cpu_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current_metrics.system_metrics.get('cpu_percent', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "CPU Usage (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        # Memory usage gauge
        memory_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current_metrics.system_metrics.get('memory_percent', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Memory Usage (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 85], 'color': "yellow"},
                    {'range': [85, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        # Processing efficiency chart
        data = self.analytics.get_historical_data(24)
        if data:
            timestamps = [d.timestamp for d in data]
            processing_times = [d.processing_metrics.get('average_processing_time', 0) for d in data]
            
            efficiency_chart = go.Figure()
            efficiency_chart.add_trace(go.Scatter(
                x=timestamps,
                y=processing_times,
                mode='lines+markers',
                name='Processing Time',
                line=dict(color='blue', width=2)
            ))
            
            efficiency_chart.update_layout(
                title='Processing Efficiency Over Time',
                xaxis_title='Time',
                yaxis_title='Processing Time (s)',
                width=600,
                height=300
            )
        else:
            efficiency_chart = go.Figure()
        
        return {
            'cpu_gauge': cpu_gauge,
            'memory_gauge': memory_gauge,
            'efficiency_chart': efficiency_chart
        }


class AdvancedDashboard:
    """Advanced dashboard with 3D visualizations and real-time analytics."""
    
    def __init__(self):
        """Initialize advanced dashboard."""
        self.analytics = RealTimeAnalytics()
        self.visualizations = AdvancedVisualizations(self.analytics)
        self.logger = get_logger(__name__)
    
    def render_dashboard(self):
        """Render the complete advanced dashboard."""
        st.set_page_config(
            page_title="RedactAI Advanced Analytics",
            page_icon="🔒",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Header
        st.title("🔒 RedactAI Advanced Analytics Dashboard")
        st.markdown("---")
        
        # Sidebar controls
        with st.sidebar:
            st.header("📊 Dashboard Controls")
            
            # Time range selector
            time_range = st.selectbox(
                "Time Range",
                ["1 Hour", "6 Hours", "24 Hours", "7 Days"],
                index=2
            )
            
            # Visualization type
            viz_type = st.selectbox(
                "Visualization Type",
                ["3D Performance", "Timeline", "Heatmap", "Geospatial", "Performance Dashboard"],
                index=0
            )
            
            # Auto-refresh
            auto_refresh = st.checkbox("Auto Refresh", value=True)
            if auto_refresh:
                refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 10)
        
        # Main content area
        if viz_type == "3D Performance":
            self._render_3d_performance()
        elif viz_type == "Timeline":
            self._render_timeline()
        elif viz_type == "Heatmap":
            self._render_heatmap()
        elif viz_type == "Geospatial":
            self._render_geospatial()
        elif viz_type == "Performance Dashboard":
            self._render_performance_dashboard()
        
        # Real-time metrics
        self._render_realtime_metrics()
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
    
    def _render_3d_performance(self):
        """Render 3D performance visualization."""
        st.header("🎯 3D Performance Surface")
        
        with st.spinner("Generating 3D visualization..."):
            fig = self.visualizations.create_3d_performance_surface(24)
            
            if fig.data:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for 3D visualization")
    
    def _render_timeline(self):
        """Render timeline visualization."""
        st.header("📈 Real-time Detection Timeline")
        
        with st.spinner("Generating timeline..."):
            fig = self.visualizations.create_animated_timeline(24)
            
            if fig.data:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for timeline visualization")
    
    def _render_heatmap(self):
        """Render heatmap visualization."""
        st.header("🔥 Processing Patterns Heatmap")
        
        with st.spinner("Generating heatmap..."):
            fig = self.visualizations.create_heatmap_visualization(24)
            
            if fig.data:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for heatmap visualization")
    
    def _render_geospatial(self):
        """Render geospatial visualization."""
        st.header("🌍 Geospatial Redaction Heatmap")
        
        # Generate sample data for demonstration
        sample_data = [
            {'latitude': 37.7749, 'longitude': -122.4194, 'intensity': 5, 'count': 10},
            {'latitude': 37.7849, 'longitude': -122.4094, 'intensity': 3, 'count': 7},
            {'latitude': 37.7649, 'longitude': -122.4294, 'intensity': 4, 'count': 5},
        ]
        
        with st.spinner("Generating geospatial map..."):
            m = self.visualizations.create_geospatial_heatmap(sample_data)
            
            # Convert to HTML and display
            map_html = m._repr_html_()
            st.components.v1.html(map_html, height=600)
    
    def _render_performance_dashboard(self):
        """Render performance dashboard."""
        st.header("⚡ Performance Dashboard")
        
        with st.spinner("Generating performance metrics..."):
            dashboard_figs = self.visualizations.create_performance_dashboard()
            
            # Display gauges in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(dashboard_figs['cpu_gauge'], use_container_width=True)
            
            with col2:
                st.plotly_chart(dashboard_figs['memory_gauge'], use_container_width=True)
            
            # Display efficiency chart
            st.plotly_chart(dashboard_figs['efficiency_chart'], use_container_width=True)
    
    def _render_realtime_metrics(self):
        """Render real-time metrics."""
        st.header("📊 Real-time Metrics")
        
        current_metrics = self.analytics.get_current_metrics()
        
        # Create metric columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Files Processed",
                current_metrics.processing_metrics.get('total_files_processed', 0),
                delta=None
            )
        
        with col2:
            st.metric(
                "Faces Detected",
                current_metrics.processing_metrics.get('total_faces_detected', 0),
                delta=None
            )
        
        with col3:
            st.metric(
                "CPU Usage",
                f"{current_metrics.system_metrics.get('cpu_percent', 0):.1f}%",
                delta=None
            )
        
        with col4:
            st.metric(
                "Memory Usage",
                f"{current_metrics.system_metrics.get('memory_percent', 0):.1f}%",
                delta=None
            )


# Global dashboard instance
_dashboard: Optional[AdvancedDashboard] = None


def get_advanced_dashboard() -> AdvancedDashboard:
    """Get the global advanced dashboard instance."""
    global _dashboard
    if _dashboard is None:
        _dashboard = AdvancedDashboard()
    return _dashboard


def run_advanced_dashboard():
    """Run the advanced dashboard."""
    dashboard = get_advanced_dashboard()
    dashboard.render_dashboard()
