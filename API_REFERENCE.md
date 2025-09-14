# RedactAI API Reference

## 🔗 **API Overview**

RedactAI provides a comprehensive REST API for all functionality, built with FastAPI and featuring automatic OpenAPI documentation, request validation, and comprehensive error handling.

**Base URL**: `http://localhost:8000`  
**API Version**: `v1`  
**Documentation**: `http://localhost:8000/docs` (Swagger UI)  
**ReDoc**: `http://localhost:8000/redoc` (Alternative documentation)

## 🔐 **Authentication**

### **API Key Authentication**

```http
Authorization: Bearer <your-api-key>
```

### **Session Authentication**

```http
Cookie: session_id=<your-session-id>
```

## 📊 **Core Endpoints**

### **Health & Status**

#### `GET /health`

Get system health status.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "uptime": 3600,
  "services": {
    "database": "healthy",
    "gpu": "available",
    "cache": "healthy"
  }
}
```

#### `GET /status`

Get detailed system status.

**Response:**

```json
{
  "system": {
    "cpu_percent": 45.2,
    "memory_percent": 67.8,
    "disk_percent": 23.1,
    "gpu_available": true,
    "gpu_memory_used": 2048,
    "gpu_memory_total": 8192
  },
  "processing": {
    "total_files_processed": 1250,
    "average_processing_time": 2.3,
    "success_rate": 98.5,
    "active_jobs": 3
  },
  "detection": {
    "faces_detected": 3420,
    "plates_detected": 890,
    "text_regions_detected": 1560,
    "names_redacted": 2340
  }
}
```

### **File Processing**

#### `POST /process`

Process a single image or video file.

**Request:**

```http
POST /process
Content-Type: multipart/form-data

file: <binary-file>
process_faces: true
process_plates: true
process_text: true
blur_type: "gaussian"
blur_strength: 15
confidence: 0.5
geotags: true
gps_lat: 37.7749
gps_lon: -122.4194
```

**Response:**

```json
{
  "success": true,
  "job_id": "job_12345",
  "output_path": "/output/processed_file.jpg",
  "processing_time": 2.3,
  "detections": {
    "faces_detected": 3,
    "plates_detected": 1,
    "text_regions_detected": 5,
    "names_redacted": 2
  },
  "metadata": {
    "original_size": "1920x1080",
    "processed_size": "1920x1080",
    "file_type": "image/jpeg",
    "file_size": 2048576
  }
}
```

#### `POST /batch/process`

Process multiple files in batch.

**Request:**

```http
POST /batch/process
Content-Type: application/json

{
  "input_directory": "/input/files",
  "output_directory": "/output/processed",
  "processing_options": {
    "process_faces": true,
    "process_plates": true,
    "process_text": true,
    "blur_type": "adaptive_gaussian",
    "blur_strength": 15,
    "confidence": 0.5
  },
  "file_extensions": [".jpg", ".jpeg", ".png", ".mp4"],
  "max_workers": 4
}
```

**Response:**

```json
{
  "success": true,
  "batch_id": "batch_67890",
  "total_files": 25,
  "queued_files": 25,
  "estimated_completion": "2024-01-15T10:35:00Z",
  "status_url": "/batch/status/batch_67890"
}
```

#### `GET /batch/status/{batch_id}`

Get batch processing status.

**Response:**

```json
{
  "batch_id": "batch_67890",
  "status": "processing",
  "progress": {
    "total_files": 25,
    "completed_files": 18,
    "failed_files": 1,
    "processing_files": 6,
    "progress_percent": 72.0
  },
  "results": {
    "total_detections": 156,
    "faces_detected": 89,
    "plates_detected": 23,
    "text_regions_detected": 44
  },
  "estimated_completion": "2024-01-15T10:35:00Z",
  "start_time": "2024-01-15T10:30:00Z"
}
```

### **Real-time Processing**

#### `POST /stream/start`

Start real-time video stream processing.

**Request:**

```http
POST /stream/start
Content-Type: application/json

{
  "source": "rtsp://camera-url/stream",
  "output_url": "rtmp://output-server/stream",
  "processing_options": {
    "detection_types": ["face", "license_plate", "text"],
    "blur_type": "adaptive_gaussian",
    "confidence": 0.6
  }
}
```

**Response:**

```json
{
  "success": true,
  "stream_id": "stream_abc123",
  "status": "starting",
  "stream_url": "rtmp://output-server/stream",
  "monitoring_url": "/stream/status/stream_abc123"
}
```

#### `GET /stream/status/{stream_id}`

Get real-time stream status.

**Response:**

```json
{
  "stream_id": "stream_abc123",
  "status": "active",
  "fps": 30.0,
  "processing_fps": 28.5,
  "detections_per_second": 12.3,
  "uptime": 3600,
  "total_frames_processed": 102600
}
```

### **Analytics & Statistics**

#### `GET /statistics`

Get comprehensive processing statistics.

**Response:**

```json
{
  "processing": {
    "total_files_processed": 1250,
    "total_faces_detected": 3420,
    "total_plates_detected": 890,
    "total_text_regions_detected": 1560,
    "total_names_redacted": 2340,
    "average_processing_time": 2.3,
    "success_rate": 98.5,
    "error_rate": 1.5
  },
  "performance": {
    "average_fps": 28.5,
    "peak_fps": 45.2,
    "gpu_utilization": 78.3,
    "memory_usage": 67.8,
    "cache_hit_rate": 85.2
  },
  "time_series": {
    "hourly_processing": [...],
    "daily_processing": [...],
    "weekly_processing": [...]
  }
}
```

#### `GET /analytics/performance`

Get detailed performance analytics.

**Response:**

```json
{
  "metrics": {
    "cpu_usage": [45.2, 47.1, 43.8, ...],
    "memory_usage": [67.8, 69.2, 65.4, ...],
    "gpu_usage": [78.3, 82.1, 75.6, ...],
    "processing_times": [2.1, 2.3, 1.9, ...]
  },
  "trends": {
    "cpu_trend": "stable",
    "memory_trend": "increasing",
    "performance_trend": "improving"
  },
  "alerts": [
    {
      "type": "high_memory_usage",
      "message": "Memory usage above 90%",
      "timestamp": "2024-01-15T10:25:00Z",
      "severity": "warning"
    }
  ]
}
```

### **Geospatial Data**

#### `GET /geospatial/heatmap`

Get geospatial heatmap data.

**Query Parameters:**

- `start_date`: Start date (ISO format)
- `end_date`: End date (ISO format)
- `resolution`: Heatmap resolution (low, medium, high)

**Response:**

```json
{
  "heatmap_data": [
    {
      "latitude": 37.7749,
      "longitude": -122.4194,
      "intensity": 5.2,
      "count": 12,
      "detection_types": {
        "faces": 8,
        "plates": 2,
        "text": 2
      }
    }
  ],
  "bounds": {
    "north": 37.7849,
    "south": 37.7649,
    "east": -122.4094,
    "west": -122.4294
  },
  "total_points": 156
}
```

#### `GET /geospatial/statistics`

Get geospatial statistics.

**Response:**

```json
{
  "total_locations": 45,
  "most_active_region": {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "activity_count": 156
  },
  "detection_density": {
    "faces_per_km2": 12.3,
    "plates_per_km2": 3.2,
    "text_per_km2": 8.7
  },
  "geographic_distribution": {
    "north_america": 45.2,
    "europe": 32.1,
    "asia": 15.8,
    "other": 6.9
  }
}
```

### **Model Management**

#### `GET /models`

List available AI models.

**Response:**

```json
{
  "models": [
    {
      "name": "face_detection_v1",
      "type": "face",
      "version": "1.0.0",
      "status": "loaded",
      "accuracy": 95.2,
      "speed": "fast",
      "memory_usage": 512
    },
    {
      "name": "plate_detection_v2",
      "type": "license_plate",
      "version": "2.1.0",
      "status": "loaded",
      "accuracy": 92.8,
      "speed": "medium",
      "memory_usage": 1024
    }
  ],
  "total_models": 5,
  "loaded_models": 3
}
```

#### `POST /models/load`

Load a specific model.

**Request:**

```http
POST /models/load
Content-Type: application/json

{
  "model_name": "face_detection_v2",
  "model_type": "face",
  "model_path": "/models/face_detection_v2.pth"
}
```

**Response:**

```json
{
  "success": true,
  "model_name": "face_detection_v2",
  "status": "loaded",
  "load_time": 2.3,
  "memory_usage": 768
}
```

### **Configuration**

#### `GET /config`

Get current configuration.

**Response:**

```json
{
  "processing": {
    "face_detection": {
      "confidence_threshold": 0.5,
      "blur_type": "gaussian",
      "blur_strength": 15
    },
    "plate_detection": {
      "confidence_threshold": 0.6,
      "blur_type": "pixelate",
      "blur_strength": 20
    }
  },
  "system": {
    "max_file_size": 104857600,
    "allowed_extensions": [".jpg", ".jpeg", ".png", ".mp4"],
    "gpu_enabled": true,
    "max_workers": 4
  }
}
```

#### `PUT /config`

Update configuration.

**Request:**

```http
PUT /config
Content-Type: application/json

{
  "processing": {
    "face_detection": {
      "confidence_threshold": 0.6,
      "blur_strength": 20
    }
  }
}
```

**Response:**

```json
{
  "success": true,
  "message": "Configuration updated successfully",
  "updated_fields": [
    "processing.face_detection.confidence_threshold",
    "processing.face_detection.blur_strength"
  ]
}
```

## 🔍 **Error Handling**

### **Error Response Format**

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid file format",
    "details": {
      "field": "file",
      "value": "document.pdf",
      "allowed_types": ["image/jpeg", "image/png", "video/mp4"]
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_12345"
  }
}
```

### **Common Error Codes**

| Code                   | HTTP Status | Description               |
| ---------------------- | ----------- | ------------------------- |
| `VALIDATION_ERROR`     | 400         | Request validation failed |
| `FILE_TOO_LARGE`       | 413         | File exceeds size limit   |
| `UNSUPPORTED_FORMAT`   | 415         | Unsupported file format   |
| `PROCESSING_ERROR`     | 422         | File processing failed    |
| `RATE_LIMIT_EXCEEDED`  | 429         | Rate limit exceeded       |
| `AUTHENTICATION_ERROR` | 401         | Authentication failed     |
| `AUTHORIZATION_ERROR`  | 403         | Insufficient permissions  |
| `NOT_FOUND`            | 404         | Resource not found        |
| `INTERNAL_ERROR`       | 500         | Internal server error     |

## 📝 **Rate Limiting**

### **Rate Limits**

- **File Processing**: 10 requests per minute
- **Batch Processing**: 5 requests per minute
- **Analytics**: 60 requests per minute
- **Configuration**: 5 requests per minute

### **Rate Limit Headers**

```http
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 7
X-RateLimit-Reset: 1642248600
```

## 🔒 **Security**

### **Security Headers**

```http
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
```

### **Input Validation**

- **File Size**: Maximum 100MB per file
- **File Types**: Only image and video formats
- **Request Size**: Maximum 1MB per request
- **Parameter Validation**: All parameters validated against schemas

## 📊 **Monitoring & Metrics**

### **Health Check Endpoints**

- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed health information
- `GET /metrics` - Prometheus-compatible metrics

### **Performance Metrics**

- Processing time per file
- Detection accuracy rates
- System resource usage
- Error rates and types
- Throughput metrics

## 🚀 **Getting Started**

### **1. Authentication**

```bash
# Get API key
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "password"}'
```

### **2. Process a File**

```bash
# Process single image
curl -X POST http://localhost:8000/process \
  -H "Authorization: Bearer <api-key>" \
  -F "file=@image.jpg" \
  -F "process_faces=true" \
  -F "process_plates=true"
```

### **3. Check Status**

```bash
# Get system status
curl -X GET http://localhost:8000/status \
  -H "Authorization: Bearer <api-key>"
```

## 📚 **SDK Examples**

### **Python SDK**

```python
from redact_ai import RedactAIClient

client = RedactAIClient(api_key="your-api-key")

# Process file
result = client.process_file(
    file_path="image.jpg",
    process_faces=True,
    process_plates=True,
    process_text=True
)

print(f"Processed {result['faces_detected']} faces")
```

### **JavaScript SDK**

```javascript
import { RedactAIClient } from "redact-ai-js";

const client = new RedactAIClient("your-api-key");

// Process file
const result = await client.processFile("image.jpg", {
  processFaces: true,
  processPlates: true,
  processText: true,
});

console.log(`Processed ${result.facesDetected} faces`);
```

---

This API provides **comprehensive functionality** for all RedactAI operations with **enterprise-grade security**, **detailed documentation**, and **multiple SDK options** for easy integration.
