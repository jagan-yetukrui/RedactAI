"""
Advanced enterprise security and audit system for RedactAI.

This module implements comprehensive security features including audit logging,
access control, data encryption, and compliance reporting for enterprise use.
"""

import hashlib
import hmac
import json
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
import sqlite3
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from ..utils.logger import get_logger
from ..utils.monitoring import get_metrics_collector

logger = get_logger(__name__)


class SecurityLevel(Enum):
    """Security levels for different operations."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class AuditEventType(Enum):
    """Types of audit events."""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    FILE_UPLOAD = "file_upload"
    FILE_PROCESS = "file_process"
    FILE_DOWNLOAD = "file_download"
    FILE_DELETE = "file_delete"
    CONFIG_CHANGE = "config_change"
    SECURITY_VIOLATION = "security_violation"
    SYSTEM_ERROR = "system_error"
    DATA_ACCESS = "data_access"
    ADMIN_ACTION = "admin_action"


@dataclass
class AuditEvent:
    """An audit event record."""
    
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource_path: Optional[str]
    action: str
    result: str  # success, failure, error
    details: Dict[str, Any] = field(default_factory=dict)
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    data_classification: str = "internal"
    risk_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'session_id': self.session_id,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'resource_path': self.resource_path,
            'action': self.action,
            'result': self.result,
            'details': json.dumps(self.details),
            'security_level': self.security_level.value,
            'data_classification': self.data_classification,
            'risk_score': self.risk_score
        }


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    
    # Access control
    require_authentication: bool = True
    session_timeout_minutes: int = 30
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    
    # Data protection
    encrypt_sensitive_data: bool = True
    data_retention_days: int = 90
    auto_delete_processed_files: bool = True
    
    # Audit requirements
    audit_all_operations: bool = True
    audit_retention_days: int = 365
    real_time_monitoring: bool = True
    
    # Security thresholds
    max_file_size_mb: int = 100
    allowed_file_types: List[str] = field(default_factory=lambda: [
        'image/jpeg', 'image/png', 'video/mp4', 'video/avi'
    ])
    blocked_ip_ranges: List[str] = field(default_factory=list)
    
    # Compliance
    gdpr_compliance: bool = True
    hipaa_compliance: bool = False
    sox_compliance: bool = False


class DataEncryption:
    """Advanced data encryption utilities."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        """Initialize encryption with master key."""
        if master_key is None:
            master_key = Fernet.generate_key()
        
        self.master_key = master_key
        self.fernet = Fernet(master_key)
        
        # Derive additional keys for different purposes
        self.audit_key = self._derive_key(b"audit", master_key)
        self.data_key = self._derive_key(b"data", master_key)
        self.metadata_key = self._derive_key(b"metadata", master_key)
    
    def _derive_key(self, purpose: bytes, master_key: bytes) -> bytes:
        """Derive a specific purpose key from master key."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=purpose,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(master_key))
    
    def encrypt_data(self, data: bytes, purpose: str = "data") -> bytes:
        """Encrypt data with purpose-specific key."""
        if purpose == "audit":
            key = self.audit_key
        elif purpose == "data":
            key = self.data_key
        elif purpose == "metadata":
            key = self.metadata_key
        else:
            key = self.master_key
        
        fernet = Fernet(key)
        return fernet.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes, purpose: str = "data") -> bytes:
        """Decrypt data with purpose-specific key."""
        if purpose == "audit":
            key = self.audit_key
        elif purpose == "data":
            key = self.data_key
        elif purpose == "metadata":
            key = self.metadata_key
        else:
            key = self.master_key
        
        fernet = Fernet(key)
        return fernet.decrypt(encrypted_data)
    
    def encrypt_file(self, file_path: Path, output_path: Path, purpose: str = "data") -> None:
        """Encrypt a file."""
        with open(file_path, 'rb') as f:
            data = f.read()
        
        encrypted_data = self.encrypt_data(data, purpose)
        
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
    
    def decrypt_file(self, encrypted_path: Path, output_path: Path, purpose: str = "data") -> None:
        """Decrypt a file."""
        with open(encrypted_path, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_data = self.decrypt_data(encrypted_data, purpose)
        
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)


class AccessControl:
    """Advanced access control system."""
    
    def __init__(self, policy: SecurityPolicy):
        """Initialize access control."""
        self.policy = policy
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.locked_accounts: Dict[str, datetime] = {}
        self.lock = threading.RLock()
    
    def authenticate_user(self, user_id: str, password_hash: str, 
                         ip_address: str) -> Tuple[bool, str, Optional[str]]:
        """Authenticate a user."""
        with self.lock:
            # Check if account is locked
            if user_id in self.locked_accounts:
                lockout_time = self.locked_accounts[user_id]
                if datetime.now(timezone.utc) - lockout_time < timedelta(minutes=self.policy.lockout_duration_minutes):
                    return False, "Account locked due to failed attempts", None
            
            # Check failed attempts
            if user_id in self.failed_attempts:
                recent_attempts = [
                    attempt for attempt in self.failed_attempts[user_id]
                    if datetime.now(timezone.utc) - attempt < timedelta(minutes=15)
                ]
                
                if len(recent_attempts) >= self.policy.max_login_attempts:
                    self.locked_accounts[user_id] = datetime.now(timezone.utc)
                    return False, "Too many failed attempts", None
            
            # Simulate authentication (in real implementation, verify against database)
            if self._verify_credentials(user_id, password_hash):
                # Clear failed attempts
                if user_id in self.failed_attempts:
                    del self.failed_attempts[user_id]
                
                # Create session
                session_id = str(uuid.uuid4())
                self.active_sessions[session_id] = {
                    'user_id': user_id,
                    'ip_address': ip_address,
                    'created_at': datetime.now(timezone.utc),
                    'last_activity': datetime.now(timezone.utc)
                }
                
                return True, "Authentication successful", session_id
            else:
                # Record failed attempt
                if user_id not in self.failed_attempts:
                    self.failed_attempts[user_id] = []
                self.failed_attempts[user_id].append(datetime.now(timezone.utc))
                
                return False, "Invalid credentials", None
    
    def _verify_credentials(self, user_id: str, password_hash: str) -> bool:
        """Verify user credentials (placeholder implementation)."""
        # In real implementation, verify against secure database
        return user_id == "admin" and password_hash == "hashed_password"
    
    def validate_session(self, session_id: str, ip_address: str) -> Tuple[bool, Optional[str]]:
        """Validate an active session."""
        with self.lock:
            if session_id not in self.active_sessions:
                return False, "Invalid session"
            
            session = self.active_sessions[session_id]
            
            # Check IP address
            if session['ip_address'] != ip_address:
                return False, "IP address mismatch"
            
            # Check session timeout
            timeout = timedelta(minutes=self.policy.session_timeout_minutes)
            if datetime.now(timezone.utc) - session['last_activity'] > timeout:
                del self.active_sessions[session_id]
                return False, "Session expired"
            
            # Update last activity
            session['last_activity'] = datetime.now(timezone.utc)
            
            return True, session['user_id']
    
    def logout_user(self, session_id: str) -> bool:
        """Logout a user."""
        with self.lock:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                return True
            return False
    
    def check_access_permission(self, user_id: str, resource: str, 
                               action: str) -> Tuple[bool, str]:
        """Check if user has permission to access resource."""
        # Simplified permission check (in real implementation, use RBAC)
        if user_id == "admin":
            return True, "Access granted"
        
        # Check resource-specific permissions
        if "admin" in resource and user_id != "admin":
            return False, "Admin access required"
        
        return True, "Access granted"


class AuditLogger:
    """Advanced audit logging system."""
    
    def __init__(self, db_path: Path, encryption: DataEncryption, policy: SecurityPolicy):
        """Initialize audit logger."""
        self.db_path = db_path
        self.encryption = encryption
        self.policy = policy
        self.lock = threading.RLock()
        
        # Initialize database
        self._init_database()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_records, daemon=True)
        self.cleanup_thread.start()
    
    def _init_database(self):
        """Initialize audit database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    resource_path TEXT,
                    action TEXT NOT NULL,
                    result TEXT NOT NULL,
                    details TEXT,
                    security_level TEXT NOT NULL,
                    data_classification TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)
            """)
    
    def log_event(self, event: AuditEvent) -> bool:
        """Log an audit event."""
        try:
            with self.lock:
                # Encrypt sensitive details
                encrypted_details = self.encryption.encrypt_data(
                    json.dumps(event.details).encode(), "audit"
                )
                
                # Store in database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO audit_events (
                            event_id, event_type, timestamp, user_id, session_id,
                            ip_address, user_agent, resource_path, action, result,
                            details, security_level, data_classification, risk_score
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event.event_id,
                        event.event_type.value,
                        event.timestamp.isoformat(),
                        event.user_id,
                        event.session_id,
                        event.ip_address,
                        event.user_agent,
                        event.resource_path,
                        event.action,
                        event.result,
                        base64.b64encode(encrypted_details).decode(),
                        event.security_level.value,
                        event.data_classification,
                        event.risk_score
                    ))
                
                logger.info(f"Audit event logged: {event.event_type.value} - {event.action}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            return False
    
    def _cleanup_old_records(self):
        """Clean up old audit records based on retention policy."""
        while True:
            try:
                time.sleep(3600)  # Run every hour
                
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.policy.audit_retention_days)
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        DELETE FROM audit_events 
                        WHERE timestamp < ?
                    """, (cutoff_date.isoformat(),))
                    
                    deleted_count = conn.total_changes
                    if deleted_count > 0:
                        logger.info(f"Cleaned up {deleted_count} old audit records")
                
            except Exception as e:
                logger.error(f"Error in audit cleanup: {e}")
    
    def query_events(self, start_date: datetime, end_date: datetime,
                    event_types: List[AuditEventType] = None,
                    user_id: str = None,
                    security_level: SecurityLevel = None) -> List[AuditEvent]:
        """Query audit events with filters."""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM audit_events WHERE timestamp BETWEEN ? AND ?"
            params = [start_date.isoformat(), end_date.isoformat()]
            
            if event_types:
                placeholders = ','.join(['?' for _ in event_types])
                query += f" AND event_type IN ({placeholders})"
                params.extend([et.value for et in event_types])
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if security_level:
                query += " AND security_level = ?"
                params.append(security_level.value)
            
            query += " ORDER BY timestamp DESC"
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            events = []
            for row in rows:
                # Decrypt details
                encrypted_details = base64.b64decode(row[11])
                decrypted_details = self.encryption.decrypt_data(encrypted_details, "audit")
                details = json.loads(decrypted_details.decode())
                
                event = AuditEvent(
                    event_id=row[1],
                    event_type=AuditEventType(row[2]),
                    timestamp=datetime.fromisoformat(row[3]),
                    user_id=row[4],
                    session_id=row[5],
                    ip_address=row[6],
                    user_agent=row[7],
                    resource_path=row[8],
                    action=row[9],
                    result=row[10],
                    details=details,
                    security_level=SecurityLevel(row[12]),
                    data_classification=row[13],
                    risk_score=row[14]
                )
                events.append(event)
            
            return events
    
    def generate_compliance_report(self, start_date: datetime, 
                                 end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for the specified period."""
        events = self.query_events(start_date, end_date)
        
        # Calculate statistics
        total_events = len(events)
        event_type_counts = {}
        user_activity = {}
        security_violations = 0
        failed_operations = 0
        
        for event in events:
            # Event type counts
            event_type = event.event_type.value
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
            
            # User activity
            if event.user_id:
                user_activity[event.user_id] = user_activity.get(event.user_id, 0) + 1
            
            # Security violations
            if event.event_type == AuditEventType.SECURITY_VIOLATION:
                security_violations += 1
            
            # Failed operations
            if event.result == "failure":
                failed_operations += 1
        
        report = {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'summary': {
                'total_events': total_events,
                'security_violations': security_violations,
                'failed_operations': failed_operations,
                'success_rate': (total_events - failed_operations) / total_events if total_events > 0 else 0
            },
            'event_breakdown': event_type_counts,
            'user_activity': user_activity,
            'compliance_status': {
                'gdpr_compliant': self.policy.gdpr_compliance,
                'hipaa_compliant': self.policy.hipaa_compliance,
                'sox_compliant': self.policy.sox_compliance
            }
        }
        
        return report


class SecurityManager:
    """Main security manager coordinating all security features."""
    
    def __init__(self, config_path: Path = None):
        """Initialize security manager."""
        self.config_path = config_path or Path("security/config.json")
        self.policy = self._load_security_policy()
        self.encryption = DataEncryption()
        self.access_control = AccessControl(self.policy)
        self.audit_logger = AuditLogger(
            Path("security/audit.db"),
            self.encryption,
            self.policy
        )
        self.metrics_collector = get_metrics_collector()
        
        logger.info("Security manager initialized")
    
    def _load_security_policy(self) -> SecurityPolicy:
        """Load security policy from configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            return SecurityPolicy(**config_data)
        else:
            # Create default policy
            policy = SecurityPolicy()
            self._save_security_policy(policy)
            return policy
    
    def _save_security_policy(self, policy: SecurityPolicy):
        """Save security policy to configuration."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(policy.__dict__, f, indent=2)
    
    def create_audit_event(self, event_type: AuditEventType, action: str,
                          user_id: str = None, session_id: str = None,
                          ip_address: str = None, user_agent: str = None,
                          resource_path: str = None, details: Dict[str, Any] = None,
                          security_level: SecurityLevel = SecurityLevel.INTERNAL) -> AuditEvent:
        """Create and log an audit event."""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource_path=resource_path,
            action=action,
            result="success",
            details=details or {},
            security_level=security_level,
            risk_score=self._calculate_risk_score(event_type, action, details)
        )
        
        self.audit_logger.log_event(event)
        return event
    
    def _calculate_risk_score(self, event_type: AuditEventType, action: str, 
                             details: Dict[str, Any]) -> float:
        """Calculate risk score for an event."""
        base_scores = {
            AuditEventType.USER_LOGIN: 0.1,
            AuditEventType.USER_LOGOUT: 0.0,
            AuditEventType.FILE_UPLOAD: 0.3,
            AuditEventType.FILE_PROCESS: 0.2,
            AuditEventType.FILE_DOWNLOAD: 0.4,
            AuditEventType.FILE_DELETE: 0.5,
            AuditEventType.CONFIG_CHANGE: 0.8,
            AuditEventType.SECURITY_VIOLATION: 1.0,
            AuditEventType.SYSTEM_ERROR: 0.6,
            AuditEventType.DATA_ACCESS: 0.7,
            AuditEventType.ADMIN_ACTION: 0.9
        }
        
        base_score = base_scores.get(event_type, 0.5)
        
        # Adjust based on action
        if "delete" in action.lower():
            base_score += 0.2
        elif "admin" in action.lower():
            base_score += 0.3
        
        # Adjust based on details
        if details:
            if details.get('file_size', 0) > 50 * 1024 * 1024:  # Large files
                base_score += 0.1
            if details.get('sensitive_data', False):
                base_score += 0.2
        
        return min(base_score, 1.0)
    
    def validate_file_upload(self, file_path: Path, user_id: str) -> Tuple[bool, str]:
        """Validate file upload against security policy."""
        # Check file size
        file_size = file_path.stat().st_size
        max_size = self.policy.max_file_size_mb * 1024 * 1024
        
        if file_size > max_size:
            self.create_audit_event(
                AuditEventType.SECURITY_VIOLATION,
                "file_upload_size_exceeded",
                user_id=user_id,
                details={'file_size': file_size, 'max_size': max_size}
            )
            return False, f"File size exceeds limit of {self.policy.max_file_size_mb}MB"
        
        # Check file type (simplified)
        file_extension = file_path.suffix.lower()
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.mp4', '.avi']
        
        if file_extension not in allowed_extensions:
            self.create_audit_event(
                AuditEventType.SECURITY_VIOLATION,
                "file_upload_invalid_type",
                user_id=user_id,
                details={'file_extension': file_extension}
            )
            return False, f"File type {file_extension} not allowed"
        
        return True, "File upload validated"
    
    def encrypt_sensitive_file(self, file_path: Path, output_path: Path) -> bool:
        """Encrypt a sensitive file."""
        try:
            self.encryption.encrypt_file(file_path, output_path, "data")
            
            # Log the encryption
            self.create_audit_event(
                AuditEventType.DATA_ACCESS,
                "file_encrypted",
                details={'original_path': str(file_path), 'encrypted_path': str(output_path)}
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to encrypt file: {e}")
            return False
    
    def get_security_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate security report for the last N days."""
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        return self.audit_logger.generate_compliance_report(start_date, end_date)


# Global security manager instance
_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """Get the global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


def create_audit_event(event_type: AuditEventType, action: str, **kwargs) -> AuditEvent:
    """Create an audit event using the global security manager."""
    return get_security_manager().create_audit_event(event_type, action, **kwargs)
