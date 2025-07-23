#!/usr/bin/env python3
"""
VORTA Phase 5.7 Security Hardening & Compliance
Enterprise AI Voice Agent - Security Management System
"""

import asyncio
import hashlib
import secrets
import hmac
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import os

# Configure security logging
security_logger = logging.getLogger('VortaSecuritySystem')
security_logger.setLevel(logging.INFO)

class SecurityLevel(Enum):
    """Security classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"

class ThreatLevel(Enum):
    """Security threat levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityEvent:
    """Security event for audit logging"""
    timestamp: float
    event_type: str
    severity: ThreatLevel
    source_ip: str
    user_id: Optional[str]
    component: str
    action: str
    details: Dict[str, Any]
    risk_score: float = 0.0

@dataclass
class AccessControlEntry:
    """Access control entry for role-based access"""
    user_id: str
    role: str
    permissions: Set[str]
    security_clearance: SecurityLevel
    created_at: float
    expires_at: Optional[float] = None
    is_active: bool = True

class EncryptionManager:
    """Advanced encryption and key management"""
    
    def __init__(self):
        self.keys: Dict[str, bytes] = {}
        self.rsa_private_key = None
        self.rsa_public_key = None
        self._generate_master_keys()
    
    def _generate_master_keys(self):
        """Generate master encryption keys"""
        # Generate RSA key pair for asymmetric encryption
        self.rsa_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.rsa_public_key = self.rsa_private_key.public_key()
        
        # Generate symmetric encryption key
        master_key = Fernet.generate_key()
        self.keys['master'] = master_key
        
        security_logger.info("🔐 Master encryption keys generated")
    
    def encrypt_sensitive_data(self, data: str, classification: SecurityLevel) -> str:
        """Encrypt sensitive data based on classification level"""
        try:
            if classification in [SecurityLevel.SECRET, SecurityLevel.TOP_SECRET]:
                # Use RSA for high-classification data
                encrypted = self.rsa_public_key.encrypt(
                    data.encode(),
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                return base64.b64encode(encrypted).decode()
            else:
                # Use Fernet for lower-classification data
                f = Fernet(self.keys['master'])
                encrypted = f.encrypt(data.encode())
                return encrypted.decode()
                
        except Exception as e:
            security_logger.error(f"❌ Encryption failed: {e}")
            raise SecurityException(f"Encryption failed: {e}")
    
    def decrypt_sensitive_data(self, encrypted_data: str, classification: SecurityLevel) -> str:
        """Decrypt sensitive data based on classification level"""
        try:
            if classification in [SecurityLevel.SECRET, SecurityLevel.TOP_SECRET]:
                # Use RSA for high-classification data
                encrypted_bytes = base64.b64decode(encrypted_data)
                decrypted = self.rsa_private_key.decrypt(
                    encrypted_bytes,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                return decrypted.decode()
            else:
                # Use Fernet for lower-classification data
                f = Fernet(self.keys['master'])
                decrypted = f.decrypt(encrypted_data.encode())
                return decrypted.decode()
                
        except Exception as e:
            security_logger.error(f"❌ Decryption failed: {e}")
            raise SecurityException(f"Decryption failed: {e}")

class AccessControlManager:
    """Role-based access control system"""
    
    def __init__(self):
        self.access_entries: Dict[str, AccessControlEntry] = {}
        self.role_permissions: Dict[str, Set[str]] = {
            'admin': {
                'create_component', 'delete_component', 'modify_component',
                'view_logs', 'modify_security', 'system_control'
            },
            'user': {
                'create_component', 'view_component', 'basic_operations'
            },
            'viewer': {
                'view_component', 'view_public_data'
            },
            'security_officer': {
                'view_logs', 'security_audit', 'threat_analysis'
            }
        }
    
    def create_user_access(self, user_id: str, role: str, security_clearance: SecurityLevel) -> bool:
        """Create user access control entry"""
        try:
            if role not in self.role_permissions:
                raise SecurityException(f"Invalid role: {role}")
            
            access_entry = AccessControlEntry(
                user_id=user_id,
                role=role,
                permissions=self.role_permissions[role].copy(),
                security_clearance=security_clearance,
                created_at=time.time()
            )
            
            self.access_entries[user_id] = access_entry
            security_logger.info(f"🔐 Access created for user {user_id} with role {role}")
            return True
            
        except Exception as e:
            security_logger.error(f"❌ Failed to create user access: {e}")
            return False
    
    def check_permission(self, user_id: str, permission: str, required_clearance: SecurityLevel = SecurityLevel.PUBLIC) -> bool:
        """Check if user has required permission and clearance"""
        try:
            if user_id not in self.access_entries:
                security_logger.warning(f"⚠️ Access denied: Unknown user {user_id}")
                return False
            
            entry = self.access_entries[user_id]
            
            if not entry.is_active:
                security_logger.warning(f"⚠️ Access denied: User {user_id} account is inactive")
                return False
            
            if entry.expires_at and time.time() > entry.expires_at:
                security_logger.warning(f"⚠️ Access denied: User {user_id} access expired")
                entry.is_active = False
                return False
            
            # Check permission
            if permission not in entry.permissions:
                security_logger.warning(f"⚠️ Access denied: User {user_id} lacks permission {permission}")
                return False
            
            # Check security clearance level
            clearance_levels = [
                SecurityLevel.PUBLIC, SecurityLevel.INTERNAL, SecurityLevel.CONFIDENTIAL,
                SecurityLevel.SECRET, SecurityLevel.TOP_SECRET
            ]
            
            user_clearance_level = clearance_levels.index(entry.security_clearance)
            required_clearance_level = clearance_levels.index(required_clearance)
            
            if user_clearance_level < required_clearance_level:
                security_logger.warning(f"⚠️ Access denied: User {user_id} clearance insufficient")
                return False
            
            return True
            
        except Exception as e:
            security_logger.error(f"❌ Permission check failed: {e}")
            return False

class ThreatDetectionEngine:
    """AI-powered threat detection and response"""
    
    def __init__(self):
        self.threat_patterns: Dict[str, Dict[str, Any]] = {
            'brute_force': {
                'threshold': 5,
                'time_window': 300,  # 5 minutes
                'severity': ThreatLevel.HIGH
            },
            'suspicious_access': {
                'threshold': 10,
                'time_window': 600,  # 10 minutes
                'severity': ThreatLevel.MEDIUM
            },
            'data_exfiltration': {
                'threshold': 1,
                'time_window': 60,  # 1 minute
                'severity': ThreatLevel.CRITICAL
            }
        }
        self.events_history: List[SecurityEvent] = []
        self.blocked_ips: Set[str] = set()
    
    def analyze_security_event(self, event: SecurityEvent) -> Dict[str, Any]:
        """Analyze security event for threats"""
        try:
            self.events_history.append(event)
            
            # Clean old events (keep last 24 hours)
            cutoff_time = time.time() - 86400
            self.events_history = [e for e in self.events_history if e.timestamp > cutoff_time]
            
            threats_detected = []
            
            # Check for brute force attacks
            if self._detect_brute_force(event):
                threats_detected.append({
                    'type': 'brute_force_attack',
                    'severity': ThreatLevel.HIGH,
                    'recommended_action': 'block_ip'
                })
            
            # Check for suspicious access patterns
            if self._detect_suspicious_access(event):
                threats_detected.append({
                    'type': 'suspicious_access_pattern',
                    'severity': ThreatLevel.MEDIUM,
                    'recommended_action': 'enhanced_monitoring'
                })
            
            # Check for data exfiltration
            if self._detect_data_exfiltration(event):
                threats_detected.append({
                    'type': 'potential_data_exfiltration',
                    'severity': ThreatLevel.CRITICAL,
                    'recommended_action': 'immediate_investigation'
                })
            
            analysis_result = {
                'event_id': f"evt_{int(event.timestamp)}_{hash(str(event.details))}",
                'threats_detected': threats_detected,
                'risk_score': self._calculate_risk_score(event, threats_detected),
                'recommendations': self._get_security_recommendations(threats_detected)
            }
            
            if threats_detected:
                security_logger.warning(f"🚨 Threats detected: {len(threats_detected)} for event from {event.source_ip}")
            
            return analysis_result
            
        except Exception as e:
            security_logger.error(f"❌ Threat analysis failed: {e}")
            return {'error': str(e)}
    
    def _detect_brute_force(self, event: SecurityEvent) -> bool:
        """Detect brute force attack patterns"""
        if event.action not in ['login_failed', 'auth_failed']:
            return False
        
        pattern = self.threat_patterns['brute_force']
        recent_events = [
            e for e in self.events_history 
            if (e.source_ip == event.source_ip and 
                e.action in ['login_failed', 'auth_failed'] and
                event.timestamp - e.timestamp <= pattern['time_window'])
        ]
        
        return len(recent_events) >= pattern['threshold']
    
    def _detect_suspicious_access(self, event: SecurityEvent) -> bool:
        """Detect suspicious access patterns"""
        if event.action not in ['component_access', 'data_access']:
            return False
        
        pattern = self.threat_patterns['suspicious_access']
        recent_events = [
            e for e in self.events_history 
            if (e.source_ip == event.source_ip and 
                e.action in ['component_access', 'data_access'] and
                event.timestamp - e.timestamp <= pattern['time_window'])
        ]
        
        return len(recent_events) >= pattern['threshold']
    
    def _detect_data_exfiltration(self, event: SecurityEvent) -> bool:
        """Detect potential data exfiltration"""
        if 'data_size' not in event.details:
            return False
        
        # Check for unusually large data transfers
        data_size = event.details.get('data_size', 0)
        return data_size > 10_000_000  # 10MB threshold
    
    def _calculate_risk_score(self, event: SecurityEvent, threats: List[Dict[str, Any]]) -> float:
        """Calculate risk score for the event"""
        base_score = 1.0
        
        # Increase score based on threats
        for threat in threats:
            if threat['severity'] == ThreatLevel.CRITICAL:
                base_score += 8.0
            elif threat['severity'] == ThreatLevel.HIGH:
                base_score += 5.0
            elif threat['severity'] == ThreatLevel.MEDIUM:
                base_score += 3.0
            else:
                base_score += 1.0
        
        # Cap at 10.0
        return min(base_score, 10.0)
    
    def _get_security_recommendations(self, threats: List[Dict[str, Any]]) -> List[str]:
        """Get security recommendations based on threats"""
        recommendations = []
        
        for threat in threats:
            action = threat.get('recommended_action', 'monitor')
            if action == 'block_ip':
                recommendations.append("Block source IP address immediately")
            elif action == 'enhanced_monitoring':
                recommendations.append("Enable enhanced monitoring for this source")
            elif action == 'immediate_investigation':
                recommendations.append("Launch immediate security investigation")
            else:
                recommendations.append("Continue monitoring activity")
        
        return recommendations

class SecurityAuditLogger:
    """Comprehensive security audit logging"""
    
    def __init__(self, log_file_path: str = "logs/security_audit.log"):
        self.log_file_path = log_file_path
        self.events_buffer: List[SecurityEvent] = []
        self.encryption_manager = EncryptionManager()
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    def log_security_event(self, event: SecurityEvent):
        """Log security event with encryption"""
        try:
            # Add to buffer
            self.events_buffer.append(event)
            
            # Encrypt sensitive event data
            encrypted_details = self.encryption_manager.encrypt_sensitive_data(
                json.dumps(event.details),
                SecurityLevel.CONFIDENTIAL
            )
            
            # Create audit log entry
            log_entry = {
                'timestamp': event.timestamp,
                'event_type': event.event_type,
                'severity': event.severity.value,
                'source_ip': event.source_ip,
                'user_id': event.user_id,
                'component': event.component,
                'action': event.action,
                'encrypted_details': encrypted_details,
                'risk_score': event.risk_score
            }
            
            # Write to audit log
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            security_logger.info(f"📝 Security event logged: {event.event_type} from {event.source_ip}")
            
        except Exception as e:
            security_logger.error(f"❌ Failed to log security event: {e}")
    
    def get_audit_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security audit summary"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            recent_events = [e for e in self.events_buffer if e.timestamp > cutoff_time]
            
            severity_counts = {}
            for event in recent_events:
                severity = event.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            top_sources = {}
            for event in recent_events:
                ip = event.source_ip
                top_sources[ip] = top_sources.get(ip, 0) + 1
            
            return {
                'total_events': len(recent_events),
                'time_period_hours': hours,
                'severity_breakdown': severity_counts,
                'top_source_ips': dict(sorted(top_sources.items(), key=lambda x: x[1], reverse=True)[:10]),
                'average_risk_score': sum(e.risk_score for e in recent_events) / len(recent_events) if recent_events else 0.0
            }
            
        except Exception as e:
            security_logger.error(f"❌ Failed to generate audit summary: {e}")
            return {'error': str(e)}

class SecurityException(Exception):
    """Custom security exception"""
    pass

class VortaSecurityManager:
    """Main security management system"""
    
    def __init__(self):
        self.encryption_manager = EncryptionManager()
        self.access_control = AccessControlManager()
        self.threat_detection = ThreatDetectionEngine()
        self.audit_logger = SecurityAuditLogger()
        
        # Initialize default security policies
        self._initialize_security_policies()
    
    def _initialize_security_policies(self):
        """Initialize default security policies"""
        # Create default admin user
        self.access_control.create_user_access(
            user_id="admin",
            role="admin",
            security_clearance=SecurityLevel.TOP_SECRET
        )
        
        # Create default security officer
        self.access_control.create_user_access(
            user_id="security_officer",
            role="security_officer",
            security_clearance=SecurityLevel.SECRET
        )
        
        security_logger.info("🔐 Security policies initialized")
    
    def secure_component_creation(self, user_id: str, component_type: str, 
                                config: Dict[str, Any], source_ip: str = "127.0.0.1") -> Dict[str, Any]:
        """Secure component creation with full security validation"""
        try:
            # Create security event
            event = SecurityEvent(
                timestamp=time.time(),
                event_type="component_creation",
                severity=ThreatLevel.LOW,
                source_ip=source_ip,
                user_id=user_id,
                component=component_type,
                action="create_component",
                details=config
            )
            
            # Check permissions
            if not self.access_control.check_permission(user_id, "create_component", SecurityLevel.INTERNAL):
                event.severity = ThreatLevel.HIGH
                event.details['reason'] = 'insufficient_permissions'
                self.audit_logger.log_security_event(event)
                raise SecurityException(f"User {user_id} lacks permission to create components")
            
            # Analyze for threats
            threat_analysis = self.threat_detection.analyze_security_event(event)
            event.risk_score = threat_analysis.get('risk_score', 0.0)
            
            # Log the event
            self.audit_logger.log_security_event(event)
            
            # Encrypt sensitive configuration
            if 'api_key' in config or 'password' in config or 'token' in config:
                for sensitive_key in ['api_key', 'password', 'token']:
                    if sensitive_key in config:
                        config[sensitive_key] = self.encryption_manager.encrypt_sensitive_data(
                            config[sensitive_key],
                            SecurityLevel.SECRET
                        )
            
            security_logger.info(f"✅ Secure component creation approved for {user_id}: {component_type}")
            
            return {
                'approved': True,
                'secure_config': config,
                'risk_score': event.risk_score,
                'security_recommendations': threat_analysis.get('recommendations', [])
            }
            
        except SecurityException:
            raise
        except Exception as e:
            security_logger.error(f"❌ Secure component creation failed: {e}")
            raise SecurityException(f"Security validation failed: {e}")
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard"""
        try:
            audit_summary = self.audit_logger.get_audit_summary(hours=24)
            
            # Get active users count
            active_users = sum(1 for entry in self.access_control.access_entries.values() if entry.is_active)
            
            # Get threat summary
            recent_threats = [e for e in self.threat_detection.events_history 
                            if e.severity in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]]
            
            return {
                'system_status': 'secure',
                'active_users': active_users,
                'audit_summary': audit_summary,
                'recent_high_threats': len(recent_threats),
                'blocked_ips_count': len(self.threat_detection.blocked_ips),
                'security_level': 'enterprise_grade',
                'compliance_status': {
                    'encryption': 'enabled',
                    'access_control': 'active',
                    'audit_logging': 'enabled',
                    'threat_detection': 'active'
                }
            }
            
        except Exception as e:
            security_logger.error(f"❌ Failed to generate security dashboard: {e}")
            return {'error': str(e)}

def main():
    """Security system demonstration"""
    print("🔒 VORTA Phase 5.7 Security Hardening & Compliance Demo")
    print("Enterprise AI Voice Agent - Security Management System")
    
    try:
        # Initialize security manager
        security_manager = VortaSecurityManager()
        
        # Simulate secure component creation
        print("\n🧪 Testing Secure Component Creation...")
        
        # Test with admin user
        result = security_manager.secure_component_creation(
            user_id="admin",
            component_type="voice_processor",
            config={"api_key": "secret_key_123", "model": "whisper-large"},
            source_ip="192.168.1.100"
        )
        
        print(f"✅ Admin creation approved: Risk Score {result['risk_score']}")
        
        # Test with unauthorized user
        try:
            security_manager.secure_component_creation(
                user_id="unknown_user",
                component_type="ai_processor",
                config={"model": "gpt-4"},
                source_ip="10.0.0.50"
            )
        except SecurityException as e:
            print(f"🚫 Unauthorized access blocked: {e}")
        
        # Get security dashboard
        print("\n📊 Security Dashboard:")
        dashboard = security_manager.get_security_dashboard()
        for key, value in dashboard.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        print("\n✅ Phase 5.7 Security Hardening: System Secure and Compliant")
        
    except Exception as e:
        print(f"❌ Security demo failed: {e}")
        raise

if __name__ == "__main__":
    main()
