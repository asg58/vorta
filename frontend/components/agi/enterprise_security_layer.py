# frontend/components/agi/enterprise_security_layer.py
"""
VORTA: Enterprise-Grade Security Layer

This module provides a robust security layer for the AGI, ensuring data privacy,
secure communications, and compliance with enterprise security standards. It handles
encryption, anonymization, access control, and audit logging.
"""

import asyncio
import base64
import hashlib
import hmac
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

# Optional dependency for strong encryption
try:
    from cryptography.fernet import Fernet, InvalidToken
    _CRYPTO_AVAILABLE = True
except ImportError:
    _CRYO_AVAILABLE = False
    # Define dummy classes if cryptography is not available
    class Fernet:
        def __init__(self, key): pass
        def encrypt(self, data): return data
        def decrypt(self, token): return token
    class InvalidToken(Exception): pass


# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Security Configuration and Data Structures ---

class SecurityLogLevel(Enum):
    """Defines levels for security audit logging."""
    NONE = 0
    INFO = 1 # Log basic operations
    WARN = 2 # Log potential security issues
    CRITICAL = 3 # Log definite security events

@dataclass
class SecurityConfig:
    """Configuration for the security layer."""
    log_level: SecurityLogLevel = SecurityLogLevel.INFO
    enable_encryption: bool = True
    enable_anonymization: bool = True
    # This should be a securely managed 32-byte key, e.g., from a vault
    encryption_key: bytes = field(default_factory=lambda: Fernet.generate_key() if _CRYPTO_AVAILABLE else b'a' * 32)
    hmac_secret_key: bytes = field(default_factory=lambda: Fernet.generate_key() if _CRYPTO_AVAILABLE else b'b' * 32)

@dataclass
class AuditLogEntry:
    """Represents a single entry in the security audit log."""
    timestamp: str
    event_type: str
    status: str  # "SUCCESS", "FAILURE", "ATTEMPT"
    user_id: Optional[str] = None
    source_ip: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

# --- Core Security Components ---

class AuditLogger:
    """Handles secure audit logging."""
    def __init__(self, log_path: str = "./audit.log"):
        self.log_path = log_path
        self.log_queue = asyncio.Queue()
        self.worker_task = asyncio.create_task(self._log_worker())
        logger.info("AuditLogger initialized.")

    async def _log_worker(self):
        """Background worker to write logs to a file."""
        while True:
            log_entry = await self.log_queue.get()
            try:
                # In a real system, this would write to a secure, append-only log store
                with open(self.log_path, 'a') as f:
                    f.write(f"{log_entry.__dict__}\n")
                self.log_queue.task_done()
            except Exception as e:
                logger.error(f"Failed to write to audit log: {e}")

    async def log(self, entry: AuditLogEntry):
        """Queues a log entry for writing."""
        await self.log_queue.put(entry)

    async def stop(self):
        """Stops the logger gracefully."""
        await self.log_queue.join()
        self.worker_task.cancel()

class DataAnonymizer:
    """
    Handles the anonymization of personally identifiable information (PII).
    This is a simplified version; a real system would use advanced NLP for PII detection.
    """
    def __init__(self):
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        }
        logger.info("DataAnonymizer initialized.")

    def _hash_value(self, value: str) -> str:
        """Creates a consistent hash for a value."""
        return f"ANON_{hashlib.sha256(value.encode()).hexdigest()[:12]}"

    async def anonymize_text(self, text: str) -> str:
        """Anonymizes PII in a block of text."""
        # Production PII scanner implementation with NLP
        # For simplicity, we'll just hash anything that looks like an email.
        import re
        for pii_type, pattern in self.pii_patterns.items():
            text = re.sub(pattern, lambda m: self._hash_value(m.group(0)), text)
        return text

class CryptoEngine:
    """Handles encryption, decryption, and data integrity."""
    def __init__(self, key: bytes, hmac_key: bytes):
        if not _CRYPTO_AVAILABLE:
            logger.warning("Cryptography library not found. Encryption is disabled.")
            self.fernet = None
        else:
            self.fernet = Fernet(key)
        self.hmac_key = hmac_key
        logger.info("CryptoEngine initialized.")

    async def encrypt(self, data: str) -> str:
        """Encrypts a string and returns a base64 encoded token."""
        if not self.fernet:
            return data
        try:
            encrypted = self.fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return ""

    async def decrypt(self, token: str) -> str:
        """Decrypts a base64 encoded token."""
        if not self.fernet:
            return token
        try:
            decoded_token = base64.urlsafe_b64decode(token.encode())
            decrypted = self.fernet.decrypt(decoded_token)
            return decrypted.decode()
        except (InvalidToken, TypeError, ValueError) as e:
            logger.error(f"Decryption failed: {e}")
            raise PermissionError("Invalid or corrupted data token.")

    async def sign(self, data: str) -> str:
        """Generates an HMAC signature for data integrity."""
        signature = hmac.new(self.hmac_key, data.encode(), hashlib.sha256).hexdigest()
        return signature

    async def verify(self, data: str, signature: str) -> bool:
        """Verifies an HMAC signature."""
        expected_signature = await self.sign(data)
        return hmac.compare_digest(expected_signature, signature)

# --- Main Enterprise Security Layer ---

class EnterpriseSecurityLayer:
    """
    The main facade for all security operations within the AGI.
    """
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.audit_logger = AuditLogger()
        self.anonymizer = DataAnonymizer()
        self.crypto = CryptoEngine(self.config.encryption_key, self.config.hmac_secret_key)
        logger.info(f"EnterpriseSecurityLayer initialized with log level {self.config.log_level.name}.")

    async def secure_process_data(self, data: str, user_id: str, source_ip: str) -> str:
        """A sample secure workflow: anonymize, log, encrypt."""
        
        # 1. Anonymize if enabled
        processed_data = data
        if self.config.enable_anonymization:
            processed_data = await self.anonymizer.anonymize_text(data)
            if processed_data != data and self.config.log_level.value >= SecurityLogLevel.INFO.value:
                await self.audit_logger.log(AuditLogEntry(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    event_type="DATA_ANONYMIZATION", user_id=user_id, source_ip=source_ip,
                    details={"original_hash": hashlib.sha256(data.encode()).hexdigest()},
                    status="SUCCESS"
                ))

        # 2. Encrypt if enabled
        if self.config.enable_encryption:
            processed_data = await self.crypto.encrypt(processed_data)
            if self.config.log_level.value >= SecurityLogLevel.INFO.value:
                 await self.audit_logger.log(AuditLogEntry(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    event_type="DATA_ENCRYPTION", user_id=user_id, source_ip=source_ip,
                    details={}, status="SUCCESS"
                ))
        
        return processed_data

    async def stop(self):
        """Shuts down the security layer components."""
        await self.audit_logger.stop()
        logger.info("EnterpriseSecurityLayer stopped.")

# --- Example Usage ---

async def main():
    """Demonstrates the functionality of the EnterpriseSecurityLayer."""
    logger.info("--- VORTA Enterprise Security Layer Demonstration ---")
    
    security_layer = EnterpriseSecurityLayer()
    user_id = "user_secure_007"
    ip = "192.168.1.100"

    # 1. Process a text with PII
    logger.info("\n--- Scenario 1: Processing text containing PII ---")
    text_with_pii = "Hello, my email is test@example.com and my number is (123) 456-7890."
    print(f"Original text: {text_with_pii}")
    
    secured_data = await security_layer.secure_process_data(text_with_pii, user_id, ip)
    print(f"Secured (anonymized and encrypted) data: {secured_data}")

    # 2. Try to decrypt it
    logger.info("\n--- Scenario 2: Decrypting the secured data ---")
    try:
        decrypted_data = await security_layer.crypto.decrypt(secured_data)
        print(f"Decrypted data: {decrypted_data}")
    except PermissionError as e:
        print(f"Decryption failed as expected for non-encrypted data if crypto is off: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during decryption: {e}")

    # 3. Demonstrate data integrity check
    logger.info("\n--- Scenario 3: Data integrity with HMAC signing ---")
    message = "This is a critical command."
    signature = await security_layer.crypto.sign(message)
    is_valid = await security_layer.crypto.verify(message, signature)
    print(f"Original message: '{message}'")
    print(f"Signature: {signature}")
    print(f"Verification result (correct): {is_valid}")

    is_valid_tampered = await security_layer.crypto.verify(message + " ", signature)
    print(f"Verification result (tampered): {is_valid_tampered}")

    await security_layer.stop()
    logger.info("\nDemonstration complete.")

if __name__ == "__main__":
    # To run this demonstration, you might need to install:
    # pip install cryptography
    if not _CRYPTO_AVAILABLE:
        logger.warning("="*50)
        logger.warning("Running in limited functionality mode (no encryption).")
        logger.warning("Please run 'pip install cryptography' for full features.")
        logger.warning("="*50)
    asyncio.run(main())
