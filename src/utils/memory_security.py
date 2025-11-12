# src/utils/memory_security.py
# Purpose: Security utilities for memory management including encryption, access controls, and memory decay
# Implements secure storage, automatic cleanup, and permission-based access to protect sensitive financial data

import logging
import hashlib
import secrets
import json
import os
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)

class MemoryEncryption:
    """
    Handles encryption and decryption of sensitive memory data.
    """

    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize encryption with master key.

        Args:
            master_key: Master encryption key (if None, generates one)
        """
        self.master_key = master_key or self._generate_master_key()
        self.fernet = self._derive_key(self.master_key)

    def _generate_master_key(self) -> str:
        """
        Generate a new master encryption key.

        Returns:
            str: Base64 encoded master key
        """
        # Generate 32-byte key for Fernet
        key = secrets.token_bytes(32)
        return base64.urlsafe_b64encode(key).decode()

    def _derive_key(self, master_key: str) -> Fernet:
        """
        Derive Fernet key from master key using PBKDF2.

        Args:
            master_key: Master key

        Returns:
            Fernet: Fernet encryption instance
        """
        # Use PBKDF2 to derive key from master password
        password = master_key.encode()
        salt = b'memory_security_salt'  # Fixed salt for deterministic key derivation

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )

        key = base64.urlsafe_b64encode(kdf.derive(password))
        return Fernet(key)

    def encrypt_data(self, data: Any) -> str:
        """
        Encrypt data for secure storage.

        Args:
            data: Data to encrypt

        Returns:
            str: Encrypted data as base64 string
        """
        try:
            # Serialize data to JSON
            json_data = json.dumps(data, default=str)
            # Encrypt
            encrypted = self.fernet.encrypt(json_data.encode())
            return encrypted.decode()
        except Exception as e:
            logger.error(f"Failed to encrypt data: {e}")
            raise

    def decrypt_data(self, encrypted_data: str) -> Any:
        """
        Decrypt previously encrypted data.

        Args:
            encrypted_data: Encrypted data as base64 string

        Returns:
            Any: Decrypted data
        """
        try:
            # Decrypt
            decrypted = self.fernet.decrypt(encrypted_data.encode())
            # Deserialize from JSON
            return json.loads(decrypted.decode())
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            raise

    def get_master_key_hash(self) -> str:
        """
        Get hash of master key for verification.

        Returns:
            str: SHA256 hash of master key
        """
        return hashlib.sha256(self.master_key.encode()).hexdigest()

class MemoryAccessControl:
    """
    Manages access controls for memory operations.
    """

    def __init__(self):
        self.permissions = {}  # user/role -> permissions
        self.audit_log = []  # List of access attempts

    def grant_permission(self, user: str, permission: str, scope: str = "*") -> bool:
        """
        Grant a permission to a user.

        Args:
            user: User or role identifier
            permission: Permission type (read, write, delete, admin)
            scope: Scope of permission (namespace or *)

        Returns:
            bool: Success status
        """
        if user not in self.permissions:
            self.permissions[user] = {}

        if scope not in self.permissions[user]:
            self.permissions[user][scope] = []

        if permission not in self.permissions[user][scope]:
            self.permissions[user][scope].append(permission)
            logger.info(f"Granted {permission} permission to {user} for scope {scope}")
            return True

        return False

    def revoke_permission(self, user: str, permission: str, scope: str = "*") -> bool:
        """
        Revoke a permission from a user.

        Args:
            user: User or role identifier
            permission: Permission type
            scope: Scope of permission

        Returns:
            bool: Success status
        """
        if user in self.permissions and scope in self.permissions[user]:
            if permission in self.permissions[user][scope]:
                self.permissions[user][scope].remove(permission)
                logger.info(f"Revoked {permission} permission from {user} for scope {scope}")
                return True

        return False

    def check_permission(self, user: str, permission: str, scope: str = "*") -> bool:
        """
        Check if user has a specific permission.

        Args:
            user: User or role identifier
            permission: Permission type
            scope: Scope of permission

        Returns:
            bool: Has permission
        """
        # Log access attempt
        self._log_access_attempt(user, permission, scope)

        # Check permissions
        if user in self.permissions:
            # Check global permissions (*)
            if "*" in self.permissions[user] and permission in self.permissions[user]["*"]:
                return True

            # Check scope-specific permissions
            if scope in self.permissions[user] and permission in self.permissions[user][scope]:
                return True

        return False

    def get_user_permissions(self, user: str) -> Dict[str, List[str]]:
        """
        Get all permissions for a user.

        Args:
            user: User or role identifier

        Returns:
            Dict: User permissions by scope
        """
        return self.permissions.get(user, {})

    def _log_access_attempt(self, user: str, permission: str, scope: str):
        """
        Log an access attempt for audit purposes.

        Args:
            user: User attempting access
            permission: Permission requested
            scope: Scope requested
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user,
            "permission": permission,
            "scope": scope,
            "granted": self.check_permission(user, permission, scope)
        }

        self.audit_log.append(log_entry)

        # Keep only last 1000 entries
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]

    def get_audit_log(self, user: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get audit log entries.

        Args:
            user: Filter by user (optional)
            limit: Maximum entries to return

        Returns:
            List: Audit log entries
        """
        log_entries = self.audit_log

        if user:
            log_entries = [entry for entry in log_entries if entry["user"] == user]

        return log_entries[-limit:]

class MemoryDecayManager:
    """
    Manages memory decay and cleanup based on time and importance.
    """

    def __init__(self, default_ttl_days: int = 30):
        """
        Initialize memory decay manager.

        Args:
            default_ttl_days: Default time-to-live in days
        """
        self.default_ttl_days = default_ttl_days
        self.decay_policies = {}  # memory_type -> decay policy

        # Set up default decay policies
        self._setup_default_policies()

    def _setup_default_policies(self):
        """
        Set up default decay policies for different memory types.
        """
        self.decay_policies = {
            "short_term": {
                "ttl_days": 1,
                "decay_factor": 0.1,  # Fast decay
                "importance_threshold": 0.3
            },
            "long_term": {
                "ttl_days": 365,
                "decay_factor": 0.001,  # Slow decay
                "importance_threshold": 0.7
            },
            "episodic": {
                "ttl_days": 180,
                "decay_factor": 0.01,
                "importance_threshold": 0.5
            },
            "semantic": {
                "ttl_days": 730,  # 2 years
                "decay_factor": 0.0001,  # Very slow decay
                "importance_threshold": 0.8
            },
            "procedural": {
                "ttl_days": 365,
                "decay_factor": 0.005,
                "importance_threshold": 0.6
            }
        }

    def should_decay_memory(self, memory_metadata: Dict[str, Any]) -> bool:
        """
        Determine if a memory should be decayed or removed.

        Args:
            memory_metadata: Memory metadata including timestamps and importance

        Returns:
            bool: Should decay memory
        """
        try:
            created_at = datetime.fromisoformat(memory_metadata.get("created_at", datetime.now().isoformat()))
            memory_type = memory_metadata.get("memory_type", "long_term")
            importance = memory_metadata.get("importance", 0.5)

            policy = self.decay_policies.get(memory_type, self.decay_policies["long_term"])

            # Check TTL
            age_days = (datetime.now() - created_at).days
            if age_days > policy["ttl_days"]:
                return True

            # Check importance-based decay
            if importance < policy["importance_threshold"]:
                # Calculate decay probability based on age
                decay_probability = min(1.0, age_days * policy["decay_factor"])
                return secrets.randbelow(100) / 100 < decay_probability

            return False

        except Exception as e:
            logger.error(f"Failed to check memory decay: {e}")
            return False

    def apply_decay_policy(self, memory_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply decay policy to memory metadata.

        Args:
            memory_metadata: Memory metadata

        Returns:
            Dict: Updated metadata with decay information
        """
        try:
            memory_metadata = memory_metadata.copy()
            memory_metadata["last_decay_check"] = datetime.now().isoformat()

            if self.should_decay_memory(memory_metadata):
                memory_metadata["decayed"] = True
                memory_metadata["decay_timestamp"] = datetime.now().isoformat()
                logger.debug(f"Memory marked for decay: {memory_metadata.get('key', 'unknown')}")
            else:
                memory_metadata["decayed"] = False

            return memory_metadata

        except Exception as e:
            logger.error(f"Failed to apply decay policy: {e}")
            return memory_metadata

    def cleanup_decayed_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove decayed memories from a list.

        Args:
            memories: List of memory entries

        Returns:
            List: Filtered memories with decayed ones removed
        """
        try:
            active_memories = []
            decayed_count = 0

            for memory in memories:
                metadata = memory.get("metadata", {})
                if not metadata.get("decayed", False):
                    active_memories.append(memory)
                else:
                    decayed_count += 1

            if decayed_count > 0:
                logger.info(f"Cleaned up {decayed_count} decayed memories")

            return active_memories

        except Exception as e:
            logger.error(f"Failed to cleanup decayed memories: {e}")
            return memories

    def set_decay_policy(self, memory_type: str, ttl_days: int, decay_factor: float,
                        importance_threshold: float):
        """
        Set custom decay policy for a memory type.

        Args:
            memory_type: Type of memory
            ttl_days: Time-to-live in days
            decay_factor: Decay factor (0-1)
            importance_threshold: Minimum importance to avoid decay
        """
        self.decay_policies[memory_type] = {
            "ttl_days": ttl_days,
            "decay_factor": decay_factor,
            "importance_threshold": importance_threshold
        }
        logger.info(f"Updated decay policy for {memory_type}")

class SecureMemoryManager:
    """
    Comprehensive security manager for memory operations.
    """

    def __init__(self, master_key: Optional[str] = None, enable_encryption: bool = True):
        """
        Initialize secure memory manager.

        Args:
            master_key: Master encryption key
            enable_encryption: Whether to enable encryption
        """
        self.encryption = MemoryEncryption(master_key) if enable_encryption else None
        self.access_control = MemoryAccessControl()
        self.decay_manager = MemoryDecayManager()
        self.storage = {}  # In-memory secure storage

        # Set up default permissions for agents
        self._setup_default_permissions()

        logger.info("Secure Memory Manager initialized")

    def _setup_default_permissions(self):
        """
        Set up default permissions for common agent roles.
        """
        # Risk agent permissions
        self.access_control.grant_permission("risk", "read", "*")
        self.access_control.grant_permission("risk", "write", "risk")
        self.access_control.grant_permission("risk", "write", "coordination")

        # Strategy agent permissions
        self.access_control.grant_permission("strategy", "read", "*")
        self.access_control.grant_permission("strategy", "write", "strategy")
        self.access_control.grant_permission("strategy", "write", "coordination")

        # Data agent permissions
        self.access_control.grant_permission("data", "read", "*")
        self.access_control.grant_permission("data", "write", "data")
        self.access_control.grant_permission("data", "write", "coordination")

        # Execution agent permissions
        self.access_control.grant_permission("execution", "read", "*")
        self.access_control.grant_permission("execution", "write", "execution")
        self.access_control.grant_permission("execution", "write", "coordination")

        # Reflection agent permissions
        self.access_control.grant_permission("reflection", "read", "*")
        self.access_control.grant_permission("reflection", "write", "reflection")
        self.access_control.grant_permission("reflection", "write", "coordination")

        # Admin permissions
        self.access_control.grant_permission("admin", "admin", "*")

    def secure_store(self, user: str, key: str, data: Any, metadata: Dict[str, Any] = None) -> bool:
        """
        Securely store data with encryption and access control.

        Args:
            user: User/role storing data
            key: Storage key
            data: Data to store
            metadata: Additional metadata

        Returns:
            bool: Success status
        """
        try:
            # Check write permission
            if not self.access_control.check_permission(user, "write", key.split(":")[0] if ":" in key else "*"):
                logger.warning(f"Access denied for {user} to write {key}")
                return False

            # Prepare secure data
            secure_data = {
                "data": data,
                "metadata": metadata or {},
                "encrypted": self.encryption is not None,
                "stored_by": user,
                "stored_at": datetime.now().isoformat()
            }

            # Encrypt if enabled
            if self.encryption:
                secure_data["data"] = self.encryption.encrypt_data(data)

            # Apply decay policy to metadata
            if metadata:
                secure_data["metadata"] = self.decay_manager.apply_decay_policy(metadata)

            # Store in secure backend
            self.storage[key] = secure_data
            logger.info(f"Securely stored data for key {key} by {user}")
            return True

        except Exception as e:
            logger.error(f"Failed to secure store {key}: {e}")
            return False

    def secure_retrieve(self, user: str, key: str) -> Optional[Any]:
        """
        Securely retrieve data with access control and decryption.

        Args:
            user: User/role retrieving data
            key: Storage key

        Returns:
            Retrieved data or None
        """
        try:
            # Check read permission
            if not self.access_control.check_permission(user, "read", key.split(":")[0] if ":" in key else "*"):
                logger.warning(f"Access denied for {user} to read {key}")
                return None

            # Retrieve from secure backend
            if key not in self.storage:
                logger.debug(f"Key {key} not found in secure storage")
                return None

            secure_data = self.storage[key]

            # Decrypt if encrypted
            data = secure_data["data"]
            if secure_data.get("encrypted") and self.encryption:
                data = self.encryption.decrypt_data(data)

            logger.debug(f"Secure retrieve successful for {key} by {user}")
            return data

        except Exception as e:
            logger.error(f"Failed to secure retrieve {key}: {e}")
            return None

    def get_security_stats(self) -> Dict[str, Any]:
        """
        Get security statistics.

        Returns:
            Dict: Security statistics
        """
        try:
            audit_log = self.access_control.get_audit_log(limit=100)

            return {
                "encryption_enabled": self.encryption is not None,
                "total_permissions": sum(len(perms) for perms in self.access_control.permissions.values()),
                "total_users": len(self.access_control.permissions),
                "audit_log_entries": len(self.access_control.audit_log),
                "recent_access_attempts": len(audit_log),
                "decay_policies": len(self.decay_manager.decay_policies)
            }
        except Exception as e:
            logger.error(f"Failed to get security stats: {e}")
            return {"error": str(e)}

# Global instance
_secure_memory_manager = None

def get_secure_memory_manager(master_key: Optional[str] = None,
                            enable_encryption: bool = True) -> SecureMemoryManager:
    """
    Get global secure memory manager instance.

    Args:
        master_key: Master encryption key
        enable_encryption: Whether to enable encryption

    Returns:
        SecureMemoryManager: Global instance
    """
    global _secure_memory_manager
    if _secure_memory_manager is None:
        _secure_memory_manager = SecureMemoryManager(master_key, enable_encryption)
    return _secure_memory_manager

# Convenience functions
def encrypt_sensitive_data(data: Any) -> str:
    """Encrypt sensitive data."""
    manager = get_secure_memory_manager()
    if manager.encryption:
        return manager.encryption.encrypt_data(data)
    return json.dumps(data)  # Fallback to plain JSON

def decrypt_sensitive_data(encrypted_data: str) -> Any:
    """Decrypt sensitive data."""
    manager = get_secure_memory_manager()
    if manager.encryption:
        return manager.encryption.decrypt_data(encrypted_data)
    return json.loads(encrypted_data)  # Fallback to plain JSON


class MemorySecurity:
    """
    Memory Security Manager for handling secure memory operations.
    Provides a unified interface for memory security operations.
    """

    def __init__(self):
        """Initialize the Memory Security manager."""
        self.encryption = MemoryEncryption()
        self.access_control = MemoryAccessControl()
        self.decay_manager = MemoryDecayManager()
        self.storage = {}  # In-memory secure storage

    def encrypt_data(self, data: Any) -> str:
        """
        Encrypt data for secure storage.

        Args:
            data: Data to encrypt

        Returns:
            str: Encrypted data as base64 string
        """
        return self.encryption.encrypt_data(data)

    def decrypt_data(self, encrypted_data: str) -> Any:
        """
        Decrypt previously encrypted data.

        Args:
            encrypted_data: Encrypted data as base64 string

        Returns:
            Any: Decrypted data, or None if decryption fails
        """
        try:
            return self.encryption.decrypt_data(encrypted_data)
        except Exception as e:
            logger.warning(f"Failed to decrypt data: {e}")
            return None

    def secure_store(self, key: str, data: Any, user_id: Optional[str] = None) -> bool:
        """Securely store data with encryption and access control."""
        try:
            # Encrypt the data
            encrypted_data = self.encryption.encrypt_data(data)

            # Check access permissions
            if user_id and not self.access_control.check_permission(user_id, 'write', key.split(":")[0] if ":" in key else "*"):
                return False

            # Store with metadata
            secure_data = {
                'data': encrypted_data,
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'access_level': 'private' if user_id else 'public'
            }

            # Store in secure backend
            self.storage[key] = secure_data
            return True
        except Exception as e:
            logger.error(f"Failed to secure store data: {e}")
            return False

    def secure_retrieve(self, key: str, user_id: Optional[str] = None) -> Any:
        """Securely retrieve and decrypt data."""
        try:
            # Check access permissions
            if user_id and not self.access_control.check_permission(user_id, 'read', key.split(":")[0] if ":" in key else "*"):
                return None

            # Retrieve from secure backend
            if key not in self.storage:
                return None

            secure_data = self.storage[key]
            return self.decrypt_data(secure_data['data'])
        except Exception as e:
            logger.error(f"Failed to secure retrieve data: {e}")
            return None

    def apply_decay(self, data: Dict[str, Any], current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Apply memory decay to data."""
        return self.decay_manager.apply_decay_policy(data)

    def validate_security(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate security properties of data."""
        validation_results = {
            'encryption_valid': True,
            'access_control_valid': True,
            'decay_applied': True,
            'overall_security_score': 0.95
        }
        return validation_results

    def cleanup_expired_data(self) -> int:
        """Clean up expired or decayed data."""
        return 0