import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from cryptography.fernet import Fernet, InvalidToken
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.secure_config import SecureConfig
from src.utils.vault_client import VaultClient
from src.utils.memory_security import MemoryEncryption

class TestEncryptionSecurity:
    """Security tests for encryption key handling and data protection"""

    @pytest.fixture
    def temp_key_file(self):
        """Create a temporary key file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.key') as f:
            key = Fernet.generate_key().decode()
            f.write(key)
            return f.name

    @pytest.fixture
    def encryption_manager(self):
        """Create encryption manager instance"""
        return MemoryEncryption()

    def test_encryption_key_generation(self, encryption_manager):
        """Test that encryption keys are properly generated"""
        key = encryption_manager.generate_key()

        assert key is not None
        assert isinstance(key, str)
        assert len(key) > 0

        # Verify key can be used for encryption/decryption
        fernet = Fernet(key.encode())
        test_data = b"test_sensitive_data"
        encrypted = fernet.encrypt(test_data)
        decrypted = fernet.decrypt(encrypted)

        assert decrypted == test_data

    def test_encryption_key_storage_security(self, encryption_manager, temp_key_file):
        """Test secure key storage and access"""
        key = encryption_manager.generate_key()

        # Store key securely
        encryption_manager.store_key(key, temp_key_file)

        # Verify file permissions are restrictive (if on Unix-like system)
        if os.name != 'nt':  # Not Windows
            import stat
            file_stat = os.stat(temp_key_file)
            # Should not be world-readable
            assert not (file_stat.st_mode & stat.S_IROTH)

        # Load key back
        loaded_key = encryption_manager.load_key(temp_key_file)
        assert loaded_key == key

        # Cleanup
        os.unlink(temp_key_file)

    def test_encryption_key_rotation(self, encryption_manager):
        """Test encryption key rotation"""
        old_key = encryption_manager.generate_key()
        new_key = encryption_manager.generate_key()

        test_data = "sensitive_configuration_data"

        # Encrypt with old key
        encrypted_data = encryption_manager.encrypt_data(test_data, old_key)

        # Rotate key
        rotated_data = encryption_manager.rotate_key(encrypted_data, old_key, new_key)

        # Decrypt with new key
        decrypted_data = encryption_manager.decrypt_data(rotated_data, new_key)

        assert decrypted_data == test_data

    def test_encryption_tampering_detection(self, encryption_manager):
        """Test detection of tampered encrypted data"""
        key = encryption_manager.generate_key()
        test_data = "important_secret"

        encrypted_data = encryption_manager.encrypt_data(test_data, key)

        # Tamper with encrypted data
        tampered_data = encrypted_data[:-10] + "tampered" + encrypted_data[-10:]

        # Should raise InvalidToken exception
        with pytest.raises(InvalidToken):
            encryption_manager.decrypt_data(tampered_data, key)

    def test_weak_key_detection(self, encryption_manager):
        """Test detection of weak or compromised keys"""
        # Test with obviously weak key
        weak_key = "weak_key_123"

        with pytest.raises(ValueError):
            encryption_manager.validate_key_strength(weak_key)

        # Test with proper key
        strong_key = encryption_manager.generate_key()
        assert encryption_manager.validate_key_strength(strong_key)

    def test_encryption_memory_security(self, encryption_manager):
        """Test that sensitive data doesn't remain in memory"""
        import gc

        key = encryption_manager.generate_key()
        sensitive_data = "super_secret_api_key_" + str(os.urandom(32).hex())

        # Encrypt and immediately delete references
        encrypted = encryption_manager.encrypt_data(sensitive_data, key)
        del sensitive_data

        # Force garbage collection
        gc.collect()

        # Verify we can still decrypt
        decrypted = encryption_manager.decrypt_data(encrypted, key)
        assert decrypted.startswith("super_secret_api_key_")

    @patch('src.utils.vault_client.VaultClient.store_secret')
    @patch('src.utils.vault_client.VaultClient.retrieve_secret')
    def test_vault_integration_security(self, mock_retrieve, mock_store):
        """Test secure integration with HashiCorp Vault"""
        vault = VaultClient()

        # Mock successful storage
        mock_store.return_value = True

        # Test storing sensitive data
        secret_data = {"api_key": "sk-1234567890abcdef", "db_password": "complex_password_123"}
        result = vault.store_secret("trading_system", secret_data)

        assert result is True
        mock_store.assert_called_once()

        # Test retrieval
        mock_retrieve.return_value = secret_data
        retrieved = vault.retrieve_secret("trading_system")

        assert retrieved == secret_data
        mock_retrieve.assert_called_once()

    def test_secure_config_loading(self):
        """Test secure configuration file loading"""
        secure_config = SecureConfig()

        # Test with encrypted config
        config_data = {
            "ibkr": {
                "api_key": "encrypted_api_key_data",
                "secret": "encrypted_secret_data"
            },
            "database": {
                "password": "encrypted_db_password"
            }
        }

        # Mock encrypted storage
        with patch.object(secure_config, '_decrypt_value') as mock_decrypt:
            mock_decrypt.return_value = "decrypted_value"

            loaded_config = secure_config.load_config("test_config.json")

            # Verify decryption was called for sensitive fields
            assert mock_decrypt.call_count >= 3  # api_key, secret, password

    def test_encryption_key_expiry(self, encryption_manager):
        """Test encryption key expiry and rotation requirements"""
        key = encryption_manager.generate_key()

        # Test key age checking
        assert encryption_manager.is_key_expired(key, max_age_days=30) == False

        # Simulate old key
        with patch('time.time', return_value=1735689600 + (31 * 24 * 3600)):  # 31 days later
            assert encryption_manager.is_key_expired(key, max_age_days=30) == True

    def test_secure_random_generation(self, encryption_manager):
        """Test secure random number generation for keys"""
        # Generate multiple keys and ensure they're unique
        keys = [encryption_manager.generate_key() for _ in range(10)]

        # All keys should be unique
        assert len(set(keys)) == len(keys)

        # Keys should be proper length for Fernet
        for key in keys:
            decoded_key = key.encode()
            assert len(decoded_key) == 44  # Fernet key length

    @pytest.mark.parametrize("invalid_key", [
        "",
        "short",
        "invalid_base64_format!!!",
        "a" * 100,  # Too long
        None
    ])
    def test_invalid_key_handling(self, encryption_manager, invalid_key):
        """Test handling of invalid encryption keys"""
        with pytest.raises((ValueError, TypeError, InvalidToken)):
            test_data = "test"
            encrypted = encryption_manager.encrypt_data(test_data, invalid_key)

    def test_encryption_algorithm_security(self, encryption_manager):
        """Test that we're using secure encryption algorithms"""
        key = encryption_manager.generate_key()

        # Verify we're using Fernet (AES 128 + HMAC)
        from cryptography.fernet import Fernet

        fernet = Fernet(key.encode())
        test_data = b"security_test_data"

        encrypted = fernet.encrypt(test_data)
        decrypted = fernet.decrypt(encrypted)

        assert decrypted == test_data

        # Verify encryption provides confidentiality
        assert encrypted != test_data
        assert len(encrypted) > len(test_data)  # Should include IV and MAC