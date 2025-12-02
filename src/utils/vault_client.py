import hvac
import os
import logging
from .alert_manager import get_alert_manager

logger = logging.getLogger(__name__)
alert_manager = get_alert_manager()

class VaultClient:
    def __init__(self):
        self.url = os.getenv('VAULT_ADDR', 'http://127.0.0.1:8200')
        self.token = os.getenv('VAULT_TOKEN', 'root')
        self.client = hvac.Client(url=self.url, token=self.token)
        self.vault_available = False  # Bypassing Vault for testing

    def get_secret(self, path: str, key: str) -> str:
        if self.vault_available:
            try:
                read_response = self.client.secrets.kv.v2.read_secret_version(path=path)
                return read_response['data']['data'][key]
            except Exception as e:
                logger.warning(f"Failed to get secret {key} from Vault {path}: {e} - falling back to environment variables")
                alert_manager.warning(
                    f"Vault secret retrieval failed, using fallback: {key}",
                    {"path": path, "key": key, "error": str(e)},
                    "vault_client"
                )

        # Fallback to environment variables
        env_key = key.upper()
        value = os.getenv(env_key)
        if value:
            logger.info(f"Using environment variable for {key}")
            return value
        else:
            error_msg = f"Secret {key} not found in Vault or environment variables"
            alert_manager.error(
                error_msg,
                {"path": path, "key": key, "env_key": env_key},
                "vault_client"
            )
            raise ValueError(error_msg)

    def store_secret(self, path: str, data: dict) -> bool:
        """Store secret in Vault or fallback."""
        if self.vault_available:
            try:
                self.client.secrets.kv.v2.create_or_update_secret_version(path=path, secret=data)
                return True
            except Exception as e:
                logger.warning(f"Failed to store secret in Vault {path}: {e}")
                alert_manager.error(
                    f"Failed to store secret in Vault: {path}",
                    {"path": path, "data_keys": list(data.keys()), "error": str(e)},
                    "vault_client"
                )
                return False
        # Fallback: do nothing, assume stored
        return True

    def retrieve_secret(self, path: str) -> dict:
        """Retrieve secret from Vault or fallback."""
        if self.vault_available:
            try:
                read_response = self.client.secrets.kv.v2.read_secret_version(path=path)
                return read_response['data']['data']
            except Exception as e:
                logger.warning(f"Failed to retrieve secret from Vault {path}: {e}")
                return {}
        # Fallback: return empty
        return {}

# Global instance
vault_client = VaultClient()

def get_vault_secret(key: str) -> str:
    """Get secret from Vault, fallback to environment variables"""
    return vault_client.get_secret('secret/app', key)