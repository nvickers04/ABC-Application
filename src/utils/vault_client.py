import hvac
import os
import logging

logger = logging.getLogger(__name__)

class VaultClient:
    def __init__(self):
        self.url = os.getenv('VAULT_ADDR', 'http://127.0.0.1:8200')
        self.token = os.getenv('VAULT_TOKEN', 'root')
        self.client = hvac.Client(url=self.url, token=self.token)
        self.vault_available = False

        try:
            if self.client.is_authenticated():
                self.vault_available = True
                logger.info("Vault connection established successfully")
            else:
                logger.warning("Vault authentication failed - falling back to environment variables")
        except Exception as e:
            logger.warning(f"Vault connection failed: {e} - falling back to environment variables")

    def get_secret(self, path: str, key: str) -> str:
        if self.vault_available:
            try:
                read_response = self.client.secrets.kv.v2.read_secret_version(path=path)
                return read_response['data']['data'][key]
            except Exception as e:
                logger.warning(f"Failed to get secret {key} from Vault {path}: {e} - falling back to environment variables")

        # Fallback to environment variables
        env_key = key.upper()
        value = os.getenv(env_key)
        if value:
            logger.info(f"Using environment variable for {key}")
            return value
        else:
            raise ValueError(f"Secret {key} not found in Vault or environment variables")

# Global instance
vault_client = VaultClient()

def get_vault_secret(key: str) -> str:
    """Get secret from Vault, fallback to environment variables"""
    return vault_client.get_secret('secret/app', key)