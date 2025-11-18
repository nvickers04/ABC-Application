import hvac
import os
import logging

logger = logging.getLogger(__name__)

class VaultClient:
    def __init__(self):
        self.url = os.getenv('VAULT_ADDR', 'http://127.0.0.1:8200')
        self.token = os.getenv('VAULT_TOKEN', 'root')
        self.client = hvac.Client(url=self.url, token=self.token)
        if not self.client.is_authenticated():
            logger.error("Failed to authenticate with Vault")
            raise ValueError("Vault authentication failed")

    def get_secret(self, path: str, key: str) -> str:
        try:
            read_response = self.client.secrets.kv.v2.read_secret_version(path=path)
            return read_response['data']['data'][key]
        except Exception as e:
            logger.error(f"Failed to get secret {key} from {path}: {e}")
            raise

# Global instance
vault_client = VaultClient()

def get_vault_secret(key: str) -> str:
    """Get secret from Vault under 'app_secrets' path"""
    return vault_client.get_secret('app_secrets', key)