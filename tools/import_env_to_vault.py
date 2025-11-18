import os
import hvac
import logging

logger = logging.getLogger(__name__)

def import_env_to_vault():
    vault_url = os.getenv('VAULT_ADDR', 'http://127.0.0.1:8200')
    vault_token = os.getenv('VAULT_TOKEN', 'root')
    client = hvac.Client(url=vault_url, token=vault_token)
    
    if not client.is_authenticated():
        logger.error("Failed to authenticate with Vault")
        return

    secrets = {
        'MARKETDATAAPP_API_KEY': os.getenv('MARKETDATAAPP_API_KEY'),
        'ALPHA_VANTAGE_API_KEY': os.getenv('ALPHA_VANTAGE_API_KEY'),
        'DISCORD_BOT_TOKEN': os.getenv('DISCORD_BOT_TOKEN'),  # Add specific ones as needed
        'IBKR_CLIENT_ID': os.getenv('IBKR_CLIENT_ID'),
        'IBKR_ACCOUNT': os.getenv('IBKR_ACCOUNT'),
        'GROK_API_KEY': os.getenv('GROK_API_KEY'),
        # Add more keys as per your .env
    }

    # Filter out None values
    secrets = {k: v for k, v in secrets.items() if v is not None}

    if not secrets:
        logger.warning("No secrets found in environment variables")
        return

    try:
        client.secrets.kv.v2.create_or_update_secret(
            path='app_secrets',
            secret=secrets,
        )
        logger.info("Successfully imported secrets to Vault")
    except Exception as e:
        logger.error(f"Failed to import secrets to Vault: {e}")

if __name__ == "__main__":
    import_env_to_vault()