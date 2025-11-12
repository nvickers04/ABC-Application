
# src/utils/secure_config.py
import os
from pathlib import Path
from cryptography.fernet import Fernet

def load_secure_environment():
    """Load encrypted environment variables"""
    project_root = Path(__file__).parent.parent.parent
    key_file = project_root / '.env.key'
    encrypted_file = project_root / '.env.enc'

    if not key_file.exists() or not encrypted_file.exists():
        raise FileNotFoundError("Encrypted environment files not found. Run security_setup.py first.")

    # Load encryption key
    with open(key_file, 'rb') as f:
        key = f.read()

    # Decrypt environment
    cipher = Fernet(key)
    with open(encrypted_file, 'rb') as f:
        encrypted_data = f.read()

    decrypted_data = cipher.decrypt(encrypted_data).decode()

    # Load into environment
    for line in decrypted_data.split('\n'):
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip()

# Load secure environment on import
load_secure_environment()
