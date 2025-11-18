# setup_production_vault.ps1
# This script sets up a production-like Vault instance on Windows for local use.
# WARNING: This is NOT fully production-ready (e.g., no clustering, basic TLS). For real production, use HashiCorp Vault Cloud, a Linux server cluster, or consult official docs.
# Prerequisites: Vault CLI installed at C:\Users\nvick\ABC-Application\tools\vault\vault.exe, admin rights, NSSM for service (install via Chocolatey or manually).
# Usage: Run as Administrator. Follow prompts to enter/rotate secrets securely.

# Configuration - Customize if needed
$VAULT_EXE = 'C:\Users\nvick\ABC-Application\tools\vault\vault.exe'
$VAULT_CONFIG_DIR = 'C:\Users\nvick\ABC-Application\vault-config'
$VAULT_DATA_DIR = 'C:\Users\nvick\ABC-Application\vault-data'
$VAULT_CONFIG_FILE = "$VAULT_CONFIG_DIR\vault-config.hcl"
$VAULT_ADDR = 'https://127.0.0.1:8200'  # Use HTTPS for prod
$VAULT_SERVICE_NAME = 'VaultService'

# Step 1: Install NSSM if not present (for running Vault as a service)
if (!(Get-Command nssm -ErrorAction SilentlyContinue)) {
    Write-Output "Installing NSSM for Windows services..."
    # Assuming Chocolatey is installed; if not, download NSSM manually from https://nssm.cc/download
    choco install nssm -y
}

# Step 2: Create directories
New-Item -ItemType Directory -Force -Path $VAULT_CONFIG_DIR, $VAULT_DATA_DIR

# Step 3: Generate self-signed TLS cert (for basic HTTPS - replace with real cert in prod)
$cert = New-SelfSignedCertificate -DnsName "localhost" -CertStoreLocation "cert:\LocalMachine\My"
$thumbprint = $cert.Thumbprint
Export-Certificate -Cert $cert -FilePath "$VAULT_CONFIG_DIR\vault.crt" -Type CERT
Export-PfxCertificate -Cert $cert -FilePath "$VAULT_CONFIG_DIR\vault.pfx" -Password (ConvertTo-SecureString "password" -AsPlainText -Force)  # Change password!

# Step 4: Create Vault config file (file storage, TLS enabled)
Set-Content -Path $VAULT_CONFIG_FILE -Value @"
storage "file" {
  path = "$VAULT_DATA_DIR"
}

listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_cert_file = "$VAULT_CONFIG_DIR\vault.crt"
  tls_key_file  = "$VAULT_CONFIG_DIR\vault.pfx"  # Use .key for private key if not PFX
  tls_client_ca_file = "$VAULT_CONFIG_DIR\vault.crt"  # Optional
  tls_require_and_verify_client_cert = "false"  # Set to true in full prod
}

api_addr = "$VAULT_ADDR"
disable_mlock = true
"@

# Step 5: Install Vault as a Windows Service using NSSM
nssm install $VAULT_SERVICE_NAME $VAULT_EXE
nssm set $VAULT_SERVICE_NAME AppParameters "server -config=$VAULT_CONFIG_FILE"
nssm set $VAULT_SERVICE_NAME AppDirectory (Split-Path $VAULT_EXE)
nssm set $VAULT_SERVICE_NAME Start SERVICE_AUTO_START
nssm start $VAULT_SERVICE_NAME

# Wait for startup
Start-Sleep -Seconds 10

# Step 6: Initialize Vault (run once - saves keys to file securely)
$initOutput = & $VAULT_EXE operator init -key-shares=3 -key-threshold=2 -format=json | ConvertFrom-Json
$unsealKeys = $initOutput.unseal_keys_b64
$rootToken = $initOutput.root_token

# Save to encrypted file (use Windows DPAPI or manual encryption)
$secureData = @{
    UnsealKeys = $unsealKeys
    RootToken = $rootToken
} | ConvertTo-Json
$secureData | Protect-CmsMessage -To "cn=localhost" -OutFile "$VAULT_CONFIG_DIR\vault-init.enc"  # Encrypt with cert

Write-Output "Vault initialized. Encrypted init data saved to $VAULT_CONFIG_DIR\vault-init.enc. Decrypt with Unprotect-CmsMessage."

# Step 7: Unseal Vault (need 2 of 3 keys)
for ($i = 0; $i -lt 2; $i++) {  # Prompt for 2 keys
    $key = Read-Host -Prompt "Enter Unseal Key $($i+1)" -AsSecureString
    $plainKey = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto([System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($key))
    & $VAULT_EXE operator unseal $plainKey
}

# Step 8: Login and Enable KV
$env:VAULT_TOKEN = $rootToken
& $VAULT_EXE secrets enable -path=secret kv-v2

# Step 9: Prompt and Store Secrets
$secrets = @('MARKETDATAAPP_API_KEY', 'ALPHA_VANTAGE_API_KEY', 'DISCORD_BOT_TOKEN', 'IBKR_CLIENT_ID', 'IBKR_ACCOUNT', 'GROK_API_KEY')  # Add yours
$vaultSecrets = @{}
foreach ($key in $secrets) {
    $value = Read-Host -Prompt "Enter new value for $key (blank to skip)" -AsSecureString
    if ($value.Length -gt 0) {
        $plainValue = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto([System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($value))
        $vaultSecrets[$key] = $plainValue
    }
}

if ($vaultSecrets.Count -gt 0) {
    $jsonSecrets = $vaultSecrets | ConvertTo-Json
    & $VAULT_EXE kv put secret/app_secrets @$jsonSecrets
}

# Step 10: Rotate Root Token (important!)
$newRootToken = Read-Host -Prompt "Enter new root token password" -AsSecureString
$plainNewToken = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto([System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($newRootToken))
& $VAULT_EXE auth tune -default-lease-ttl=1h -max-lease-ttl=1h token
# Note: Actual root rotation requires policy setup - see Vault docs.

Write-Output "Setup complete! Vault running as service '$VAULT_SERVICE_NAME'. Address: $VAULT_ADDR"
Write-Output "To stop: nssm stop $VAULT_SERVICE_NAME"
Write-Output "Secure your unseal keys and new root token offline. Update app to use Vault client with TLS."
