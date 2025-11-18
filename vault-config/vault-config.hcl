storage "file" {
  path = "C:\Users\nvick\ABC-Application\vault-data"
}

listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_cert_file = "C:\Users\nvick\ABC-Application\vault-config\vault.crt"
  tls_key_file  = "C:\Users\nvick\ABC-Application\vault-config\vault.pfx"  # Use .key for private key if not PFX
  tls_client_ca_file = "C:\Users\nvick\ABC-Application\vault-config\vault.crt"  # Optional
  tls_require_and_verify_client_cert = "false"  # Set to true in full prod
}

api_addr = "https://127.0.0.1:8200"
disable_mlock = true
