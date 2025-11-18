storage "file" {
  path = "C:\Users\nvick\ABC-Application\vault-data"
}

listener "tcp" {
  address     = "0.0.0.0:8200"  # Listen on all interfaces (secure with firewall)
  tls_disable = "true"  # DISABLED FOR SIMPLICITY - ENABLE TLS IN REAL PROD!
}

api_addr = "http://127.0.0.1:8200"
disable_mlock = true
