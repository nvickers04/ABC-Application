
# src/utils/audit_logger.py
import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
import os

class AuditLogger:
    def __init__(self, audit_file='logs/audit.log'):
        self.audit_file = Path(audit_file)
        self.setup_audit_file()

    def setup_audit_file(self):
        """Ensure audit file exists with secure permissions"""
        self.audit_file.parent.mkdir(exist_ok=True)
        if not self.audit_file.exists():
            self.audit_file.touch()
        os.chmod(self.audit_file, 0o600)  # Owner read/write only

    def log_event(self, event_type, data, user='system'):
        """Log auditable event with integrity hash"""
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'data': data,
            'user': user,
            'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown'
        }

        # Create integrity hash
        entry_str = json.dumps(entry, sort_keys=True)
        integrity_hash = hashlib.sha256(entry_str.encode()).hexdigest()
        entry['integrity_hash'] = integrity_hash

        # Append to audit log
        with open(self.audit_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        logging.info(f"Audit logged: {event_type}")

    def log_trade(self, trade_data):
        """Log trade execution"""
        self.log_event('trade_execution', trade_data)

    def log_system_event(self, event_data):
        """Log system events"""
        self.log_event('system_event', event_data)

# Global audit logger instance
audit_logger = AuditLogger()
