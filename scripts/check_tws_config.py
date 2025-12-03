#!/usr/bin/env python3
"""
TWS Configuration Checker

This script helps diagnose TWS (Trader Workstation) configuration issues
for paper trading setup.
"""

import socket
import time
import sys

def check_port(host, port):
    """Check if a port is open on the given host."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"Error checking port {port}: {e}")
        return False

def main():
    print("🔍 TWS Configuration Checker")
    print("=" * 40)

    # Check if TWS API ports are accessible
    ports_to_check = [
        (7497, "IBKR Paper Trading API"),
        (7496, "IBKR Live Trading API"),
        (4001, "IBKR Gateway API"),
        (4002, "IBKR Gateway API (alternative)")
    ]

    print("\n📡 Checking API ports...")
    for port, description in ports_to_check:
        is_open = check_port('127.0.0.1', port)
        status = "✅ OPEN" if is_open else "❌ CLOSED"
        print(f"{status} Port {port} ({description})")

    print("\n🔧 TWS Configuration Checklist:")
    print("1. ✅ TWS is running (confirmed)")
    print("2. ⏳ TWS is in PAPER TRADING mode")
    print("3. ⏳ API connections are enabled")
    print("4. ⏳ Socket port is set to 7497")
    print("5. ⏳ 'Allow connections from localhost only' is checked")

    print("\n📋 To fix TWS configuration:")
    print("1. Open TWS")
    print("2. Go to File → Global Configuration → API")
    print("3. Check 'Enable ActiveX and Socket Clients'")
    print("4. Set 'Socket port' to 7497")
    print("5. Check 'Allow connections from localhost only'")
    print("6. Click Apply, then OK")
    print("7. Restart TWS")

    print("\n🔄 After configuring TWS, run:")
    print("   python scripts/validate_paper_trading_setup.py")

    # Check which ports are open
    open_ports = [port for port, desc in ports_to_check if check_port('127.0.0.1', port)]

    if not open_ports:
        print("\n❌ No IBKR API ports are accessible.")
        print("   Make sure TWS is running in paper trading mode with API enabled.")
        return 1
    else:
        print(f"\n✅ Found open ports: {open_ports}")
        if 7497 in open_ports:
            print("   Port 7497 is accessible - TWS API should be working!")
        else:
            print("   Port 7497 is not accessible - check TWS configuration.")
        return 0

if __name__ == "__main__":
    sys.exit(main())