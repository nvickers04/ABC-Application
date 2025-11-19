#!/usr/bin/env python3
"""
TWS API Setup Verification Script
"""

import socket
import time

def check_tws_running():
    """Check if TWS is running on port 7497"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('127.0.0.1', 7497))
        sock.close()
        return result == 0
    except:
        return False

def test_api_enabled():
    """Test if API is enabled by attempting handshake"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        sock.connect(('127.0.0.1', 7497))

        # Send minimal API handshake
        version_msg = b'\x00\x00\x00\x01\x00\x00\x00d'
        sock.send(version_msg)

        # Wait for response with short timeout
        sock.settimeout(2)
        response = sock.recv(1024)
        sock.close()

        return len(response) > 0
    except:
        return False

def main():
    print("🔍 TWS API Setup Verification")
    print("=" * 40)

    # Check 1: TWS running
    print("\n1. Checking if TWS is running...")
    tws_running = check_tws_running()
    if tws_running:
        print("✅ TWS is running on port 7497")
    else:
        print("❌ TWS is NOT running on port 7497")
        print("   → Start Paper Trading TWS first")
        return

    # Check 2: API enabled
    print("\n2. Checking if API is enabled...")
    api_enabled = test_api_enabled()
    if api_enabled:
        print("✅ TWS API is responding!")
        print("\n🎉 SUCCESS! TWS API is properly configured.")
        print("You can now run: python test_paper_trading.py")
    else:
        print("❌ TWS API is NOT enabled")
        print("\n📋 REQUIRED ACTION:")
        print("1. In TWS: File → Global Configuration → API")
        print("2. Check 'Enable ActiveX and Socket Clients'")
        print("3. Click OK to save")
        print("4. Restart TWS completely")
        print("5. Run this script again")

if __name__ == "__main__":
    main()