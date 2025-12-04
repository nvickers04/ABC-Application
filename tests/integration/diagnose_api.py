#!/usr/bin/env python3
"""
Simple TWS API diagnostic - check if API is responding
"""

import socket
import time

def test_api_handshake():
    """Test the basic API handshake"""
    print("🔍 Testing TWS API Handshake")
    print("=" * 40)

    try:
        # Connect to TWS
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(('127.0.0.1', 7497))

        print("✅ Socket connected to TWS")

        # Send API version handshake
        # This is the minimal handshake IBKR expects
        version_msg = b'\x00\x00\x00\x01\x00\x00\x00d'  # Client version 100
        sock.send(version_msg)

        print("📤 Sent API version handshake")

        # Try to receive response
        sock.settimeout(10)  # Wait up to 10 seconds for response
        response = sock.recv(1024)

        if response:
            print(f"📥 Received API response: {len(response)} bytes")
            print(f"Response (hex): {response.hex()}")
            sock.close()
            return True
        else:
            print("❌ No response from TWS API")
            sock.close()
            return False

    except socket.timeout:
        print("❌ API handshake timed out - TWS API not responding")
        print("This confirms the API is not enabled in TWS settings")
        return False
    except ConnectionRefusedError:
        print("❌ Connection refused - TWS not running on port 7497")
        return False
    except Exception as e:
        print(f"❌ API handshake failed: {e}")
        return False

def main():
    print("🩺 TWS API Diagnostic Tool")
    print("This will test if the TWS API is properly enabled")
    print()

    success = test_api_handshake()

    if success:
        print("\n✅ TWS API is responding!")
        print("The issue might be in the Python library or authentication.")
        print("Try running: python test_paper_trading.py")
    else:
        print("\n❌ TWS API is NOT responding.")
        print("Go back to TWS and check:")
        print("File → Global Configuration → API")
        print("Make sure 'Enable ActiveX and Socket Clients' is checked")

if __name__ == "__main__":
    main()