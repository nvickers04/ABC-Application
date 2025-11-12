#!/usr/bin/env python3
"""
Minimal IBKR connectivity test to diagnose TWS API issues
"""
import socket
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_socket_connection(host='127.0.0.1', port=7497, timeout=5):
    """Test basic socket connectivity to TWS"""
    try:
        logger.info(f"Testing socket connection to {host}:{port}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()

        if result == 0:
            logger.info("✅ Socket connection successful")
            return True
        else:
            logger.error(f"❌ Socket connection failed with error code: {result}")
            return False
    except Exception as e:
        logger.error(f"❌ Socket connection error: {e}")
        return False

def test_ibkr_minimal():
    """Minimal IBKR test using raw socket"""
    try:
        logger.info("Testing minimal IBKR API handshake...")

        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)

        # Connect
        sock.connect(('127.0.0.1', 7497))
        logger.info("✅ Connected to TWS socket")

        # Send minimal API handshake (client version 100)
        # IBKR API protocol: client sends version, server responds
        version_msg = b'\x00\x00\x00\x01\x00\x00\x00d'  # Minimal handshake
        sock.send(version_msg)

        # Try to receive response
        response = sock.recv(1024)
        logger.info(f"Received response: {response}")

        sock.close()
        return True

    except Exception as e:
        logger.error(f"❌ Minimal IBKR test failed: {e}")
        return False

def test_ibkr_simple():
    """Simple IBKR connection test"""
    try:
        from ib_insync import IB

        logger.info("Testing simple IBKR connection...")

        # Create IB instance with minimal settings
        ib = IB()

        # Try connection with very short timeout first
        logger.info("Attempting connection...")
        ib.connect('127.0.0.1', 7497, clientId=1)

        if ib.isConnected():
            logger.info("✅ IBKR connection successful!")
            logger.info(f"Managed accounts: {ib.managedAccounts()}")
            ib.disconnect()
            return True
        else:
            logger.error("❌ IBKR connection failed - not connected after connect()")
            return False

    except Exception as e:
        logger.error(f"❌ IBKR simple test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("🔍 IBKR Connectivity Diagnostic Tool")
    print("=" * 50)

    # Test 1: Basic socket connectivity
    print("\n1. Testing socket connectivity...")
    socket_ok = test_socket_connection()

    if not socket_ok:
        print("\n❌ TWS is not running or not accepting connections on port 7497")
        print("Please ensure:")
        print("- TWS is running")
        print("- API connections are enabled in TWS")
        print("- Port 7497 is not blocked by firewall")
        exit(1)

    # Test 2: Minimal API handshake
    print("\n2. Testing minimal API handshake...")
    minimal_ok = test_ibkr_minimal()

    # Test 3: Simple IBKR connection
    print("\n3. Testing IBKR library connection...")
    simple_ok = test_ibkr_simple()

    print("\n" + "=" * 50)
    if simple_ok:
        print("✅ All tests passed! IBKR connection should work.")
    else:
        print("❌ Connection issues detected.")
        print("Common solutions:")
        print("- Restart TWS")
        print("- Check TWS API settings (File > Global Configuration > API)")
        print("- Ensure paper trading account is logged in")
        print("- Try different client ID")
        print("- Check firewall/antivirus settings")