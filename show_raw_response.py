#!/usr/bin/env python3
"""
Show exactly what TWS is responding with
"""

import socket
import time

def show_raw_response():
    """Show the raw response from TWS"""
    print("🔍 Raw TWS Response Analysis")
    print("=" * 40)

    try:
        # Connect
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(('127.0.0.1', 7497))
        print("✅ Connected to TWS socket")

        # Send handshake
        version_msg = b'\x00\x00\x00\x01\x00\x00\x00d'
        sock.send(version_msg)
        print(f"📤 Sent handshake: {version_msg.hex()}")

        # Try to receive
        sock.settimeout(3)
        try:
            response = sock.recv(1024)
            if response:
                print(f"📥 TWS responded with {len(response)} bytes:")
                print(f"   Hex: {response.hex()}")
                print(f"   Raw: {response}")
                print("\n✅ API IS ENABLED! TWS is responding.")
                return True
            else:
                print("📥 No response received")
        except socket.timeout:
            print("⏰ Response timeout - API not enabled")
        except Exception as e:
            print(f"❌ Receive error: {e}")

        sock.close()
        print("\n❌ API is NOT enabled. TWS is not responding to API calls.")
        return False

    except ConnectionRefusedError:
        print("❌ Cannot connect to TWS - is it running?")
        return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def main():
    print("This will show you exactly what TWS sends back.")
    print("If API is disabled, you'll see 'Response timeout'")
    print("If API is enabled, you'll see actual response data.")
    print()

    api_enabled = show_raw_response()

    if api_enabled:
        print("\n🎉 SUCCESS! API is enabled.")
        print("Run: python test_paper_trading.py")
    else:
        print("\n❌ API is still disabled.")
        print("Go to TWS: File → Global Configuration → API")
        print("Check 'Enable ActiveX and Socket Clients'")

if __name__ == "__main__":
    main()