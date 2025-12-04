#!/usr/bin/env python3
"""
Mock IBKR TWS Server for Testing
Simulates IBKR TWS API responses for integration testing.
"""

import socket
import threading
import time
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockIBKRServer:
    """Mock IBKR TWS server for testing."""

    def __init__(self, host='0.0.0.0', port=7497):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False

    def start(self):
        """Start the mock server."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.running = True

        logger.info(f"Mock IBKR server started on {self.host}:{self.port}")

        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                logger.info(f"Accepted connection from {address}")
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
            except OSError:
                break

    def stop(self):
        """Stop the mock server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        logger.info("Mock IBKR server stopped")

    def handle_client(self, client_socket, address):
        """Handle client connection."""
        try:
            # Send welcome message
            welcome_msg = "Mock IBKR TWS v1.0 Ready\n"
            client_socket.send(welcome_msg.encode())

            buffer = ""
            while self.running:
                try:
                    data = client_socket.recv(1024)
                    if not data:
                        break

                    buffer += data.decode('utf-8', errors='ignore')

                    # Process complete messages
                    while '\n' in buffer:
                        message, buffer = buffer.split('\n', 1)
                        response = self.process_message(message.strip())
                        if response:
                            client_socket.send((response + '\n').encode())

                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Error handling client {address}: {e}")
                    break

        finally:
            client_socket.close()
            logger.info(f"Closed connection from {address}")

    def process_message(self, message):
        """Process incoming message and return response."""
        logger.info(f"Received message: {message}")

        # Simple mock responses based on message content
        if 'CONNECT' in message.upper():
            return "CONNECTED: Mock IBKR TWS"

        elif 'ACCOUNT' in message.upper():
            return json.dumps({
                "account": "TEST123",
                "balance": 100000.00,
                "currency": "USD",
                "status": "ACTIVE"
            })

        elif 'MARKET_DATA' in message.upper() or 'QUOTE' in message.upper():
            return json.dumps({
                "symbol": "AAPL",
                "price": 150.25,
                "volume": 1000000,
                "timestamp": datetime.now().isoformat()
            })

        elif 'ORDER' in message.upper():
            return json.dumps({
                "order_id": "TEST_ORDER_123",
                "status": "SUBMITTED",
                "symbol": "AAPL",
                "quantity": 100,
                "price": 150.25
            })

        elif 'PING' in message.upper():
            return "PONG"

        else:
            return f"ACK: {message}"

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Mock IBKR TWS Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=7497, help='Port to bind to')

    args = parser.parse_args()

    server = MockIBKRServer(args.host, args.port)

    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        server.stop()

if __name__ == '__main__':
    main()</content>
</xai:function_call name="create_file">
<parameter name="filePath">c:\Users\nvick\ABC-Application\setup\Dockerfile.health_monitor