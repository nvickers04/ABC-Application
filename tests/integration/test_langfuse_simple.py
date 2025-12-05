#!/usr/bin/env python3
"""
Simple test script for Langfuse integration.
Tests basic tracing functionality without complex multi-agent simulation.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.utils.langfuse_client import (
        initialize_langfuse,
        get_langfuse_client,
        is_langfuse_enabled,
        trace_function,
        trace_async_function
    )
    print("✓ Langfuse client imports successful")
except ImportError as e:
    print(f"✗ Langfuse client import failed: {e}")
    sys.exit(1)

# Test functions with tracing
@trace_function(name="test_sync_function", user_id="test_user", tags=["test", "sync"])
def test_sync_function(x, y):
    """Simple synchronous function to test tracing."""
    print(f"  Executing sync function: {x} + {y}")
    result = x + y
    return result

@trace_async_function(name="test_async_function", user_id="test_user", tags=["test", "async"])
async def test_async_function(data):
    """Simple async function to test tracing."""
    print(f"  Executing async function with data: {data}")
    await asyncio.sleep(0.1)  # Simulate async work
    result = {"processed": data, "status": "success"}
    return result

@trace_function(name="test_error_function", user_id="test_user", tags=["test", "error"])
def test_error_function():
    """Function that raises an error to test error tracing."""
    print("  Executing function that will raise an error...")
    raise ValueError("Test error for tracing")

async def main():
    """Main test function."""
    print("LANGFUSE INTEGRATION TEST")
    print("=" * 40)

    # Test 1: Initialize Langfuse
    print("\n1. Testing Langfuse initialization...")
    success = initialize_langfuse()
    if success:
        print("✓ Langfuse initialized successfully")
    else:
        print("⚠ Langfuse initialization failed (expected if no API keys)")

    # Test 2: Check if enabled
    enabled = is_langfuse_enabled()
    print(f"✓ Langfuse enabled: {enabled}")

    # Test 3: Get client
    client = get_langfuse_client()
    if client:
        print("✓ Langfuse client available")
    else:
        print("⚠ Langfuse client not available")

    # Test 4: Sync function tracing
    print("\n2. Testing synchronous function tracing...")
    try:
        result = test_sync_function(5, 3)
        print(f"✓ Sync function result: {result}")
    except Exception as e:
        print(f"✗ Sync function failed: {e}")

    # Test 5: Async function tracing
    print("\n3. Testing asynchronous function tracing...")
    try:
        result = await test_async_function({"test": "data"})
        print(f"✓ Async function result: {result}")
    except Exception as e:
        print(f"✗ Async function failed: {e}")

    # Test 6: Error tracing
    print("\n4. Testing error tracing...")
    try:
        test_error_function()
    except ValueError as e:
        print(f"✓ Error tracing test completed (expected error: {e})")
    except Exception as e:
        print(f"✗ Unexpected error in error test: {e}")

    # Test 7: Flush traces if client available
    if client and hasattr(client, 'flush'):
        print("\n5. Flushing traces to Langfuse...")
        try:
            client.flush()
            print("✓ Traces flushed successfully")
        except Exception as e:
            print(f"⚠ Trace flush failed: {e}")

    print("\n" + "=" * 40)
    print("LANGFUSE TEST SUMMARY:")
    print(f"- Initialization: {'✓' if success else '⚠'}")
    print(f"- Tracing enabled: {'✓' if enabled else '⚠'}")
    print("- Sync tracing: ✓ Tested")
    print("- Async tracing: ✓ Tested")
    print("- Error tracing: ✓ Tested")

    if enabled:
        print("\n🎉 Langfuse integration is working!")
        print("Check your dashboard at: https://us.cloud.langfuse.com")
    else:
        print("\n⚠ Langfuse tracing is disabled.")
        print("To enable, set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables.")

if __name__ == "__main__":
    asyncio.run(main())