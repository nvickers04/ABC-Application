#!/usr/bin/env python3
"""
Test script for Acontext integration initialization and basic functionality.
Tests the integration setup and fallback mode as specified in Phase 1.
"""

import os
import sys
import asyncio
import yaml
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.integrations.acontext_integration import get_acontext_integration, ACONTEXT_AVAILABLE
    print("✓ Acontext integration import successful")
except ImportError as e:
    print(f"✗ Acontext integration import failed: {e}")
    sys.exit(1)

def load_config():
    """Load Acontext configuration from config file."""
    config_path = Path("config/acontext_config.yaml")
    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        return None

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Config file loaded successfully")
        return config
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return None

async def test_integration_init():
    """Test Acontext integration initialization."""
    config = load_config()
    if not config:
        return False

    print(f"✓ Acontext SDK available: {ACONTEXT_AVAILABLE}")

    # Get API key from environment
    api_key = os.getenv('ACONTEXT_API_KEY')
    if not api_key:
        print("⚠ No ACONTEXT_API_KEY environment variable set - will use fallback mode")
    else:
        print("✓ ACONTEXT_API_KEY environment variable found")

    try:
        # Get integration instance
        integration = get_acontext_integration()
        print("✓ Integration instance created successfully")

        # Initialize integration
        init_result = await integration.initialize()
        if init_result:
            print("✓ Acontext connection established successfully")
            return True
        else:
            print("⚠ Acontext using fallback mode (expected without API key)")
            # Test fallback functionality
            return await test_fallback_mode(integration)

    except Exception as e:
        print(f"✗ Integration initialization failed: {e}")
        return False

async def test_fallback_mode(integration):
    """Test fallback mode functionality."""
    try:
        # Test storing a directive in fallback mode
        from src.integrations.acontext_integration import TradingDirective

        test_directive = TradingDirective(
            id="test-directive-001",
            category="test",
            name="Test Directive",
            description="A test directive for fallback mode",
            content={"test_key": "test_value"},
            applies_to=["strategy", "risk"],
            source="learning",
            priority="medium"
        )

        # Try to store SOP (should work in fallback mode)
        stored_id = await integration.store_sop(test_directive)
        if stored_id:
            print("✓ Fallback mode SOP storage works")
        else:
            print("⚠ Fallback mode SOP storage failed")
            return False

        # Try to query SOPs
        retrieved = await integration.query_sops(category="test")
        if retrieved and len(retrieved) > 0:
            print("✓ Fallback mode SOP query works")
            return True
        else:
            print("⚠ Fallback mode SOP query failed")
            return False

    except Exception as e:
        print(f"✗ Fallback mode test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Acontext integration setup...")
    success = asyncio.run(test_integration_init())
    if success:
        print("\n🎉 Acontext integration is working! Ready for Phase 2.")
    else:
        print("\n⚠ Integration tests failed. Check configuration and dependencies.")
    sys.exit(0 if success else 1)