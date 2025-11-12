#!/usr/bin/env python3
# integrations/test_massive_api.py
# Purpose: Test Massive API endpoints to verify correct URLs and functionality
# Run this script to validate API integration before using in production

import sys
import os
import requests
import json
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from utils.config import get_massive_api_key
except ImportError:
    # Fallback for direct execution
    import dotenv
    dotenv.load_dotenv()
    def get_massive_api_key():
        return os.getenv('MASSIVE_API_KEY', '')

def test_massive_api_endpoints():
    """Test Massive API endpoints with the configured API key"""

    print("🔍 Testing Massive API Endpoints")
    print("=" * 50)

    # Get API key
    api_key = get_massive_api_key()
    if not api_key:
        print("❌ No Massive API key found. Please set MASSIVE_API_KEY in .env file.")
        return False

    print(f"✅ API Key loaded: {api_key[:8]}...{api_key[-4:]}")

    # Test symbols
    test_symbols = ['AAPL', 'SPY', 'TSLA']
    base_url = "https://api.massive.app/v3"

    # Test each data type
    data_types = ['quotes', 'trades', 'options', 'dark_pool']

    results = {}

    for data_type in data_types:
        print(f"\n📊 Testing {data_type.upper()} endpoints...")
        results[data_type] = {}

        for symbol in test_symbols:
            print(f"  Testing {symbol}...")

            # Import the tool function directly
            try:
                from utils.tools import massive_api_tool
                # Call the tool's underlying function
                result = massive_api_tool.func(symbol, data_type)
            except AttributeError:
                # Fallback: call the tool directly if it's not a StructuredTool
                result = massive_api_tool(symbol, data_type)

            if 'error' in result:
                print(f"    ❌ {symbol}: {result['error']}")
                results[data_type][symbol] = {'status': 'error', 'error': result['error']}
            else:
                print(f"    ✅ {symbol}: Success - {result.get('source', 'unknown')}")
                if 'endpoint_used' in result:
                    print(f"       Endpoint: {result['endpoint_used']}")
                results[data_type][symbol] = {'status': 'success', 'data': result}

    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)

    total_tests = len(data_types) * len(test_symbols)
    successful_tests = 0

    for data_type, symbols in results.items():
        print(f"\n{data_type.upper()}:")
        for symbol, result in symbols.items():
            status = result['status']
            if status == 'success':
                successful_tests += 1
                print(f"  ✅ {symbol}: Working")
            else:
                print(f"  ❌ {symbol}: Failed - {result.get('error', 'Unknown error')}")

    success_rate = (successful_tests / total_tests) * 100
    print(f"Success Rate: {success_rate:.1f}%")
    if success_rate > 50:
        print("🎉 Overall: GOOD - API appears functional")
    elif success_rate > 25:
        print("⚠️  Overall: FAIR - Some endpoints working, may need adjustments")
    else:
        print("❌ Overall: POOR - API integration needs fixing")

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"massive_api_test_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'api_key_configured': bool(api_key),
            'test_results': results,
            'summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': success_rate
            }
        }, f, indent=2, default=str)

    print(f"\n📄 Detailed results saved to: {results_file}")

    return success_rate > 25  # Return True if at least 25% success rate

def test_manual_endpoints():
    """Manually test potential endpoints if the tool fails"""

    print("\n🔧 Manual Endpoint Testing")
    print("=" * 30)

    api_key = get_massive_api_key()
    if not api_key:
        return

    # Common financial API patterns to try
    potential_endpoints = [
        "https://files.massive.com/quotes/AAPL",
        "https://files.massive.com/api/quotes/AAPL",
        "https://files.massive.com/v1/quotes/AAPL",
        "https://files.massive.com/market/quotes/AAPL",
        "https://files.massive.com/trades/AAPL",
        "https://files.massive.com/api/trades/AAPL",
        "https://api.massive.com/v1/quotes/AAPL",  # Alternative domain
        "https://api.massive.com/v1/trades/AAPL",
    ]

    for url in potential_endpoints:
        # Try different API key parameter names
        for key_param in ["apiKey", "api_key", "key"]:
            try:
                params = {key_param: api_key}
                response = requests.get(url, params=params, timeout=10)
                print(f"Testing: {url} (param: {key_param})")
                print(f"  Status: {response.status_code}")

                if response.status_code == 200:
                    data = response.json()
                    print(f"  ✅ Success! Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not dict'}")
                    break  # Success, no need to try other params
                else:
                    print(f"  ❌ Failed: {response.text[:100]}")

            except Exception as e:
                print(f"Testing: {url} (param: {key_param})")
                print(f"  ❌ Error: {str(e)}")

        print()

if __name__ == "__main__":
    print("🚀 Massive API Endpoint Verification Tool")
    print("This script tests your Massive API integration")
    print()

    # Run main tests
    success = test_massive_api_endpoints()

    if not success:
        print("\n🔧 Running manual endpoint tests...")
        test_manual_endpoints()

    print("\n💡 Tips:")
    print("  - Check Massive.com API documentation for correct endpoints")
    print("  - Verify your API key is valid and has proper permissions")
    print("  - Consider rate limits if getting 429 errors")
    print("  - Some endpoints may require premium subscription")