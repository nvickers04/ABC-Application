#!/usr/bin/env python3
"""
Test script for trading permissions integration.
Tests the IBKR connector's trading permissions functionality and strategy agent filtering.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_trading_permissions():
    """Test trading permissions functionality."""
    print("Testing Trading Permissions Integration")
    print("=" * 50)

    try:
        # Test IBKR connector trading permissions
        from integrations.ibkr_connector import get_ibkr_connector
        ibkr_connector = get_ibkr_connector()

        print("\n1. Testing account permissions retrieval...")
        permissions = await ibkr_connector.get_account_permissions()
        if 'error' in permissions:
            print(f"❌ Error getting account permissions: {permissions['error']}")
        else:
            print("✅ Account permissions retrieved successfully")
            account_type = permissions.get('account_type', 'unknown')
            print(f"   Account Type: {account_type}")

        print("\n2. Testing trading permissions config...")
        permissions_config = await ibkr_connector.get_trading_permissions_config()
        if 'error' in permissions_config:
            print(f"❌ Error getting permissions config: {permissions_config['error']}")
        else:
            print("✅ Trading permissions config loaded successfully")
            combined_permissions = permissions_config.get('combined_permissions', {})
            print(f"   Available instrument types: {list(combined_permissions.keys())}")

        print("\n3. Testing instrument tradability checks...")
        test_instruments = [
            ('SPY', 'equity'),
            ('AAPL', 'equity'),
            ('SPY240119C00500000', 'option'),  # SPY option
            ('ES=F', 'future'),  # E-mini S&P 500
            ('EUR/USD', 'forex'),
            ('BTC/USD', 'crypto')
        ]

        for symbol, instrument_type in test_instruments:
            try:
                check_result = await ibkr_connector.can_trade_instrument(symbol, instrument_type)
                can_trade = check_result.get('can_trade', False)
                status = "✅ Can trade" if can_trade else "❌ Cannot trade"
                reason = f" - {check_result.get('reason', '')}" if not can_trade else ""
                print(f"   {symbol} ({instrument_type}): {status}{reason}")
            except Exception as e:
                print(f"   {symbol} ({instrument_type}): ❌ Error - {str(e)}")

        print("\n4. Testing strategy agent signal filtering...")
        from src.agents.strategy import StrategyAgent
        from src.utils.a2a_protocol import A2AProtocol

        # Create strategy agent
        a2a_protocol = A2AProtocol()
        strategy_agent = StrategyAgent(a2a_protocol=a2a_protocol)

        # Create test signals
        test_signals = [
            {
                'symbol': 'SPY',
                'direction': 'long',
                'quantity': 100,
                'strategy_type': 'Conservative',
                'confidence': 0.8,
                'roi_estimate': 0.12,
                'analysis': 'Conservative long position in SPY',
                'timestamp': '2025-11-20T10:00:00Z',
                'agent_source': 'test'
            },
            {
                'symbol': 'AAPL',
                'direction': 'long',
                'quantity': 50,
                'strategy_type': 'Growth',
                'confidence': 0.9,
                'roi_estimate': 0.18,
                'analysis': 'Growth position in AAPL',
                'timestamp': '2025-11-20T10:00:00Z',
                'agent_source': 'test'
            },
            {
                'symbol': 'SPY240119C00500000',
                'direction': 'long',
                'quantity': 10,
                'strategy_type': 'Options',
                'confidence': 0.7,
                'roi_estimate': 0.25,
                'analysis': 'Call option strategy',
                'timestamp': '2025-11-20T10:00:00Z',
                'agent_source': 'test'
            }
        ]

        # Test signal filtering
        filtered_signals = await strategy_agent._filter_signals_by_trading_permissions(test_signals)

        print(f"   Original signals: {len(test_signals)}")
        print(f"   Filtered signals: {len(filtered_signals)}")

        for signal in filtered_signals:
            permissions = signal.get('trading_permissions', {})
            can_trade = permissions.get('can_trade', False)
            instrument_type = permissions.get('instrument_type', 'unknown')
            print(f"   ✅ {signal['symbol']} ({instrument_type}): Approved for trading")

        # Show filtered out signals
        filtered_out = [s for s in test_signals if not s.get('trading_permissions', {}).get('can_trade', True)]
        for signal in filtered_out:
            permissions = signal.get('trading_permissions', {})
            reason = permissions.get('reason', 'Unknown reason')
            instrument_type = permissions.get('instrument_type', 'unknown')
            print(f"   ❌ {signal['symbol']} ({instrument_type}): {reason}")

        print("\n5. Testing complete market opportunity analysis...")
        try:
            opportunities = await strategy_agent.analyze_market_opportunities()
            print(f"   Found {len(opportunities)} market opportunities")
            if opportunities:
                top_opportunity = opportunities[0]
                permissions = top_opportunity.get('trading_permissions', {})
                can_trade = permissions.get('can_trade', False)
                print(f"   Top opportunity: {top_opportunity.get('symbol', 'N/A')} - Can trade: {can_trade}")
        except Exception as e:
            print(f"   ❌ Error in market analysis: {str(e)}")

        print("\n" + "=" * 50)
        print("Trading Permissions Integration Test Complete")

    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_trading_permissions())