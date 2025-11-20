# premarket_prep_simulated.py
# Purpose: Simulated IBKR Premarket Preparation Script
# Demonstrates premarket preparation routines without requiring live IBKR connection
# Useful for testing and development when TWS/Gateway is not available

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any
import sys
import os
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimulatedPremarketPrep:
    """
    Simulated premarket preparation that mimics IBKR functionality
    """

    def __init__(self):
        self.premarket_symbols = [
            'SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA',
            'META', 'NFLX', 'AMD', 'INTC', 'BA', 'DIS', 'V', 'JPM'
        ]

        # Simulated market data (realistic prices for Nov 20, 2025)
        self.simulated_prices = {
            'SPY': 450.25, 'QQQ': 380.80, 'AAPL': 195.60, 'MSFT': 425.30,
            'GOOGL': 175.90, 'AMZN': 185.45, 'TSLA': 285.70, 'NVDA': 145.20,
            'META': 485.60, 'NFLX': 725.80, 'AMD': 165.40, 'INTC': 22.15,
            'BA': 185.90, 'DIS': 95.25, 'V': 305.80, 'JPM': 225.60
        }

    async def run_premarket_prep(self) -> Dict[str, Any]:
        """
        Run simulated premarket preparation routine

        Returns:
            Dict with simulated preparation results
        """
        logger.info("🚀 Starting SIMULATED IBKR Premarket Preparation...")
        logger.info("=" * 60)

        results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'connection_status': True,  # Simulated as successful
            'account_summary': {},
            'positions': [],
            'market_data': {},
            'trading_permissions': {},
            'news_bulletins': [],
            'system_health': {},
            'recommendations': [],
            'simulated': True
        }

        try:
            # 1. Simulate IBKR connection
            logger.info("📡 Simulating IBKR Paper Trading connection...")
            await asyncio.sleep(0.5)  # Simulate connection time
            logger.info("✅ Successfully connected to IBKR (SIMULATED)")

            # 2. Get simulated account summary
            logger.info("📊 Retrieving simulated account summary...")
            account_summary = self._simulate_account_summary()
            results['account_summary'] = account_summary

            cash = account_summary.get('cash_balance', 0)
            positions_count = account_summary.get('total_positions', 0)
            logger.info(".2f")
            logger.info(f"📈 Open Positions: {positions_count}")

            # 3. Get simulated positions
            logger.info("📋 Checking simulated current positions...")
            positions = self._simulate_positions()
            results['positions'] = positions
            logger.info(f"📊 Found {len(positions)} open positions")

            # 4. Get simulated market data
            logger.info("📈 Gathering simulated premarket data for key symbols...")
            market_data = {}
            for symbol in self.premarket_symbols:
                try:
                    data = self._simulate_market_data(symbol)
                    market_data[symbol] = data
                    logger.info(".2f")
                    await asyncio.sleep(0.1)  # Simulate API delay
                except Exception as e:
                    logger.warning(f"⚠️  Error getting simulated data for {symbol}: {e}")

            results['market_data'] = market_data

            # 5. Get simulated trading permissions
            logger.info("🔐 Verifying simulated trading permissions...")
            permissions = self._simulate_trading_permissions()
            results['trading_permissions'] = permissions

            account_type = permissions.get('account_type', 'unknown')
            logger.info(f"🏦 Account Type: {account_type}")

            trading_perms = permissions.get('trading_permissions', {})
            enabled_assets = [asset for asset, config in trading_perms.items()
                            if config.get('enabled', False)]
            logger.info(f"✅ Enabled Trading: {', '.join(enabled_assets)}")

            # 6. Get simulated news bulletins
            logger.info("📰 Checking for simulated news bulletins...")
            bulletins = self._simulate_news_bulletins()
            results['news_bulletins'] = bulletins
            if bulletins:
                logger.info(f"📰 Found {len(bulletins)} recent news bulletins")
                for bulletin in bulletins[:3]:
                    logger.info(f"   • {bulletin['exchange']}: {bulletin['message'][:100]}...")
            else:
                logger.info("📰 No recent news bulletins")

            # 7. System health check
            logger.info("🏥 Performing system health check...")
            try:
                from src.utils.advanced_memory import get_memory_health_status
                memory_health = get_memory_health_status()
                results['system_health'] = {
                    'memory': memory_health,
                    'ibkr_connected': True
                }
                logger.info("✅ System health check completed")
            except Exception as e:
                logger.warning(f"⚠️  System health check error: {e}")

            # 8. Generate recommendations
            logger.info("💡 Generating trading recommendations...")
            recommendations = self._generate_recommendations(results)
            results['recommendations'] = recommendations

            logger.info("🎯 Simulated premarket preparation completed successfully!")
            logger.info("=" * 60)

            return results

        except Exception as e:
            logger.error(f"❌ Simulated premarket preparation failed: {e}")
            results['error'] = str(e)
            return results

    def _simulate_account_summary(self) -> Dict[str, Any]:
        """Generate simulated account summary"""
        return {
            'account_id': 'DUN976979',
            'cash_balance': 98543.67,
            'currency': 'USD',
            'total_positions': 3,
            'buying_power': 197087.34,
            'total_value': 152341.89,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    def _simulate_positions(self) -> List[Dict[str, Any]]:
        """Generate simulated positions"""
        return [
            {
                'symbol': 'SPY',
                'position': 50,
                'avg_cost': 445.20,
                'market_value': 22512.50,
                'unrealized_pnl': 262.50,
                'realized_pnl': 0.0
            },
            {
                'symbol': 'AAPL',
                'position': 25,
                'avg_cost': 190.50,
                'market_value': 4890.00,
                'unrealized_pnl': 127.50,
                'realized_pnl': 0.0
            },
            {
                'symbol': 'MSFT',
                'position': 15,
                'avg_cost': 415.80,
                'market_value': 6379.50,
                'unrealized_pnl': 142.50,
                'realized_pnl': 0.0
            }
        ]

    def _simulate_market_data(self, symbol: str) -> Dict[str, Any]:
        """Generate simulated market data"""
        base_price = self.simulated_prices.get(symbol, 100.0)

        # Add some random variation (±2%)
        variation = random.uniform(-0.02, 0.02)
        current_price = base_price * (1 + variation)

        # Generate OHLC data
        high = current_price * random.uniform(1.001, 1.01)
        low = current_price * random.uniform(0.99, 0.999)
        open_price = (high + low) / 2 * random.uniform(0.995, 1.005)
        volume = random.randint(100000, 5000000)

        return {
            'symbol': symbol,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(current_price, 2),
            'volume': volume
        }

    def _simulate_trading_permissions(self) -> Dict[str, Any]:
        """Generate simulated trading permissions"""
        return {
            'account_id': 'DUN976979',
            'account_type': 'paper_trading',
            'trading_permissions': {
                'equities': {'enabled': True, 'exchanges': ['NASDAQ', 'NYSE', 'AMEX', 'ARCA']},
                'options': {'enabled': True, 'types': ['CALL', 'PUT', 'SPREAD', 'STRADDLE', 'STRANGLE']},
                'futures': {'enabled': True, 'exchanges': ['CME', 'CBOT', 'NYMEX', 'COMEX']},
                'forex': {'enabled': True, 'pairs': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'USD/CAD']},
                'crypto': {'enabled': False},
                'margin': {'enabled': True, 'leverage_max': 4.0},
                'short_selling': {'enabled': True}
            },
            'account_features': {
                'buying_power': 197087.34,
                'day_trades_remaining': 3,
                'cushion': 0.05
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    def _simulate_news_bulletins(self) -> List[Dict[str, Any]]:
        """Generate simulated news bulletins"""
        bulletins = [
            {
                'id': 'BULLETIN_001',
                'message': 'NASDAQ: Normal market conditions expected for today\'s trading session.',
                'exchange': 'NASDAQ',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'type': 'bulletin'
            },
            {
                'id': 'BULLETIN_002',
                'message': 'NYSE: Pre-market trading volume showing normal patterns.',
                'exchange': 'NYSE',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'type': 'bulletin'
            }
        ]
        return bulletins

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate trading recommendations based on simulated premarket data

        Args:
            results: Premarket preparation results

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Account health recommendations
        account = results.get('account_summary', {})
        cash_balance = account.get('cash_balance', 0)
        if cash_balance > 50000:
            recommendations.append("💰 HEALTHY CASH: Good liquidity for trading operations")
        elif cash_balance > 25000:
            recommendations.append("💰 ADEQUATE CASH: Sufficient funds for moderate trading")

        # Position recommendations
        positions = results.get('positions', [])
        if len(positions) > 5:
            recommendations.append("⚠️  MULTIPLE POSITIONS: Monitor portfolio diversification")
        elif len(positions) > 0:
            recommendations.append("📊 ACTIVE POSITIONS: Review current holdings for adjustments")

        # Market data recommendations
        market_data = results.get('market_data', {})
        if market_data:
            recommendations.append(f"📈 MARKET DATA: Retrieved data for {len(market_data)} symbols")

        # Permissions recommendations
        permissions = results.get('trading_permissions', {})
        account_type = permissions.get('account_type', '')
        if account_type == 'paper_trading':
            recommendations.append("🎓 PAPER TRADING: Practice environment - all features available")

        # News recommendations
        bulletins = results.get('news_bulletins', [])
        if bulletins:
            recommendations.append(f"📰 NEWS ALERT: {len(bulletins)} market bulletins - review for trading insights")

        # System health recommendations
        system_health = results.get('system_health', {})
        memory_health = system_health.get('memory', {})
        if memory_health.get('overall_health') == 'healthy':
            recommendations.append("✅ MEMORY HEALTHY: All memory backends operational")

        if not recommendations:
            recommendations.append("✅ ALL SYSTEMS GO: Ready for simulated trading operations")

        recommendations.append("🔧 SIMULATION MODE: This is simulated data for development/testing")

        return recommendations

    def print_summary_report(self, results: Dict[str, Any]):
        """
        Print a formatted summary report of premarket preparation

        Args:
            results: Premarket preparation results
        """
        print("\n" + "="*80)
        print("📊 SIMULATED IBKR PREMARKET PREPARATION SUMMARY")
        print("="*80)
        print(f"⏰ Time: {results['timestamp']}")
        print("🎭 Mode: SIMULATION (No live IBKR connection required)")
        print()

        # Connection status
        if results['connection_status']:
            print("✅ IBKR Connection: SUCCESSFUL (SIMULATED)")
        else:
            print("❌ IBKR Connection: FAILED")
            if 'error' in results:
                print(f"   Error: {results['error']}")
        print()

        # Account summary
        account = results.get('account_summary', {})
        if account:
            print("💰 ACCOUNT SUMMARY:")
            print(".2f")
            print(f"   Positions: {account.get('total_positions', 0)}")
            print(".2f")
            print(".2f")
            print(f"   Currency: {account.get('currency', 'USD')}")
        else:
            print("❌ Account Summary: No data available")
        print()

        # Positions
        positions = results.get('positions', [])
        if positions:
            print("📊 CURRENT POSITIONS:")
            total_value = 0
            total_pnl = 0
            for pos in positions:
                value = pos.get('market_value', 0)
                pnl = pos.get('unrealized_pnl', 0)
                total_value += value
                total_pnl += pnl
                print(".2f"
                      ".2f")
            print(".2f")
            print(".2f")
        else:
            print("📊 No open positions")
        print()

        # Market data
        market_data = results.get('market_data', {})
        if market_data:
            print("📈 KEY MARKET DATA:")
            for symbol, data in list(market_data.items())[:10]:  # Show first 10
                print(".2f")
        else:
            print("⚠️  No market data available")
        print()

        # Trading permissions
        permissions = results.get('trading_permissions', {})
        if permissions:
            account_type = permissions.get('account_type', 'unknown')
            print(f"🏦 ACCOUNT TYPE: {account_type.upper()}")
            trading_perms = permissions.get('trading_permissions', {})
            enabled_trading = [asset for asset, config in trading_perms.items()
                             if config.get('enabled', False)]
            print(f"✅ ENABLED TRADING: {', '.join(enabled_trading)}")
        print()

        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            print("💡 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"   • {rec}")
        else:
            print("💡 No specific recommendations")
        print()

        print("="*80)
        print("🔧 DEVELOPMENT NOTES:")
        print("   • To use live IBKR connection, run: python premarket_prep.py")
        print("   • Make sure IBKR TWS/Gateway is running on port 7497")
        print("   • Enable API connections in TWS settings")
        print("="*80)

async def main():
    """Main simulated premarket preparation function"""
    prep = SimulatedPremarketPrep()
    results = await prep.run_premarket_prep()
    prep.print_summary_report(results)

    # Return success/failure
    return results.get('error') is None and results['connection_status']

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)