# premarket_prep.py
# Purpose: IBKR Premarket Preparation Script
# Connects to IBKR paper trading account and performs morning preparation routines
# Checks account status, positions, market data, and trading readiness

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.ibkr_connector import get_ibkr_connector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PremarketPrep:
    """
    Handles premarket preparation routines for IBKR trading
    """

    def __init__(self):
        self.connector = get_ibkr_connector()
        self.premarket_symbols = [
            'SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA',
            'META', 'NFLX', 'AMD', 'INTC', 'BA', 'DIS', 'V', 'JPM'
        ]

    async def run_premarket_prep(self) -> Dict[str, Any]:
        """
        Run complete premarket preparation routine

        Returns:
            Dict with preparation results
        """
        logger.info("🚀 Starting IBKR Premarket Preparation...")
        logger.info("=" * 60)

        results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'connection_status': False,
            'account_summary': {},
            'positions': [],
            'market_data': {},
            'trading_permissions': {},
            'news_bulletins': [],
            'system_health': {},
            'recommendations': []
        }

        try:
            # 1. Connect to IBKR
            logger.info("📡 Connecting to IBKR Paper Trading...")
            connected = await self.connector.connect()
            results['connection_status'] = connected

            if not connected:
                results['error'] = "Failed to connect to IBKR"
                return results

            logger.info("✅ Successfully connected to IBKR")

            # 2. Get account summary
            logger.info("📊 Retrieving account summary...")
            account_summary = await self.connector.get_account_summary()
            results['account_summary'] = account_summary

            if 'error' in account_summary:
                logger.warning(f"⚠️  Account summary error: {account_summary['error']}")
            else:
                cash = account_summary.get('cash_balance', 0)
                positions_count = account_summary.get('total_positions', 0)
                logger.info(f"💰 Cash Balance: ${cash:,.2f}")
                logger.info(f"📈 Open Positions: {positions_count}")

            # 3. Get current positions
            logger.info("📋 Checking current positions...")
            positions = await self.connector.get_positions()
            results['positions'] = positions
            logger.info(f"📊 Found {len(positions)} open positions")

            # 4. Get market data for key symbols
            logger.info("📈 Gathering premarket data for key symbols...")
            market_data = {}
            for symbol in self.premarket_symbols:
                try:
                    data = await self.connector.get_market_data(symbol)
                    if data:
                        market_data[symbol] = data
                        logger.info(f"📊 {symbol}: ${data['close']:.2f}")
                    else:
                        logger.warning(f"⚠️  No data available for {symbol}")
                except Exception as e:
                    logger.warning(f"⚠️  Error getting data for {symbol}: {e}")

            results['market_data'] = market_data

            # 5. Check trading permissions
            logger.info("🔐 Verifying trading permissions...")
            permissions = await self.connector.get_account_permissions()
            results['trading_permissions'] = permissions

            if 'error' in permissions:
                logger.warning(f"⚠️  Permissions check error: {permissions['error']}")
            else:
                account_type = permissions.get('account_type', 'unknown')
                logger.info(f"🏦 Account Type: {account_type}")

                trading_perms = permissions.get('trading_permissions', {})
                enabled_assets = [asset for asset, config in trading_perms.items()
                                if config.get('enabled', False)]
                logger.info(f"✅ Enabled Trading: {', '.join(enabled_assets)}")

            # 6. Get news bulletins
            logger.info("📰 Checking for news bulletins...")
            try:
                bulletins = await self.connector.get_news_bulletins(all_messages=False)
                results['news_bulletins'] = bulletins
                if bulletins:
                    logger.info(f"📰 Found {len(bulletins)} recent news bulletins")
                    for bulletin in bulletins[:3]:  # Show first 3
                        logger.info(f"   • {bulletin['exchange']}: {bulletin['message'][:100]}...")
                else:
                    logger.info("📰 No recent news bulletins")
            except Exception as e:
                logger.warning(f"⚠️  Error getting news bulletins: {e}")

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

            logger.info("🎯 Premarket preparation completed successfully!")
            logger.info("=" * 60)

            return results

        except Exception as e:
            logger.error(f"❌ Premarket preparation failed: {e}")
            results['error'] = str(e)
            return results

        finally:
            # Always disconnect
            try:
                await self.connector.disconnect()
                logger.info("🔌 Disconnected from IBKR")
            except Exception as e:
                logger.warning(f"⚠️  Error disconnecting: {e}")

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate trading recommendations based on premarket data

        Args:
            results: Premarket preparation results

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Account health recommendations
        account = results.get('account_summary', {})
        if 'error' not in account:
            cash_balance = account.get('cash_balance', 0)
            if cash_balance < 1000:
                recommendations.append("⚠️  LOW CASH: Consider funding account for better trading flexibility")
            elif cash_balance > 50000:
                recommendations.append("💰 HEALTHY CASH: Good liquidity for trading operations")

        # Position recommendations
        positions = results.get('positions', [])
        if len(positions) > 10:
            recommendations.append("⚠️  HIGH POSITIONS: Consider reducing position count for better risk management")
        elif len(positions) == 0:
            recommendations.append("📊 NO POSITIONS: Ready for new trading opportunities")

        # Market data recommendations
        market_data = results.get('market_data', {})
        if len(market_data) < len(self.premarket_symbols) * 0.5:
            recommendations.append("⚠️  LIMITED MARKET DATA: Check IBKR market data subscriptions")

        # Permissions recommendations
        permissions = results.get('trading_permissions', {})
        if 'error' not in permissions:
            account_type = permissions.get('account_type', '')
            if account_type == 'paper_trading':
                recommendations.append("🎓 PAPER TRADING: All features available for practice trading")
            elif account_type == 'individual_cash':
                recommendations.append("💼 CASH ACCOUNT: Focus on long equity positions")

        # News recommendations
        bulletins = results.get('news_bulletins', [])
        if bulletins:
            recommendations.append(f"📰 NEWS ALERT: {len(bulletins)} recent bulletins - review for trading opportunities")

        # System health recommendations
        system_health = results.get('system_health', {})
        memory_health = system_health.get('memory', {})
        if memory_health.get('overall_health') == 'critical':
            recommendations.append("🚨 MEMORY CRITICAL: Address memory backend issues before trading")

        if not recommendations:
            recommendations.append("✅ ALL SYSTEMS GO: Ready for trading operations")

        return recommendations

    def print_summary_report(self, results: Dict[str, Any]):
        """
        Print a formatted summary report of premarket preparation

        Args:
            results: Premarket preparation results
        """
        print("\n" + "="*80)
        print("📊 IBKR PREMARKET PREPARATION SUMMARY")
        print("="*80)
        print(f"⏰ Time: {results['timestamp']}")
        print()

        # Connection status
        if results['connection_status']:
            print("✅ IBKR Connection: SUCCESSFUL")
        else:
            print("❌ IBKR Connection: FAILED")
            if 'error' in results:
                print(f"   Error: {results['error']}")
        print()

        # Account summary
        account = results.get('account_summary', {})
        if 'error' not in account:
            print("💰 ACCOUNT SUMMARY:")
            print(".2f")
            print(f"   Positions: {account.get('total_positions', 0)}")
            print(f"   Currency: {account.get('currency', 'USD')}")
        else:
            print(f"❌ Account Summary: {account['error']}")
        print()

        # Positions
        positions = results.get('positions', [])
        if positions:
            print("📊 CURRENT POSITIONS:")
            for pos in positions[:5]:  # Show first 5
                print(".2f"
                      ".2f")
            if len(positions) > 5:
                print(f"   ... and {len(positions) - 5} more positions")
        else:
            print("📊 No open positions")
        print()

        # Market data
        market_data = results.get('market_data', {})
        if market_data:
            print("📈 KEY MARKET DATA:")
            for symbol, data in list(market_data.items())[:8]:  # Show first 8
                print(".2f")
        else:
            print("⚠️  No market data available")
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

async def main():
    """Main premarket preparation function"""
    prep = PremarketPrep()
    results = await prep.run_premarket_prep()
    prep.print_summary_report(results)

    # Return success/failure
    return results.get('error') is None and results['connection_status']

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)