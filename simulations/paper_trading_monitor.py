#!/usr/bin/env python3
"""
Paper Trading Monitor Dashboard
Real-time monitoring and visualization for paper trading operations
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.alert_manager import get_alert_manager, AlertLevel

logger = logging.getLogger(__name__)

@dataclass
class TradingMetrics:
    """Real-time trading performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_commission: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    portfolio_value: float = 100000.0  # Starting capital
    daily_pnl: List[float] = field(default_factory=list)
    trade_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class SystemHealth:
    """System health and connectivity metrics"""
    ibkr_connected: bool = False
    last_ibkr_check: Optional[datetime] = None
    alert_queue_size: int = 0
    circuit_breaker_status: Dict[str, Any] = field(default_factory=dict)
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    active_positions: int = 0
    pending_orders: int = 0

class PaperTradingMonitor:
    """Real-time paper trading monitoring dashboard"""

    def __init__(self):
        self.metrics = TradingMetrics()
        self.health = SystemHealth()
        self.alert_manager = get_alert_manager()
        self.monitoring_active = False
        self.update_interval = 5  # seconds
        self.dashboard_file = "data/paper_trading_dashboard.json"

        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)

    async def start_monitoring(self):
        """Start the monitoring dashboard"""
        logger.info("Starting paper trading monitor dashboard")
        self.monitoring_active = True

        # Start background monitoring tasks
        asyncio.create_task(self._periodic_health_check())
        asyncio.create_task(self._update_dashboard())

        await self.alert_manager.info("Paper trading monitor started", {
            "component": "paper_trading_monitor",
            "update_interval": self.update_interval
        })

    async def stop_monitoring(self):
        """Stop the monitoring dashboard"""
        logger.info("Stopping paper trading monitor dashboard")
        self.monitoring_active = False

        await self.alert_manager.info("Paper trading monitor stopped", {
            "component": "paper_trading_monitor"
        })

    async def record_trade(self, trade_data: Dict[str, Any]):
        """Record a completed trade for metrics calculation"""
        try:
            # Extract trade information
            symbol = trade_data.get('symbol', 'UNKNOWN')
            action = trade_data.get('action', 'UNKNOWN')
            quantity = trade_data.get('executed_quantity', 0)
            price = trade_data.get('executed_price', 0.0)
            commission = trade_data.get('commission', 0.0)
            pnl = trade_data.get('realized_pnl', 0.0)

            # Create trade record
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'commission': commission,
                'pnl': pnl,
                'trade_id': trade_data.get('order_id', 'UNKNOWN')
            }

            # Update metrics
            self.metrics.total_trades += 1
            self.metrics.total_commission += commission
            self.metrics.total_pnl += pnl

            if pnl > 0:
                self.metrics.winning_trades += 1
                self.metrics.avg_win = ((self.metrics.avg_win * (self.metrics.winning_trades - 1)) + pnl) / self.metrics.winning_trades
                self.metrics.largest_win = max(self.metrics.largest_win, pnl)
            elif pnl < 0:
                self.metrics.losing_trades += 1
                self.metrics.avg_loss = ((self.metrics.avg_loss * (self.metrics.losing_trades - 1)) + abs(pnl)) / self.metrics.losing_trades
                self.metrics.largest_loss = max(self.metrics.largest_loss, abs(pnl))

            # Update win rate
            if self.metrics.total_trades > 0:
                self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades

            # Update portfolio value
            self.metrics.portfolio_value += pnl - commission

            # Add to trade history
            self.metrics.trade_history.append(trade_record)

            # Keep only last 1000 trades in memory
            if len(self.metrics.trade_history) > 1000:
                self.metrics.trade_history = self.metrics.trade_history[-1000:]

            logger.info(f"Trade recorded: {symbol} {action} {quantity} @ {price} | PnL: ${pnl:.2f}")

            # Send alert for significant trades
            if abs(pnl) > 100:  # Alert on trades > $100 PnL
                level = AlertLevel.WARNING if pnl > 0 else AlertLevel.ERROR
                await self.alert_manager.send_alert(
                    level=level,
                    component="paper_trading_monitor",
                    message=f"Significant paper trade: {symbol} {action} | PnL: ${pnl:.2f}",
                    context={
                        'symbol': symbol,
                        'pnl': pnl,
                        'portfolio_value': self.metrics.portfolio_value
                    }
                )

            # Force dashboard update after trade recording
            self.force_dashboard_update()

        except Exception as e:
            logger.error(f"Failed to record trade: {e}")
            await self.alert_manager.error(Exception(f"Trade recording failed: {e}"), {
                "component": "paper_trading_monitor",
                "trade_data": trade_data
            })

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data for display"""
        return {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'total_trades': self.metrics.total_trades,
                'winning_trades': self.metrics.winning_trades,
                'losing_trades': self.metrics.losing_trades,
                'win_rate': round(self.metrics.win_rate * 100, 2),
                'total_pnl': round(self.metrics.total_pnl, 2),
                'total_commission': round(self.metrics.total_commission, 2),
                'avg_win': round(self.metrics.avg_win, 2),
                'avg_loss': round(self.metrics.avg_loss, 2),
                'largest_win': round(self.metrics.largest_win, 2),
                'largest_loss': round(self.metrics.largest_loss, 2),
                'portfolio_value': round(self.metrics.portfolio_value, 2),
                'sharpe_ratio': round(self.metrics.sharpe_ratio, 2),
                'max_drawdown': round(self.metrics.max_drawdown, 2),
                'current_drawdown': round(self.metrics.current_drawdown, 2)
            },
            'health': {
                'ibkr_connected': self.health.ibkr_connected,
                'last_ibkr_check': self.health.last_ibkr_check.isoformat() if self.health.last_ibkr_check else None,
                'alert_queue_size': self.health.alert_queue_size,
                'circuit_breaker_status': self.health.circuit_breaker_status,
                'memory_usage': round(self.health.memory_usage, 2),
                'cpu_usage': round(self.health.cpu_usage, 2),
                'active_positions': self.health.active_positions,
                'pending_orders': self.health.pending_orders
            },
            'recent_trades': [
                {
                    'timestamp': trade['timestamp'].isoformat(),
                    'symbol': trade['symbol'],
                    'action': trade['action'],
                    'quantity': trade['quantity'],
                    'price': round(trade['price'], 2),
                    'pnl': round(trade['pnl'], 2)
                }
                for trade in self.metrics.trade_history[-10:]  # Last 10 trades
            ]
        }

    async def _periodic_health_check(self):
        """Perform periodic health checks"""
        while self.monitoring_active:
            try:
                # Update alert queue size
                self.health.alert_queue_size = len(self.alert_manager.error_queue)

                # Mock IBKR connectivity check (would be real in production)
                self.health.ibkr_connected = True  # Paper trading always "connected"
                self.health.last_ibkr_check = datetime.now()

                # Mock system resource usage
                import psutil
                try:
                    self.health.memory_usage = psutil.virtual_memory().percent
                    self.health.cpu_usage = psutil.cpu_percent(interval=1)
                except ImportError:
                    # psutil not available, use mock values
                    self.health.memory_usage = 45.0
                    self.health.cpu_usage = 15.0

                # Check for health issues
                if self.health.memory_usage > 90:
                    await self.alert_manager.warning("High memory usage detected", {
                        "component": "paper_trading_monitor",
                        "memory_usage": self.health.memory_usage
                    })

                if self.health.cpu_usage > 95:
                    await self.alert_manager.error(Exception("High CPU usage detected"), {
                        "component": "paper_trading_monitor",
                        "cpu_usage": self.health.cpu_usage
                    })

            except Exception as e:
                logger.error(f"Health check failed: {e}")

            await asyncio.sleep(self.update_interval)

    async def _update_dashboard(self):
        """Update dashboard file periodically"""
        while self.monitoring_active:
            try:
                dashboard_data = await self.get_dashboard_data()

                # Write to file
                with open(self.dashboard_file, 'w') as f:
                    json.dump(dashboard_data, f, indent=2, default=str)

                # Log summary every 60 seconds
                if int(time.time()) % 60 == 0:
                    logger.info(f"Dashboard updated - Portfolio: ${self.metrics.portfolio_value:.2f}, "
                              f"Trades: {self.metrics.total_trades}, Win Rate: {self.metrics.win_rate:.1%}")

            except Exception as e:
                logger.error(f"Dashboard update failed: {e}")

            await asyncio.sleep(self.update_interval)

    def force_dashboard_update(self):
        """Force an immediate dashboard update"""
        try:
            import asyncio
            # Create a new event loop if one doesn't exist
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, create task
                    asyncio.create_task(self._update_dashboard_once())
                else:
                    loop.run_until_complete(self._update_dashboard_once())
            except RuntimeError:
                # No event loop, create one
                asyncio.run(self._update_dashboard_once())
        except Exception as e:
            logger.error(f"Force dashboard update failed: {e}")

    async def _update_dashboard_once(self):
        """Update dashboard file once"""
        try:
            dashboard_data = await self.get_dashboard_data()
            with open(self.dashboard_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Dashboard update failed: {e}")

    def print_dashboard(self):
        """Print current dashboard to console"""
        try:
            with open(self.dashboard_file, 'r') as f:
                data = json.load(f)

            print("\n" + "="*80)
            print("PAPER TRADING MONITOR DASHBOARD")
            print("="*80)

            metrics = data['metrics']
            health = data['health']

            print("\n📊 PERFORMANCE METRICS:")
            print(f"  Total Trades: {metrics['total_trades']}")
            print(f"  Win Rate: {metrics['win_rate']}%")
            print(f"  Total P&L: ${metrics['total_pnl']}")
            print(f"  Portfolio Value: ${metrics['portfolio_value']}")
            print(f"  Avg Win: ${metrics['avg_win']}")
            print(f"  Avg Loss: ${metrics['avg_loss']}")

            print("\n🏥 SYSTEM HEALTH:")
            print(f"  IBKR Connected: {'✅' if health['ibkr_connected'] else '❌'}")
            print(f"  Memory Usage: {health['memory_usage']}%")
            print(f"  CPU Usage: {health['cpu_usage']}%")
            print(f"  Alert Queue: {health['alert_queue_size']}")

            if data['recent_trades']:
                print("\n📈 RECENT TRADES:")
                for trade in data['recent_trades'][-5:]:  # Show last 5
                    pnl_symbol = "📈" if trade['pnl'] >= 0 else "📉"
                    print(f"  {trade['timestamp'][:19]} | {trade['symbol']} {trade['action']} {trade['quantity']} @ ${trade['price']} | {pnl_symbol} ${trade['pnl']}")

            print(f"\n⏰ Last Updated: {data['timestamp'][:19]}")
            print("="*80)

        except FileNotFoundError:
            print("Dashboard file not found. Start monitoring first.")
        except Exception as e:
            print(f"Error displaying dashboard: {e}")

# Global monitor instance
_monitor_instance = None

def get_paper_trading_monitor() -> PaperTradingMonitor:
    """Get singleton paper trading monitor instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = PaperTradingMonitor()
    return _monitor_instance

async def main():
    """Main function for testing the monitor"""
    monitor = get_paper_trading_monitor()

    # Start monitoring
    await monitor.start_monitoring()

    # Simulate some trades
    sample_trades = [
        {'symbol': 'AAPL', 'action': 'BUY', 'executed_quantity': 100, 'executed_price': 150.25, 'commission': 1.0, 'realized_pnl': 0.0, 'order_id': 'PAPER_001'},
        {'symbol': 'AAPL', 'action': 'SELL', 'executed_quantity': 100, 'executed_price': 152.10, 'commission': 1.0, 'realized_pnl': 185.0, 'order_id': 'PAPER_002'},
        {'symbol': 'GOOGL', 'action': 'BUY', 'executed_quantity': 50, 'executed_price': 2800.00, 'commission': 0.5, 'realized_pnl': 0.0, 'order_id': 'PAPER_003'},
        {'symbol': 'GOOGL', 'action': 'SELL', 'executed_quantity': 50, 'executed_price': 2785.00, 'commission': 0.5, 'realized_pnl': -75.0, 'order_id': 'PAPER_004'},
    ]

    for trade in sample_trades:
        await monitor.record_trade(trade)
        await asyncio.sleep(1)  # Simulate time between trades

    # Display dashboard
    monitor.print_dashboard()

    # Keep running for a bit to show monitoring
    await asyncio.sleep(10)

    # Stop monitoring
    await monitor.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())