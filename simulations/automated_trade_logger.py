#!/usr/bin/env python3
"""
Automated Trade Logger and Reporter
Handles comprehensive trade logging, reporting, and analytics for paper trading
"""

import asyncio
import logging
import json
import csv
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.alert_manager import get_alert_manager, AlertLevel
from simulations.paper_trading_monitor import get_paper_trading_monitor

logger = logging.getLogger(__name__)

@dataclass
class TradeLogEntry:
    """Structured trade log entry"""
    timestamp: datetime
    order_id: str
    symbol: str
    action: str  # BUY, SELL
    quantity: int
    price: float
    commission: float
    realized_pnl: float
    strategy: Optional[str] = None
    confidence: Optional[float] = None
    market_conditions: Optional[Dict[str, Any]] = None
    execution_details: Optional[Dict[str, Any]] = None

@dataclass
class DailyReport:
    """Daily trading performance report"""
    date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_commission: float
    net_pnl: float
    largest_win: float
    largest_loss: float
    avg_win: float
    avg_loss: float
    sharpe_ratio: float
    portfolio_value_start: float
    portfolio_value_end: float
    top_performers: List[Dict[str, Any]] = field(default_factory=list)
    worst_performers: List[Dict[str, Any]] = field(default_factory=list)

class AutomatedTradeLogger:
    """Automated trade logging and reporting system"""

    def __init__(self):
        self.log_file = "data/trade_log.jsonl"
        self.daily_reports_file = "data/daily_reports.json"
        self.alert_manager = get_alert_manager()
        self.monitor = get_paper_trading_monitor()

        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)

        # Load existing trade log
        self.trade_log: List[TradeLogEntry] = self._load_trade_log()

        # Daily reporting schedule
        self.reporting_active = False
        self.daily_report_time = "23:59"  # Generate daily report at 11:59 PM

    async def log_trade(self, trade_data: Dict[str, Any], strategy: Optional[str] = None,
                       market_conditions: Optional[Dict[str, Any]] = None) -> bool:
        """Log a completed trade with comprehensive details"""
        try:
            # Create structured log entry
            log_entry = TradeLogEntry(
                timestamp=datetime.now(),
                order_id=trade_data.get('order_id', 'UNKNOWN'),
                symbol=trade_data.get('symbol', 'UNKNOWN'),
                action=trade_data.get('action', 'UNKNOWN'),
                quantity=trade_data.get('executed_quantity', 0),
                price=trade_data.get('executed_price', 0.0),
                commission=trade_data.get('commission', 0.0),
                realized_pnl=trade_data.get('realized_pnl', 0.0),
                strategy=strategy,
                confidence=trade_data.get('confidence'),
                market_conditions=market_conditions,
                execution_details={
                    'execution_time': trade_data.get('execution_time'),
                    'slippage': trade_data.get('slippage', 0.0),
                    'market_price': trade_data.get('market_price'),
                    'limit_price': trade_data.get('limit_price')
                }
            )

            # Add to in-memory log
            self.trade_log.append(log_entry)

            # Append to log file
            self._append_to_log_file(log_entry)

            # Update monitor
            await self.monitor.record_trade(trade_data)

            # Check for trading milestones
            await self._check_trading_milestones(log_entry)

            logger.info(f"Trade logged: {log_entry.symbol} {log_entry.action} "
                       f"{log_entry.quantity} @ ${log_entry.price:.2f} | "
                       f"PnL: ${log_entry.realized_pnl:.2f}")

            return True

        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
            await self.alert_manager.error(Exception(f"Trade logging failed: {e}"), {
                "component": "automated_trade_logger",
                "trade_data": trade_data
            })
            return False

    async def generate_daily_report(self, target_date: Optional[str] = None) -> Optional[DailyReport]:
        """Generate comprehensive daily trading report"""
        try:
            # Use target date or today
            if target_date is None:
                target_date = datetime.now().strftime('%Y-%m-%d')

            # Filter trades for the target date
            day_trades = [
                trade for trade in self.trade_log
                if trade.timestamp.strftime('%Y-%m-%d') == target_date
            ]

            if not day_trades:
                logger.info(f"No trades found for {target_date}")
                return None

            # Calculate metrics
            total_trades = len(day_trades)
            winning_trades = len([t for t in day_trades if t.realized_pnl > 0])
            losing_trades = len([t for t in day_trades if t.realized_pnl < 0])

            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

            total_pnl = sum(t.realized_pnl for t in day_trades)
            total_commission = sum(t.commission for t in day_trades)
            net_pnl = total_pnl - total_commission

            winning_amounts = [t.realized_pnl for t in day_trades if t.realized_pnl > 0]
            losing_amounts = [abs(t.realized_pnl) for t in day_trades if t.realized_pnl < 0]

            largest_win = max(winning_amounts) if winning_amounts else 0.0
            largest_loss = max(losing_amounts) if losing_amounts else 0.0
            avg_win = sum(winning_amounts) / len(winning_amounts) if winning_amounts else 0.0
            avg_loss = sum(losing_amounts) / len(losing_amounts) if losing_amounts else 0.0

            # Calculate Sharpe ratio (simplified)
            pnl_series = [t.realized_pnl for t in day_trades]
            if len(pnl_series) > 1:
                mean_return = sum(pnl_series) / len(pnl_series)
                std_return = (sum((r - mean_return) ** 2 for r in pnl_series) / len(pnl_series)) ** 0.5
                sharpe_ratio = mean_return / std_return * (252 ** 0.5) if std_return > 0 else 0.0
            else:
                sharpe_ratio = 0.0

            # Get portfolio values (simplified)
            portfolio_value_start = 100000.0  # Would come from monitor
            portfolio_value_end = portfolio_value_start + net_pnl

            # Top and worst performers by symbol
            symbol_performance = {}
            for trade in day_trades:
                if trade.symbol not in symbol_performance:
                    symbol_performance[trade.symbol] = {'pnl': 0.0, 'trades': 0}
                symbol_performance[trade.symbol]['pnl'] += trade.realized_pnl
                symbol_performance[trade.symbol]['trades'] += 1

            top_performers = sorted(
                [{'symbol': s, 'pnl': d['pnl'], 'trades': d['trades']}
                 for s, d in symbol_performance.items()],
                key=lambda x: x['pnl'], reverse=True
            )[:5]

            worst_performers = sorted(
                [{'symbol': s, 'pnl': d['pnl'], 'trades': d['trades']}
                 for s, d in symbol_performance.items()],
                key=lambda x: x['pnl']
            )[:5]

            # Create report
            report = DailyReport(
                date=target_date,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=round(win_rate, 2),
                total_pnl=round(total_pnl, 2),
                total_commission=round(total_commission, 2),
                net_pnl=round(net_pnl, 2),
                largest_win=round(largest_win, 2),
                largest_loss=round(largest_loss, 2),
                avg_win=round(avg_win, 2),
                avg_loss=round(avg_loss, 2),
                sharpe_ratio=round(sharpe_ratio, 2),
                portfolio_value_start=round(portfolio_value_start, 2),
                portfolio_value_end=round(portfolio_value_end, 2),
                top_performers=top_performers,
                worst_performers=worst_performers
            )

            # Save report
            self._save_daily_report(report)

            # Send alert for significant daily performance
            if abs(net_pnl) > 500:  # Alert on days with > $500 net P&L
                level = AlertLevel.WARNING if net_pnl > 0 else AlertLevel.ERROR
                await self.alert_manager.send_alert(
                    level=level,
                    component="automated_trade_logger",
                    message=f"Significant daily performance: ${net_pnl:.2f} net P&L on {target_date}",
                    context={
                        'date': target_date,
                        'net_pnl': net_pnl,
                        'win_rate': win_rate,
                        'total_trades': total_trades
                    }
                )

            logger.info(f"Daily report generated for {target_date}: "
                       f"{total_trades} trades, ${net_pnl:.2f} net P&L, {win_rate:.1f}% win rate")

            return report

        except Exception as e:
            logger.error(f"Failed to generate daily report: {e}")
            await self.alert_manager.error(Exception(f"Daily report generation failed: {e}"), {
                "component": "automated_trade_logger",
                "target_date": target_date
            })
            return None

    async def export_trades_to_csv(self, filename: Optional[str] = None,
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None) -> str:
        """Export trade log to CSV file"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"data/trade_export_{timestamp}.csv"

            # Filter trades by date range
            filtered_trades = self.trade_log
            if start_date:
                start_dt = datetime.fromisoformat(start_date)
                filtered_trades = [t for t in filtered_trades if t.timestamp >= start_dt]
            if end_date:
                end_dt = datetime.fromisoformat(end_date)
                filtered_trades = [t for t in filtered_trades if t.timestamp <= end_dt]

            # Write to CSV
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'order_id', 'symbol', 'action', 'quantity',
                            'price', 'commission', 'realized_pnl', 'strategy', 'confidence']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for trade in filtered_trades:
                    writer.writerow({
                        'timestamp': trade.timestamp.isoformat(),
                        'order_id': trade.order_id,
                        'symbol': trade.symbol,
                        'action': trade.action,
                        'quantity': trade.quantity,
                        'price': round(trade.price, 2),
                        'commission': round(trade.commission, 2),
                        'realized_pnl': round(trade.realized_pnl, 2),
                        'strategy': trade.strategy or '',
                        'confidence': trade.confidence or ''
                    })

            logger.info(f"Exported {len(filtered_trades)} trades to {filename}")
            return filename

        except Exception as e:
            logger.error(f"Failed to export trades to CSV: {e}")
            await self.alert_manager.error(Exception(f"Trade export failed: {e}"), {
                "component": "automated_trade_logger",
                "filename": filename
            })
            return ""

    async def get_trading_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive trading statistics for the last N days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_trades = [t for t in self.trade_log if t.timestamp >= cutoff_date]

            if not recent_trades:
                return {"message": f"No trades found in the last {days} days"}

            # Calculate statistics
            total_trades = len(recent_trades)
            winning_trades = len([t for t in recent_trades if t.realized_pnl > 0])
            losing_trades = len([t for t in recent_trades if t.realized_pnl < 0])

            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

            total_pnl = sum(t.realized_pnl for t in recent_trades)
            total_commission = sum(t.commission for t in recent_trades)
            net_pnl = total_pnl - total_commission

            # Daily breakdown
            daily_pnl = {}
            for trade in recent_trades:
                day = trade.timestamp.strftime('%Y-%m-%d')
                if day not in daily_pnl:
                    daily_pnl[day] = 0.0
                daily_pnl[day] += trade.realized_pnl

            profitable_days = len([pnl for pnl in daily_pnl.values() if pnl > 0])
            total_days = len(daily_pnl)

            return {
                'period_days': days,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': round(win_rate, 2),
                'total_pnl': round(total_pnl, 2),
                'total_commission': round(total_commission, 2),
                'net_pnl': round(net_pnl, 2),
                'profitable_days': profitable_days,
                'total_trading_days': total_days,
                'daily_win_rate': round((profitable_days / total_days * 100) if total_days > 0 else 0, 2),
                'avg_daily_pnl': round(sum(daily_pnl.values()) / len(daily_pnl) if daily_pnl else 0, 2)
            }

        except Exception as e:
            logger.error(f"Failed to get trading statistics: {e}")
            return {"error": str(e)}

    async def start_automated_reporting(self):
        """Start automated daily reporting"""
        self.reporting_active = True

        while self.reporting_active:
            try:
                # Check if it's time for daily report
                now = datetime.now()
                report_time = datetime.strptime(self.daily_report_time, '%H:%M').time()

                if now.time() >= report_time:
                    # Generate report for yesterday
                    yesterday = (now - timedelta(days=1)).strftime('%Y-%m-%d')
                    report = await self.generate_daily_report(yesterday)

                    if report:
                        logger.info(f"Automated daily report generated for {yesterday}")

                    # Wait until tomorrow
                    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                    sleep_time = (tomorrow - now).total_seconds()
                    await asyncio.sleep(sleep_time)
                else:
                    # Wait until report time
                    report_datetime = datetime.combine(now.date(), report_time)
                    if now.time() > report_time:
                        report_datetime += timedelta(days=1)

                    sleep_time = (report_datetime - now).total_seconds()
                    await asyncio.sleep(min(sleep_time, 3600))  # Check every hour max

            except Exception as e:
                logger.error(f"Automated reporting error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying

    async def stop_automated_reporting(self):
        """Stop automated daily reporting"""
        self.reporting_active = False

    def _load_trade_log(self) -> List[TradeLogEntry]:
        """Load existing trade log from file"""
        trades = []
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            # Convert timestamp string back to datetime
                            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                            trades.append(TradeLogEntry(**data))
        except Exception as e:
            logger.error(f"Failed to load trade log: {e}")

        return trades

    def _append_to_log_file(self, log_entry: TradeLogEntry):
        """Append trade log entry to file"""
        try:
            with open(self.log_file, 'a') as f:
                # Convert to dict for JSON serialization
                entry_dict = {
                    'timestamp': log_entry.timestamp.isoformat(),
                    'order_id': log_entry.order_id,
                    'symbol': log_entry.symbol,
                    'action': log_entry.action,
                    'quantity': log_entry.quantity,
                    'price': log_entry.price,
                    'commission': log_entry.commission,
                    'realized_pnl': log_entry.realized_pnl,
                    'strategy': log_entry.strategy,
                    'confidence': log_entry.confidence,
                    'market_conditions': log_entry.market_conditions,
                    'execution_details': log_entry.execution_details
                }
                f.write(json.dumps(entry_dict) + '\n')
        except Exception as e:
            logger.error(f"Failed to append to log file: {e}")

    def _save_daily_report(self, report: DailyReport):
        """Save daily report to file"""
        try:
            # Load existing reports
            reports = {}
            if os.path.exists(self.daily_reports_file):
                with open(self.daily_reports_file, 'r') as f:
                    reports = json.load(f)

            # Add new report
            reports[report.date] = {
                'total_trades': report.total_trades,
                'winning_trades': report.winning_trades,
                'losing_trades': report.losing_trades,
                'win_rate': report.win_rate,
                'total_pnl': report.total_pnl,
                'total_commission': report.total_commission,
                'net_pnl': report.net_pnl,
                'largest_win': report.largest_win,
                'largest_loss': report.largest_loss,
                'avg_win': report.avg_win,
                'avg_loss': report.avg_loss,
                'sharpe_ratio': report.sharpe_ratio,
                'portfolio_value_start': report.portfolio_value_start,
                'portfolio_value_end': report.portfolio_value_end,
                'top_performers': report.top_performers,
                'worst_performers': report.worst_performers
            }

            # Save back to file
            with open(self.daily_reports_file, 'w') as f:
                json.dump(reports, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save daily report: {e}")

    async def _check_trading_milestones(self, trade: TradeLogEntry):
        """Check for trading milestones and send alerts"""
        try:
            # Check for trade count milestones
            if len(self.trade_log) in [10, 50, 100, 500, 1000]:
                await self.alert_manager.send_alert(
                    level=AlertLevel.INFO,
                    component="automated_trade_logger",
                    message=f"Trading milestone reached: {len(self.trade_log)} total trades logged",
                    context={'total_trades': len(self.trade_log)}
                )

            # Check for significant P&L
            if abs(trade.realized_pnl) > 1000:  # $1000+ per trade
                level = AlertLevel.WARNING if trade.realized_pnl > 0 else AlertLevel.ERROR
                await self.alert_manager.send_alert(
                    level=level,
                    component="automated_trade_logger",
                    message=f"Exceptional trade: ${trade.realized_pnl:.2f} P&L on {trade.symbol}",
                    context={
                        'symbol': trade.symbol,
                        'pnl': trade.realized_pnl,
                        'order_id': trade.order_id
                    }
                )

        except Exception as e:
            logger.error(f"Failed to check trading milestones: {e}")

# Global logger instance
_logger_instance = None

def get_automated_trade_logger() -> AutomatedTradeLogger:
    """Get singleton automated trade logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = AutomatedTradeLogger()
    return _logger_instance

async def main():
    """Main function for testing the trade logger"""
    logger = get_automated_trade_logger()

    # Log some sample trades
    sample_trades = [
        {
            'order_id': 'PAPER_001',
            'symbol': 'AAPL',
            'action': 'BUY',
            'executed_quantity': 100,
            'executed_price': 150.25,
            'commission': 1.0,
            'realized_pnl': 0.0
        },
        {
            'order_id': 'PAPER_002',
            'symbol': 'AAPL',
            'action': 'SELL',
            'executed_quantity': 100,
            'executed_price': 152.10,
            'commission': 1.0,
            'realized_pnl': 185.0
        },
        {
            'order_id': 'PAPER_003',
            'symbol': 'GOOGL',
            'action': 'BUY',
            'executed_quantity': 50,
            'executed_price': 2800.00,
            'commission': 0.5,
            'realized_pnl': 0.0
        }
    ]

    for trade in sample_trades:
        await logger.log_trade(trade, strategy="momentum", market_conditions={'volatility': 0.15})

    # Generate daily report
    report = await logger.generate_daily_report()
    if report:
        print(f"Daily Report for {report.date}:")
        print(f"  Total Trades: {report.total_trades}")
        print(f"  Win Rate: {report.win_rate}%")
        print(f"  Net P&L: ${report.net_pnl}")
        print(f"  Sharpe Ratio: {report.sharpe_ratio}")

    # Export to CSV
    csv_file = await logger.export_trades_to_csv()
    print(f"Trades exported to: {csv_file}")

    # Get statistics
    stats = await logger.get_trading_statistics(days=1)
    print(f"Trading Statistics: {stats}")

if __name__ == "__main__":
    asyncio.run(main())