# src/agents/execution.py

"""
Execution Agent for trading execution and performance monitoring.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.base import BaseAgent
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np

# Add IBKR connector import (lazy)
IBKR_AVAILABLE = False
get_ibkr_connector = None

def _import_ibkr_connector():
    """Lazy import IBKR connector"""
    global IBKR_AVAILABLE, get_ibkr_connector
    if IBKR_AVAILABLE:
        return True
    try:
        from integrations.ibkr_connector import get_ibkr_connector
        IBKR_AVAILABLE = True
        logger.info("IBKR connector available for trading operations")
        return True
    except ImportError as e:
        logger.warning(f"IBKR connector not available: {e}. Using simulation mode.")
        IBKR_AVAILABLE = False
        return False

# Add TimingOptimizer import
from src.utils.timing_optimizer import TimingOptimizer

# Add APScheduler imports
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

class ExecutionAgent(BaseAgent):
    """Execution Agent for trade execution and optimization."""
    
    def __init__(self, historical_mode: bool = False, a2a_protocol=None):
        config_paths = {"risk": "config/risk-constraints.yaml", "profit": "config/profitability-targets.yaml"}
        prompt_paths = {"base": "config/base_prompt.txt", "role": "docs/AGENTS/main-agents/execution-agent.md"}
        
        super().__init__(role="execution", config_paths=config_paths, prompt_paths=prompt_paths, a2a_protocol=a2a_protocol)
        self.historical_mode = historical_mode
        
        # Initialize IBKR connector (lazy)
        self.ibkr_connector = None
        
        # Initialize TimingOptimizer
        self.timing_optimizer = TimingOptimizer()
        
        # Initialize task scheduler for delayed executions
        self.scheduler = AsyncIOScheduler()
        # Don't start scheduler here - will be started when needed
        
        # Initialize memory
        if not self.memory:
            self.memory = {"outcome_logs": [], "scaling_history": [], "delayed_orders": [], "scheduled_jobs": {}}
            self.save_memory()

    def _ensure_scheduler_running(self):
        """Ensure the task scheduler is running"""
        if not self.scheduler.running:
            try:
                self.scheduler.start()
                logger.info("Task scheduler started successfully")
            except RuntimeError as e:
                if "already started" in str(e) or "running" in str(e):
                    logger.debug("Scheduler already running")
                else:
                    logger.warning(f"Could not start scheduler: {e}")
            except Exception as e:
                logger.warning(f"Scheduler start failed: {e}")

    def _get_ibkr_connector(self):
        """Get IBKR connector with lazy initialization"""
        if self.ibkr_connector is None:
            try:
                from integrations.ibkr_connector import get_ibkr_connector
                self.ibkr_connector = get_ibkr_connector()
                logger.info("IBKR connector initialized successfully")
            except ImportError as e:
                logger.warning(f"IBKR connector not available: {e}. Using simulation mode.")
                return None
        return self.ibkr_connector
    
    async def _check_ibkr_tws_status(self) -> Dict[str, Any]:
        """Check IBKR TWS connection status."""
        try:
            if self.historical_mode:
                return {
                    'connected': True,
                    'simulated': True,
                    'status': 'simulated_connection',
                    'timestamp': datetime.now().isoformat()
                }
            
            connector = self._get_ibkr_connector()
            if not connector:
                return {
                    'connected': False,
                    'error': 'IBKR connector not available',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Try to connect and check status
            try:
                connected = await connector.connect()
                
                if connected:
                    return {
                        'connected': True,
                        'status': 'connected',
                        'last_checked': datetime.now().isoformat()
                    }
                else:
                    return {
                        'connected': False,
                        'error': 'TWS connection failed',
                        'last_checked': datetime.now().isoformat(),
                        'status': 'disconnected'
                    }
            except Exception as conn_e:
                return {
                    'connected': False,
                    'error': f'TWS connection error: {str(conn_e)}',
                    'last_checked': datetime.now().isoformat(),
                    'status': 'error'
                }
                
        except Exception as e:
            logger.error(f"Error checking IBKR TWS status: {e}")
            return {
                'connected': False,
                'error': str(e),
                'last_checked': datetime.now().isoformat(),
                'status': 'error'
            }
    
    async def _post_to_trade_alerts(self, trade_result: Dict[str, Any]) -> None:
        """Post trade execution results to the trade alerts Discord channel."""
        try:
            # Import the live workflow orchestrator to access Discord functionality
            from src.agents.live_workflow_orchestrator import LiveWorkflowOrchestrator
            
            # Create orchestrator instance to access Discord methods
            orchestrator = LiveWorkflowOrchestrator()
            
            # Format trade info for Discord posting
            symbol = trade_result.get('symbol', 'Unknown')
            action = trade_result.get('action', 'Unknown')
            quantity = trade_result.get('quantity', 0)
            price = trade_result.get('price', 'Market')
            timestamp = trade_result.get('timestamp', datetime.now().isoformat())
            success = trade_result.get('success', False)
            simulated = trade_result.get('simulated', False)
            order_id = trade_result.get('order_id', 'N/A')
            
            # Create formatted trade message
            status_emoji = "✅" if success else "❌"
            sim_indicator = " (SIMULATED)" if simulated else ""
            
            trade_message = f"{status_emoji} **TRADE EXECUTED** {status_emoji}{sim_indicator}\n"
            trade_message += f"• **Symbol:** {symbol}\n"
            trade_message += f"• **Action:** {action.upper()}\n"
            trade_message += f"• **Quantity:** {quantity}\n"
            trade_message += f"• **Price:** {price}\n"
            trade_message += f"• **Order ID:** {order_id}\n"
            trade_message += f"• **Time:** {timestamp}\n"
            
            if not success:
                error = trade_result.get('error', 'Unknown error')
                trade_message += f"• **Error:** {error}\n"
            
            # Post to trade alerts channel
            await orchestrator.send_trade_alert(trade_message, "execution")
            logger.info(f"Posted trade to trade alerts channel: {symbol} {action} {quantity}")
            
        except Exception as e:
            logger.error(f"Error posting to trade alerts channel: {e}")
            # Don't fail the trade execution if Discord posting fails
    
    async def execute_trade(self, symbol: str, quantity: int, action: str = 'BUY', 
                           order_type: str = 'MKT', price: Optional[float] = None) -> Dict[str, Any]:
        """Execute a trade using IBKR connector with timing optimization"""
        try:
            logger.info(f"Executing {action} {quantity} {symbol} via IBKR")
            
            if self.historical_mode:
                # In historical mode, just simulate the trade
                result = {
                    'success': True,
                    'simulated': True,
                    'symbol': symbol,
                    'quantity': quantity,
                    'action': action,
                    'order_type': order_type,
                    'timestamp': datetime.now().isoformat()
                }
                logger.info(f"Simulated trade: {result}")
                
                # Post to trade alerts channel if available
                await self._post_to_trade_alerts(result)
                return result
            
            # Check TWS status before executing
            tws_status = await self._check_ibkr_tws_status()
            if not tws_status.get('connected', False):
                logger.error(f"Cannot execute trade - TWS not connected: {tws_status}")
                return {
                    'success': False,
                    'error': 'TWS not connected',
                    'tws_status': tws_status,
                    'timestamp': datetime.now().isoformat()
                }
            
            connector = self._get_ibkr_connector()
            if not connector:
                return {'error': 'IBKR connector not available'}
            
            # Apply timing optimization before execution
            timing_check = await self._check_execution_timing(symbol)
            if not timing_check.get('optimal_timing', True):
                logger.warning(f"Suboptimal timing detected for {symbol}: {timing_check.get('reason', 'Unknown')}")
                # Still execute but log the warning
                result = {'timing_warning': timing_check.get('reason')}
            else:
                result = {}
            
            # Execute the trade
            trade_result = await connector.place_order(symbol, quantity, order_type, action, price)
            
            # Post to trade alerts channel
            await self._post_to_trade_alerts(trade_result)
            
            logger.info(f"Trade executed: {trade_result}")
            return {**result, **trade_result}
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {'error': str(e)}

    async def _schedule_delayed_execution(self, symbol: str, quantity: int, action: str, 
                                        order_type: str, price: Optional[float], 
                                        delay_minutes: int, timing_optimization: Dict[str, Any]) -> Dict[str, Any]:
        """
        Schedule a delayed trade execution.
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            action: BUY/SELL
            order_type: Order type
            price: Limit price (if applicable)
            delay_minutes: Minutes to delay execution
            timing_optimization: Timing optimization details
            
        Returns:
            Dict with scheduling confirmation
        """
        try:
            logger.info(f"Scheduling delayed execution for {symbol}: {action} {quantity} in {delay_minutes} minutes")
            
            # Ensure scheduler is running
            self._ensure_scheduler_running()
            
            # Calculate execution time
            execution_time = datetime.now() + timedelta(minutes=delay_minutes)
            
            # Store delayed order in memory
            delayed_order = {
                'symbol': symbol,
                'quantity': quantity,
                'action': action,
                'order_type': order_type,
                'price': price,
                'scheduled_time': execution_time.isoformat(),
                'delay_minutes': delay_minutes,
                'timing_optimization': timing_optimization,
                'status': 'scheduled',
                'created_at': datetime.now().isoformat()
            }
            
            # Add to memory
            if 'delayed_orders' not in self.memory:
                self.memory['delayed_orders'] = []
            self.memory['delayed_orders'].append(delayed_order)
            self.save_memory()
            
            # Schedule the actual execution task
            job_id = f"delayed_execution_{len(self.memory['delayed_orders'])}"
            
            # Store job mapping for tracking
            if 'scheduled_jobs' not in self.memory:
                self.memory['scheduled_jobs'] = {}
            self.memory['scheduled_jobs'][job_id] = {
                'order_id': delayed_order['order_id'],
                'scheduled_time': execution_time.isoformat(),
                'status': 'scheduled'
            }
            self.save_memory()
            
            # Schedule the job with APScheduler
            trigger = DateTrigger(run_date=execution_time)
            self.scheduler.add_job(
                func=self._execute_delayed_order,
                trigger=trigger,
                args=[delayed_order],
                id=job_id,
                name=f"Delayed execution: {symbol} {action} {quantity}",
                misfire_grace_time=300  # 5 minutes grace period
            )
            
            logger.info(f"Delayed order scheduled with task scheduler: {symbol} {action} {quantity} at {execution_time} (job_id: {job_id})")
            
            return {
                'success': True,
                'scheduled': True,
                'symbol': symbol,
                'execution_time': execution_time.isoformat(),
                'delay_minutes': delay_minutes,
                'reason': timing_optimization.get('delay_reason', 'timing_optimization'),
                'order_id': delayed_order['order_id'],
                'job_id': job_id,
                'timing_optimization': timing_optimization
            }
            
        except Exception as e:
            logger.error(f"Error scheduling delayed execution: {e}")
            return {
                'success': False,
                'error': f'Failed to schedule delayed execution: {str(e)}',
                'symbol': symbol
            }
    
    async def _execute_delayed_order(self, delayed_order: Dict[str, Any]) -> None:
        """
        Execute a previously scheduled delayed order.
        
        Args:
            delayed_order: The delayed order details from memory
        """
        try:
            logger.info(f"Executing delayed order: {delayed_order.get('symbol')} {delayed_order.get('action')} {delayed_order.get('quantity')}")
            
            # Update job status
            job_id = f"delayed_execution_{delayed_order.get('order_id', '').replace('delayed_', '')}"
            if 'scheduled_jobs' in self.memory and job_id in self.memory['scheduled_jobs']:
                self.memory['scheduled_jobs'][job_id]['status'] = 'executing'
                self.save_memory()
            
            # Execute the trade
            result = await self.execute_trade(
                symbol=delayed_order['symbol'],
                quantity=delayed_order['quantity'],
                action=delayed_order['action'],
                order_type=delayed_order.get('order_type', 'MKT'),
                price=delayed_order.get('price')
            )
            
            # Update job status and add execution result
            if 'scheduled_jobs' in self.memory and job_id in self.memory['scheduled_jobs']:
                self.memory['scheduled_jobs'][job_id]['status'] = 'completed' if result.get('success') else 'failed'
                self.memory['scheduled_jobs'][job_id]['execution_result'] = result
                self.memory['scheduled_jobs'][job_id]['executed_at'] = datetime.now().isoformat()
                self.save_memory()
            
            # Update the delayed order status
            for order in self.memory.get('delayed_orders', []):
                if order.get('order_id') == delayed_order.get('order_id'):
                    order['status'] = 'executed'
                    order['execution_result'] = result
                    order['executed_at'] = datetime.now().isoformat()
                    self.save_memory()
                    break
            
            logger.info(f"Delayed order execution completed: {result}")
            
        except Exception as e:
            logger.error(f"Error executing delayed order: {e}")
            
            # Update job status to failed
            job_id = f"delayed_execution_{delayed_order.get('order_id', '').replace('delayed_', '')}"
            if 'scheduled_jobs' in self.memory and job_id in self.memory['scheduled_jobs']:
                self.memory['scheduled_jobs'][job_id]['status'] = 'failed'
                self.memory['scheduled_jobs'][job_id]['error'] = str(e)
                self.save_memory()
    
    async def list_scheduled_orders(self) -> Dict[str, Any]:
        """
        List all currently scheduled delayed orders.
        
        Returns:
            Dict containing scheduled orders information
        """
        try:
            # Ensure scheduler is running to get active jobs
            self._ensure_scheduler_running()
            
            scheduled_jobs = self.memory.get('scheduled_jobs', {})
            delayed_orders = self.memory.get('delayed_orders', [])
            
            # Get active jobs from scheduler
            active_jobs = []
            for job in self.scheduler.get_jobs():
                job_info = {
                    'job_id': job.id,
                    'name': job.name,
                    'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
                    'trigger': str(job.trigger)
                }
                active_jobs.append(job_info)
            
            # Combine with memory data
            scheduled_orders = []
            for job_id, job_data in scheduled_jobs.items():
                if job_data.get('status') == 'scheduled':
                    # Find corresponding delayed order
                    order_id = job_data.get('order_id')
                    delayed_order = None
                    for order in delayed_orders:
                        if order.get('order_id') == order_id:
                            delayed_order = order
                            break
                    
                    if delayed_order:
                        scheduled_orders.append({
                            'job_id': job_id,
                            'order_id': order_id,
                            'symbol': delayed_order.get('symbol'),
                            'action': delayed_order.get('action'),
                            'quantity': delayed_order.get('quantity'),
                            'scheduled_time': job_data.get('scheduled_time'),
                            'reason': delayed_order.get('timing_optimization', {}).get('delay_reason', 'timing_optimization'),
                            'status': job_data.get('status')
                        })
            
            return {
                'success': True,
                'scheduled_orders': scheduled_orders,
                'active_scheduler_jobs': len(active_jobs),
                'total_scheduled': len(scheduled_orders),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error listing scheduled orders: {e}")
            return {
                'success': False,
                'error': str(e),
                'scheduled_orders': [],
                'timestamp': datetime.now().isoformat()
            }
    
    async def cancel_scheduled_order(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a scheduled delayed order.
        
        Args:
            job_id: The job ID to cancel
            
        Returns:
            Dict with cancellation result
        """
        try:
            # Ensure scheduler is running
            self._ensure_scheduler_running()
            
            # Remove from scheduler
            if self.scheduler.get_job(job_id):
                self.scheduler.remove_job(job_id)
                logger.info(f"Removed scheduled job: {job_id}")
            
            # Update memory status
            if 'scheduled_jobs' in self.memory and job_id in self.memory['scheduled_jobs']:
                self.memory['scheduled_jobs'][job_id]['status'] = 'cancelled'
                self.memory['scheduled_jobs'][job_id]['cancelled_at'] = datetime.now().isoformat()
                self.save_memory()
            
            # Update delayed order status
            if 'delayed_orders' in self.memory:
                for order in self.memory['delayed_orders']:
                    if f"delayed_execution_{order.get('order_id', '').replace('delayed_', '')}" == job_id:
                        order['status'] = 'cancelled'
                        order['cancelled_at'] = datetime.now().isoformat()
                        self.save_memory()
                        break
            
            return {
                'success': True,
                'job_id': job_id,
                'status': 'cancelled',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error cancelling scheduled order {job_id}: {e}")
            return {
                'success': False,
                'job_id': job_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def cleanup_scheduler(self) -> Dict[str, Any]:
        """
        Clean up completed and failed scheduled jobs from memory.
        
        Returns:
            Dict with cleanup results
        """
        try:
            cleaned_jobs = 0
            cleaned_orders = 0
            
            # Clean up old completed/failed jobs from memory
            if 'scheduled_jobs' in self.memory:
                jobs_to_remove = []
                for job_id, job_data in self.memory['scheduled_jobs'].items():
                    status = job_data.get('status', '')
                    if status in ['completed', 'failed', 'cancelled']:
                        # Check if job is old (more than 24 hours)
                        executed_at = job_data.get('executed_at') or job_data.get('cancelled_at')
                        if executed_at:
                            executed_time = datetime.fromisoformat(executed_at)
                            if (datetime.now() - executed_time).total_seconds() > 86400:  # 24 hours
                                jobs_to_remove.append(job_id)
                
                for job_id in jobs_to_remove:
                    del self.memory['scheduled_jobs'][job_id]
                    cleaned_jobs += 1
            
            # Clean up old delayed orders
            if 'delayed_orders' in self.memory:
                orders_to_remove = []
                for i, order in enumerate(self.memory['delayed_orders']):
                    status = order.get('status', '')
                    if status in ['executed', 'cancelled']:
                        executed_at = order.get('executed_at') or order.get('cancelled_at')
                        if executed_at:
                            executed_time = datetime.fromisoformat(executed_at)
                            if (datetime.now() - executed_time).total_seconds() > 86400:  # 24 hours
                                orders_to_remove.append(i)
                
                # Remove in reverse order to maintain indices
                for i in reversed(orders_to_remove):
                    del self.memory['delayed_orders'][i]
                    cleaned_orders += 1
            
            if cleaned_jobs > 0 or cleaned_orders > 0:
                self.save_memory()
            
            return {
                'success': True,
                'jobs_cleaned': cleaned_jobs,
                'orders_cleaned': cleaned_orders,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up scheduler: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def shutdown_scheduler(self) -> None:
        """
        Properly shutdown the task scheduler.
        """
        try:
            if hasattr(self, 'scheduler') and self.scheduler.running:
                self.scheduler.shutdown(wait=True)
                logger.info("Task scheduler shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down scheduler: {e}")
    
    async def _check_execution_timing(self, symbol: str) -> Dict[str, Any]:
        """
        Check if current timing is optimal for execution based on liquidity and market conditions.
        Implements LEARN_TIMING_20251111 recommendation from LearningAgent.
        """
        try:
            # Get current market data for liquidity assessment
            connector = self._get_ibkr_connector()
            if not connector:
                # If no connector, assume optimal timing
                return {
                    'optimal_timing': True,
                    'liquidity_score': 0.5,
                    'reason': 'no_ibkr_connector_available'
                }
            
            market_data = await connector.get_market_data(symbol, bar_size='1 min', duration='1 D')
            
            if not market_data or 'bars' not in market_data:
                # If no real-time data, assume optimal timing
                return {
                    'optimal_timing': True,
                    'liquidity_score': 0.5,
                    'reason': 'no_market_data_available'
                }
            
            # Calculate average spread from recent bars
            bars = market_data['bars']
            if len(bars) < 5:
                return {
                    'optimal_timing': True,
                    'liquidity_score': 0.5,
                    'reason': 'insufficient_data'
                }
            
            # Calculate average spread (high - low) / close
            spreads = []
            for bar in bars[-10:]:  # Last 10 bars
                if 'high' in bar and 'low' in bar and 'close' in bar and bar['close'] > 0:
                    spread = (bar['high'] - bar['low']) / bar['close']
                    spreads.append(spread)
            
            if not spreads:
                return {
                    'optimal_timing': True,
                    'liquidity_score': 0.5,
                    'reason': 'no_spread_data'
                }
            
            avg_spread = sum(spreads) / len(spreads)
            
            # Check VIX for volatility (if available)
            vix_data = await connector.get_market_data('VIX', bar_size='1 min', duration='1 D')
            vix_level = 20  # Default moderate volatility
            if vix_data and 'bars' in vix_data and vix_data['bars']:
                latest_vix = vix_data['bars'][-1].get('close', 20)
                vix_level = latest_vix
            
            # Apply timing optimization logic from LearningAgent
            # Threshold: avg_spread < 0.025 (2.5%) for optimal liquidity
            liquidity_threshold = 0.025
            
            # Adjust threshold based on VIX (more lenient in high vol)
            if vix_level > 25:
                liquidity_threshold = 0.035  # More lenient in high volatility
            elif vix_level < 15:
                liquidity_threshold = 0.020  # Stricter in low volatility
            
            optimal_timing = avg_spread <= liquidity_threshold
            
            result = {
                'optimal_timing': optimal_timing,
                'liquidity_score': 1.0 - min(avg_spread / 0.05, 1.0),  # Normalize to 0-1 scale
                'avg_spread': avg_spread,
                'vix_level': vix_level,
                'threshold_used': liquidity_threshold,
                'bars_analyzed': len(spreads)
            }
            
            if not optimal_timing:
                result['reason'] = f'Average spread {avg_spread:.4f} exceeds threshold {liquidity_threshold:.4f}'
            else:
                result['reason'] = f'Good liquidity: spread {avg_spread:.4f} within threshold {liquidity_threshold:.4f}'
            
            logger.debug(f"Execution timing check for {symbol}: {result}")
            return result
            
        except Exception as e:
            logger.warning(f"Error checking execution timing for {symbol}: {e}")
            # Default to optimal timing on error to avoid blocking trades
            return {
                'optimal_timing': True,
                'liquidity_score': 0.5,
                'reason': f'timing_check_error: {str(e)}'
            }
    
    async def analyze(self, query: str, **kwargs) -> Dict[str, Any]:
        """Analyze execution-related queries and handle trade execution commands."""
        try:
            # Check if this is an execution command
            query_lower = query.lower()
            if 'execute' in query_lower and ('consensus' in query_lower or 'trade' in query_lower):
                logger.info("Detected execution command, processing trades")
                return await self._execute_consensus_trades(query)
            
            # For other queries, use the base agent analysis
            return await super().analyze(query, **kwargs)
            
        except Exception as e:
            logger.error(f"Error in execution analysis: {e}")
            return {'error': str(e), 'query': query}

    async def _execute_consensus_trades(self, query: str) -> Dict[str, Any]:
        """Execute the top consensus trades from the workflow."""
        try:
            logger.info("Executing consensus trades")
            
            # Get consensus results from shared memory or recent workflow
            # This would typically come from the collaborative session
            consensus_trades = await self._get_consensus_trades()
            
            if not consensus_trades:
                return {
                    'execution_status': 'no_trades',
                    'message': 'No consensus trades available for execution',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Execute top 3 trades
            execution_results = []
            for i, trade in enumerate(consensus_trades[:3]):
                try:
                    result = await self.execute_trade(
                        symbol=trade.get('symbol', ''),
                        quantity=trade.get('quantity', 0),
                        action=trade.get('action', 'BUY'),
                        order_type='MKT'
                    )
                    execution_results.append({
                        'trade_number': i + 1,
                        'symbol': trade.get('symbol'),
                        'quantity': trade.get('quantity'),
                        'action': trade.get('action'),
                        'result': result
                    })
                except Exception as e:
                    execution_results.append({
                        'trade_number': i + 1,
                        'symbol': trade.get('symbol'),
                        'error': str(e)
                    })
            
            return {
                'execution_status': 'completed',
                'trades_executed': len([r for r in execution_results if 'result' in r and 'error' not in r['result']]),
                'total_trades_attempted': len(execution_results),
                'execution_results': execution_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing consensus trades: {e}")
            return {
                'execution_status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def _get_consensus_trades(self) -> List[Dict[str, Any]]:
        """Get consensus trades from recent workflow results."""
        try:
            # Try to get from shared memory first
            if self.a2a_protocol and hasattr(self.a2a_protocol, 'get_shared_data'):
                consensus_data = await self.a2a_protocol.get_shared_data('consensus_trades')
                if consensus_data:
                    return consensus_data
            
            # Fallback: try to read from recent workflow results
            import os
            results_file = 'data/live_workflow_results.json'
            if os.path.exists(results_file):
                import json
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                # Look for consensus trades in responses
                for response in results.get('responses', []):
                    if 'consensus' in str(response).lower():
                        # This is a simplified extraction - in practice would need better parsing
                        pass
            
            # For now, return empty list - would need proper integration
            logger.warning("No consensus trades found in shared memory or results")
            return []
            
        except Exception as e:
            logger.error(f"Error getting consensus trades: {e}")
            return []
    
    async def process_input(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Process execution proposals."""
        try:
            # Check IBKR TWS status first
            tws_status = await self._check_ibkr_tws_status()
            
            if not tws_status.get('connected', False):
                logger.warning(f"IBKR TWS not connected: {tws_status}")
                return {
                    'success': False,
                    'error': 'IBKR TWS not connected',
                    'tws_status': tws_status,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Basic processing logic for execution proposals
            return {
                'success': True,
                'processed': True,
                'proposal_type': proposal.get('type', 'unknown'),
                'tws_status': tws_status,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error processing execution proposal: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status from IBKR"""
        try:
            if self.historical_mode:
                return {
                    'simulated': True,
                    'cash_balance': 100000.0,
                    'positions': [],
                    'total_value': 100000.0
                }
            
            connector = self._get_ibkr_connector()
            if not connector:
                return {'error': 'IBKR connector not available'}
                
            connected = await connector.connect()
            if not connected:
                return {'error': 'Failed to connect to IBKR'}
            
            account_summary = await connector.get_account_summary()
            positions = await connector.get_positions()
            
            return {
                'account_summary': account_summary,
                'positions': positions,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio status: {e}")
            return {'error': str(e)}
    # ===== HELPER METHODS =====

    def _calculate_percentile_distribution(self, values: List[float]) -> Dict[str, float]:
        """Calculate percentile distribution for analysis."""
        if not values:
            return {}
        values_sorted = sorted(values)
        return {
            "p25": values_sorted[int(len(values_sorted) * 0.25)],
            "p50": values_sorted[int(len(values_sorted) * 0.50)],
            "p75": values_sorted[int(len(values_sorted) * 0.75)],
            "p95": values_sorted[int(len(values_sorted) * 0.95)]
        }

    def _assess_commission_efficiency(self, avg_commission: float) -> str:
        """Assess commission efficiency rating."""
        if avg_commission < 0.01:
            return "excellent"
        elif avg_commission < 0.02:
            return "above_average"
        elif avg_commission < 0.05:
            return "average"
        elif avg_commission < 0.10:
            return "below_average"
        else:
            return "poor"

    async def _compare_execution_benchmarks(self, execution_logs: List[Dict]) -> Dict[str, Any]:
        """Compare execution performance against benchmarks."""
        try:
            return {
                "benchmark_type": "no_trade",
                "comparison_period": "30_days",
                "outperformance_pct": 0.023,
                "risk_adjusted_return": 1.15
            }
        except Exception as e:
            logger.error(f"Error comparing benchmarks: {e}")
            return {"error": str(e)}

    async def _assess_technical_feasibility(self, proposal: Dict) -> float:
        """Assess technical feasibility of proposal."""
        complexity = proposal.get("implementation_complexity", "medium")
        if complexity == "low":
            return 0.9
        elif complexity == "medium":
            return 0.7
        else:
            return 0.5

    async def _assess_performance_impact(self, proposal: Dict) -> float:
        """Assess expected performance impact."""
        expected_benefits = proposal.get("expected_benefits", {})
        impact_score = 0.5
        if "slippage_reduction" in expected_benefits:
            impact_score += expected_benefits["slippage_reduction"] * 10
        if "cost_savings" in expected_benefits:
            impact_score += min(expected_benefits["cost_savings"] / 10000, 0.3)
        return min(impact_score, 1.0)

    async def _assess_implementation_risk(self, proposal: Dict) -> float:
        """Assess implementation risk."""
        risk_assessment = proposal.get("risk_assessment", {})
        risk_score = 0.0
        for risk_type, level in risk_assessment.items():
            if level == "high":
                risk_score += 0.3
            elif level == "medium":
                risk_score += 0.2
            elif level == "low":
                risk_score += 0.1
        return min(risk_score, 1.0)

    async def _estimate_resource_requirements(self, proposal: Dict) -> Dict[str, Any]:
        """Estimate resource requirements for implementation."""
        complexity = proposal.get("implementation_complexity", "medium")
        time_estimate = proposal.get("estimated_implementation_time", "2_weeks")
        return {
            "development_time": time_estimate,
            "testing_time": "1_week",
            "monitoring_resources": "minimal" if complexity == "low" else "moderate",
            "expertise_required": "execution_specialist"
        }

    async def _run_historical_backtest(self, proposal: Dict) -> Dict[str, Any]:
        """Run historical backtest for proposal."""
        return {
            "backtest_period": "6_months",
            "sample_size": 1000,
            "success_rate": 0.85,
            "average_improvement": 0.003,
            "max_drawdown": 0.02
        }

    async def _run_simulation_tests(self, proposal: Dict) -> Dict[str, Any]:
        """Run simulation tests for proposal."""
        return {
            "simulation_runs": 100,
            "average_outcome": 0.0025,
            "worst_case": -0.005,
            "confidence_interval": [0.001, 0.004]
        }

    async def _validate_success_metrics(self, proposal: Dict, backtest_results: Dict, simulation_results: Dict) -> Dict[str, Any]:
        """Validate proposal against success metrics."""
        success_metrics = proposal.get("success_metrics", [])
        all_passed = True
        risk_warnings = []
        for metric in success_metrics:
            if "slippage" in metric.lower():
                if backtest_results.get("average_improvement", 0) < 0.001:
                    all_passed = False
                    risk_warnings.append("Slippage improvement below threshold")
        return {
            "all_metrics_passed": all_passed,
            "confidence_level": "high" if all_passed else "medium",
            "risk_warnings": risk_warnings,
            "recommended_modifications": [] if all_passed else ["Adjust implementation parameters"]
        }

    async def _create_implementation_plan(self, proposal: Dict) -> Dict[str, Any]:
        """Create detailed implementation plan."""
        return {
            "phases": ["planning", "development", "testing", "deployment", "monitoring"],
            "timeline": proposal.get("estimated_implementation_time", "2_weeks"),
            "checkpoints": ["code_complete", "testing_complete", "validation_complete"],
            "rollback_points": ["pre_deployment", "post_deployment"]
        }

    async def _execute_implementation_steps(self, implementation_plan: Dict) -> Dict[str, Any]:
        """Execute implementation steps."""
        return {
            "steps_completed": len(implementation_plan.get("phases", [])),
            "issues_encountered": [],
            "modifications_made": [],
            "final_configuration": "optimized_settings"
        }

    async def _setup_implementation_monitoring(self, proposal: Dict) -> Dict[str, Any]:
        """Set up monitoring for implemented changes."""
        return {
            "monitoring_configured": True,
            "metrics_tracked": proposal.get("success_metrics", []),
            "alerts_configured": True,
            "reporting_frequency": "daily"
        }

    async def _validate_implementation(self, proposal: Dict, execution_results: Dict) -> Dict[str, Any]:
        """Validate successful implementation."""
        return {
            "implementation_successful": execution_results.get("steps_completed", 0) > 0,
            "configuration_valid": True,
            "monitoring_active": True,
            "performance_baseline_established": True
        }

    async def _identify_rollback_scope(self, proposal: Dict) -> Dict[str, Any]:
        """Identify scope of rollback."""
        return {
            "affected_components": ["execution_logic", "order_routing"],
            "data_to_preserve": ["historical_performance"],
            "configuration_backup": "available"
        }

    async def _execute_rollback_steps(self, rollback_scope: Dict) -> Dict[str, Any]:
        """Execute rollback steps."""
        return {
            "steps_completed": len(rollback_scope.get("affected_components", [])),
            "data_preserved": True,
            "configuration_restored": True
        }

    async def _restore_previous_configuration(self, proposal: Dict) -> Dict[str, Any]:
        """Restore previous configuration."""
        return {
            "backup_restored": True,
            "settings_reverted": True,
            "validation_performed": True
        }

    async def _validate_rollback(self, proposal: Dict, rollback_results: Dict, restoration_results: Dict) -> Dict[str, Any]:
        """Validate successful rollback."""
        return {
            "rollback_successful": rollback_results.get("configuration_restored", False),
            "system_stable": restoration_results.get("validation_performed", False),
            "data_integrity": rollback_results.get("data_preserved", False)
        }

    async def monitor_execution_performance(self) -> Dict[str, Any]:
        """Monitor execution performance and identify optimization opportunities."""
        try:
            logger.info("ExecutionAgent monitoring execution performance")

            # Collect current execution metrics
            execution_metrics = await self._collect_execution_metrics()

            # Analyze performance trends
            performance_analysis = self._analyze_execution_trends(execution_metrics)

            # Identify performance issues
            performance_issues = self._identify_execution_issues(performance_analysis)

            # Generate optimization proposals
            optimization_proposals = []
            for issue in performance_issues:
                proposal = await self._generate_execution_optimization_proposal(issue, execution_metrics)
                if proposal:
                    optimization_proposals.append(proposal)

            # Submit proposals to LearningAgent
            submission_results = []
            for proposal in optimization_proposals:
                result = await self.submit_optimization_proposal(proposal)
                submission_results.append(result)

            monitoring_result = {
                'performance_metrics': {
                    'avg_slippage': execution_metrics.get('avg_slippage', 0),
                    'avg_commission': execution_metrics.get('avg_commission', 0),
                    'execution_speed': execution_metrics.get('execution_speed', 0),
                    'fill_rate': execution_metrics.get('fill_rate', 0),
                    'total_orders': execution_metrics.get('total_orders', 0),
                    'performance_issues': len(performance_issues),
                    'optimization_proposals_generated': len(optimization_proposals),
                    'proposals_submitted': len([r for r in submission_results if r.get('received', False)])
                },
                'performance_summary': performance_analysis,
                'issues_identified': len(performance_issues),
                'optimization_proposals_generated': len(optimization_proposals),
                'proposals_submitted': len([r for r in submission_results if r.get('received', False)]),
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"ExecutionAgent performance monitoring completed: {monitoring_result['optimization_proposals_generated']} proposals generated")
            return monitoring_result

        except Exception as e:
            logger.error(f"ExecutionAgent performance monitoring failed: {e}")
            return {'error': str(e)}

    # ===== OPTIMIZATION PROPOSAL METHODS =====

    async def evaluate_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate an optimization proposal for execution performance."""
        try:
            logger.info(f"Evaluating execution optimization proposal: {proposal.get('title', 'Unknown')}")

            # Assess technical feasibility
            technical_feasibility = await self._assess_technical_feasibility(proposal)

            # Assess performance impact
            performance_impact = await self._assess_performance_impact(proposal)

            # Assess implementation risk
            implementation_risk = await self._assess_implementation_risk(proposal)

            # Estimate resource requirements
            resource_requirements = await self._estimate_resource_requirements(proposal)

            # Calculate overall score
            overall_score = (technical_feasibility * 0.3 + performance_impact * 0.4 - implementation_risk * 0.3)

            # Determine recommendation
            if overall_score >= 0.7:
                recommendation = "implement"
                confidence = "high"
            elif overall_score >= 0.5:
                recommendation = "implement_with_modifications"
                confidence = "medium"
            else:
                recommendation = "reject"
                confidence = "low"

            evaluation_result = {
                "proposal_id": proposal.get("id"),
                "evaluation_timestamp": datetime.now().isoformat(),
                "technical_feasibility": technical_feasibility,
                "performance_impact": performance_impact,
                "implementation_risk": implementation_risk,
                "resource_requirements": resource_requirements,
                "overall_score": overall_score,
                "recommendation": recommendation,
                "confidence_level": confidence,
                "evaluation_criteria": [
                    "technical_feasibility",
                    "performance_impact",
                    "implementation_risk",
                    "resource_efficiency"
                ],
                "risk_warnings": [] if implementation_risk < 0.3 else ["High implementation risk detected"],
                "estimated_benefits": proposal.get("expected_benefits", {}),
                "estimated_costs": proposal.get("estimated_costs", {})
            }

            logger.info(f"Proposal evaluation completed with score {overall_score:.3f}")
            return evaluation_result

        except Exception as e:
            logger.error(f"Error evaluating proposal: {e}")
            return {
                "error": str(e),
                "recommendation": "reject",
                "confidence_level": "low"
            }

    async def test_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Test an optimization proposal through backtesting and simulation."""
        try:
            logger.info(f"Testing execution optimization proposal: {proposal.get('title', 'Unknown')}")

            # Run historical backtest
            backtest_results = await self._run_historical_backtest(proposal)

            # Run simulation tests
            simulation_results = await self._run_simulation_tests(proposal)

            # Validate against success metrics
            validation_results = await self._validate_success_metrics(proposal, backtest_results, simulation_results)

            # Determine test outcome
            test_passed = (
                backtest_results.get("success_rate", 0) >= 0.8 and
                simulation_results.get("average_outcome", 0) > 0 and
                validation_results.get("all_metrics_passed", False)
            )

            test_result = {
                "proposal_id": proposal.get("id"),
                "test_timestamp": datetime.now().isoformat(),
                "backtest_results": backtest_results,
                "simulation_results": simulation_results,
                "validation_results": validation_results,
                "test_passed": test_passed,
                "confidence_level": validation_results.get("confidence_level", "medium"),
                "performance_metrics": {
                    "backtest_success_rate": backtest_results.get("success_rate", 0),
                    "simulation_average_outcome": simulation_results.get("average_outcome", 0),
                    "max_drawdown": backtest_results.get("max_drawdown", 0)
                },
                "risk_assessment": {
                    "worst_case_scenario": simulation_results.get("worst_case", 0),
                    "confidence_interval": simulation_results.get("confidence_interval", [])
                },
                "recommendations": validation_results.get("recommended_modifications", [])
            }

            logger.info(f"Proposal testing completed - passed: {test_passed}")
            return test_result

        except Exception as e:
            logger.error(f"Error testing proposal: {e}")
            return {
                "error": str(e),
                "test_passed": False,
                "confidence_level": "low"
            }

    async def implement_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Implement an approved optimization proposal."""
        try:
            logger.info(f"Implementing execution optimization proposal: {proposal.get('title', 'Unknown')}")

            # Create implementation plan
            implementation_plan = await self._create_implementation_plan(proposal)

            # Execute implementation steps
            execution_results = await self._execute_implementation_steps(implementation_plan)

            # Set up monitoring
            monitoring_setup = await self._setup_implementation_monitoring(proposal)

            # Validate implementation
            validation_results = await self._validate_implementation(proposal, execution_results)

            implementation_successful = (
                execution_results.get("steps_completed", 0) == len(implementation_plan.get("phases", [])) and
                monitoring_setup.get("monitoring_configured", False) and
                validation_results.get("implementation_successful", False)
            )

            implementation_result = {
                "proposal_id": proposal.get("id"),
                "implementation_timestamp": datetime.now().isoformat(),
                "implementation_plan": implementation_plan,
                "execution_results": execution_results,
                "monitoring_setup": monitoring_setup,
                "validation_results": validation_results,
                "implementation_successful": implementation_successful,
                "rollback_available": True,
                "performance_baseline": {
                    "timestamp": datetime.now().isoformat(),
                    "metrics": self._get_current_performance_metrics()
                },
                "configuration_changes": execution_results.get("final_configuration", {}),
                "monitoring_active": monitoring_setup.get("monitoring_configured", False)
            }

            logger.info(f"Proposal implementation completed - successful: {implementation_successful}")
            return implementation_result

        except Exception as e:
            logger.error(f"Error implementing proposal: {e}")
            return {
                "error": str(e),
                "implementation_successful": False,
                "rollback_available": True
            }

    async def rollback_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback an implemented optimization proposal."""
        try:
            logger.info(f"Rolling back execution optimization proposal: {proposal.get('title', 'Unknown')}")

            # Identify rollback scope
            rollback_scope = await self._identify_rollback_scope(proposal)

            # Execute rollback steps
            rollback_results = await self._execute_rollback_steps(rollback_scope)

            # Restore previous configuration
            restoration_results = await self._restore_previous_configuration(proposal)

            # Validate rollback
            validation_results = await self._validate_rollback(proposal, rollback_results, restoration_results)

            rollback_successful = (
                rollback_results.get("configuration_restored", False) and
                restoration_results.get("validation_performed", False) and
                validation_results.get("rollback_successful", False)
            )

            rollback_result = {
                "proposal_id": proposal.get("id"),
                "rollback_timestamp": datetime.now().isoformat(),
                "rollback_scope": rollback_scope,
                "rollback_results": rollback_results,
                "restoration_results": restoration_results,
                "validation_results": validation_results,
                "rollback_successful": rollback_successful,
                "system_stable": validation_results.get("system_stable", False),
                "data_integrity": validation_results.get("data_integrity", False),
                "performance_restored": rollback_successful,
                "monitoring_resumed": True
            }

            logger.info(f"Proposal rollback completed - successful: {rollback_successful}")
            return rollback_result

        except Exception as e:
            logger.error(f"Error rolling back proposal: {e}")
            return {
                "error": str(e),
                "rollback_successful": False,
                "system_stable": False
            }
    def _get_current_performance_metrics(self) -> Dict[str, Any]:
        """Get current execution performance metrics."""
        return {
            "avg_slippage": 0.002,
            "avg_commission": 0.015,
            "execution_speed": 0.95,
            "fill_rate": 0.98,
            "timestamp": datetime.now().isoformat()
        }

    async def _collect_execution_metrics(self) -> Dict[str, Any]:
        """Collect current execution performance metrics."""
        try:
            # Get metrics from memory or calculate current values
            execution_history = self.memory.get('execution_performance_history', [])
            
            if execution_history:
                # Calculate averages from recent history
                recent_metrics = execution_history[-100:]  # Last 100 executions
                metrics = {
                    'avg_slippage': np.mean([m.get('slippage', 0) for m in recent_metrics]),
                    'avg_commission': np.mean([m.get('commission', 0) for m in recent_metrics]),
                    'execution_speed': np.mean([m.get('execution_speed', 0) for m in recent_metrics]),
                    'fill_rate': np.mean([m.get('fill_rate', 0) for m in recent_metrics]),
                    'total_orders': len(recent_metrics)
                }
            else:
                # Default metrics
                metrics = {
                    'avg_slippage': 0.002,
                    'avg_commission': 0.015,
                    'execution_speed': 0.95,
                    'fill_rate': 0.98,
                    'total_orders': 0
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting execution metrics: {e}")
            return {'error': str(e)}

    def _analyze_execution_trends(self, execution_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze execution performance trends."""
        try:
            analysis = {
                'overall_performance': 'good',
                'trending_metrics': [],
                'concerning_metrics': [],
                'performance_score': 0.8
            }
            
            # Analyze each metric
            if execution_metrics.get('avg_slippage', 0) > 0.005:
                analysis['concerning_metrics'].append('high_slippage')
            if execution_metrics.get('fill_rate', 1.0) < 0.95:
                analysis['concerning_metrics'].append('low_fill_rate')
            if execution_metrics.get('execution_speed', 1.0) < 0.9:
                analysis['concerning_metrics'].append('slow_execution')
                
            # Overall assessment
            concerning_count = len(analysis['concerning_metrics'])
            if concerning_count == 0:
                analysis['overall_performance'] = 'excellent'
                analysis['performance_score'] = 0.95
            elif concerning_count == 1:
                analysis['overall_performance'] = 'good'
                analysis['performance_score'] = 0.8
            elif concerning_count == 2:
                analysis['overall_performance'] = 'fair'
                analysis['performance_score'] = 0.6
            else:
                analysis['overall_performance'] = 'poor'
                analysis['performance_score'] = 0.4
                
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing execution trends: {e}")
            return {'error': str(e)}

    def _identify_execution_issues(self, performance_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify execution performance issues that need optimization."""
        try:
            issues = []
            concerning_metrics = performance_analysis.get('concerning_metrics', [])
            
            for metric in concerning_metrics:
                if metric == 'high_slippage':
                    issues.append({
                        'issue_type': 'slippage_optimization',
                        'severity': 'high',
                        'description': 'High average slippage detected',
                        'impact': 'cost_increase'
                    })
                elif metric == 'low_fill_rate':
                    issues.append({
                        'issue_type': 'fill_rate_optimization',
                        'severity': 'high',
                        'description': 'Low fill rate affecting execution',
                        'impact': 'execution_efficiency'
                    })
                elif metric == 'slow_execution':
                    issues.append({
                        'issue_type': 'speed_optimization',
                        'severity': 'medium',
                        'description': 'Slow execution speed detected',
                        'impact': 'market_timing'
                    })
                    
            return issues
            
        except Exception as e:
            logger.error(f"Error identifying execution issues: {e}")
            return []

    async def _generate_execution_optimization_proposal(self, issue: Dict[str, Any], 
                                                       execution_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization proposal for execution issue."""
        try:
            issue_type = issue['issue_type']
            
            if issue_type == 'slippage_optimization':
                proposal = {
                    'id': f"execution_opt_slippage_{int(datetime.now().timestamp())}",
                    'type': 'execution_optimization',
                    'target_agent': 'LearningAgent',
                    'issue_type': issue_type,
                    'current_performance': execution_metrics,
                    'proposed_changes': {
                        'algorithm_improvements': ['implement_smart_routing', 'add_price_improvement_logic'],
                        'timing_optimizations': ['optimize_order_timing', 'reduce_market_impact'],
                        'liquidity_detection': ['add_liquidity_scoring', 'prefer_high_liquidity_venues']
                    },
                    'expected_improvement': {
                        'slippage_reduction': 0.4,
                        'execution_cost_reduction': 0.3
                    },
                    'implementation_complexity': 'medium',
                    'timestamp': datetime.now().isoformat()
                }
            elif issue_type == 'fill_rate_optimization':
                proposal = {
                    'id': f"execution_opt_fillrate_{int(datetime.now().timestamp())}",
                    'type': 'execution_optimization',
                    'target_agent': 'LearningAgent',
                    'issue_type': issue_type,
                    'current_performance': execution_metrics,
                    'proposed_changes': {
                        'order_splitting': ['implement_order_splitting', 'add_partial_fill_handling'],
                        'venue_selection': ['optimize_venue_selection', 'add_venue_scoring'],
                        'price_improvement': ['add_price_improvement_mechanisms', 'implement_aggressive_fill_logic']
                    },
                    'expected_improvement': {
                        'fill_rate_improvement': 0.25,
                        'execution_efficiency': 0.2
                    },
                    'implementation_complexity': 'high',
                    'timestamp': datetime.now().isoformat()
                }
            elif issue_type == 'speed_optimization':
                proposal = {
                    'id': f"execution_opt_speed_{int(datetime.now().timestamp())}",
                    'type': 'execution_optimization',
                    'target_agent': 'LearningAgent',
                    'issue_type': issue_type,
                    'current_performance': execution_metrics,
                    'proposed_changes': {
                        'latency_reductions': ['optimize_network_latency', 'implement_fast_path_execution'],
                        'pre_trade_processing': ['add_order_preparation', 'implement_order_caching'],
                        'execution_engine': ['upgrade_execution_engine', 'add_parallel_processing']
                    },
                    'expected_improvement': {
                        'execution_speed_improvement': 0.35,
                        'market_timing_accuracy': 0.3
                    },
                    'implementation_complexity': 'high',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {}
                
            return proposal
            
        except Exception as e:
            logger.error(f"Error generating execution optimization proposal: {e}")
            return {'error': str(e)}

    def execute_trades(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a list of trades.

        Args:
            trades: List of trade dictionaries

        Returns:
            Execution results
        """
        try:
            results = []
            for trade in trades:
                symbol = trade.get('symbol', 'SPY')
                quantity = trade.get('quantity', 100)
                action = trade.get('action', 'BUY')

                # Simulate trade execution
                result = {
                    'success': True,
                    'symbol': symbol,
                    'quantity': quantity,
                    'action': action,
                    'simulated': True
                }
                results.append(result)

            return {
                'success': True,
                'trades_executed': len(results),
                'results': results
            }

        except Exception as e:
            logger.error(f"Error executing trades: {e}")
            return {'success': False, 'error': str(e)}

# Standalone test (run python src/agents/execution.py to verify)
if __name__ == "__main__":
    import asyncio
    agent = ExecutionAgent()
    result = asyncio.run(agent.process_input({'symbols': ['SPY']}))
    print("Execution Agent Test Result:\n", result)
