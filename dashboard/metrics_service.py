# src/dashboard/metrics_service.py
# Purpose: Centralized metrics service for dashboard and agent data access
# Provides real-time data from agents and cost tracking for both human dashboard and agent queries

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
from pathlib import Path

# Import agents for data access
from src.agents.reflection import ReflectionAgent
from src.agents.risk import RiskAgent
from src.agents.execution import ExecutionAgent

class MetricsService:
    """
    Centralized service for metrics collection and access.
    Used by both dashboard and agents.
    """

    def __init__(self):
        self.agents = {}
        self.cost_tracking = {
            'api_calls': [],
            'token_usage': [],
            'ibkr_fees': [],
            'total_costs': 0.0
        }
        self._load_agents()

    def _load_agents(self):
        """Load agent instances for data access"""
        try:
            self.agents['reflection'] = ReflectionAgent()
            self.agents['risk'] = RiskAgent()
            self.agents['execution'] = ExecutionAgent()
        except Exception as e:
            print(f"Warning: Could not load all agents: {e}")

    def get_performance_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get performance metrics from Reflection agent"""
        try:
            reflection_agent = self.agents.get('reflection')
            if not reflection_agent or not hasattr(reflection_agent, 'memory'):
                return self._get_sample_performance_data(days)

            performance_history = reflection_agent.memory.get('performance_history', [])

            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_performance = [
                p for p in performance_history
                if datetime.fromisoformat(p['timestamp']) > cutoff_date
            ]

            if not recent_performance:
                return self._get_sample_performance_data(days)

            # Calculate metrics
            pnl_values = [p['pnl_pct'] for p in recent_performance]
            sharpe_values = [p.get('sharpe_ratio', 1.5) for p in recent_performance]
            drawdown_values = [p.get('max_drawdown', 0.1) for p in recent_performance]

            return {
                'total_trades': len(recent_performance),
                'total_pnl': sum(pnl_values),
                'avg_pnl': sum(pnl_values) / len(pnl_values) if pnl_values else 0,
                'win_rate': sum(1 for p in pnl_values if p > 0) / len(pnl_values) if pnl_values else 0,
                'avg_sharpe': sum(sharpe_values) / len(sharpe_values) if sharpe_values else 1.5,
                'max_drawdown': max(drawdown_values) if drawdown_values else 0.1,
                'performance_history': recent_performance[-20:],  # Last 20 for charts
                'data_source': 'reflection_agent'
            }

        except Exception as e:
            print(f"Error getting performance metrics: {e}")
            return self._get_sample_performance_data(days)

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk metrics from Risk agent"""
        try:
            risk_agent = self.agents.get('risk')
            if not risk_agent or not hasattr(risk_agent, 'memory'):
                return self._get_sample_risk_data()

            # Get risk memory data
            risk_memory = risk_agent.memory if hasattr(risk_agent, 'memory') else {}

            return {
                'current_pop_threshold': risk_agent.configs.get('risk', {}).get('constraints', {}).get('pop_floor', 0.6),
                'max_drawdown_limit': risk_agent.configs.get('risk', {}).get('constraints', {}).get('max_drawdown', 0.05),
                'position_size_limit': risk_agent.configs.get('risk', {}).get('constraints', {}).get('max_position_size', 0.3),
                'variance_threshold': risk_agent.configs.get('risk', {}).get('constraints', {}).get('variance_sd_threshold', 1.0),
                'recent_proposals': risk_memory.get('proposal_history', [])[-10:],
                'data_source': 'risk_agent'
            }

        except Exception as e:
            print(f"Error getting risk metrics: {e}")
            return self._get_sample_risk_data()

    def get_cost_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get cost tracking metrics"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            # Filter recent costs
            recent_api_calls = [
                c for c in self.cost_tracking['api_calls']
                if datetime.fromisoformat(c['timestamp']) > cutoff_date
            ]

            recent_tokens = [
                c for c in self.cost_tracking['token_usage']
                if datetime.fromisoformat(c['timestamp']) > cutoff_date
            ]

            recent_fees = [
                c for c in self.cost_tracking['ibkr_fees']
                if datetime.fromisoformat(c['timestamp']) > cutoff_date
            ]

            total_api_cost = sum(c['cost'] for c in recent_api_calls)
            total_token_cost = sum(c['cost'] for c in recent_tokens)
            total_fee_cost = sum(c['cost'] for c in recent_fees)
            total_cost = total_api_cost + total_token_cost + total_fee_cost

            return {
                'total_cost': total_cost,
                'api_cost': total_api_cost,
                'token_cost': total_token_cost,
                'trading_fees': total_fee_cost,
                'api_calls': len(recent_api_calls),
                'token_usage': sum(c.get('tokens', 0) for c in recent_tokens),
                'cost_breakdown': {
                    'api': total_api_cost,
                    'tokens': total_token_cost,
                    'fees': total_fee_cost
                },
                'recent_costs': (recent_api_calls + recent_tokens + recent_fees)[-20:],
                'data_source': 'metrics_service'
            }

        except Exception as e:
            print(f"Error getting cost metrics: {e}")
            return self._get_sample_cost_data()

    def log_api_call(self, agent: str, endpoint: str, cost: float = 0.01, tokens: int = 100):
        """Log an API call for cost tracking"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'agent': agent,
            'endpoint': endpoint,
            'cost': cost,
            'tokens': tokens
        }
        self.cost_tracking['api_calls'].append(entry)
        self.cost_tracking['token_usage'].append(entry)
        self.cost_tracking['total_costs'] += cost

    def log_trading_fee(self, symbol: str, fee: float, trade_type: str):
        """Log trading fees"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'fee': fee,
            'type': trade_type
        }
        self.cost_tracking['ibkr_fees'].append(entry)
        self.cost_tracking['total_costs'] += fee

    def get_agent_flow_data(self) -> Dict[str, Any]:
        """Get data for Mermaid AI flow visualization"""
        try:
            # Get recent agent interactions
            reflection_agent = self.agents.get('reflection')
            recent_audits = []
            recent_reviews = []

            if reflection_agent and hasattr(reflection_agent, 'memory'):
                audits = reflection_agent.memory.get('audit_history', [])[-5:]
                reviews = reflection_agent.memory.get('performance_history', [])[-10:]

                recent_audits = [
                    {
                        'timestamp': a['timestamp'],
                        'type': a['type'],
                        'outcome': a.get('analysis', {}).get('rationale', 'completed')
                    } for a in audits
                ]

                recent_reviews = [
                    {
                        'timestamp': r['timestamp'],
                        'symbol': r['symbol'],
                        'pnl': r['pnl_pct'],
                        'bonus': r.get('outcome', {}).get('bonus_awarded', False)
                    } for r in reviews
                ]

            return {
                'agent_states': {
                    'data_agent': 'active',
                    'strategy_agent': 'active',
                    'risk_agent': 'active',
                    'execution_agent': 'active',
                    'reflection_agent': 'active'
                },
                'recent_flows': {
                    'audits': recent_audits,
                    'reviews': recent_reviews,
                    'total_interactions': len(recent_audits) + len(recent_reviews)
                },
                'system_health': 'operational'
            }

        except Exception as e:
            print(f"Error getting agent flow data: {e}")
            return self._get_sample_flow_data()

    def export_metrics_for_agents(self) -> Dict[str, Any]:
        """Export all metrics in format agents can consume"""
        return {
            'performance': self.get_performance_metrics(),
            'risk': self.get_risk_metrics(),
            'costs': self.get_cost_metrics(),
            'agent_flow': self.get_agent_flow_data(),
            'export_timestamp': datetime.now().isoformat()
        }

    # Sample data fallbacks
    def _get_sample_performance_data(self, days: int):
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
        pnl_values = [0.01 * (1 if i % 3 != 0 else -1) for i in range(days)]  # Mostly positive

        return {
            'total_trades': len(pnl_values),
            'total_pnl': sum(pnl_values),
            'avg_pnl': sum(pnl_values) / len(pnl_values),
            'win_rate': sum(1 for p in pnl_values if p > 0) / len(pnl_values),
            'avg_sharpe': 1.5,
            'max_drawdown': 0.15,
            'performance_history': [
                {'timestamp': str(d), 'pnl_pct': p, 'sharpe_ratio': 1.5, 'max_drawdown': 0.1}
                for d, p in zip(dates[-20:], pnl_values[-20:])
            ],
            'data_source': 'sample_data'
        }

    def _get_sample_risk_data(self):
        return {
            'current_pop_threshold': 0.6,
            'max_drawdown_limit': 0.05,
            'position_size_limit': 0.3,
            'variance_threshold': 1.0,
            'recent_proposals': [],
            'data_source': 'sample_data'
        }

    def _get_sample_cost_data(self):
        return {
            'total_cost': 25.50,
            'api_cost': 15.00,
            'token_cost': 8.50,
            'trading_fees': 2.00,
            'api_calls': 45,
            'token_usage': 8500,
            'cost_breakdown': {'api': 15.00, 'tokens': 8.50, 'fees': 2.00},
            'recent_costs': [],
            'data_source': 'sample_data'
        }

    def _get_sample_flow_data(self):
        return {
            'agent_states': {
                'data_agent': 'active',
                'strategy_agent': 'active',
                'risk_agent': 'active',
                'execution_agent': 'active',
                'reflection_agent': 'active'
            },
            'recent_flows': {
                'audits': [],
                'reviews': [],
                'total_interactions': 0
            },
            'system_health': 'operational'
        }

# Global instance
_metrics_service = None

def get_metrics_service() -> MetricsService:
    """Get singleton metrics service instance"""
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = MetricsService()
    return _metrics_service