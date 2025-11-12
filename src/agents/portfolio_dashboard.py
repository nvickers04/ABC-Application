# src/agents/portfolio_dashboard.py
# Portfolio Dashboard Agent for monitoring system performance and generating alerts
# Implements a class-based dashboard for portfolio management and risk monitoring

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
from unittest.mock import Mock
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.base import BaseAgent
from src.utils.tools import institutional_holdings_analysis_tool
from src.utils.advanced_memory import get_advanced_memory_manager

class PortfolioDashboard(BaseAgent):
    """
    Portfolio Dashboard Agent for monitoring and visualization.
    Provides comprehensive portfolio analytics, risk assessment, and alert generation.
    """

    def __init__(self):
        """Initialize the Portfolio Dashboard agent."""
        self.logger = logging.getLogger(__name__)
        config_paths = {
            'risk': 'config/risk-constraints.yaml',
            'profit': 'config/profitability-targets.yaml'
        }
        prompt_paths = {
            'base': 'base_prompt.txt',
            'role': 'agents/portfolio-dashboard.txt'
        }
        tools = []

        super().__init__(
            role='portfolio_dashboard',
            config_paths=config_paths,
            prompt_paths=prompt_paths,
            tools=tools
        )

        # Initialize memory manager
        try:
            self.memory_manager = get_advanced_memory_manager()
        except Exception:
            self.memory_manager = Mock()  # Mock for testing

        # Load configurations
        self.risk_config = self.load_risk_constraints()
        self.profit_config = self.load_profit_targets()
        self.config = {
            'risk': self.risk_config,
            'profit': self.profit_config
        }

    def load_risk_constraints(self) -> Dict[str, Any]:
        """Load risk constraints from YAML configuration."""
        try:
            config_path = os.path.join(project_root, 'config', 'risk-constraints.yaml')
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Warning: Could not load risk constraints: {e}")
            return {"constraints": {"max_drawdown": 0.05, "max_position_size": 0.1}}

    def load_profit_targets(self) -> Dict[str, Any]:
        """Load profit targets from YAML configuration."""
        try:
            config_path = os.path.join(project_root, 'config', 'profitability-targets.yaml')
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Warning: Could not load profit targets: {e}")
            return {"targets": {"roi_monthly": "10-15%", "roi_annual": "120-150%", "sharpe_min": 1.5}}

    async def generate_sample_data(self) -> pd.DataFrame:
        """Generate portfolio data from real shared memory data."""
        try:
            # Try to get real portfolio data from shared memory
            portfolio_data = await self.retrieve_shared_memory("portfolio", "current_state")
            trade_history = await self.retrieve_shared_memory("portfolio", "trade_history")

            if portfolio_data and trade_history:
                # Use real data
                dates = pd.to_datetime([trade['timestamp'] for trade in trade_history[-30:]])  # Last 30 trades
                roi_values = []
                cumulative_roi = 0

                for trade in trade_history[-30:]:
                    pnl = trade.get('pnl', 0)
                    cumulative_roi += pnl
                    roi_values.append(cumulative_roi)

                if len(roi_values) == 0:
                    roi_values = [0] * 30
                    dates = pd.date_range(start='2025-01-01', periods=30, freq='D')

                # Calculate drawdown
                roi_series = pd.Series(roi_values)
                drawdown = (roi_series - roi_series.cummax()) / (roi_series.cummax() + 1e-8)

                # Calculate returns and Sharpe
                returns = roi_series.pct_change().fillna(0)
                sharpe = returns.rolling(window=len(returns), min_periods=1).mean() / returns.rolling(window=len(returns), min_periods=1).std()
                sharpe = sharpe.fillna(0)

                # Estimate drag (transaction costs) from trade data
                drag_values = [abs(trade.get('commission', 0.001)) for trade in trade_history[-30:]]
                if len(drag_values) < 30:
                    drag_values.extend([0.001] * (30 - len(drag_values)))

                data = pd.DataFrame({
                    'Date': dates,
                    'ROI': roi_values,
                    'Drawdown': drawdown,
                    'Sharpe': sharpe,
                    'Drag': drag_values
                })
                return data

        except Exception as e:
            self.logger.warning(f"Failed to retrieve real portfolio data: {e}")

        # Fallback to neutral/default data instead of synthetic
        dates = pd.date_range(start='2025-01-01', periods=30, freq='D')
        roi = [0] * 30  # Neutral ROI
        drawdown = [0] * 30  # No drawdown
        returns = pd.Series(roi).pct_change().fillna(0)
        sharpe = [0] * 30  # Neutral Sharpe
        drag = [0.001] * 30  # Minimal drag

        data = pd.DataFrame({
            'Date': dates,
            'ROI': roi,
            'Drawdown': drawdown,
            'Sharpe': sharpe,
            'Drag': drag
        })
        return data

    def calculate_performance_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate key performance metrics from portfolio data."""
        metrics = {
            'total_roi': data['ROI'].iloc[-1] if len(data) > 0 else 0,
            'max_drawdown': data['Drawdown'].min() if len(data) > 0 else 0,
            'avg_sharpe': data['Sharpe'].mean() if len(data) > 0 else 0,
            'volatility': data['ROI'].std() if len(data) > 0 else 0,
            'win_rate': (data['ROI'].pct_change() > 0).mean() if len(data) > 1 else 0
        }
        return metrics

    def assess_risk_levels(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess current risk levels against constraints."""
        breaches = []
        risk_level = "low"

        current_drawdown = portfolio_data.get('current_drawdown', 0)
        max_drawdown_limit = portfolio_data.get('max_drawdown_limit', 0.05)

        if current_drawdown < -max_drawdown_limit:
            breaches.append(f"Drawdown {current_drawdown:.1%} exceeds limit of {max_drawdown_limit:.1%}")
            risk_level = "high"

        current_positions = portfolio_data.get('current_positions', 0)
        max_positions_limit = portfolio_data.get('max_positions_limit', 10)

        if current_positions > max_positions_limit:
            breaches.append(f"Position count {current_positions} exceeds limit of {max_positions_limit}")
            risk_level = "high"

        return {
            'risk_level': risk_level,
            'breaches': breaches
        }

    def generate_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on performance metrics."""
        alerts = []

        current_drawdown = metrics.get('current_drawdown', 0)
        max_drawdown_limit = metrics.get('max_drawdown_limit', 0.05)

        if current_drawdown < -max_drawdown_limit:
            alerts.append({
                'type': 'critical',
                'message': f"Drawdown {current_drawdown:.1%} exceeds limit of {max_drawdown_limit:.1%}",
                'severity': 'critical'
            })

        sharpe_ratio = metrics.get('sharpe_ratio', 1.0)
        min_sharpe = metrics.get('min_sharpe', 1.0)

        if sharpe_ratio < min_sharpe:
            alerts.append({
                'type': 'warning',
                'message': f"Sharpe ratio {sharpe_ratio:.2f} below minimum of {min_sharpe:.2f}",
                'severity': 'warning'
            })

        return alerts

    def analyze_institutional_holdings(self, holdings_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze institutional holdings data."""
        symbol = holdings_data.get('symbol', 'UNKNOWN')
        institutional_ownership = holdings_data.get('institutional_ownership', 0)

        analysis = {
            'symbol': symbol,
            'ownership_trends': 'stable',
            'key_investors': ['Vanguard', 'BlackRock'],
            'concentration_risk': 'low' if institutional_ownership < 0.7 else 'high'
        }

        return analysis

    def get_memory_insights(self) -> Dict[str, Any]:
        """Retrieve insights from memory system."""
        try:
            # This would integrate with the actual memory system
            return {
                'recent_trades': [],
                'performance_history': [],
                'risk_metrics': {}
            }
        except Exception:
            return {}

    async def process_input(self, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process dashboard input and generate comprehensive analysis."""
        # Generate sample data for demonstration
        data = await self.generate_sample_data()
        metrics = self.calculate_performance_metrics(data)

        # Assess risk levels
        portfolio_data = {
            'current_drawdown': metrics['max_drawdown'],
            'max_drawdown_limit': self.risk_config['constraints']['max_drawdown'],
            'current_positions': 5,
            'max_positions_limit': 10
        }
        risk_assessment = self.assess_risk_levels(portfolio_data)

        # Generate alerts
        alert_metrics = {
            'current_drawdown': metrics['max_drawdown'],
            'max_drawdown_limit': self.risk_config['constraints']['max_drawdown'],
            'sharpe_ratio': metrics['avg_sharpe'],
            'min_sharpe': self.profit_config['targets']['sharpe_min']
        }
        alerts = self.generate_alerts(alert_metrics)

        return {
            'performance_metrics': metrics,
            'risk_assessment': risk_assessment,
            'alerts': alerts,
            'data': data.to_dict('records')
        }

    def reflect(self, adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """Reflect on performance and generate improvement suggestions."""
        return {
            'insights': ['Monitor drawdown closely', 'Consider rebalancing'],
            'adjustments': adjustments
        }
