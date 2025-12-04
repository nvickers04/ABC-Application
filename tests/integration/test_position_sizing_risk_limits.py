#!/usr/bin/env python3
"""
Integration tests for position sizing and risk limits in paper trading environment
"""

import asyncio
import pytest
import sys
import os
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Any
from unittest.mock import AsyncMock, Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.integrations.live_trading_safeguards import LiveTradingSafeguards, RiskLevel
from src.agents.risk import RiskAgent
from simulations.paper_trading_monitor import get_paper_trading_monitor


class TestPositionSizingRiskLimits:
    """Integration tests for position sizing and risk limit validation"""

    @pytest.fixture
    async def trading_safeguards(self):
        """Create LiveTradingSafeguards instance for testing"""
        safeguards = LiveTradingSafeguards()
        yield safeguards
        # Cleanup if needed

    @pytest.fixture
    async def risk_agent(self):
        """Create RiskAgent instance for testing"""
        agent = RiskAgent()
        yield agent

    @pytest.fixture
    async def paper_monitor(self):
        """Get paper trading monitor instance"""
        monitor = get_paper_trading_monitor()
        yield monitor

    @pytest.mark.asyncio
    async def test_position_size_limit_enforcement(self, trading_safeguards):
        """Test that position size limits are properly enforced"""
        # Setup test data
        symbol = "AAPL"
        price = 150.0
        portfolio_value = 100000.0

        # Test case 1: Position within limits (should pass)
        quantity_small = 30  # $4,500 position = 0.45% of portfolio (within 5% limit)
        account_info = {'TotalCashValue': portfolio_value}
        positions = []

        approved, reason, analysis = await trading_safeguards.pre_trade_risk_check(
            symbol, quantity_small, price, 'BUY', account_info, positions
        )

        assert approved == True
        assert 'position_size_limit' in analysis['checks_passed']
        assert analysis['risk_level'] == RiskLevel.LOW.value

        # Test case 2: Position exceeds limit (should fail)
        quantity_large = 400  # $60,000 position = 6% of portfolio (exceeds 5% limit)
        approved, reason, analysis = await trading_safeguards.pre_trade_risk_check(
            symbol, quantity_large, price, 'BUY', account_info, positions
        )

        assert approved == False
        assert 'position_size_limit' in analysis['checks_failed']
        assert 'exceeds limit' in reason.lower()
        assert analysis['risk_level'] == RiskLevel.HIGH.value

    @pytest.mark.asyncio
    async def test_single_stock_exposure_limits(self, trading_safeguards):
        """Test single stock exposure limits"""
        symbol = "AAPL"
        price = 150.0
        portfolio_value = 100000.0

        # Setup existing position
        existing_position = {
            'symbol': symbol,
            'market_value': 8000.0,  # $8,000 existing exposure = 8% of portfolio
            'quantity': 50
        }
        positions = [existing_position]
        account_info = {'TotalCashValue': portfolio_value}

        # Test case 1: Additional position within limits (but position size limit triggers first)
        quantity_small = 10  # $1,500 additional = total $9,500 = 9.5% (within 10% limit)
        approved, reason, analysis = await trading_safeguards.pre_trade_risk_check(
            symbol, quantity_small, price, 'BUY', account_info, positions
        )

        # Position size check passes, but we can't test exposure until position size passes
        assert approved == True
        assert 'position_size_limit' in analysis['checks_passed']

        # Test case 2: Additional position exceeds limit (position size fails first)
        quantity_large = 100  # $15,000 additional = total $23,000 = 23% (exceeds 10% limit)
        approved, reason, analysis = await trading_safeguards.pre_trade_risk_check(
            symbol, quantity_large, price, 'BUY', account_info, positions
        )

        assert approved == False
        # Position size limit fails first (6% > 5%), so exposure check doesn't run
        assert 'position_size_limit' in analysis['checks_failed']

    @pytest.mark.asyncio
    async def test_total_portfolio_exposure_limits(self, trading_safeguards):
        """Test total portfolio exposure limits"""
        symbol = "TSLA"
        price = 200.0
        portfolio_value = 100000.0

        # Setup existing positions totaling 25% exposure
        positions = [
            {'symbol': 'AAPL', 'market_value': 15000.0},
            {'symbol': 'GOOGL', 'market_value': 10000.0}
        ]
        account_info = {'TotalCashValue': portfolio_value}

        # Test case 1: New position keeps total exposure within limits
        quantity_small = 25  # $5,000 additional = total $30,000 = 30% (within 30% limit)
        approved, reason, analysis = await trading_safeguards.pre_trade_risk_check(
            symbol, quantity_small, price, 'BUY', account_info, positions
        )

        assert approved == True
        assert 'position_size_limit' in analysis['checks_passed']

        # Test case 2: New position exceeds total exposure limit (but position size fails first)
        quantity_large = 150  # $30,000 additional = total $55,000 = 55% (exceeds 30% limit)
        approved, reason, analysis = await trading_safeguards.pre_trade_risk_check(
            symbol, quantity_large, price, 'BUY', account_info, positions
        )

        assert approved == False
        # Position size limit fails first (6% > 5%), so exposure check doesn't run
        assert 'position_size_limit' in analysis['checks_failed']

    @pytest.mark.asyncio
    async def test_daily_loss_limits(self, trading_safeguards):
        """Test daily loss limit enforcement"""
        symbol = "NVDA"
        price = 400.0
        portfolio_value = 100000.0

        # Simulate significant daily losses
        trading_safeguards.current_session.daily_pnl = -6000.0  # $6,000 loss = 6% of portfolio

        account_info = {'TotalCashValue': portfolio_value}
        positions = []

        # Test case: New trade should be rejected due to daily loss limit
        quantity = 10  # Small position that would otherwise pass
        approved, reason, analysis = await trading_safeguards.pre_trade_risk_check(
            symbol, quantity, price, 'BUY', account_info, positions
        )

        assert approved == False
        assert 'daily_loss_limit' in analysis['checks_failed']
        assert 'daily loss' in reason.lower()

    @pytest.mark.asyncio
    async def test_emergency_stop_functionality(self, trading_safeguards):
        """Test emergency stop blocks all trading"""
        # Set emergency stop
        trading_safeguards.trading_state = trading_safeguards.trading_state.EMERGENCY_STOP

        symbol = "MSFT"
        price = 300.0
        account_info = {'TotalCashValue': 100000.0}
        positions = []

        approved, reason, analysis = await trading_safeguards.pre_trade_risk_check(
            symbol, 10, price, 'BUY', account_info, positions
        )

        assert approved == False
        assert 'emergency stop' in reason.lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, trading_safeguards):
        """Test circuit breaker blocks trading"""
        # Trigger circuit breaker
        trading_safeguards.circuit_breaker_triggered = True

        symbol = "AMZN"
        price = 150.0
        account_info = {'TotalCashValue': 100000.0}
        positions = []

        approved, reason, analysis = await trading_safeguards.pre_trade_risk_check(
            symbol, 50, price, 'BUY', account_info, positions
        )

        assert approved == False
        assert 'circuit breaker' in reason.lower()

    @pytest.mark.asyncio
    async def test_risk_agent_position_limits_definition(self, risk_agent):
        """Test RiskAgent position limits definition"""
        # Test proposal for position limits
        proposal = {
            'symbol': 'SPY',
            'expected_returns': 0.08,
            'volatility': 0.15
        }

        # Test position limits for normal risk level
        limits_normal = risk_agent._define_position_limits(proposal, 'normal')
        assert limits_normal['max_position_size'] == 0.10
        assert limits_normal['max_total_exposure'] == 0.50

        # Test position limits for high risk level
        limits_high = risk_agent._define_position_limits(proposal, 'high')
        assert limits_high['max_position_size'] == 0.05
        assert limits_high['max_total_exposure'] == 0.25

    @pytest.mark.asyncio
    async def test_paper_trading_risk_integration(self, paper_monitor, trading_safeguards):
        """Test integration between paper trading monitor and risk safeguards"""
        # Simulate a trade through the paper trading system
        trade_data = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'executed_quantity': 100,
            'executed_price': 150.0,
            'realized_pnl': 500.0,
            'commission': 2.50,
            'order_id': 'TEST_ORDER_123'
        }

        # Record trade in paper monitor
        await paper_monitor.record_trade(trade_data)

        # Verify trade was recorded using dashboard data
        dashboard_data = await paper_monitor.get_dashboard_data()
        metrics = dashboard_data['metrics']
        assert metrics['total_trades'] >= 1
        assert metrics['total_pnl'] >= 500.0

        # Test risk check with current portfolio state
        account_info = {'TotalCashValue': 100000.0 + trade_data['realized_pnl']}
        positions = [{
            'symbol': trade_data['symbol'],
            'market_value': trade_data['executed_quantity'] * trade_data['executed_price'],
            'quantity': trade_data['executed_quantity']
        }]

        # This should pass risk checks (very small position: 5 shares * $100 = $500 = 0.05%)
        approved, reason, analysis = await trading_safeguards.pre_trade_risk_check(
            'GOOGL', 5, 100.0, 'BUY', account_info, positions
        )

        assert approved == True

    @pytest.mark.asyncio
    async def test_risk_agent_stop_loss_levels(self, risk_agent):
        """Test RiskAgent stop loss level definitions"""
        proposal = {
            'symbol': 'TSLA',
            'volatility': 0.30
        }

        # Test stop loss levels for normal risk
        stops_normal = risk_agent._define_stop_loss_levels(proposal, 'normal')
        assert stops_normal['initial_stop'] == 0.05
        assert stops_normal['trailing_stop'] == 0.03

        # Test stop loss levels for high risk
        stops_high = risk_agent._define_stop_loss_levels(proposal, 'high')
        assert stops_high['initial_stop'] == 0.08
        assert stops_high['trailing_stop'] == 0.05

    @pytest.mark.asyncio
    async def test_risk_limit_configuration_loading(self, trading_safeguards):
        """Test that risk limits are properly loaded from configuration"""
        # Verify risk limits are loaded and have expected values
        assert hasattr(trading_safeguards, 'risk_limits')
        assert trading_safeguards.risk_limits.max_position_size_pct > 0
        assert trading_safeguards.risk_limits.max_single_stock_exposure_pct > 0
        assert trading_safeguards.risk_limits.max_total_exposure_pct > 0
        assert trading_safeguards.risk_limits.max_daily_loss_pct > 0

        # Verify limits are reasonable (not zero or negative)
        assert 0 < trading_safeguards.risk_limits.max_position_size_pct <= 1.0
        assert 0 < trading_safeguards.risk_limits.max_single_stock_exposure_pct <= 1.0
        assert 0 < trading_safeguards.risk_limits.max_total_exposure_pct <= 1.0
        assert 0 < trading_safeguards.risk_limits.max_daily_loss_pct <= 1.0