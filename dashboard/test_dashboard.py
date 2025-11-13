#!/usr/bin/env python3
"""
Unit tests for dashboard components.
Tests portfolio dashboard and enhanced dashboard functionality.
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import dashboard components
from src.agents.portfolio_dashboard import PortfolioDashboard
# from src.agents.portfolio_dashboard_enhanced import PortfolioDashboardEnhanced


class TestPortfolioDashboard:
    """Test cases for PortfolioDashboard functionality."""

    @pytest.fixture
    def portfolio_dashboard(self):
        """Create a PortfolioDashboard instance for testing."""
        dashboard = PortfolioDashboard()
        return dashboard

    def test_initialization(self, portfolio_dashboard):
        """Test PortfolioDashboard initialization."""
        assert hasattr(portfolio_dashboard, 'memory_manager')
        assert hasattr(portfolio_dashboard, 'config')

    @patch('src.agents.portfolio_dashboard.load_yaml')
    def test_configuration_loading(self, mock_load_yaml, portfolio_dashboard):
        """Test configuration loading from YAML files."""
        mock_risk_config = {"max_drawdown": 0.05, "max_position_size": 0.1}
        mock_profit_config = {"target_roi": 0.15, "min_sharpe": 1.5}

        mock_load_yaml.side_effect = [mock_risk_config, mock_profit_config]

        # Test config loading
        risk_config = portfolio_dashboard.load_risk_constraints()
        profit_config = portfolio_dashboard.load_profit_targets()

        assert risk_config["max_drawdown"] == 0.05
        assert profit_config["target_roi"] == 0.15

    def test_sample_data_generation(self, portfolio_dashboard):
        """Test sample data generation functionality."""
        data = portfolio_dashboard.generate_sample_data()

        assert isinstance(data, pd.DataFrame)
        assert 'Date' in data.columns
        assert 'ROI' in data.columns
        assert 'Drawdown' in data.columns
        assert 'Sharpe' in data.columns
        assert len(data) > 0

    def test_performance_metrics_calculation(self, portfolio_dashboard):
        """Test performance metrics calculation."""
        # Create sample portfolio data
        sample_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=30),
            'ROI': np.cumsum(np.random.normal(0.005, 0.01, 30)),
            'Drawdown': np.random.uniform(-0.05, 0.0, 30),
            'Sharpe': np.random.normal(1.5, 0.5, 30)
        })

        metrics = portfolio_dashboard.calculate_performance_metrics(sample_data)

        assert isinstance(metrics, dict)
        assert 'total_roi' in metrics
        assert 'max_drawdown' in metrics
        assert 'avg_sharpe' in metrics
        assert 'volatility' in metrics

    def test_risk_assessment(self, portfolio_dashboard):
        """Test risk assessment functionality."""
        portfolio_data = {
            "current_drawdown": 0.03,
            "max_drawdown_limit": 0.05,
            "current_positions": 5,
            "max_positions_limit": 10
        }

        risk_assessment = portfolio_dashboard.assess_risk_levels(portfolio_data)

        assert isinstance(risk_assessment, dict)
        assert 'risk_level' in risk_assessment
        assert 'breaches' in risk_assessment

    def test_alert_generation(self, portfolio_dashboard):
        """Test alert generation functionality."""
        metrics = {
            "current_drawdown": 0.06,
            "max_drawdown_limit": 0.05,
            "sharpe_ratio": 0.8,
            "min_sharpe": 1.0
        }

        alerts = portfolio_dashboard.generate_alerts(metrics)

        assert isinstance(alerts, list)
        # Should generate alerts for breaches
        assert len(alerts) > 0

    @patch('src.agents.portfolio_dashboard.institutional_holdings_analysis_tool')
    def test_institutional_holdings_analysis(self, mock_institutional_tool, portfolio_dashboard):
        """Test institutional holdings analysis."""
        mock_institutional_tool.return_value = {
            "ownership_trends": "increasing",
            "key_investors": ["Vanguard", "BlackRock"],
            "concentration_risk": "low"
        }

        holdings_data = {"symbol": "AAPL", "institutional_ownership": 0.62}

        analysis = portfolio_dashboard.analyze_institutional_holdings(holdings_data)

        assert isinstance(analysis, dict)
        assert "ownership_trends" in analysis

    def test_memory_integration(self, portfolio_dashboard):
        """Test memory system integration."""
        with patch('src.agents.portfolio_dashboard.get_advanced_memory_manager') as mock_memory:
            mock_memory_manager = Mock()
            mock_memory.return_value = mock_memory_manager

            # Test memory retrieval
            memory_data = portfolio_dashboard.get_memory_insights()
            assert isinstance(memory_data, dict)


# class TestPortfolioDashboardEnhanced:
#     """Test cases for PortfolioDashboardEnhanced functionality."""

#     @pytest.fixture
#     def enhanced_dashboard(self):
#         """Create a PortfolioDashboardEnhanced instance for testing."""
#         dashboard = PortfolioDashboardEnhanced()
#         return dashboard

    def test_initialization(self, portfolio_dashboard):
        """Test PortfolioDashboard initialization."""
        assert hasattr(portfolio_dashboard, 'memory_manager')
        assert hasattr(portfolio_dashboard, 'config')

    @patch('src.agents.portfolio_dashboard_enhanced.load_yaml')
    def test_enhanced_configuration_loading(self, mock_load_yaml, enhanced_dashboard):
        """Test enhanced configuration loading."""
        mock_config = {
            "dashboard": {
                "update_interval": 30,
                "alert_thresholds": {"high": 0.8, "critical": 0.95}
            },
            "visualization": {
                "theme": "dark",
                "charts": ["performance", "risk", "holdings"]
            }
        }

        mock_load_yaml.return_value = mock_config

        config = enhanced_dashboard.load_dashboard_config()
        assert config["dashboard"]["update_interval"] == 30
        assert config["visualization"]["theme"] == "dark"

    def test_real_time_data_processing(self, enhanced_dashboard):
        """Test real-time data processing functionality."""
        real_time_data = {
            "timestamp": datetime.now(),
            "portfolio_value": 125000.50,
            "daily_pnl": 1250.75,
            "positions": [
                {"symbol": "AAPL", "quantity": 100, "current_price": 152.50},
                {"symbol": "GOOGL", "quantity": 25, "current_price": 2850.00}
            ]
        }

        processed_data = enhanced_dashboard.process_real_time_data(real_time_data)

        assert isinstance(processed_data, dict)
        assert "processed_timestamp" in processed_data
        assert "total_value" in processed_data
        assert "position_count" in processed_data

    def test_advanced_analytics(self, enhanced_dashboard):
        """Test advanced analytics functionality."""
        historical_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'returns': np.random.normal(0.001, 0.02, 100),
            'volume': np.random.randint(100000, 1000000, 100)
        })

        analytics = enhanced_dashboard.calculate_advanced_analytics(historical_data)

        assert isinstance(analytics, dict)
        assert 'volatility' in analytics
        assert 'skewness' in analytics
        assert 'kurtosis' in analytics
        assert 'var_95' in analytics

    def test_predictive_modeling(self, enhanced_dashboard):
        """Test predictive modeling functionality."""
        market_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=50),
            'price': np.cumsum(np.random.normal(0.001, 0.02, 50)) + 100,
            'volume': np.random.randint(500000, 2000000, 50),
            'sentiment': np.random.uniform(-1, 1, 50)
        })

        predictions = enhanced_dashboard.generate_predictions(market_data)

        assert isinstance(predictions, dict)
        assert 'predicted_price' in predictions
        assert 'confidence_interval' in predictions
        assert 'prediction_horizon' in predictions

    def test_scenario_analysis(self, enhanced_dashboard):
        """Test scenario analysis functionality."""
        base_portfolio = {
            "positions": [
                {"symbol": "AAPL", "weight": 0.4, "current_price": 150.0},
                {"symbol": "GOOGL", "weight": 0.3, "current_price": 2800.0},
                {"symbol": "MSFT", "weight": 0.3, "current_price": 300.0}
            ],
            "total_value": 100000.0
        }

        scenarios = ["bull_market", "bear_market", "high_volatility", "recession"]

        analysis_results = enhanced_dashboard.run_scenario_analysis(base_portfolio, scenarios)

        assert isinstance(analysis_results, dict)
        for scenario in scenarios:
            assert scenario in analysis_results
            assert 'expected_return' in analysis_results[scenario]
            assert 'expected_risk' in analysis_results[scenario]

    def test_alert_system_enhanced(self, enhanced_dashboard):
        """Test enhanced alert system."""
        system_status = {
            "api_health": {
                "grok_api": "healthy",
                "ibkr_connection": "degraded",
                "news_api": "healthy"
            },
            "performance_metrics": {
                "sharpe_ratio": 0.8,
                "max_drawdown": 0.06,
                "win_rate": 0.55
            },
            "system_resources": {
                "memory_usage": 0.85,
                "cpu_usage": 0.75,
                "disk_space": 0.60
            }
        }

        alerts = enhanced_dashboard.generate_enhanced_alerts(system_status)

        assert isinstance(alerts, list)
        # Should detect IBKR connection issue and high memory usage
        critical_alerts = [alert for alert in alerts if alert.get('severity') == 'critical']
        assert len(critical_alerts) > 0

    def test_collaborative_insights(self, enhanced_dashboard):
        """Test collaborative insights generation."""
        agent_insights = {
            "strategy_agent": {
                "signal_strength": 0.8,
                "recommended_action": "hold",
                "confidence": 0.75
            },
            "risk_agent": {
                "risk_score": 0.3,
                "approval_status": "approved",
                "concerns": ["volatility"]
            },
            "learning_agent": {
                "adaptation_needed": True,
                "improvement_areas": ["timing", "position_sizing"]
            }
        }

        collaborative_insights = enhanced_dashboard.generate_collaborative_insights(agent_insights)

        assert isinstance(collaborative_insights, dict)
        assert 'consensus_signal' in collaborative_insights
        assert 'risk_adjusted_recommendation' in collaborative_insights
        assert 'learning_opportunities' in collaborative_insights

    def test_visualization_data_preparation(self, enhanced_dashboard):
        """Test visualization data preparation."""
        raw_data = {
            "performance": pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=30),
                'portfolio_value': np.cumsum(np.random.normal(1000, 500, 30)) + 100000,
                'benchmark': np.cumsum(np.random.normal(800, 400, 30)) + 100000
            }),
            "risk_metrics": {
                "var_95": 0.05,
                "expected_shortfall": 0.08,
                "beta": 1.2
            },
            "holdings": [
                {"symbol": "AAPL", "weight": 0.35, "sector": "Technology"},
                {"symbol": "JPM", "weight": 0.25, "sector": "Financials"},
                {"symbol": "JNJ", "weight": 0.20, "sector": "Healthcare"}
            ]
        }

        viz_data = enhanced_dashboard.prepare_visualization_data(raw_data)

        assert isinstance(viz_data, dict)
        assert 'performance_chart_data' in viz_data
        assert 'risk_gauge_data' in viz_data
        assert 'holdings_pie_data' in viz_data

    def test_real_time_updates(self, enhanced_dashboard):
        """Test real-time update functionality."""
        # Test enabling real-time updates
        enhanced_dashboard.enable_real_time_updates(interval_seconds=30)

        assert enhanced_dashboard.real_time_updates is True
        assert enhanced_dashboard.update_interval == 30

        # Test disabling real-time updates
        enhanced_dashboard.disable_real_time_updates()

        assert enhanced_dashboard.real_time_updates is False

    def test_export_functionality(self, enhanced_dashboard):
        """Test data export functionality."""
        export_data = {
            "performance_report": pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=10),
                'value': range(100000, 110000, 1000)
            }),
            "risk_report": {"var": 0.05, "sharpe": 1.8},
            "alerts": [{"message": "Test alert", "severity": "info"}]
        }

        # Test JSON export
        json_export = enhanced_dashboard.export_to_json(export_data)
        assert isinstance(json_export, str)
        assert 'performance_report' in json_export

        # Test CSV export
        csv_export = enhanced_dashboard.export_to_csv(export_data["performance_report"])
        assert isinstance(csv_export, str)
        assert 'date,value' in csv_export


class TestDashboardIntegration:
    """Integration tests for dashboard components."""

    def test_dashboard_data_flow(self):
        """Test data flow between dashboard components."""
        # Test that data flows correctly from memory to dashboard
        pass

    def test_real_time_updates_integration(self):
        """Test real-time updates integration."""
        # Test that real-time updates work with live data
        pass

    def test_alert_system_integration(self):
        """Test alert system integration with other components."""
        # Test that alerts are triggered based on system state
        pass

    def test_visualization_integration(self):
        """Test visualization integration with data sources."""
        # Test that visualizations update with new data
        pass


if __name__ == "__main__":
    pytest.main([__file__])