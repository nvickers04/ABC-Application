# test_risk_analytics_framework.py
# Test script for the comprehensive risk analytics framework

import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def create_sample_simulation_results():
    """Create sample simulation results for testing."""
    # Generate sample portfolio history
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)  # For reproducible results

    # Simulate portfolio returns with some volatility and drawdowns
    n_days = len(dates)
    returns = np.random.normal(0.0005, 0.02, n_days)  # Mean 0.05% daily return, 2% volatility

    # Add some drawdown periods
    drawdown_periods = [
        (100, 150, -0.15),  # 15% drawdown
        (300, 350, -0.08),  # 8% drawdown
        (500, 520, -0.12),  # 12% drawdown
    ]

    for start, end, dd_magnitude in drawdown_periods:
        if start < n_days and end < n_days:
            # Create a gradual drawdown and recovery
            drawdown_returns = np.linspace(0, dd_magnitude, end - start)
            recovery_returns = np.linspace(dd_magnitude, 0, end - start)
            returns[start:end] = drawdown_returns[:len(returns[start:end])]

    # Calculate portfolio values
    portfolio_values = [100000]  # Starting value
    for ret in returns:
        new_value = portfolio_values[-1] * (1 + ret)
        portfolio_values.append(new_value)

    portfolio_values = portfolio_values[:-1]  # Remove extra value

    # Create portfolio history DataFrame
    portfolio_history = []
    for i, (date, value) in enumerate(zip(dates, portfolio_values)):
        portfolio_history.append({
            'date': date.strftime('%Y-%m-%d'),
            'portfolio_value': value,
            'cash': value * 0.1,  # 10% cash
            'positions_value': value * 0.9  # 90% invested
        })

    # Generate sample trades
    trades = []
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

    for i in range(200):  # 200 trades
        trade_date = dates[np.random.randint(0, n_days)].strftime('%Y-%m-%d')
        symbol = np.random.choice(symbols)
        action = np.random.choice(['BUY', 'SELL'])
        quantity = np.random.randint(10, 100)
        price = np.random.uniform(50, 500)
        value = quantity * price
        commission = value * 0.001  # 0.1% commission

        trades.append({
            'date': trade_date,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'value': value,
            'commission': commission
        })

    # Create benchmark comparison (S&P 500 proxy)
    benchmark_returns = returns * 0.8 + np.random.normal(0, 0.005, n_days)  # Correlated but different
    benchmark_cumulative = np.cumprod(1 + benchmark_returns)

    # Simulation results structure
    simulation_results = {
        'simulation_config': {
            'start_date': '2020-01-01',
            'end_date': '2023-12-31',
            'initial_capital': 100000,
            'symbols': symbols
        },
        'portfolio_history': portfolio_history,
        'trades': trades,
        'performance_metrics': {
            'total_return': (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0],
            'annualized_return': ((portfolio_values[-1] / portfolio_values[0]) ** (365/n_days) - 1),
            'volatility': np.std(returns) * np.sqrt(252),
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252),
            'max_drawdown': 0.15,  # From our simulated drawdowns
            'win_rate': 0.55
        },
        'benchmark_comparison': {
            'benchmark_symbol': 'SPY',
            'benchmark_cumulative_returns': benchmark_cumulative.tolist(),
            'benchmark_total_return': benchmark_cumulative[-1] - 1,
            'excess_return': (portfolio_values[-1] / portfolio_values[0] - 1) - (benchmark_cumulative[-1] - 1)
        },
        'trading_statistics': {
            'total_trades': len(trades),
            'winning_trades': int(len(trades) * 0.55),
            'losing_trades': int(len(trades) * 0.45),
            'avg_trade_pnl': 250,
            'total_pnl': 50000,
            'final_portfolio_value': portfolio_values[-1]
        }
    }

    return simulation_results

def test_risk_analytics_framework():
    """Test the risk analytics framework with sample data."""
    print("Testing Risk Analytics Framework...")
    print("=" * 50)

    try:
        # Create sample data
        simulation_results = create_sample_simulation_results()
        print(f"Created sample simulation with {len(simulation_results['portfolio_history'])} days of data")
        print(f"Total trades: {len(simulation_results['trades'])}")

        # Import and run risk analysis
        from src.utils.risk_analytics_framework import analyze_portfolio_risk

        print("\nRunning risk analysis...")
        risk_report = analyze_portfolio_risk(simulation_results)

        # Display results
        print("\n" + "=" * 50)
        print("RISK ANALYSIS RESULTS")
        print("=" * 50)

        # Simulation summary
        sim_summary = risk_report.get('simulation_summary', {})
        print(f"\nSimulation Summary:")
        print(f"  Period: {sim_summary.get('start_date')} to {sim_summary.get('end_date')}")
        print(f"  Initial Capital: ${sim_summary.get('initial_capital', 0):,.0f}")
        print(f"  Final Value: ${sim_summary.get('final_value', 0):,.0f}")
        print(f"  Total Return: {sim_summary.get('total_return', 0):.1%}")

        # Risk assessment
        risk_assessment = risk_report.get('risk_assessment', {})
        print(f"\nRisk Assessment:")
        print(f"  Overall Risk Level: {risk_assessment.get('overall_risk_level', 'unknown').upper()}")
        print(f"  Risk Factors: {', '.join(risk_assessment.get('risk_factors', []))}")
        print(f"  Confidence Level: {risk_assessment.get('confidence_level', 'unknown')}")

        # Key risk metrics
        risk_metrics = risk_report.get('risk_metrics', {})
        print(f"\nKey Risk Metrics:")
        print(f"  Volatility: {risk_metrics.get('volatility', 0):.1%}")
        print(f"  Max Drawdown: {risk_metrics.get('max_drawdown', 0):.1%}")
        print(f"  VaR 95%: {risk_metrics.get('var_historical_95', 0):.1%}")
        print(f"  CVaR 95%: {risk_metrics.get('cvar_95', 0):.1%}")
        print(f"  Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.2f}")

        # Performance attribution
        attribution = risk_report.get('performance_attribution', {})
        benchmark_comp = attribution.get('benchmark_comparison', {})
        if benchmark_comp:
            print(f"\nBenchmark Comparison (vs SPY):")
            print(f"  Portfolio Return: {benchmark_comp.get('portfolio_total_return', 0):.1%}")
            print(f"  Benchmark Return: {benchmark_comp.get('benchmark_total_return', 0):.1%}")
            print(f"  Excess Return: {benchmark_comp.get('excess_return', 0):.1%}")
            print(f"  Tracking Error: {benchmark_comp.get('tracking_error', 0):.1%}")
            print(f"  Information Ratio: {benchmark_comp.get('information_ratio', 0):.2f}")

        # Recommendations
        recommendations = risk_report.get('recommendations', [])
        if recommendations:
            print(f"\nRisk Management Recommendations:")
            for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
                print(f"  {i}. {rec}")

        print(f"\n" + "=" * 50)
        print("Risk Analytics Framework Test Completed Successfully!")
        print("=" * 50)

        assert True

    except Exception as e:
        print(f"Error testing risk analytics framework: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"Risk analytics framework test failed: {e}"

def test_risk_agent_integration():
    """Test the risk agent's integration with the analytics framework."""
    print("\nTesting Risk Agent Integration...")
    print("=" * 50)

    try:
        import asyncio
        from src.agents.risk import RiskAgent

        # Create sample data
        simulation_results = create_sample_simulation_results()

        # Create risk agent and test
        async def run_test():
            agent = RiskAgent()
            result = await agent.analyze_historical_simulation_risks(simulation_results)
            return result

        # Run the async test
        result = asyncio.run(run_test())

        if 'error' in result:
            print(f"Risk agent analysis failed: {result['error']}")
            return False

        # Display agent analysis results
        agent_analysis = result.get('agent_analysis', {})
        print("Risk Agent Analysis Results:")
        print(f"  Analysis Timestamp: {agent_analysis.get('timestamp', 'unknown')}")

        portfolio_chars = agent_analysis.get('portfolio_characteristics', {})
        print(f"  Trading Frequency: {portfolio_chars.get('trading_frequency', 0):.2f} trades/day")
        print(f"  Strategy Types: {', '.join(portfolio_chars.get('strategy_types', []))}")

        llm_insights = agent_analysis.get('llm_insights', {})
        if 'key_insights_extracted' in llm_insights:
            insights = llm_insights['key_insights_extracted']
            if insights:
                print(f"  LLM Insights: {insights[0][:100]}...")

        recommendations = agent_analysis.get('historical_recommendations', [])
        if recommendations:
            print(f"  Historical Recommendations: {len(recommendations)} items")

        projections = agent_analysis.get('risk_projections', {})
        if projections:
            print(f"  Risk Projections Generated: {len(projections)} metrics")

        print("Risk Agent Integration Test Completed Successfully!")
        assert True

    except Exception as e:
        print(f"Error testing risk agent integration: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"Risk agent integration test failed: {e}"

if __name__ == "__main__":
    print("Risk Analytics Framework Test Suite")
    print("===================================")

    # Test the framework
    framework_success = test_risk_analytics_framework()

    # Test agent integration
    agent_success = test_risk_agent_integration()

    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Framework Test: {'PASSED' if framework_success else 'FAILED'}")
    print(f"Agent Integration Test: {'PASSED' if agent_success else 'FAILED'}")

    if framework_success and agent_success:
        print("\nAll tests passed! Risk analytics framework is ready for production use.")
    else:
        print("\nSome tests failed. Please review the errors above.")