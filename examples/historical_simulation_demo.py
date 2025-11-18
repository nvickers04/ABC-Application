# historical_simulation_demo.py
# Demonstration script for running historical portfolio simulations
# Shows how to use the historical simulation tools and risk analytics

import sys
import os
from pathlib import Path
import json
import asyncio

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.tools import (
    historical_portfolio_simulation_tool,
    multi_strategy_portfolio_comparison_tool,
    risk_analytics_tool
)
from src.agents.risk import RiskAgent

# Import the underlying functions directly to avoid Langchain tool wrapper issues
from src.utils.historical_simulation_engine import run_historical_portfolio_simulation, run_multi_strategy_comparison
from src.utils.risk_analytics_framework import analyze_portfolio_risk

def demo_basic_portfolio_simulation():
    """Demonstrate basic portfolio simulation with a simple buy-and-hold strategy."""
    print("=" * 60)
    print("BASIC PORTFOLIO SIMULATION DEMO")
    print("=" * 60)

    # Define simulation parameters
    symbols = "AAPL,MSFT,GOOGL,AMZN,TSLA"  # Tech portfolio
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    initial_capital = "100000"

    print(f"Portfolio: {symbols}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital}")
    print("\nRunning simulation...")

    # Run the simulation using the underlying function
    result = run_historical_portfolio_simulation(
        symbols=[s.strip() for s in symbols.split(',') if s.strip()],
        start_date=start_date,
        end_date=end_date,
        initial_capital=float(initial_capital)
    )

    # Parse and display results
    try:
        simulation_data = result  # Already a dict, not JSON string
        display_simulation_results(simulation_data, "Basic Portfolio Simulation")
        return simulation_data
    except Exception as e:
        print(f"Error processing simulation results: {e}")
        print(f"Raw result: {result}")
        return None

def demo_multi_strategy_comparison():
    """Demonstrate comparing multiple trading strategies."""
    print("\n" + "=" * 60)
    print("MULTI-STRATEGY COMPARISON DEMO")
    print("=" * 60)

    # Define strategies to compare
    strategies_config = json.dumps([
        {
            "name": "Equal Weight Rebalancing",
            "description": "Monthly rebalancing to equal weights",
            "rebalance_frequency": "monthly",
            "transaction_costs": 0.001,
            "slippage": 0.0005
        },
        {
            "name": "Buy and Hold",
            "description": "No rebalancing, buy and hold",
            "rebalance_frequency": "none",
            "transaction_costs": 0.001,
            "slippage": 0.0005
        },
        {
            "name": "Quarterly Rebalancing",
            "description": "Quarterly rebalancing to equal weights",
            "rebalance_frequency": "quarterly",
            "transaction_costs": 0.001,
            "slippage": 0.0005
        }
    ])

    symbols = "SPY,QQQ,IWM,EFA,AGG"  # Diversified portfolio
    start_date = "2019-01-01"
    end_date = "2023-12-31"
    initial_capital = "100000"

    print(f"Portfolio: {symbols}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital}")
    print(f"Strategies: {len(json.loads(strategies_config))} different approaches")
    print("\nRunning strategy comparison...")

    # Run the comparison using the underlying function
    result = run_multi_strategy_comparison(
        symbols=[s.strip() for s in symbols.split(',') if s.strip()],
        strategies=json.loads(strategies_config),
        start_date=start_date,
        end_date=end_date,
        initial_capital=float(initial_capital)
    )

    # Parse and display results
    try:
        comparison_data = result  # Already a dict, not JSON string
        display_strategy_comparison(comparison_data)
        return comparison_data
    except Exception as e:
        print(f"Error processing comparison results: {e}")
        print(f"Raw result: {result}")
        return None

async def demo_risk_analytics(simulation_data):
    """Demonstrate risk analytics on simulation results."""
    if not simulation_data:
        print("No simulation data available for risk analysis")
        return

    print("\n" + "=" * 60)
    print("RISK ANALYTICS DEMO")
    print("=" * 60)

    print("Running comprehensive risk analysis...")

    # Convert simulation data to JSON string for the tool
    simulation_json = json.dumps(simulation_data)

    # Run risk analytics using the underlying function
    risk_report = analyze_portfolio_risk(simulation_data)

    display_risk_analysis(risk_report)
    return risk_report

async def demo_agent_risk_analysis(simulation_data):
    """Demonstrate agent-based risk analysis with LLM insights."""
    if not simulation_data:
        print("No simulation data available for agent analysis")
        return

    print("\n" + "=" * 60)
    print("AGENT-BASED RISK ANALYSIS DEMO")
    print("=" * 60)

    print("Running agent-based risk analysis with LLM insights...")

    try:
        # Create risk agent
        risk_agent = RiskAgent()

        # Run agent analysis
        agent_result = await risk_agent.analyze_historical_simulation_risks(simulation_data)

        if 'error' in agent_result:
            print(f"Agent analysis failed: {agent_result['error']}")
            return None

        display_agent_analysis(agent_result)
        return agent_result

    except Exception as e:
        print(f"Error in agent analysis: {e}")
        return None

def display_simulation_results(data, title):
    """Display simulation results in a readable format."""
    print(f"\n{title} Results:")
    print("-" * 40)

    if 'error' in data:
        print(f"Error: {data['error']}")
        return

    # Basic metrics
    perf = data.get('performance_metrics', {})
    trading = data.get('trading_statistics', {})

    print(f"Total Return: {perf.get('total_return', 0):.1%}")
    print(f"Annualized Return: {perf.get('annualized_return', 0):.1%}")
    print(f"Volatility: {perf.get('volatility', 0):.1%}")
    print(f"Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {perf.get('max_drawdown', 0):.1%}")
    print(f"Win Rate: {perf.get('win_rate', 0):.1%}")

    print(f"\nTrading Statistics:")
    print(f"Total Trades: {trading.get('total_trades', 0)}")
    print(f"Final Portfolio Value: ${trading.get('final_portfolio_value', 0):,.0f}")

    # Portfolio history summary
    history = data.get('portfolio_history', [])
    if history:
        print(f"\nPortfolio History: {len(history)} data points")
        print(f"Start Value: ${history[0].get('portfolio_value', 0):,.0f}")
        print(f"End Value: ${history[-1].get('portfolio_value', 0):,.0f}")

def display_strategy_comparison(data):
    """Display strategy comparison results."""
    print("\nStrategy Comparison Results:")
    print("-" * 40)

    if 'error' in data:
        print(f"Error: {data['error']}")
        return

    # Check for comparison summary
    summary = data.get('_comparison_summary', {})
    if summary:
        print("Strategy Performance Summary:")
        print("<25")
        for strategy_name, metrics in summary.items():
            print("<25")
    else:
        print("No comparison summary available")

    # Individual strategy results
    for strategy_name, result in data.items():
        if strategy_name.startswith('_'):
            continue

        print(f"\n{strategy_name}:")
        perf = result.get('performance_metrics', {})
        print(f"  Total Return: {perf.get('total_return', 0):.1%}")
        print(f"  Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {perf.get('max_drawdown', 0):.1%}")

def display_risk_analysis(risk_report):
    """Display risk analysis results."""
    print("\nRisk Analysis Results:")
    print("-" * 40)

    if 'error' in risk_report:
        print(f"Error: {risk_report['error']}")
        return

    # Risk assessment
    assessment = risk_report.get('risk_assessment', {})
    print(f"Overall Risk Level: {assessment.get('overall_risk_level', 'unknown').upper()}")
    print(f"Risk Factors: {', '.join(assessment.get('risk_factors', []))}")
    print(f"Confidence Level: {assessment.get('confidence_level', 'unknown')}")

    # Key risk metrics
    metrics = risk_report.get('risk_metrics', {})
    print("\nKey Risk Metrics:")
    print(f"  Volatility: {metrics.get('volatility', 0):.1%}")
    print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.1%}")
    print(f"  VaR 95%: {metrics.get('var_historical_95', 0):.1%}")
    print(f"  CVaR 95%: {metrics.get('cvar_95', 0):.1%}")

    # Recommendations
    recommendations = risk_report.get('recommendations', [])
    if recommendations:
        print("\nRisk Recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
            print(f"  {i}. {rec}")

def display_agent_analysis(agent_result):
    """Display agent analysis results."""
    print("\nAgent Analysis Results:")
    print("-" * 40)

    if 'error' in agent_result:
        print(f"Error: {agent_result['error']}")
        return

    # Agent analysis
    agent_analysis = agent_result.get('agent_analysis', {})
    print(f"Analysis Timestamp: {agent_analysis.get('timestamp', 'unknown')}")

    # Portfolio characteristics
    chars = agent_analysis.get('portfolio_characteristics', {})
    print(f"Trading Frequency: {chars.get('trading_frequency', 0):.2f} trades/day")
    print(f"Strategy Types: {', '.join(chars.get('strategy_types', []))}")

    # LLM insights
    llm_insights = agent_analysis.get('llm_insights', {})
    insights = llm_insights.get('key_insights_extracted', [])
    if insights:
        print(f"\nLLM Insights:")
        for insight in insights[:2]:  # Show top 2
            print(f"  • {insight}")

    # Historical recommendations
    hist_recs = agent_analysis.get('historical_recommendations', [])
    if hist_recs:
        print(f"\nHistorical Recommendations:")
        for rec in hist_recs[:2]:  # Show top 2
            print(f"  • {rec}")

async def main():
    """Run the complete historical simulation demonstration."""
    print("HISTORICAL PORTFOLIO SIMULATION DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases the complete historical simulation pipeline:")
    print("1. Basic portfolio simulation")
    print("2. Multi-strategy comparison")
    print("3. Risk analytics framework")
    print("4. Agent-based risk analysis with LLM insights")
    print("=" * 60)

    try:
        # Run basic simulation
        simulation_data = demo_basic_portfolio_simulation()

        # Run strategy comparison
        comparison_data = demo_multi_strategy_comparison()

        # Run risk analytics on basic simulation
        if simulation_data:
            risk_report = await demo_risk_analytics(simulation_data)

            # Run agent analysis
            agent_analysis = await demo_agent_risk_analysis(simulation_data)

        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE!")
        print("=" * 60)
        print("The historical simulation system is fully operational.")
        print("You can now run portfolio backtests, compare strategies,")
        print("and perform comprehensive risk analysis on historical data.")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())