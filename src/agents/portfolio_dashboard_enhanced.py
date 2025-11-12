# src/agents/portfolio_dashboard.py
# Enhanced Version: Real agent data integration via MetricsService, Mermaid AI flow visualization,
# cost tracking, and agent-accessible data exports. Dual-purpose: human monitoring + agent queries.
# Purpose: Visual dashboard for monitoring AI Portfolio Manager with real-time agent data and Mermaid diagrams
# Run with: streamlit run src/agents/portfolio_dashboard.py
# Dependencies: streamlit, plotly, pandas, numpy
# New Features: Real agent data, Mermaid AI flow, cost tracking, agent API exports

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import streamlit.components.v1 as components

# Import metrics service for real data
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dashboard.metrics_service import get_metrics_service

# Initialize metrics service
metrics_service = get_metrics_service()

# Page config
st.set_page_config(
    page_title="AI Portfolio Manager Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.mermaid-container {
    background: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 10px 0;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin: 10px 0;
}
.alert-card {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    color: white;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}
.success-card {
    background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
    color: white;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}
.agent-status {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 15px;
    font-size: 12px;
    font-weight: bold;
    margin: 2px;
}
.agent-active { background: #51cf66; color: white; }
.agent-inactive { background: #ff6b6b; color: white; }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("AI Portfolio Manager")
st.sidebar.markdown("---")

# Data refresh controls
if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

days_filter = st.sidebar.slider("Time Range (days)", 7, 90, 30)

# Agent status indicator
st.sidebar.markdown("### Agent Status")
flow_data = metrics_service.get_agent_flow_data()
for agent, status in flow_data['agent_states'].items():
    status_class = "agent-active" if status == "active" else "agent-inactive"
    st.sidebar.markdown(f'<span class="agent-status {status_class}">{agent.replace("_", " ").title()}</span>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**System Health:** {flow_data['system_health'].upper()}")

# Main dashboard
st.title("🤖 AI Portfolio Manager Dashboard")
st.markdown("### Real-time Performance, Risk & Cost Monitoring")

# Load data
performance_data = metrics_service.get_performance_metrics(days_filter)
risk_data = metrics_service.get_risk_metrics()
cost_data = metrics_service.get_cost_metrics(days_filter)

# Key Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    pnl_value = performance_data['total_pnl']
    pnl_color = "🟢" if pnl_value > 0 else "🔴"
    st.markdown(f"""
    <div class="metric-card">
        <h3>{pnl_color} Total P&L</h3>
        <h2>{pnl_value:.2%}</h2>
        <p>{performance_data['total_trades']} trades</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    win_rate = performance_data['win_rate']
    win_color = "🟢" if win_rate > 0.5 else "🟡" if win_rate > 0.4 else "🔴"
    st.markdown(f"""
    <div class="metric-card">
        <h3>{win_color} Win Rate</h3>
        <h2>{win_rate:.1%}</h2>
        <p>{performance_data['avg_pnl']:.2%} avg per trade</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    sharpe = performance_data['avg_sharpe']
    sharpe_color = "🟢" if sharpe > 1.5 else "🟡" if sharpe > 1.0 else "🔴"
    st.markdown(f"""
    <div class="metric-card">
        <h3>{sharpe_color} Sharpe Ratio</h3>
        <h2>{sharpe:.2f}</h2>
        <p>Risk-adjusted returns</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    total_cost = cost_data['total_cost']
    cost_color = "🟢" if total_cost < 50 else "🟡" if total_cost < 100 else "🔴"
    st.markdown(f"""
    <div class="metric-card">
        <h3>{cost_color} Total Costs</h3>
        <h2>${total_cost:.2f}</h2>
        <p>{cost_data['api_calls']} API calls</p>
    </div>
    """, unsafe_allow_html=True)

# Performance Charts
st.header("📈 Performance Analysis")

# Create performance DataFrame
if performance_data['performance_history']:
    perf_df = pd.DataFrame(performance_data['performance_history'])
    perf_df['timestamp'] = pd.to_datetime(perf_df['timestamp'])
    perf_df = perf_df.sort_values('timestamp')

    # P&L Over Time Chart
    fig1 = px.line(perf_df, x='timestamp', y='pnl_pct',
                   title='P&L Over Time',
                   labels={'pnl_pct': 'P&L %', 'timestamp': 'Date'})
    fig1.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
    st.plotly_chart(fig1, use_container_width=True)

    # Sharpe Ratio Chart
    fig2 = px.line(perf_df, x='timestamp', y='sharpe_ratio',
                   title='Sharpe Ratio Trend',
                   labels={'sharpe_ratio': 'Sharpe Ratio', 'timestamp': 'Date'})
    fig2.add_hline(y=1.5, line_dash="dash", line_color="green", opacity=0.5)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No performance data available yet. Start trading to see charts!")

# Risk & Cost Analysis
col1, col2 = st.columns(2)

with col1:
    st.subheader("⚠️ Risk Metrics")

    # Risk metrics cards
    risk_metrics = [
        ("POP Threshold", f"{risk_data['current_pop_threshold']:.1%}", "Minimum probability of profit"),
        ("Max Drawdown", f"{risk_data['max_drawdown_limit']:.1%}", "Maximum allowed drawdown"),
        ("Position Size", f"{risk_data['position_size_limit']:.1%}", "Max position size"),
        ("Variance SD", f"{risk_data['variance_threshold']:.1f}", "Volatility threshold")
    ]

    for label, value, desc in risk_metrics:
        st.metric(label, value, help=desc)

    # Current drawdown check
    current_drawdown = performance_data['max_drawdown']
    if current_drawdown > risk_data['max_drawdown_limit']:
        st.markdown('<div class="alert-card">⚠️ Drawdown Alert: Current drawdown exceeds limit!</div>', unsafe_allow_html=True)

with col2:
    st.subheader("💰 Cost Breakdown")

    # Cost breakdown
    cost_breakdown = cost_data['cost_breakdown']
    labels = list(cost_breakdown.keys())
    values = list(cost_breakdown.values())

    if sum(values) > 0:
        fig3 = px.pie(values=values, names=labels, title="Cost Distribution")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No costs recorded yet")

    # Cost metrics
    st.metric("API Calls", cost_data['api_calls'])
    st.metric("Token Usage", f"{cost_data['token_usage']:,}")
    st.metric("Trading Fees", f"${cost_data['trading_fees']:.2f}")

# AI Agent Flow Visualization
st.header("🔄 AI Agent Interactions")

# Mermaid diagram
mermaid_code = f"""
graph TD
    A[Data Agent] --> B[Strategy Agent]
    B --> C[Risk Agent]
    C --> D[Execution Agent]
    D --> E[Reflection Agent]
    E --> B

    A --> F[Market Data]
    B --> G[Trade Proposals]
    C --> H[Risk Assessment]
    D --> I[Order Execution]
    E --> J[Performance Review]

    K[Metrics Service] --> A
    K --> B
    K --> C
    K --> D
    K --> E

    L[Cost Tracking] --> K
    M[Audit Logs] --> K

    classDef active fill:#51cf66,color:#fff
    classDef inactive fill:#ff6b6b,color:#fff

    class A,B,C,D,E,K active
"""

# Recent activity summary
recent_activity = flow_data['recent_flows']
st.markdown(f"**Recent Activity:** {recent_activity['total_interactions']} interactions")
st.markdown(f"- Audits: {len(recent_activity['audits'])}")
st.markdown(f"- Reviews: {len(recent_activity['reviews'])}")

# Display Mermaid
components.html(f"""
<div class="mermaid-container">
    <pre class="mermaid">
{mermaid_code}
    </pre>
</div>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.2.3/dist/mermaid.min.js"></script>
<script>
    mermaid.initialize({{
        startOnLoad: true,
        theme: 'default',
        securityLevel: 'loose'
    }});
</script>
""", height=400)

# Alerts & Notifications
st.header("🚨 Alerts & Recommendations")

alerts = []

# Performance alerts
if performance_data['win_rate'] < 0.4:
    alerts.append(("warning", "Low Win Rate", f"Win rate is {performance_data['win_rate']:.1%} - consider strategy review"))

if performance_data['avg_sharpe'] < 1.0:
    alerts.append(("error", "Poor Risk-Adjusted Returns", f"Sharpe ratio {performance_data['avg_sharpe']:.2f} below 1.0"))

if performance_data['max_drawdown'] > risk_data['max_drawdown_limit']:
    alerts.append(("error", "Drawdown Limit Exceeded", f"Current drawdown {performance_data['max_drawdown']:.1%} > limit {risk_data['max_drawdown_limit']:.1%}"))

# Cost alerts
if cost_data['total_cost'] > 100:
    alerts.append(("warning", "High Costs", f"Total costs ${cost_data['total_cost']:.2f} - monitor efficiency"))

if cost_data['api_calls'] > 1000:
    alerts.append(("info", "High API Usage", f"{cost_data['api_calls']} API calls - consider optimization"))

# Display alerts
if alerts:
    for alert_type, title, message in alerts:
        if alert_type == "error":
            st.error(f"**{title}**: {message}")
        elif alert_type == "warning":
            st.warning(f"**{title}**: {message}")
        else:
            st.info(f"**{title}**: {message}")
else:
    st.success("✅ All systems operating within normal parameters")

# Agent Data Export
st.header("📤 Agent Data Access")

col1, col2 = st.columns(2)

with col1:
    if st.button("Export Metrics for Agents"):
        agent_data = metrics_service.export_metrics_for_agents()
        json_data = json.dumps(agent_data, indent=2, default=str)

        st.download_button(
            label="📥 Download Agent Data (JSON)",
            data=json_data,
            file_name=f"agent_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

with col2:
    if st.button("Log Sample API Call"):
        metrics_service.log_api_call("dashboard", "sample_endpoint", 0.02, 150)
        st.success("✅ Sample API call logged for cost tracking")

# Recent Activity
st.header("📋 Recent Activity")

if performance_data['performance_history']:
    st.subheader("Latest Trades")
    recent_trades = perf_df.tail(5)[['timestamp', 'pnl_pct', 'sharpe_ratio']].copy()
    recent_trades['pnl_pct'] = recent_trades['pnl_pct'].map('{:.2%}'.format)
    recent_trades['sharpe_ratio'] = recent_trades['sharpe_ratio'].map('{:.2f}'.format)
    recent_trades.columns = ['Timestamp', 'P&L', 'Sharpe']
    st.dataframe(recent_trades, use_container_width=True)

if cost_data['recent_costs']:
    st.subheader("Recent Costs")
    cost_df = pd.DataFrame(cost_data['recent_costs'])
    if not cost_df.empty:
        cost_df['timestamp'] = pd.to_datetime(cost_df['timestamp'])
        cost_df = cost_df.sort_values('timestamp', ascending=False).head(5)
        st.dataframe(cost_df[['timestamp', 'agent', 'cost']], use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Dashboard Status:** Connected to live agent data | Auto-refresh: Manual | Data Source: Agent Memory & Metrics Service")
st.markdown("**Built for:** Human monitoring + Agent programmatic access | **Purpose:** Real-time AI portfolio management oversight")