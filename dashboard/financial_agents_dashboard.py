import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Import your metrics service
from src.dashboard.metrics_service import get_metrics_service

# Import for memory monitoring
from src.utils.optimized_pipeline import OptimizedPipelineProcessor

# Import portfolio dashboard
from src.agents.portfolio_dashboard import PortfolioDashboard

# Initialize services
metrics_service = get_metrics_service()
pipeline = OptimizedPipelineProcessor()
memory_dashboard = pipeline.get_memory_monitoring_dashboard()
portfolio_dashboard = PortfolioDashboard()

# Fetch real data (falls back to samples if agents not loaded)
performance_metrics = metrics_service.get_performance_metrics(days=30)
risk_metrics = metrics_service.get_risk_metrics()
cost_metrics = metrics_service.get_cost_metrics(days=30)
agent_flow_data = metrics_service.get_agent_flow_data()

# Fetch portfolio data
portfolio_sample_data = portfolio_dashboard.generate_sample_data()
portfolio_performance_metrics = portfolio_dashboard.calculate_performance_metrics(portfolio_sample_data)
portfolio_risk_data = {
    "current_drawdown": abs(min(portfolio_sample_data['Drawdown'])),
    "max_drawdown_limit": 0.05,
    "current_positions": len(portfolio_sample_data),  # Dummy
    "max_positions_limit": 100
}
portfolio_risk_assessment = portfolio_dashboard.assess_risk_levels(portfolio_risk_data)
portfolio_alerts = portfolio_dashboard.generate_alerts({
    "current_drawdown": portfolio_risk_data["current_drawdown"],
    "max_drawdown_limit": portfolio_risk_data["max_drawdown_limit"],
    "sharpe_ratio": performance_metrics['avg_sharpe'],
    "min_sharpe": 1.0
})

# Prepare data for charts and tables
# PNL History for line chart
pnl_history_df = pd.DataFrame(performance_metrics['performance_history'])
pnl_history_df['timestamp'] = pd.to_datetime(pnl_history_df['timestamp'])
pnl_history_df = pnl_history_df.sort_values('timestamp')

# Cost Breakdown for pie chart
cost_breakdown_data = {
    'Category': ['API', 'Tokens', 'Fees'],
    'Value': [
        cost_metrics['cost_breakdown']['api'],
        cost_metrics['cost_breakdown']['tokens'],
        cost_metrics['cost_breakdown']['fees']
    ]
}
cost_breakdown_df = pd.DataFrame(cost_breakdown_data)

# Recent Proposals Table
recent_proposals = risk_metrics['recent_proposals']
if not recent_proposals:
    recent_proposals = performance_metrics['performance_history'][-5:]
proposals_df = pd.DataFrame(recent_proposals)
if proposals_df.empty:
    proposals_df = pd.DataFrame([
        {'timestamp': datetime.now().isoformat(), 'symbol': 'AAPL', 'action': 'Buy', 'status': 'Approved'},
        {'timestamp': datetime.now().isoformat(), 'symbol': 'BTC', 'action': 'Sell', 'status': 'Rejected'},
    ])

# Agent states
agent_states = agent_flow_data['agent_states']

# Portfolio sample for additional chart (e.g., ROI over time)
portfolio_df = portfolio_sample_data.copy()
portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])

# Dashboard setup
st.set_page_config(page_title='Financial Agents Dashboard', layout='wide', initial_sidebar_state='expanded')

# Custom CSS for dark theme and clean styling (mimicking Tempo)
st.markdown("""
<style>
    /* Overall dark theme */
    .stApp {
        background-color: #1E1E1E;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #2D2D2D;
    }
    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #2D2D2D;
        border-radius: 8px;
        padding: 16px;
        margin: 8px;
    }
    /* Table styling */
    .stDataFrame {
        background-color: #2D2D2D;
        color: white;
    }
    /* Chart backgrounds */
    section[data-testid="stElementContainer"] > div > div > div > div > div {
        background-color: #2D2D2D;
    }
    /* Buttons and selects */
    .stButton > button {
        background-color: #3A3A3A;
        color: white;
        border: none;
        border-radius: 4px;
    }
    .stSelectbox > div {
        background-color: #3A3A3A;
        color: white;
    }
    /* Mermaid diagram */
    .mermaid {
        background-color: #2D2D2D !important;
    }
    /* Alert styling */
    .alert-critical {
        color: #EF4444;
    }
    .alert-warning {
        color: #F59E0B;
    }
    .alert-info {
        color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar (like Tempo's menu)
with st.sidebar:
    st.title('⚡ Financial Dashboard')
    st.subheader('Main Menu')
    st.markdown('- Dashboard\n- Notifications\n- Performance\n- Risk\n- Costs\n- Memory\n- Agents\n- Reports\n- Transactions\n- More')
    st.subheader('General')
    st.markdown('- Settings\n- Help Center\n- Feedback')
    st.info(f"System Health: {agent_flow_data['system_health'].capitalize()}\nData Source: {performance_metrics['data_source']}")

# Top header
col_header1, col_header2, col_header3, col_header4 = st.columns([2, 1, 1, 1])
with col_header1:
    st.title('Dashboard')
with col_header2:
    period = st.selectbox('Select Period', ['30 Days', '3 Months', '1 Year'])
    # Adjust days based on period if needed
with col_header3:
    st.button('Export Metrics')
with col_header4:
    st.button('New Decision')

# Key metrics row (integrated from metrics_service and portfolio)
cols_metrics = st.columns(4)
with cols_metrics[0]:
    st.metric(label='Total PNL', value=f"{performance_metrics['total_pnl']:.2%}", delta=f"Avg: {performance_metrics['avg_pnl']:.2%}")
with cols_metrics[1]:
    st.metric(label='Win Rate', value=f"{performance_metrics['win_rate']:.1%}", delta=f"Trades: {performance_metrics['total_trades']}")
with cols_metrics[2]:
    st.metric(label='Sharpe Ratio', value=f"{performance_metrics['avg_sharpe']:.2f}", delta=f"Drawdown: {performance_metrics['max_drawdown']:.2%}")
with cols_metrics[3]:
    st.metric(label='Total Costs', value=f"${cost_metrics['total_cost']:.2f}", delta=f"API Calls: {cost_metrics['api_calls']}")

# Charts row
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader('PNL Flow')
    st.metric('Total PNL', f"{performance_metrics['total_pnl']:.2%}", f"Win Rate: {performance_metrics['win_rate']:.1%}")
    fig_line = px.line(pnl_history_df, x='timestamp', y='pnl_pct', title='')
    fig_line.update_layout(
        paper_bgcolor='#2D2D2D', plot_bgcolor='#2D2D2D', font_color='white',
        xaxis_title=None, yaxis_title=None, showlegend=False
    )
    st.plotly_chart(fig_line, use_container_width=True)
    if performance_metrics['data_source'] == 'sample_data':
        st.info('Using sample data. Load agents for real metrics.')

with col_chart2:
    st.subheader('Cost Breakdown')
    total_cost = sum(cost_breakdown_df['Value'])
    fig_pie = px.pie(cost_breakdown_df, values='Value', names='Category', title=f'Total: ${total_cost:.2f}', hole=0.4)
    fig_pie.update_traces(marker_colors=['#A78BFA', '#818CF8', '#6366F1'])
    fig_pie.update_layout(paper_bgcolor='#2D2D2D', font_color='white', showlegend=True)
    st.plotly_chart(fig_pie, use_container_width=True)

# Portfolio Section
st.subheader('Portfolio Performance')
portfolio_cols = st.columns(4)
with portfolio_cols[0]:
    st.metric('Total ROI', f"{portfolio_performance_metrics['total_roi']:.2%}")
with portfolio_cols[1]:
    st.metric('Avg Sharpe', f"{portfolio_performance_metrics['avg_sharpe']:.2f}")
with portfolio_cols[2]:
    st.metric('Max Drawdown', f"{portfolio_performance_metrics['max_drawdown']:.2%}")
with portfolio_cols[3]:
    st.metric('Volatility', f"{portfolio_performance_metrics['volatility']:.2%}")

col_port_chart1, col_port_chart2 = st.columns(2)
with col_port_chart1:
    fig_roi = px.line(portfolio_df, x='Date', y='ROI', title='ROI Over Time')
    fig_roi.update_layout(paper_bgcolor='#2D2D2D', plot_bgcolor='#2D2D2D', font_color='white')
    st.plotly_chart(fig_roi, use_container_width=True)

with col_port_chart2:
    fig_drawdown = px.line(portfolio_df, x='Date', y='Drawdown', title='Drawdown Over Time')
    fig_drawdown.update_layout(paper_bgcolor='#2D2D2D', plot_bgcolor='#2D2D2D', font_color='white')
    st.plotly_chart(fig_drawdown, use_container_width=True)

st.subheader('Portfolio Risk Assessment')
st.write(f"Risk Level: {portfolio_risk_assessment.get('risk_level', 'N/A')}")
if portfolio_risk_assessment.get('breaches'):
    st.warning(f"Breaches: {', '.join(portfolio_risk_assessment['breaches'])}")

# Risk Metrics Section
st.subheader('Risk Metrics')
risk_cols = st.columns(4)
with risk_cols[0]:
    st.metric('PoP Threshold', f"{risk_metrics['current_pop_threshold']:.2f}")
with risk_cols[1]:
    st.metric('Max Drawdown Limit', f"{risk_metrics['max_drawdown_limit']:.2%}")
with risk_cols[2]:
    st.metric('Position Size Limit', f"{risk_metrics['position_size_limit']:.2%}")
with risk_cols[3]:
    st.metric('Variance Threshold', f"{risk_metrics['variance_threshold']:.2f}")

# Recent Proposals Table
st.subheader(f"{len(recent_proposals)} Recent Proposals/Decisions")
st.dataframe(proposals_df, use_container_width=True, hide_index=True)

# Memory Monitoring Section
st.subheader('Memory Monitoring')
memory_stats = memory_dashboard['current_stats']
memory_cols = st.columns(4)
with memory_cols[0]:
    st.metric('Used Memory', f"{memory_stats['used_mb']:.1f} MB")
with memory_cols[1]:
    st.metric('Peak Memory', f"{memory_stats['peak_mb']:.1f} MB")
with memory_cols[2]:
    st.metric('Utilization', f"{memory_stats['utilization_percent']:.1f}%")
with memory_cols[3]:
    st.metric('Active Objects', f"{memory_stats['active_objects']}")

st.subheader('Memory Trends')
trends = memory_dashboard['trends']
st.write(f"Usage Trend: {trends['usage_trend']}")
st.write(f"Efficiency Score: {trends['efficiency_score']:.2f}")
if trends['leak_indicators']:
    st.warning("Leak Indicators: " + "; ".join(trends['leak_indicators']))
if trends['optimization_opportunities']:
    st.info("Optimization Opportunities: " + "; ".join(trends['optimization_opportunities']))

st.subheader('Memory Alerts')
alerts = memory_dashboard['alerts']
if alerts:
    for alert in alerts:
        if alert['severity'] == 'critical':
            st.markdown(f"<p class='alert-critical'>🔴 {alert['severity'].upper()}: {alert['message']}<br>Recommendation: {alert['recommendation']}</p>", unsafe_allow_html=True)
        elif alert['severity'] == 'warning':
            st.markdown(f"<p class='alert-warning'>🟡 {alert['severity'].upper()}: {alert['message']}<br>Recommendation: {alert['recommendation']}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='alert-info'>ℹ️ {alert['severity'].upper()}: {alert['message']}<br>Recommendation: {alert['recommendation']}</p>", unsafe_allow_html=True)
else:
    st.success("✅ No active alerts")

st.subheader('Memory Recommendations')
for rec in memory_dashboard['recommendations']:
    st.write(f"- {rec}")

st.subheader('Performance Metrics')
perf_metrics = memory_dashboard['performance_metrics']
perf_cols = st.columns(3)
with perf_cols[0]:
    st.metric('Avg Processing Time', f"{perf_metrics['avg_processing_time']:.1f}s per symbol")
with perf_cols[1]:
    st.metric('Memory per Symbol', f"{perf_metrics['memory_per_symbol']:.1f} MB")
with perf_cols[2]:
    st.metric('Cache Hit Rate', f"{perf_metrics['cache_hit_rate']:.1%}")

# Portfolio Alerts
st.subheader('Portfolio Alerts')
if portfolio_alerts:
    for alert in portfolio_alerts:
        st.warning(alert)  # Assuming alerts are strings; adjust if dict
else:
    st.success("No portfolio alerts")

# Agent Flow Visualization (using Mermaid)
st.subheader('Agent Flow Diagram')
mermaid_code = """
graph TD
    A[Data Agent] -->|Data Input| B[Strategy Agent]
    B -->|Proposal| C[Risk Agent]
    C -->|Approval| D[Execution Agent]
    D -->|Trade Outcome| E[Reflection Agent]
    E -->|Feedback| B
    style A fill:#4F46E5,stroke:#6366F1
    style B fill:#4F46E5,stroke:#6366F1
    style C fill:#4F46E5,stroke:#6366F1
    style D fill:#4F46E5,stroke:#6366F1
    style E fill:#4F46E5,stroke:#6366F1
"""
for agent, state in agent_states.items():
    if state != 'active':
        agent_letter = agent.split('_')[0].upper()[0]
        mermaid_code += f"\n    style {agent_letter} fill:#EF4444,stroke:#DC2626"

st.markdown(f"```mermaid\n{mermaid_code}\n```", unsafe_allow_html=True)
st.caption("Visual representation of agent workflow. Inspired by tools like Langflow for AI pipelines.")