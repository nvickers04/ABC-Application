# MacroAgent - Foundational Market Context & Opportunity Prioritization

## Overview

The MacroAgent serves as the **foundational cornerstone** of the collaborative reasoning framework, establishing market context and prioritizing investment opportunities before any detailed analysis begins. It performs high-level market analysis and asset class selection, implementing the "macro foundation" phase that guides all subsequent agent activities in the two-iteration framework.

## Core Responsibilities

### **Strategic Foundation Setting**
- **Market Regime Assessment**: Establishes current market conditions, volatility levels, and macroeconomic trends that inform all subsequent analysis
- **Opportunity Universe Definition**: Scans 39+ sectors/assets to identify the most promising opportunities for focused analysis
- **Context Provision**: Provides market regime intelligence that constrains and guides all agent activities
- **Risk Environment Establishment**: Sets baseline risk parameters and volatility expectations for the entire reasoning process

### Asset Universe Analysis
- **Comprehensive Coverage**: Monitor 39+ sectors/assets across equities, bonds, commodities, currencies, crypto
- **Performance Ranking**: Rank assets based on relative strength, momentum, and risk-adjusted returns
- **Top Asset Selection**: Identify highest-performing assets for micro analysis
- **Allocation Weighting**: Calculate optimal allocation weights for selected assets
- **IBKR Compatibility**: Automatic filtering for Interactive Brokers trading platform compatibility

### Framework Integration
- **Iteration 1 Foundation**: Provides the prioritized opportunity set for comprehensive multi-agent deliberation
- **Context for All Agents**: Supplies market regime context that influences data collection, strategy development, and risk assessment
- **Executive Oversight Input**: Contributes to Iteration 2 strategic review with updated market context

## Key Capabilities

### Multi-Timeframe Analysis
- **Performance Metrics**: 1-week, 1-month, 3-month, 6-month, 1-year, 2-year analysis
- **Relative Strength**: Performance vs. SPY benchmark across timeframes
- **Momentum Analysis**: Rate of change and trend acceleration metrics
- **Risk-Adjusted Returns**: Sharpe-like ratios for comprehensive evaluation

### Composite Scoring Algorithm
- **Relative Strength (40%)**: Performance ratio vs. benchmark
- **Momentum (30%)**: Trend strength and acceleration factors
- **Risk-Adjusted Returns (30%)**: Volatility-adjusted performance metrics

### LLM-Enhanced Analysis
- **Deep Regime Intelligence**: AI-powered market regime assessment beyond simple classification
- **Predictive Insights**: Forward-looking sector predictions and rotation strategies
- **Collaborative Intelligence**: Debate sector selections with Strategy and Data agents
- **Pattern Recognition**: Identify complex macroeconomic relationships
- **FinanceDatabase Integration**: Automated symbol selection based on criteria filtering

## Architecture

### Asset Universe (39+ Assets)

#### Equity Sectors (11)
XLY, XLC, XLF, XLB, XLE, XLK, XLU, XLV, XLRE, XLP, XLI

#### Fixed Income (5)
VLGSX, SPIP, JNK, EMB, GOVT

#### International/Global (3)
EFA, EEM, EUFN

#### Commodities (9)
GC=F, SI=F, CL=F, NG=F, HG=F, PL=F, CORN, WEAT, SOYB, CANE

#### Currencies (6)
FXE, FXB, FXY, FXA, FXC, FXF, UUP

#### Cryptocurrency (2)
BTC-USD, ETH-USD

### Processing Pipeline
1. **Data Collection**: Fetch historical data for all assets
2. **Ratio Calculation**: Compute performance vs. SPY benchmark
3. **Performance Analysis**: Calculate metrics across timeframes
4. **Asset Ranking**: Apply composite scoring algorithm
5. **Top Selection**: Choose top 5 assets for micro analysis
6. **Weight Calculation**: Determine allocation weights

## Integration Points

### DataAgent Collaboration
- **Sector Data Provision**: Receive comprehensive sector performance data
- **Intelligence Sharing**: Exchange macroeconomic insights and market regime analysis
- **Validation**: Cross-verify sector performance metrics

### StrategyAgent Debate
- **Sector Selection**: Debate optimal sector choices based on market regime
- **Allocation Strategy**: Collaborate on portfolio positioning and sector weights
- **Risk Integration**: Align sector selections with risk management constraints

### RiskAgent Coordination
- **Regime Context**: Provide market regime insights for risk assessment
- **Volatility Analysis**: Share macroeconomic volatility and correlation data
- **Stress Testing**: Contribute to portfolio stress testing scenarios

## Memory Integration

### Advanced Memory Systems
- **Debate Outcomes**: Record results of agent debates and consensus decisions
- **Regime History**: Track historical regime transitions and outcomes
- **Sector Patterns**: Long-term sector rotation and performance trends
- **Collaborative Insights**: Shared intelligence from multi-agent interactions

## A2A Communication Protocol

### Sector Selection Format
```json
{
  "agent": "MacroAgent",
  "selected_sectors": [
    {"ticker": "PL=F", "name": "Platinum Futures", "score": 2.6},
    {"ticker": "ETH-USD", "name": "Ethereum", "score": 2.21},
    {"ticker": "XLK", "name": "Technology", "score": 0.54}
  ],
  "allocation_weights": {
    "Platinum Futures": 0.20,
    "Ethereum": 0.20,
    "Technology": 0.20
  },
  "market_regime": "bull_moderate",
  "macro_intelligence": "..."
}
```

## Configuration

### Analysis Parameters
- **Timeframes**: [1mo, 3mo, 6mo] (configurable)
- **Top Selection**: 5 assets (configurable)
- **Scoring Weights**: Relative Strength 40%, Momentum 30%, Risk-Adjusted 30%
- **Benchmark**: SPY (S&P 500 ETF)

### Performance Optimization
- **Redis Caching**: High-performance data caching with TTL management
- **Async Processing**: Concurrent data fetching for improved performance
- **Fallback Mechanisms**: Graceful degradation when data sources unavailable

## Future Enhancements

### Advanced Features
- **Economic Indicator Integration**: Direct incorporation of FRED economic data
- **Machine Learning Regime Detection**: ML-based market regime classification
- **Real-Time Macro Monitoring**: Live sector performance tracking
- **Alternative Asset Classes**: Expansion to additional asset categories

---

*For detailed macro-micro framework documentation, see FRAMEWORKS/macro-micro-analysis.md*