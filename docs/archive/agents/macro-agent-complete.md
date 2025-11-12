# Macro Agent - Complete Implementation Guide
# This file contains the complete Macro Agent implementation and capabilities
# Agents should read this file to understand their role in the comprehensive AI-driven trading system

## Agent Overview
**Role**: Macroeconomic analysis and sector rotation strategy with deep LLM-driven market regime assessment.

**Purpose**: Comprehensive macro analysis enhanced by LLM intelligence for market regime identification, sector rotation strategies, and collaborative trade discovery through multi-agent debate and consensus building.

## Implementation Status - What Has Been Done ✅

### ✅ COMPLETED FEATURES:
- **Multi-Timeframe Analysis**: 1wk, 1mo, 3mo, 6mo, 1y, 2y performance analysis
- **Sector Universe Coverage**: 39+ sectors/assets covering equities, bonds, commodities, currencies, crypto
- **Advanced Memory Integration**: Debate outcomes, agent perspectives, and market regime tracking
- **LLM-Enhanced Analysis**: Deep market regime assessment and predictive insights
- **A2A Communication**: Collaborative debate with Strategy and Data agents
- **Caching System**: Redis-based performance optimization with fallback mechanisms

### 🚧 PARTIALLY IMPLEMENTED:
- **Real-Time Data Integration**: Basic yfinance integration, could expand to more sources

### ❌ NOT YET IMPLEMENTED:
- **Advanced Predictive Modeling**: Could integrate machine learning for regime prediction

## Comprehensive AI-Driven Approach

### FOUNDATION MACRO ANALYSIS (Always Performed):
- Analyze sector performance across multiple timeframes vs SPY benchmark
- Calculate momentum, volatility, and risk-adjusted returns for all assets
- Identify market regime (bull/bear/neutral) based on aggregate performance
- Generate sector rankings and allocation weights based on relative strength

### LLM COMPREHENSIVE ANALYSIS (Always Applied):
- **Deep Regime Intelligence**: Analyze market psychology, economic drivers, and behavioral patterns
- **Predictive Sector Insights**: Generate forward-looking sector predictions and rotation strategies
- **Collaborative Intelligence**: Debate with Strategy and Data agents for consensus building
- **Pattern Recognition**: Identify complex macroeconomic relationships and emerging trends
- **Over-Analysis**: Thorough examination of all macro dimensions for comprehensive understanding

### Collaborative Trade Discovery:
- **Debate sector selections with Strategy Agent** for refined investment decisions based on market regime and risk tolerance
- **Collaborate with Data Agent** on macroeconomic intelligence, sentiment analysis, and sector-specific data validation
- **Share regime insights with Risk Agent** for portfolio positioning and risk management
- **Guide Execution Agent** with sector rotation timing and allocation strategies
- **Provide macro context to Learning Agent** for model refinement

**IMPORTANT**: MacroAgent selects SECTORS for analysis, not individual stocks. The sector ETFs (XLK, XLY, etc.) are used only as proxies for sector performance analysis. Individual stock selection from selected sectors is handled collaboratively by DataAgent and StrategyAgent through debate and consensus building.

## Macro Analysis Framework

### Sector Universe Coverage:
- **Equity Sectors (SPDR Sector ETFs) - 11 Assets**
  - **XLY**: Consumer Discretionary
  - **XLC**: Communication Services
  - **XLF**: Financials
  - **XLB**: Materials
  - **XLE**: Energy
  - **XLK**: Technology
  - **XLU**: Utilities
  - **XLV**: Health Care
  - **XLRE**: Real Estate
  - **XLP**: Consumer Staples
  - **XLI**: Industrials

- **Fixed Income Assets - 5 Assets**
  - **VLGSX**: Vanguard Long-Term Treasury Fund
  - **SPIP**: SPDR Portfolio TIPS ETF
  - **JNK**: SPDR Bloomberg High Yield Bond ETF
  - **EMB**: iShares J.P. Morgan USD Emerging Markets Bond ETF
  - **GOVT**: iShares U.S. Treasury Bond ETF

- **International/Global Assets - 3 Assets**
  - **EFA**: iShares MSCI EAFE ETF
  - **EEM**: iShares MSCI Emerging Markets ETF
  - **EUFN**: iShares MSCI Europe Financials ETF

- **Dividend/Income Focused - 1 Asset**
  - **SDY**: SPDR S&P Dividend ETF

- **Commodities - 9 Assets**
  - **GC=F**: Gold Futures
  - **SI=F**: Silver Futures
  - **CL=F**: WTI Crude Oil Futures
  - **NG=F**: Natural Gas Futures
  - **HG=F**: Copper Futures
  - **PL=F**: Platinum Futures
  - **CORN**: Teucrium Corn Fund
  - **WEAT**: Teucrium Wheat Fund
  - **SOYB**: Teucrium Soybean Fund
  - **CANE**: Teucrium Sugar Fund

- **Currency Assets (vs USD) - 6 Assets**
  - **FXE**: Invesco CurrencyShares Euro Trust
  - **FXB**: Invesco CurrencyShares British Pound Trust
  - **FXY**: Invesco CurrencyShares Japanese Yen Trust
  - **FXA**: Invesco CurrencyShares Australian Dollar Trust
  - **FXC**: Invesco CurrencyShares Canadian Dollar Trust
  - **FXF**: Invesco CurrencyShares Swiss Franc Trust
  - **UUP**: Invesco DB USD Index Bullish Fund

- **Cryptocurrency Assets - 2 Assets**
  - **BTC-USD**: Bitcoin vs US Dollar
  - **ETH-USD**: Ethereum vs US Dollar

**Total Universe**: 39+ assets covering equities, fixed income, commodities, currencies, and cryptocurrencies.

### Performance Metrics:
- **Relative Strength**: Performance vs SPY benchmark across timeframes
- **Momentum Analysis**: Rate of change in performance ratios
- **Volatility Assessment**: Risk metrics across different time horizons
- **Risk-Adjusted Returns**: Sharpe-like ratios for sector evaluation
- **Composite Scoring**: Weighted combination of all performance factors

## LLM-Enhanced Macro Intelligence

### Deep Analysis Capabilities:
- **Market Regime Assessment**: Multi-dimensional regime classification beyond simple bull/bear
- **Economic Driver Analysis**: Fundamental economic factors influencing sector performance
- **Behavioral Finance Insights**: Market psychology and crowd behavior patterns
- **Predictive Modeling**: Forward-looking sector rotation predictions
- **Risk Intelligence**: Macro-level volatility and correlation analysis

### Collaborative Intelligence Sharing:
- **Strategy Agent**: Debate sector selections and refine investment strategies
- **Data Agent**: Share macroeconomic insights and validate data-driven hypotheses
- **Risk Agent**: Provide regime context for portfolio risk management
- **Learning Agent**: Contribute to macroeconomic pattern recognition
- **Execution Agent**: Guide sector allocation and rotation timing

## Memory Integration

### Advanced Memory Systems:
- **Debate Memory**: Records outcomes of agent debates and consensus decisions
- **Agent Perspectives**: Stores individual agent feedback and reasoning
- **Market Regime History**: Tracks historical regime transitions and outcomes
- **Sector Performance Patterns**: Long-term sector rotation and performance trends
- **Collaborative Insights**: Shared intelligence from multi-agent interactions

### Memory Types Used:
- **Long-term Memory**: Sector performance patterns and regime transitions
- **Episodic Memory**: Specific debate outcomes and agent interactions
- **Semantic Memory**: Macroeconomic concepts and relationships
- **Shared Memory**: Cross-agent intelligence and collaborative insights

## A2A Communication Protocol

### Debate Communication Format:
```json
{
  "debate_context": {
    "sector_rankings": [...],
    "market_regime": "bull_moderate",
    "timeframes": ["1mo", "3mo", "6mo"],
    "performance_metrics": {...}
  },
  "agent_feedback": {
    "strategy_agent": {
      "sector_preferences": {...},
      "risk_assessment": "...",
      "recommended_adjustments": [...]
    },
    "data_agent": {
      "sentiment_analysis": "...",
      "fundamental_insights": {...},
      "data_validation": [...]
    }
  },
  "consensus_decision": {
    "refined_rankings": [...],
    "allocation_weights": {...},
    "rationale": "..."
  }
}
```

### Collaborative Workflows:
- **Sector Selection Debate**: Multi-agent discussion for refined sector rankings
- **Regime Assessment Collaboration**: Cross-agent validation of market regime classification
- **Strategy Integration**: Debate-based refinement of investment strategies
- **Risk Coordination**: Macro context sharing for risk management decisions

## Technical Architecture

### Data Processing Pipeline:
- **Multi-Source Aggregation**: yfinance primary with caching and fallback mechanisms
- **Performance Calculation**: Relative strength analysis across multiple timeframes
- **Risk Assessment**: Volatility and correlation analysis for all assets
- **Ranking Algorithm**: Composite scoring with momentum and risk-adjustment factors

### Caching System:
- **Redis Integration**: High-performance caching for market data
- **TTL Management**: Configurable cache expiration for different data types
- **Fallback Mechanisms**: Graceful degradation when cache/redis unavailable
- **Performance Optimization**: Reduced API calls through intelligent caching

### LLM Integration:
- **Reasoning with Context**: Comprehensive macro context for intelligent analysis
- **Collaborative Synthesis**: Multi-agent debate and consensus building
- **Predictive Enhancement**: Forward-looking analysis beyond foundation metrics
- **Iterative Refinement**: Continuous improvement through agent collaboration

## Future Enhancements

### Planned Improvements:
- Advanced machine learning models for regime prediction
- Real-time data integration from additional sources
- Enhanced collaborative debate mechanisms
- More sophisticated macroeconomic modeling

---

# Macro Agent Implementation (Comprehensive AI Approach)

{base_prompt}
Analyze macroeconomic conditions comprehensively using AI-driven analysis: foundation macro processing provides quantitative sector analysis, while LLM reasoning delivers deep market regime intelligence and collaborative insights for sector rotation strategies.

FOUNDATION MACRO ANALYSIS (Always Performed):
- Analyze sector performance across multiple timeframes vs SPY benchmark
- Calculate momentum, volatility, and risk-adjusted returns for all tradable assets
- Identify market regime based on aggregate performance metrics
- Generate sector rankings and allocation weights based on relative strength

LLM COMPREHENSIVE ANALYSIS (Always Applied):
- Assess market regime with deep intelligence beyond simple bull/bear classification
- Analyze economic drivers, behavioral patterns, and market psychology
- Generate forward-looking sector predictions and rotation strategies
- Debate with Strategy and Data agents for consensus building and refined decisions
- Provide thorough examination of macroeconomic factors for optimal sector selection

Work collaboratively with other agents for comprehensive sector rotation:
- Debate sector selections with Strategy Agent for refined investment strategies
- Collaborate with Data Agent on macroeconomic intelligence and data validation
- Share regime insights with Risk Agent for portfolio positioning and risk management
- Contribute macro patterns to Learning Agent for continuous model improvement
- Guide Execution Agent with sector allocation timing and rotation strategies

Output: Comprehensive macro intelligence for A2A collaboration; include foundation metrics + deep LLM insights for sector rotation (e.g., "Deep Analysis: Bull moderate regime with 72% confidence, technology sector momentum signals rotation opportunity; collaborating with Strategy for optimal entry timing").</content>
<parameter name="filePath">c:\Users\nvick\Desktop\GROK-IBKR\agents\macro-agent-complete.md