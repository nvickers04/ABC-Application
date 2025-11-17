# Macro-Micro Analysis Framework

## Overview

The Macro-Micro Analysis Framework is the core analytical methodology of the ABC Application system, implementing a hierarchical approach to systematic trading. The framework combines high-level market scanning (macro phase) with detailed security analysis (micro phase) to identify and capitalize on asymmetric trading opportunities.

## Framework Architecture

### Two-Phase Analysis Process

#### Macro Phase (MacroAgent)
**Objective**: Broad market scanning and opportunity identification
- **Asset Universe**: 39+ sectors/assets across all major asset classes
- **Analysis Scope**: Relative strength, momentum, risk-adjusted returns
- **Time Horizons**: Multi-timeframe analysis (1wk to 2yr)
- **Output**: Top 5 performing assets for micro analysis

#### Micro Phase (DataAgent + StrategyAgent)
**Objective**: Deep analysis of selected opportunities
- **Security Selection**: Individual stock/ticker identification
- **Comprehensive Analysis**: Fundamentals, sentiment, flow, microstructure
- **Strategy Generation**: Sophisticated trade setup development
- **Output**: Executable trading strategies with risk management

### Hierarchical Intelligence Flow

```
Macro Scanning (39+ Assets) → Opportunity Selection (Top 5)
                              ↓
Micro Analysis (Individual Securities) → Strategy Development
                              ↓
Risk Assessment → Execution Planning → Performance Monitoring
```

## Macro Phase Implementation

### Asset Universe Coverage

#### Equity Sectors (11 SPDR ETFs)
- **XLY**: Consumer Discretionary - Consumer spending trends
- **XLC**: Communication Services - Media and telecom
- **XLF**: Financials - Banking and financial services
- **XLB**: Materials - Industrial materials and mining
- **XLE**: Energy - Oil, gas, and energy production
- **XLK**: Technology - Software, hardware, semiconductors
- **XLU**: Utilities - Electric and gas utilities
- **XLV**: Health Care - Pharmaceuticals and medical services
- **XLRE**: Real Estate - REITs and real estate services
- **XLP**: Consumer Staples - Essential consumer goods
- **XLI**: Industrials - Manufacturing and industrial services

#### Fixed Income (5 Assets)
- **VLGSX**: Long-term Treasury bonds
- **SPIP**: Treasury Inflation-Protected Securities (TIPS)
- **JNK**: High-yield corporate bonds
- **EMB**: Emerging markets bonds
- **GOVT**: Broad U.S. Treasury bond index

#### Commodities (9 Assets)
- **GC=F**: Gold futures - Precious metals
- **SI=F**: Silver futures - Industrial metals
- **CL=F**: WTI Crude Oil futures - Energy commodities
- **NG=F**: Natural Gas futures - Energy commodities
- **HG=F**: Copper futures - Base metals
- **PL=F**: Platinum futures - Precious metals
- **CORN**: Corn futures - Agricultural commodities
- **WEAT**: Wheat futures - Agricultural commodities
- **SOYB**: Soybean futures - Agricultural commodities
- **CANE**: Sugar futures - Agricultural commodities

#### Currencies (6 Assets)
- **FXE**: Euro vs. USD
- **FXB**: British Pound vs. USD
- **FXY**: Japanese Yen vs. USD
- **FXA**: Australian Dollar vs. USD
- **FXC**: Canadian Dollar vs. USD
- **FXF**: Swiss Franc vs. USD
- **UUP**: U.S. Dollar Index

#### Cryptocurrency (2 Assets)
- **BTC-USD**: Bitcoin vs. U.S. Dollar
- **ETH-USD**: Ethereum vs. U.S. Dollar

### Performance Analysis Methodology

#### Relative Strength Analysis
- **Benchmark**: SPY (S&P 500 ETF) as market benchmark
- **Timeframes**: 1-month, 3-month, 6-month performance comparison
- **Calculation**: (Asset Return - SPY Return) / SPY Return
- **Interpretation**: Positive ratios indicate outperformance

#### Momentum Assessment
- **Rate of Change**: Acceleration/deceleration of performance trends
- **Trend Strength**: Consistency of outperformance periods
- **Volatility-Adjusted**: Risk-adjusted momentum metrics

#### Risk-Adjusted Returns
- **Sharpe Ratio**: Excess return per unit of volatility
- **Sortino Ratio**: Downside deviation-adjusted returns
- **Information Ratio**: Active return relative to tracking error

### Composite Scoring Algorithm

The MacroAgent uses a weighted scoring system:

```
Composite Score = (Relative Strength × 0.40) + (Momentum × 0.30) + (Risk-Adjusted Returns × 0.30)
```

- **Relative Strength (40%)**: Performance vs. benchmark across timeframes
- **Momentum (30%)**: Trend strength and acceleration factors
- **Risk-Adjusted Returns (30%)**: Volatility-adjusted performance metrics

### Selection Process
1. **Calculate Metrics**: Compute all performance metrics for 39+ assets
2. **Apply Scoring**: Generate composite scores using weighted algorithm
3. **Rank Assets**: Sort by composite score (highest to lowest)
4. **Select Top 5**: Choose highest-ranked assets for micro analysis
5. **Weight Allocation**: Equal-weighted allocation (20% each) by default

## Micro Phase Implementation

### Comprehensive Security Analysis

#### Fundamental Analysis (FundamentalDatasub)
- **Financial Statements**: Balance sheet, income statement, cash flow analysis
- **Valuation Metrics**: P/E, P/B, EV/EBITDA, DCF analysis
- **Earnings Quality**: Revenue growth, margin trends, cash flow generation
- **Competitive Position**: Market share, competitive advantages, moat assessment

#### Sentiment Analysis (SentimentDatasub)
- **News Sentiment**: Real-time news impact and sentiment classification
- **Social Media**: Twitter, Reddit, and social sentiment tracking
- **Market Sentiment**: Put/call ratios, VIX analysis, AAII surveys
- **Behavioral Factors**: Crowd psychology and market behavioral patterns

#### Flow Analysis (InstitutionalDatasub + MicrostructureDatasub)
- **Institutional Holdings**: 13F filings and ETF flows
- **Order Flow**: Bid/ask imbalance and trade classification
- **Dark Pool Activity**: Large block trade detection
- **Liquidity Analysis**: Market depth and trading cost assessment

#### Technical Analysis (YfinanceDatasub + MarketDataAppDatasub)
- **Price Patterns**: Chart patterns and trend analysis
- **Technical Indicators**: RSI, MACD, moving averages, Bollinger Bands
- **Volume Analysis**: Volume patterns and accumulation/distribution
- **Market Microstructure**: Order book analysis and trade timing

### Strategy Generation Process

#### Options Strategies (OptionsStrategyAnalyzer)
- **Single-Leg**: Covered calls, protective puts, cash-secured puts
- **Multi-Leg**: Spreads, straddles, condors, butterflies
- **Dynamic Hedging**: Real-time delta and gamma management
- **Volatility Plays**: Long/short volatility positioning

#### Flow-Based Strategies (FlowStrategyAnalyzer)
- **Order Imbalance**: Bid/ask stack analysis for directional bias
- **Institutional Flows**: Smart money positioning and accumulation
- **Dark Pool Detection**: Large order identification and analysis
- **Liquidity Edges**: Optimal execution in high-liquidity periods

#### ML-Driven Strategies (MLStrategyAnalyzer)
- **Predictive Modeling**: Time series forecasting and pattern recognition
- **Feature Engineering**: Advanced technical and fundamental indicators
- **Model Validation**: Out-of-sample testing and overfitting prevention
- **Adaptive Learning**: Strategy adjustment based on market feedback

#### Multi-Asset Strategies (MultiInstrumentStrategyAnalyzer)
- **Statistical Arbitrage**: Mean-reversion and cointegration strategies
- **Cross-Market Plays**: Related asset class correlation exploitation
- **Thematic Investing**: Sector rotation based on macro themes
- **Risk Parity**: Volatility-based asset allocation

## Risk Management Integration

### Portfolio-Level Controls
- **Diversification**: Sector and asset class concentration limits
- **Volatility Targets**: Maximum portfolio volatility constraints
- **Drawdown Limits**: Automatic position reduction during losses
- **Correlation Monitoring**: Dynamic correlation analysis and limits

### Position-Level Controls
- **Position Sizing**: Risk-based allocation algorithms (Kelly Criterion)
- **Stop Losses**: Automatic loss-limiting mechanisms
- **Take Profits**: Profit-taking thresholds and trailing stops
- **Hedging Requirements**: Mandatory hedging for concentrated positions

### Execution Risk Controls
- **Transaction Costs**: Commission and slippage minimization
- **Market Impact**: Position size limits to prevent price movement
- **Liquidity Requirements**: Minimum liquidity thresholds for execution
- **Timing Optimization**: Optimal execution windows and algorithms

## Performance Objectives

### Return Targets
- **Monthly Target**: 10-20% monthly returns
- **Risk Budget**: <5% maximum drawdown
- **Sharpe Ratio**: >2.0 risk-adjusted returns
- **Win Rate**: >60% profitable trades

### Success Metrics
- **Selection Accuracy**: Percentage of selected assets that outperform
- **Strategy Effectiveness**: Risk-adjusted returns of generated strategies
- **Execution Quality**: Transaction cost minimization and timing optimization
- **System Reliability**: 99.9% uptime with automated recovery

## Implementation Workflow

### Daily Cycle
1. **Macro Scanning**: Evaluate 39+ assets using composite scoring
2. **Opportunity Selection**: Choose top 5 assets for detailed analysis
3. **Micro Analysis**: Comprehensive analysis of selected securities
4. **Strategy Generation**: Develop sophisticated trading strategies
5. **Risk Assessment**: Validate strategies against risk constraints
6. **Execution Planning**: Optimize execution parameters and timing
7. **Performance Monitoring**: Track execution and outcomes
8. **Learning Integration**: Feed results back for system improvement

### Weekly/Monthly Cycles
- **Backtesting**: Comprehensive strategy validation on historical data
- **Parameter Optimization**: Refine algorithms based on performance
- **Model Updates**: Update ML models with new data and outcomes
- **System Calibration**: Adjust risk parameters and thresholds

## Advantages of Macro-Micro Framework

### Efficiency Benefits
- **Resource Optimization**: Focus computational resources on highest-probability opportunities
- **Scalability**: Systematic scanning of broad market while maintaining analysis depth
- **Risk Control**: Hierarchical risk management at both macro and micro levels
- **Performance Attribution**: Clear separation of alpha sources (selection vs. execution)

### Intelligence Benefits
- **Market Perspective**: Broad market scanning provides context for individual decisions
- **Pattern Recognition**: Multi-level analysis reveals complex market relationships
- **Adaptive Learning**: Framework learns from both macro trends and micro outcomes
- **Collaborative Intelligence**: Agent debate enhances decision quality at each level

## Future Enhancements

### Advanced Analytics
- **Machine Learning Integration**: ML-based regime detection and asset selection
- **Alternative Data**: Non-traditional data sources for enhanced intelligence
- **Real-Time Adaptation**: Dynamic framework adjustment based on live market data
- **Cross-Asset Expansion**: Additional asset classes and global markets

### Framework Evolution
- **Multi-Timeframe Integration**: Simultaneous analysis across time horizons
- **Network Analysis**: Interconnected asset relationship modeling
- **Behavioral Integration**: Market psychology incorporation at all levels

## Conclusion

The Macro-Micro Analysis Framework provides a systematic, hierarchical approach to trading that combines the breadth of market scanning with the depth of fundamental analysis. By implementing this framework through 22 specialized agents, the ABC Application system achieves institutional-quality investment decisions with the efficiency of automated execution.

The framework's success depends on the seamless integration of macro opportunity identification with micro strategy development, enabled through sophisticated agent collaboration and continuous learning. This approach positions the system to consistently identify and capitalize on asymmetric market opportunities while maintaining strict risk controls.

---

*For implementation details, see the individual agent documentation in AGENTS/.*