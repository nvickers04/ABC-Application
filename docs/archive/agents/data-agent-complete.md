# Data Agent - Complete Implementation Guide
# This file contains the complete Data Agent implementation and capabilities
# Agents should read this file to understand their role in the comprehensive AI-driven trading system

## Agent Overview
**Role**: Comprehensive market data aggregation, analysis, and intelligence gathering with deep LLM-driven insights.

**Purpose**: Foundation data processing enhanced by LLM analysis for market intelligence, sentiment interpretation, and predictive insights that drive collaborative trade discovery.

## Implementation Status - What Has Been Done ✅

### ✅ COMPLETED FEATURES:
- **Multi-Source Data Integration**: yfinance, MarketDataApp, FRED, NewsAPI, CurrentsAPI, Twitter API, Kalshi
- **Real-Time Data Processing**: Live market feeds and sentiment analysis via LangChain tools
- **Subagent Architecture**: Specialized data collection and processing agents
- **LLM-Enhanced Analysis**: Deep market intelligence and predictive insights using xAI/Grok
- **A2A Communication**: Comprehensive data sharing with all agents via JSON/DataFrame formats
- **Memory Systems**: Historical data patterns and market intelligence storage
- **LangChain Tool Integration**: Full access to @tool decorated functions for data processing

### 🚧 PARTIALLY IMPLEMENTED:
- **Advanced Predictive Modeling**: Basic LLM insights, could expand to more sophisticated predictions

### ❌ NOT YET IMPLEMENTED:
- **Real-Time Predictive Analytics**: Advanced forward-looking market analysis

## Comprehensive AI-Driven Approach

### FOUNDATION DATA PROCESSING (Always Performed):
- Aggregate data from all sources using LangChain tools (yfinance_data_tool, marketdataapp_api_tool, economic_data_tool, news_data_tool, sentiment_analysis_tool, twitter_sentiment_tool, currents_news_tool, kalshi_odds_tool)
- Process market data, economic indicators, and sentiment feeds
- Generate quantitative metrics and technical indicators
- Validate data quality and cross-reference sources

### LLM COMPREHENSIVE ANALYSIS (Always Applied):
- **Deep Market Intelligence**: Analyze market sentiment, news impact, and behavioral patterns using xAI/Grok API
- **Predictive Insights**: Generate forward-looking market predictions and risk assessments
- **Collaborative Intelligence**: Share insights with Strategy, Risk, and other agents for trade discovery
- **Pattern Recognition**: Identify complex market relationships and emerging trends
- **Over-Analysis**: Thorough examination of all data dimensions for comprehensive understanding

### Collaborative Trade Discovery:
- **Debate sector selections with Strategy Agent** using macroeconomic data, sentiment analysis, and market intelligence to determine which sectors MacroAgent identified are most suitable for trading
- **Collaborate with MacroAgent** on sector performance validation and regime assessment
- **Work with Strategy Agent** to identify specific individual tickers from selected sectors that show the strongest trade setups
- **Share intelligence with Risk Agent** on market conditions and individual stock risk profiles
- **Provide comprehensive data context** for all agent decision-making

**SECTOR TO TICKER SELECTION PROCESS**:
1. Receive sector selections from MacroAgent (e.g., "Technology", "Energy", "Consumer Discretionary")
2. Debate with StrategyAgent on which sectors to prioritize based on data-driven analysis
3. For selected sectors, use data analysis tools to identify individual stocks
4. Analyze individual tickers with full data pipeline (fundamentals, sentiment, microstructure, etc.)
5. Collaborate with StrategyAgent to select optimal individual tickers for trading

## Data Sources & Integration

### Primary Data Feeds:
- **yfinance**: Stock prices, fundamentals, options data via yfinance_data_tool
- **MarketDataApp**: Premium real-time trading data, market depth, order flow via marketdataapp_api_tool and marketdataapp_websocket_tool
- **FRED**: Economic indicators, interest rates, inflation data via economic_data_tool
- **NewsAPI/CurrentsAPI**: Real-time news and market-moving headlines via news_data_tool and currents_news_tool
- **Twitter API**: Social sentiment and market discussions via twitter_sentiment_tool
- **Kalshi**: Prediction market odds and event contracts via kalshi_odds_tool
- **xAI/Grok**: Sentiment analysis and market intelligence via sentiment_analysis_tool

### Data Processing Pipeline:
- **Real-Time Aggregation**: Continuous data collection and normalization via LangChain tools
- **Quality Validation**: Cross-source verification and anomaly detection
- **Feature Engineering**: Technical indicators, sentiment scores, market microstructure
- **Intelligence Enhancement**: LLM-driven pattern recognition and predictive analysis

## LLM-Enhanced Market Intelligence

### Deep Analysis Capabilities:
- **Sentiment Analysis**: Multi-dimensional sentiment interpretation across news, social media, and market data using xAI/Grok
- **Market Psychology**: Behavioral finance insights and crowd behavior analysis
- **Predictive Modeling**: Forward-looking market predictions based on comprehensive data analysis
- **Risk Intelligence**: Market condition assessments and volatility forecasting
- **Opportunity Discovery**: Identification of asymmetric opportunities and market inefficiencies

### Collaborative Intelligence Sharing:
- **Strategy Agent**: Provide market insights for trade opportunity identification
- **Risk Agent**: Share volatility and market condition intelligence
- **Learning Agent**: Contribute to model refinement and pattern recognition
- **Execution Agent**: Provide real-time market context for trade execution

## Subagent Architecture

### Data Collection Subagents:
- **Market Data Subagent**: Real-time price and volume data collection via yfinance_data_tool and marketdataapp_api_tool
- **Economic Data Subagent**: Macroeconomic indicators and policy data via economic_data_tool
- **Sentiment Subagent**: News, social media, and market sentiment analysis via sentiment_analysis_tool, twitter_sentiment_tool, currents_news_tool
- **Options Data Subagent**: Derivatives data and volatility surface analysis via options_greeks_calc_tool

### Processing Subagents:
- **Technical Analysis Subagent**: Chart patterns, indicators, and trend analysis
- **Fundamental Analysis Subagent**: Company financials and valuation metrics via fundamental_data_tool
- **Flow Analysis Subagent**: Order flow and market microstructure analysis via microstructure_analysis_tool and flow_alpha_calc_tool
- **Intelligence Subagent**: LLM-driven comprehensive market analysis using xAI/Grok

---

# Data Agent Subagents - Complete Implementation Guide

## YFinance Data Subagent
**File**: `src/agents/data_subs/yfinance_datasub.py`
**Role**: Primary market data collection and real-time price feeds
**LLM Integration**: Uses xAI/Grok memory for initial research on data patterns and market behavior

### Capabilities:
- **Real-time Price Data**: Live stock prices, volume, and market data via yfinance_data_tool
- **Historical Data**: Comprehensive historical price series for analysis
- **Options Data**: Complete options chains with Greeks calculations via options_greeks_calc_tool
- **Fundamentals**: Company financials, earnings, and key metrics via fundamental_data_tool
- **Technical Indicators**: RSI, MACD, moving averages, and custom indicators

### Memory System:
- **Subagent-Level Memory**: Collaborative data patterns and market insights shared between data subagents
- **Research Integration**: xAI/Grok memory used for initial market behavior analysis
- **Pattern Recognition**: Historical price patterns and volatility regimes
- **Cross-Subagent Sharing**: Data insights shared with economic, sentiment, and fundamental subagents

### LLM-Driven Analysis:
- **Market Behavior Research**: Initial xAI/Grok analysis of price patterns and market psychology
- **Data Quality Assessment**: AI-powered validation of data integrity and anomalies
- **Predictive Insights**: Forward-looking price movement analysis
- **Collaborative Intelligence**: Shares insights with other data subagents for comprehensive analysis

---

## Economic Data Subagent
**File**: `src/agents/data_subs/economic_datasub.py`
**Role**: Macroeconomic data aggregation and policy impact analysis
**LLM Integration**: Uses xAI/Grok memory for economic research and policy analysis

### Capabilities:
- **Economic Indicators**: GDP, unemployment, inflation, interest rates from FRED via economic_data_tool
- **Policy Data**: Federal Reserve announcements, fiscal policy changes
- **Global Economics**: International trade data, currency impacts
- **Market Correlations**: Economic data impact on asset classes
- **Forward Guidance**: Central bank projections and market expectations

### Memory System:
- **Economic Pattern Memory**: Historical economic cycles and policy responses
- **Collaborative Research**: Shares economic insights with market data and sentiment subagents
- **Policy Impact Tracking**: Records and analyzes past policy decisions
- **Cross-Market Correlations**: Economic data relationships with various asset classes

### LLM-Driven Analysis:
- **Policy Research**: Deep analysis of economic policy implications using xAI/Grok
- **Market Impact Assessment**: AI analysis of economic data effects on markets
- **Predictive Economics**: Forward-looking economic scenario analysis
- **Collaborative Synthesis**: Integrates economic insights with market and sentiment data

---

## Sentiment Data Subagent
**File**: `src/agents/data_subs/sentiment_datasub.py`
**Role**: Multi-dimensional sentiment analysis across news, social media, and market data
**LLM Integration**: Uses xAI/Grok memory for sentiment pattern research and behavioral analysis

### Capabilities:
- **News Sentiment**: Real-time news analysis and impact assessment via news_data_tool and currents_news_tool
- **Social Media**: Twitter, Reddit, and social sentiment tracking via twitter_sentiment_tool
- **Market Sentiment**: Put/call ratios, VIX analysis, volatility sentiment
- **Behavioral Finance**: Crowd psychology and market behavioral patterns
- **Event Impact**: News event sentiment analysis and market reactions

### Memory System:
- **Sentiment Pattern Memory**: Historical sentiment patterns and market reactions
- **Collaborative Sharing**: Sentiment insights shared with economic and market data subagents
- **Event Response Tracking**: Records market reactions to sentiment events
- **Behavioral Pattern Recognition**: Identifies recurring sentiment-driven market behaviors

### LLM-Driven Analysis:
- **Sentiment Research**: xAI/Grok-powered analysis of sentiment patterns and implications
- **Behavioral Psychology**: Deep analysis of market psychology and crowd behavior
- **Impact Prediction**: AI assessment of sentiment impact on price movements
- **Collaborative Intelligence**: Integrates sentiment with economic and market data

---

## Fundamental Data Subagent
**File**: `src/agents/data_subs/fundamental_datasub.py`
**Role**: Company financial analysis and valuation metrics
**LLM Integration**: Uses xAI/Grok memory for fundamental research and valuation analysis

### Capabilities:
- **Financial Statements**: Balance sheets, income statements, cash flow analysis via fundamental_data_tool and sec_edgar_fundamentals_tool
- **Valuation Metrics**: P/E, P/B, EV/EBITDA, and custom valuation models
- **Earnings Analysis**: Historical earnings and forward estimates
- **Competitive Analysis**: Industry comparisons and competitive positioning
- **Growth Metrics**: Revenue growth, margin analysis, and sustainability assessment

### Memory System:
- **Fundamental Pattern Memory**: Historical valuation patterns and earnings trends
- **Company-Specific Memory**: Individual company financial histories and patterns
- **Industry Analysis Memory**: Sector-specific fundamental trends and relationships
- **Collaborative Valuation**: Shares fundamental insights with market and economic subagents

### LLM-Driven Analysis:
- **Fundamental Research**: xAI/Grok-powered company and industry analysis
- **Valuation Research**: Deep analysis of valuation methodologies and market implications
- **Growth Assessment**: AI evaluation of company growth prospects and risks
- **Collaborative Synthesis**: Integrates fundamental analysis with market and economic data

---

## Institutional Data Subagent
**File**: `src/agents/data_subs/institutional_datasub.py`
**Role**: Institutional holdings analysis and flow detection
**LLM Integration**: Uses xAI/Grok memory for institutional behavior research and flow analysis

### Capabilities:
- **13F Filings**: Institutional holdings and position changes via thirteen_f_filings_tool and sec_edgar_13f_tool
- **ETF Flows**: Institutional ETF inflow/outflow analysis
- **Dark Pool Activity**: Large block trade detection and analysis via marketdataapp_api_tool
- **Institutional Sentiment**: Smart money positioning and trends via institutional_holdings_analysis_tool
- **Flow Analysis**: Order flow patterns and market impact assessment

### Memory System:
- **Institutional Pattern Memory**: Historical institutional behavior patterns
- **Flow Tracking Memory**: Records and analyzes institutional flow patterns
- **Position Change Memory**: Tracks institutional position adjustments over time
- **Collaborative Intelligence**: Shares institutional insights with flow and market subagents

### LLM-Driven Analysis:
- **Institutional Research**: xAI/Grok analysis of institutional behavior and implications
- **Flow Pattern Analysis**: AI-powered detection of institutional flow patterns
- **Market Impact Assessment**: Deep analysis of institutional activity effects
- **Collaborative Synthesis**: Integrates institutional data with market and flow analysis

---

## News Data Subagent
**File**: `src/agents/data_subs/news_datasub.py`
**Role**: Real-time news aggregation and impact analysis
**LLM Integration**: Uses xAI/Grok memory for news pattern research and impact analysis

### Capabilities:
- **Real-time News**: Breaking news and market-moving headlines via news_data_tool and currents_news_tool
- **News Categorization**: Automated news classification and prioritization
- **Impact Assessment**: Historical news impact on price movements
- **Event Detection**: Major news events and market reaction patterns
- **Source Credibility**: News source validation and reliability assessment

### Memory System:
- **News Pattern Memory**: Historical news impact patterns and market reactions
- **Event Response Memory**: Records market reactions to various news events
- **Source Credibility Memory**: Tracks news source reliability and accuracy
- **Collaborative Sharing**: News insights shared with sentiment and market subagents

### LLM-Driven Analysis:
- **News Research**: xAI/Grok-powered analysis of news patterns and implications
- **Impact Prediction**: AI assessment of news event potential market impact
- **Event Analysis**: Deep analysis of news events and market psychology
- **Collaborative Intelligence**: Integrates news analysis with sentiment and market data

---

## Microstructure Data Subagent
**File**: `src/agents/data_subs/microstructure_datasub.py`
**Role**: Market microstructure analysis and order flow intelligence
**LLM Integration**: Uses xAI/Grok memory for microstructure research and flow analysis

### Capabilities:
- **Order Book Analysis**: Bid/ask spread analysis and market depth via microstructure_analysis_tool
- **Trade Classification**: Algorithmic vs human trade detection
- **Liquidity Analysis**: Market liquidity assessment and impact costs
- **High-Frequency Data**: Microsecond-level market data analysis via marketdataapp_websocket_tool
- **Flow Toxicity**: Detection of aggressive vs passive order flow via flow_alpha_calc_tool

### Memory System:
- **Microstructure Pattern Memory**: Historical order flow and microstructure patterns
- **Liquidity Memory**: Records market liquidity conditions and changes
- **Flow Pattern Memory**: Tracks various order flow patterns and implications
- **Collaborative Intelligence**: Shares microstructure insights with institutional and market subagents

### LLM-Driven Analysis:
- **Microstructure Research**: xAI/Grok analysis of market microstructure patterns
- **Flow Intelligence**: AI-powered order flow analysis and prediction
- **Liquidity Assessment**: Deep analysis of market liquidity dynamics
- **Collaborative Synthesis**: Integrates microstructure data with institutional and market analysis

---

## MarketDataApp Data Subagent
**File**: `src/agents/data_subs/marketdataapp_datasub.py`
**Role**: Premium market data from MarketDataApp API
**LLM Integration**: Uses xAI/Grok memory for premium data research and analysis

### Capabilities:
- **Premium Quotes**: High-quality real-time and historical quotes via marketdataapp_api_tool
- **Trade Data**: Detailed trade execution data and timestamps via marketdataapp_websocket_tool
- **Options Data**: Complete options chains and market data
- **Dark Pool Detection**: Large block trade identification
- **Market Depth**: Level 2 order book data and analysis

### Memory System:
- **Premium Data Memory**: Historical premium market data patterns
- **Trade Pattern Memory**: Records and analyzes trade execution patterns
- **Options Memory**: Options market data and volatility patterns
- **Collaborative Intelligence**: Shares premium insights with market and flow subagents

### LLM-Driven Analysis:
- **Premium Data Research**: xAI/Grok-powered analysis of high-quality market data
- **Trade Analysis**: Deep analysis of trade execution patterns and implications
- **Options Intelligence**: AI assessment of options market dynamics
- **Collaborative Synthesis**: Integrates premium data with market and flow analysis

---

## Kalshi Data Subagent
**File**: `src/agents/data_subs/kalshi_datasub.py`
**Role**: Event contract data and prediction market intelligence
**LLM Integration**: Uses xAI/Grok memory for prediction market research and event analysis

### Capabilities:
- **Event Contracts**: Political, economic, and market event predictions via kalshi_odds_tool
- **Prediction Markets**: Crowd-sourced probability assessments
- **Event Impact**: Historical event outcome analysis
- **Market Expectations**: Real-time market probability assessments
- **Risk Premia**: Event risk pricing and market sentiment

### Memory System:
- **Event Pattern Memory**: Historical event outcomes and market reactions
- **Prediction Memory**: Records prediction market accuracy and patterns
- **Event Impact Memory**: Tracks market reactions to various events
- **Collaborative Intelligence**: Shares prediction insights with sentiment and economic subagents

### LLM-Driven Analysis:
- **Event Research**: xAI/Grok analysis of prediction markets and event implications
- **Probability Assessment**: AI evaluation of market probabilities and expectations
- **Event Impact Analysis**: Deep analysis of event outcomes and market effects
- **Collaborative Synthesis**: Integrates prediction data with sentiment and economic analysis

---

## A2A Communication Protocol

### Data Sharing Formats:
```json
{
  "market_data": {
    "prices": {...},
    "sentiment": {...},
    "volatility": {...}
  },
  "intelligence": {
    "predictions": [...],
    "risk_assessments": [...],
    "opportunities": [...]
  },
  "collaboration_signals": {
    "strategy_insights": [...],
    "risk_warnings": [...],
    "execution_context": {...}
  }
}
```

### Collaborative Workflows:
- **Trade Discovery Loops**: Iterative analysis with Strategy Agent for opportunity identification
- **Risk Assessment Collaboration**: Real-time market intelligence sharing with Risk Agent
- **Learning Integration**: Data pattern sharing with Learning Agent for model improvement
- **Execution Support**: Real-time market context provision to Execution Agent

## Memory and LLM Integration

### Advanced Memory Systems
- **Multi-Level Memory**: Short-term, long-term, episodic, semantic, and procedural memory
- **Collaborative Memory**: Shared memory spaces for cross-agent intelligence
- **Pattern Storage**: Historical market patterns and predictive insights
- **Context Preservation**: Maintaining analysis context across sessions

### LLM Reasoning Capabilities
- **Deep Market Intelligence**: xAI/Grok-powered market analysis
- **Predictive Reasoning**: Forward-looking market predictions and risk assessments
- **Collaborative Synthesis**: Multi-agent debate and consensus building
- **Contextual Understanding**: Comprehensive market context for intelligent decisions

### Integration Architecture
- **LangChain Tool Integration**: Full access to @tool decorated functions for data processing
- **A2A Memory Sharing**: Cross-agent memory coordination and intelligence sharing
- **Reasoning with Context**: LLM analysis enhanced by historical patterns and collaborative insights
- **Adaptive Learning**: Continuous improvement through memory-driven learning

## Technical Architecture

### Data Processing Engine:
- **Real-Time Streaming**: Continuous data ingestion and processing via LangChain tools
- **Multi-Threaded Analysis**: Parallel processing of different data types
- **Memory Management**: Efficient storage and retrieval of historical data
- **Quality Assurance**: Automated data validation and error correction

### LLM Integration:
- **Context Provision**: Comprehensive data context for all LLM reasoning
- **Collaborative Analysis**: Cross-agent intelligence sharing and synthesis
- **Predictive Enhancement**: Forward-looking analysis beyond foundation metrics
- **Iterative Refinement**: Continuous improvement through agent collaboration

## Future Enhancements

### Planned Improvements:
- Advanced predictive analytics with machine learning models
- Enhanced real-time collaboration capabilities
- Expanded data source integration
- More sophisticated market intelligence algorithms

---

# Data Agent Implementation (Comprehensive AI Approach)

{base_prompt}
Process market data comprehensively using AI-driven analysis: foundation data processing provides quantitative metrics, while LLM reasoning delivers deep market intelligence and collaborative insights for trade discovery.

FOUNDATION DATA PROCESSING (Always Performed):
- Aggregate data from all sources using LangChain tools (yfinance_data_tool, marketdataapp_api_tool, economic_data_tool, news_data_tool, sentiment_analysis_tool, twitter_sentiment_tool, currents_news_tool, kalshi_odds_tool)
- Process market data, economic indicators, and sentiment feeds
- Generate quantitative metrics and technical indicators
- Validate data quality and cross-reference sources

LLM COMPREHENSIVE ANALYSIS (Always Applied):
- Analyze market sentiment, news impact, and behavioral patterns with deep intelligence using xAI/Grok
- Generate forward-looking market predictions and risk assessments
- Identify complex market relationships and emerging trends through over-analysis
- Collaborate with Strategy, Risk, and other agents for comprehensive trade discovery
- Provide thorough examination of all data dimensions for winning trade identification

Work collaboratively with other agents to discover winning trades:
- Share deep market intelligence with Strategy Agent for opportunity identification
- Collaborate with Risk Agent on comprehensive market condition assessments
- Contribute intelligence to Learning Agent for continuous model refinement
- Provide real-time market context to Execution Agent for optimal trade execution

Output: Comprehensive data intelligence for A2A collaboration; include foundation metrics + deep LLM insights for trade discovery (e.g., "Deep Analysis: Market sentiment bullish with 78% confidence, volatility contraction signals opportunity in SPY calls; collaborating with Strategy for asymmetric trade setup").

# Data Agent Subagents - Complete Implementation Guide

## YFinance Data Subagent
**File**: `src/agents/data_subs/yfinance_datasub.py`
**Role**: Primary market data collection and real-time price feeds
**LLM Integration**: Uses Grok memory for initial research on data patterns and market behavior

### Capabilities:
- **Real-time Price Data**: Live stock prices, volume, and market data via yfinance
- **Historical Data**: Comprehensive historical price series for analysis
- **Options Data**: Complete options chains with Greeks calculations
- **Fundamentals**: Company financials, earnings, and key metrics
- **Technical Indicators**: RSI, MACD, moving averages, and custom indicators

### Memory System:
- **Subagent-Level Memory**: Collaborative data patterns and market insights shared between data subagents
- **Research Integration**: Grok memory used for initial market behavior analysis
- **Pattern Recognition**: Historical price patterns and volatility regimes
- **Cross-Subagent Sharing**: Data insights shared with economic, sentiment, and fundamental subagents

### LLM-Driven Analysis:
- **Market Behavior Research**: Initial Grok analysis of price patterns and market psychology
- **Data Quality Assessment**: AI-powered validation of data integrity and anomalies
- **Predictive Insights**: Forward-looking price movement analysis
- **Collaborative Intelligence**: Shares insights with other data subagents for comprehensive analysis

---

## Economic Data Subagent
**File**: `src/agents/data_subs/economic_datasub.py`
**Role**: Macroeconomic data aggregation and policy impact analysis
**LLM Integration**: Uses Grok memory for economic research and policy analysis

### Capabilities:
- **Economic Indicators**: GDP, unemployment, inflation, interest rates from FRED
- **Policy Data**: Federal Reserve announcements, fiscal policy changes
- **Global Economics**: International trade data, currency impacts
- **Market Correlations**: Economic data impact on asset classes
- **Forward Guidance**: Central bank projections and market expectations

### Memory System:
- **Economic Pattern Memory**: Historical economic cycles and policy responses
- **Collaborative Research**: Shares economic insights with market data and sentiment subagents
- **Policy Impact Tracking**: Records and analyzes past policy decisions
- **Cross-Market Correlations**: Economic data relationships with various asset classes

### LLM-Driven Analysis:
- **Policy Research**: Deep analysis of economic policy implications using Grok
- **Market Impact Assessment**: AI analysis of economic data effects on markets
- **Predictive Economics**: Forward-looking economic scenario analysis
- **Collaborative Synthesis**: Integrates economic insights with market and sentiment data

---

## Sentiment Data Subagent
**File**: `src/agents/data_subs/sentiment_datasub.py`
**Role**: Multi-dimensional sentiment analysis across news, social media, and market data
**LLM Integration**: Uses Grok memory for sentiment pattern research and behavioral analysis

### Capabilities:
- **News Sentiment**: Real-time news analysis and impact assessment
- **Social Media**: Twitter, Reddit, and social sentiment tracking
- **Market Sentiment**: Put/call ratios, VIX analysis, volatility sentiment
- **Behavioral Finance**: Crowd psychology and market behavioral patterns
- **Event Impact**: News event sentiment analysis and market reactions

### Memory System:
- **Sentiment Pattern Memory**: Historical sentiment patterns and market reactions
- **Collaborative Sharing**: Sentiment insights shared with economic and market data subagents
- **Event Response Tracking**: Records market reactions to sentiment events
- **Behavioral Pattern Recognition**: Identifies recurring sentiment-driven market behaviors

### LLM-Driven Analysis:
- **Sentiment Research**: Grok-powered analysis of sentiment patterns and implications
- **Behavioral Psychology**: Deep analysis of market psychology and crowd behavior
- **Impact Prediction**: AI assessment of sentiment impact on price movements
- **Collaborative Intelligence**: Integrates sentiment with economic and market data

---

## Fundamental Data Subagent
**File**: `src/agents/data_subs/fundamental_datasub.py`
**Role**: Company financial analysis and valuation metrics
**LLM Integration**: Uses Grok memory for fundamental research and valuation analysis

### Capabilities:
- **Financial Statements**: Balance sheets, income statements, cash flow analysis
- **Valuation Metrics**: P/E, P/B, EV/EBITDA, and custom valuation models
- **Earnings Analysis**: Historical earnings and forward estimates
- **Competitive Analysis**: Industry comparisons and competitive positioning
- **Growth Metrics**: Revenue growth, margin analysis, and sustainability assessment

### Memory System:
- **Fundamental Pattern Memory**: Historical valuation patterns and earnings trends
- **Company-Specific Memory**: Individual company financial histories and patterns
- **Industry Analysis Memory**: Sector-specific fundamental trends and relationships
- **Collaborative Valuation**: Shares fundamental insights with market and economic subagents

### LLM-Driven Analysis:
- **Fundamental Research**: Grok-powered company and industry analysis
- **Valuation Research**: Deep analysis of valuation methodologies and market implications
- **Growth Assessment**: AI evaluation of company growth prospects and risks
- **Collaborative Synthesis**: Integrates fundamental analysis with market and economic data

---

## Institutional Data Subagent
**File**: `src/agents/data_subs/institutional_datasub.py`
**Role**: Institutional holdings analysis and flow detection
**LLM Integration**: Uses Grok memory for institutional behavior research and flow analysis

### Capabilities:
- **13F Filings**: Institutional holdings and position changes
- **ETF Flows**: Institutional ETF inflow/outflow analysis
- **Dark Pool Activity**: Large block trade detection and analysis
- **Institutional Sentiment**: Smart money positioning and trends
- **Flow Analysis**: Order flow patterns and market impact assessment

### Memory System:
- **Institutional Pattern Memory**: Historical institutional behavior patterns
- **Flow Tracking Memory**: Records and analyzes institutional flow patterns
- **Position Change Memory**: Tracks institutional position adjustments over time
- **Collaborative Intelligence**: Shares institutional insights with flow and market subagents

### LLM-Driven Analysis:
- **Institutional Research**: Grok analysis of institutional behavior and implications
- **Flow Pattern Analysis**: AI-powered detection of institutional flow patterns
- **Market Impact Assessment**: Deep analysis of institutional activity effects
- **Collaborative Synthesis**: Integrates institutional data with market and flow analysis

---

## News Data Subagent
**File**: `src/agents/data_subs/news_datasub.py`
**Role**: Real-time news aggregation and impact analysis
**LLM Integration**: Uses Grok memory for news pattern research and impact analysis

### Capabilities:
- **Real-time News**: Breaking news and market-moving headlines
- **News Categorization**: Automated news classification and prioritization
- **Impact Assessment**: Historical news impact on price movements
- **Event Detection**: Major news events and market reaction patterns
- **Source Credibility**: News source validation and reliability assessment

### Memory System:
- **News Pattern Memory**: Historical news impact patterns and market reactions
- **Event Response Memory**: Records market reactions to various news events
- **Source Credibility Memory**: Tracks news source reliability and accuracy
- **Collaborative Sharing**: News insights shared with sentiment and market subagents

### LLM-Driven Analysis:
- **News Research**: Grok-powered analysis of news patterns and implications
- **Impact Prediction**: AI assessment of news event potential market impact
- **Event Analysis**: Deep analysis of news events and market psychology
- **Collaborative Intelligence**: Integrates news analysis with sentiment and market data

---

## Microstructure Data Subagent
**File**: `src/agents/data_subs/microstructure_datasub.py`
**Role**: Market microstructure analysis and order flow intelligence
**LLM Integration**: Uses Grok memory for microstructure research and flow analysis

### Capabilities:
- **Order Book Analysis**: Bid/ask spread analysis and market depth
- **Trade Classification**: Algorithmic vs human trade detection
- **Liquidity Analysis**: Market liquidity assessment and impact costs
- **High-Frequency Data**: Microsecond-level market data analysis
- **Flow Toxicity**: Detection of aggressive vs passive order flow

### Memory System:
- **Microstructure Pattern Memory**: Historical order flow and microstructure patterns
- **Liquidity Memory**: Records market liquidity conditions and changes
- **Flow Pattern Memory**: Tracks various order flow patterns and implications
- **Collaborative Intelligence**: Shares microstructure insights with institutional and market subagents

### LLM-Driven Analysis:
- **Microstructure Research**: Grok analysis of market microstructure patterns
- **Flow Intelligence**: AI-powered order flow analysis and prediction
- **Liquidity Assessment**: Deep analysis of market liquidity dynamics
- **Collaborative Synthesis**: Integrates microstructure data with institutional and market analysis

---

## MarketDataApp Data Subagent
**File**: `src/agents/data_subs/marketdataapp_datasub.py`
**Role**: Premium market data from MarketDataApp API
**LLM Integration**: Uses Grok memory for premium data research and analysis

### Capabilities:
- **Premium Quotes**: High-quality real-time and historical quotes
- **Trade Data**: Detailed trade execution data and timestamps
- **Options Data**: Complete options chains and market data
- **Dark Pool Detection**: Large block trade identification
- **Market Depth**: Level 2 order book data and analysis

### Memory System:
- **Premium Data Memory**: Historical premium market data patterns
- **Trade Pattern Memory**: Records and analyzes trade execution patterns
- **Options Memory**: Options market data and volatility patterns
- **Collaborative Intelligence**: Shares premium insights with market and flow subagents

### LLM-Driven Analysis:
- **Premium Data Research**: Grok-powered analysis of high-quality market data
- **Trade Analysis**: Deep analysis of trade execution patterns and implications
- **Options Intelligence**: AI assessment of options market dynamics
- **Collaborative Synthesis**: Integrates premium data with market and flow analysis

---

## Kalshi Data Subagent
**File**: `src/agents/data_subs/kalshi_datasub.py`
**Role**: Event contract data and prediction market intelligence
**LLM Integration**: Uses Grok memory for prediction market research and event analysis

### Capabilities:
- **Event Contracts**: Political, economic, and market event predictions
- **Prediction Markets**: Crowd-sourced probability assessments
- **Event Impact**: Historical event outcome analysis
- **Market Expectations**: Real-time market probability assessments
- **Risk Premia**: Event risk pricing and market sentiment

### Memory System:
- **Event Pattern Memory**: Historical event outcomes and market reactions
- **Prediction Memory**: Records prediction market accuracy and patterns
- **Event Impact Memory**: Tracks market reactions to various events
- **Collaborative Intelligence**: Shares prediction insights with sentiment and economic subagents

### LLM-Driven Analysis:
- **Event Research**: Grok analysis of prediction markets and event implications
- **Probability Assessment**: AI evaluation of market probabilities and expectations
- **Event Impact Analysis**: Deep analysis of event outcomes and market effects
- **Collaborative Synthesis**: Integrates prediction data with sentiment and economic analysis

---

## Options Data Subagent
**File**: `src/agents/data_subs/options_datasub.py`
**Role**: Options market data collection and derivatives analysis
**LLM Integration**: Uses Grok memory for options pricing research and volatility analysis

### Capabilities:
- **Options Chains**: Complete options data for stocks, ETFs, and indices via options_greeks_calc_tool
- **Greeks Calculation**: Delta, gamma, theta, vega, rho calculations for all options
- **Implied Volatility**: Volatility surface analysis and skew detection
- **Open Interest**: Options positioning and flow analysis
- **Volume Analysis**: Options trading volume and unusual activity detection

### Memory System:
- **Options Pattern Memory**: Historical options pricing patterns and volatility regimes
- **Greeks Memory**: Records and analyzes options Greeks over time
- **Volatility Memory**: Tracks implied volatility patterns and surface changes
- **Collaborative Intelligence**: Shares options insights with market and flow subagents

### LLM-Driven Analysis:
- **Options Research**: Grok-powered analysis of options market dynamics
- **Volatility Analysis**: Deep analysis of implied volatility and pricing anomalies
- **Flow Intelligence**: AI assessment of options flow and positioning
- **Collaborative Synthesis**: Integrates options data with market and microstructure analysis

---

## A2A Communication Protocol

### Data Sharing Formats:
```json
{
  "market_data": {
    "prices": {...},
    "sentiment": {...},
    "volatility": {...}
  },
  "intelligence": {
    "predictions": [...],
    "risk_assessments": [...],
    "opportunities": [...]
  },
  "collaboration_signals": {
    "strategy_insights": [...],
    "risk_warnings": [...],
    "execution_context": {...}
  }
}
```

### Collaborative Workflows:
- **Trade Discovery Loops**: Iterative analysis with Strategy Agent for opportunity identification
- **Risk Assessment Collaboration**: Real-time market intelligence sharing with Risk Agent
- **Learning Integration**: Data pattern sharing with Learning Agent for model improvement
- **Execution Support**: Real-time market context provision to Execution Agent

## Memory and LLM Integration

### Advanced Memory Systems
- **Multi-Level Memory**: Short-term, long-term, episodic, semantic, and procedural memory
- **Collaborative Memory**: Shared memory spaces for cross-agent intelligence
- **Pattern Storage**: Historical market patterns and predictive insights
- **Context Preservation**: Maintaining analysis context across sessions

### LLM Reasoning Capabilities
- **Deep Market Intelligence**: ChatOpenAI/ChatXAI-powered market analysis
- **Predictive Reasoning**: Forward-looking market predictions and risk assessments
- **Collaborative Synthesis**: Multi-agent debate and consensus building
- **Contextual Understanding**: Comprehensive market context for intelligent decisions

### Integration Architecture
- **BaseAgent Inheritance**: Full access to advanced memory and LLM systems
- **A2A Memory Sharing**: Cross-agent memory coordination and intelligence sharing
- **Reasoning with Context**: LLM analysis enhanced by historical patterns and collaborative insights
- **Adaptive Learning**: Continuous improvement through memory-driven learning

## Technical Architecture

### Data Processing Engine:
- **Real-Time Streaming**: Continuous data ingestion and processing
- **Multi-Threaded Analysis**: Parallel processing of different data types
- **Memory Management**: Efficient storage and retrieval of historical data
- **Quality Assurance**: Automated data validation and error correction

### LLM Integration:
- **Context Provision**: Comprehensive data context for all LLM reasoning
- **Collaborative Analysis**: Cross-agent intelligence sharing and synthesis
- **Predictive Enhancement**: Forward-looking analysis beyond foundation metrics
- **Iterative Refinement**: Continuous improvement through agent collaboration

## Future Enhancements

### Planned Improvements:
- Advanced predictive analytics with machine learning models
- Enhanced real-time collaboration capabilities
- Expanded data source integration
- More sophisticated market intelligence algorithms

---

# Data Agent Implementation (Comprehensive AI Approach)

{base_prompt}
Process market data comprehensively using AI-driven analysis: foundation data processing provides quantitative metrics, while LLM reasoning delivers deep market intelligence and collaborative insights for trade discovery.

FOUNDATION DATA PROCESSING (Always Performed):
- Aggregate data from all sources (yfinance, IBKR, FRED, FMP, options APIs)
- Process market data, economic indicators, and sentiment feeds
- Generate quantitative metrics and technical indicators
- Validate data quality and cross-reference sources

LLM COMPREHENSIVE ANALYSIS (Always Applied):
- Analyze market sentiment, news impact, and behavioral patterns with deep intelligence
- Generate forward-looking market predictions and risk assessments
- Identify complex market relationships and emerging trends through over-analysis
- Collaborate with Strategy, Risk, and other agents for comprehensive trade discovery
- Provide thorough examination of all data dimensions for winning trade identification

Work collaboratively with other agents to discover winning trades:
- Share deep market intelligence with Strategy Agent for opportunity identification
- Collaborate with Risk Agent on comprehensive market condition assessments
- Contribute intelligence to Learning Agent for continuous model refinement
- Provide real-time market context to Execution Agent for optimal trade execution

Output: Comprehensive data intelligence for A2A collaboration; include foundation metrics + deep LLM insights for trade discovery (e.g., "Deep Analysis: Market sentiment bullish with 78% confidence, volatility contraction signals opportunity in SPY calls; collaborating with Strategy for asymmetric trade setup").