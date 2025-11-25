# integrations/agent-model-mapping.md
# Agent-Model Integration Mapping
# Purpose: Designate which AI models and APIs each agent should use
# Updated: November 4, 2025

## Available xAI Models
Choose from these xAI/Grok models for agent integration:

### Production Models
- **grok-code-fast-1**: 256K context, 2M TPM • 480 RPM, $0.20 input • $1.50 output
- **grok-4-1-fast-reasoning**: 2M context, 4M TPM • 480 RPM, $0.20 input • $0.50 output  
- **grok-4-fast-non-reasoning**: 2M context, 4M TPM • 480 RPM, $0.20 input • $0.50 output
- **grok-4-0709**: 256K context, 2M TPM • 480 RPM, $3.00 input • $15.00 output
- **grok-3-mini**: 131K context, 480 RPM, $0.30 input • $0.50 output

## Agent Model Configuration

### DataAgent
**Purpose**: Market data collection and sentiment analysis
**Assigned Models**:
- **Primary LLM**: grok-4-1-fast-reasoning
- **Sentiment Analysis**: grok-4-1-fast-reasoning (via sentiment_analysis_tool)
- **APIs**: Massive (REST + WebSocket), Twitter/X (sentiment)

**Current Status**: ✅ Twitter API configured, ✅ Massive API configured

### StrategyAgent
**Purpose**: Trading strategy generation and proposal creation
**Assigned Models**:
- **Primary LLM**: grok-4-1-fast-reasoning
- **Strategy Reasoning**: grok-4-1-fast-reasoning
- **APIs**: Massive (market data), yfinance (historical)

**Current Status**: ✅ LLM integrated, ✅ Algorithmic strategies

### RiskAgent
**Purpose**: Risk assessment and position sizing
**Assigned Models**:
- **Primary LLM**: grok-4-1-fast-reasoning
- **Risk Analysis**: grok-4-1-fast-reasoning
- **APIs**: Massive (volatility data), IBKR (position data)

**Current Status**: ✅ LLM integrated, ✅ Risk calculations

### ExecutionAgent
**Purpose**: Order execution and IBKR integration
**Assigned Models**:
- **Primary LLM**: grok-4-1-fast-reasoning
- **Execution Logic**: grok-4-1-fast-reasoning
- **APIs**: IBKR (trading), Massive (real-time quotes)

**Current Status**: ✅ IBKR integration, ✅ LLM integrated

### LearningAgent
**Purpose**: Performance analysis and strategy adaptation
**Assigned Models**:
- **Primary LLM**: grok-4-1-fast-reasoning
- **Learning Analysis**: grok-4-1-fast-reasoning
- **APIs**: Massive (performance data), yfinance (backtesting)

**Current Status**: ✅ LLM integrated, ✅ Performance metrics

### ReflectionAgent
**Purpose**: System self-assessment and improvement
**Assigned Models**:
- **Primary LLM**: grok-4-1-fast-reasoning
- **Reflective Reasoning**: grok-4-1-fast-reasoning
- **APIs**: All agent data (A2A protocol)

**Current Status**: ✅ LLM integrated, ✅ Audit polling

## API Integration Status

### ✅ Active & Working
- **xAI/Grok API**: LLM integration implemented
  - Key: `GROK_API_KEY=[CONFIGURED]`
  - Models: grok-2, grok-3, grok-4 variants
  - Status: ✅ Working (tested with grok-2)
- **Massive API**: REST + WebSocket tools implemented
  - Key: `MASSIVE_API_KEY=WxzAxYZbs6lglNP5mMZkfxNTQm9mjJLP`
  - Tools: `massive_api_tool`, `massive_websocket_tool`
  - Status: Endpoints need verification (docs show https://api.massive.com/v1/ structure)
  - Client Library: Available at https://github.com/massive-com/client-python
- **Twitter/X API**: Sentiment analysis ready
  - Key: `TWITTER_BEARER_TOKEN=[NEEDS TO BE SET IN .env FILE]`
  - Tool: `twitter_sentiment_tool`
  - Setup: Get bearer token from https://developer.twitter.com/
- **IBKR Integration**: Paper trading ready
  - Credentials: Username, password, account configured

### ⚠️ Configured but Needs Testing
- **IBKR Integration**: Paper trading ready but needs execution testing

### ❌ Stub Implementations (To Replace)
- **News API**: `news_data_tool` (placeholder headlines)
  - **Free Alternatives Available**:
    - **NewsAPI.org**: 100 requests/day free, financial news filtering
    - **Currents API**: 600 requests/day free, real-time news
    - **Mediastack**: 500 requests/month free, global news coverage
    - **NewsData.io**: 200 requests/day free, financial categories
    - **ContextualWeb**: 1000 requests/day free, news search
- **FRED API**: `economic_data_tool` (placeholder data)
- **Strategy Tools**: `strategy_proposal_tool` (rule-based)

## Alternative Market Data Sources (Backup/Placeholder)

### Free Tier APIs (No Subscription Required)
- **yfinance**: Historical data, technical indicators
  - Status: ✅ Implemented (`yfinance_data_tool`)
  - Limitations: Delayed data, rate limits, no real-time quotes
  - Use Case: Backtesting, historical analysis, free alternative to paid APIs

- **Alpha Vantage**: Free API key for stocks, forex, crypto
  - Status: ✅ Implemented (`alpha_vantage_tool`)
  - Free Tier: 25 API calls/day, 5 calls/minute
  - Endpoints: TIME_SERIES_DAILY, GLOBAL_QUOTE, NEWS_SENTIMENT
  - Setup: Get API key at https://www.alphavantage.co/support/#api-key

- **IEX Cloud**: Free tier for stocks and market data
  - Status: Placeholder (not implemented)
  - Free Tier: 50,000 API calls/month
  - Endpoints: Quotes, charts, company info, dividends
  - Setup: Get API key at https://iexcloud.io/console/tokens

### Paid Subscription Alternatives
- **Massive API**: Premium real-time and historical data
  - Status: ✅ Configured but endpoints need verification
  - Pricing: Starter $49/month (1M requests), Professional $149/month (10M requests)
  - Use Case: Production market data when budget allows

- **Polygon.io**: Real-time and historical market data
  - Status: Placeholder (not implemented)
  - Pricing: Stocks Starter $39/month (5M requests), All Assets $199/month (25M requests)
  - Features: Real-time quotes, aggregates, options data, forex, crypto

- **Twelve Data**: Global market data API
  - Status: Placeholder (not implemented)
  - Pricing: Basic $9.99/month (800K calls), Professional $29.99/month (5M calls)
  - Features: Real-time data, 1-minute bars, fundamental data, global coverage

- **NewsAPI.org**: Financial news aggregation
  - Status: Placeholder (not implemented)
  - Pricing: Free 100/day, Basic $39/month (10K requests), Professional $199/month (200K requests)
  - Features: Financial news filtering, company mentions, sentiment analysis

- **Currents API**: Real-time global news
  - Status: Placeholder (not implemented)
  - Pricing: Free 600/day, Standard $49/month (25K requests), Professional $99/month (100K requests)
  - Features: Real-time news updates, global coverage, category filtering

- **Mediastack**: Global news with categories
  - Status: Placeholder (not implemented)
  - Pricing: Free 500/month, Basic $19/month (5K requests), Professional $99/month (100K requests)
  - Features: News by country/language, category filtering, source credibility

### Implementation Priority for Backups
1. **Alpha Vantage**: Easy setup, good free tier for basic needs
2. **IEX Cloud**: More generous free tier, better for development
3. **yfinance**: Already implemented, use as immediate fallback
4. **Paid Services**: Massive, Polygon, Twelve Data (when budget available)

## Model Usage Guidelines

### When to Use Each Model
- **grok-4-1-fast-reasoning**: Complex reasoning, strategy development, sentiment analysis, learning tasks (2M context, high TPM)
- **grok-4-fast-non-reasoning**: Fast execution tasks, real-time decisions, high-volume processing (2M context, high TPM)
- **grok-code-fast-1**: Code generation, technical analysis, programming tasks (256K context, balanced pricing)
- **grok-4-0709**: High-quality reasoning when cost is not a concern (256K context, premium pricing)
- **grok-3-mini**: Lightweight tasks, simple analysis, cost-effective processing (131K context, low cost)

### Fallback Strategy
1. Primary model (assigned above)
2. OpenAI GPT-3.5-turbo (if xAI fails)
3. Rule-based algorithms (if all APIs fail)

## Implementation Priority

### Phase 1: Core AI Integration
1. **DataAgent**: xAI sentiment analysis
2. **StrategyAgent**: xAI strategy reasoning
3. **RiskAgent**: xAI risk assessment

### Phase 2: Advanced Features
4. **LearningAgent**: xAI performance analysis
5. **ReflectionAgent**: xAI self-improvement
6. **ExecutionAgent**: xAI execution logic

### Phase 3: Full API Integration
7. **Real News Data**: Replace news stubs
8. **Economic Indicators**: FRED API integration
9. **Multi-source Validation**: Cross-reference all data

## Testing Commands

```bash
# Test Twitter sentiment
python -c "from src.utils.tools import twitter_sentiment_tool; print(twitter_sentiment_tool.func('AAPL stock', 10))"

# Test Massive REST API
python -c "from src.utils.tools import massive_api_tool; print(massive_api_tool.func('AAPL', 'quotes'))"

# Test Massive WebSocket (30 second sample)
python -c "from src.utils.tools import massive_websocket_tool; print(massive_websocket_tool.func('AAPL', 'quotes', 30))"

# Test Alpha Vantage (backup market data)
python -c "from src.utils.tools import alpha_vantage_tool; print(alpha_vantage_tool.func('AAPL', 'GLOBAL_QUOTE'))"
```

## Configuration Changes Needed

To change model assignments, edit the "Assigned Models" section above for each agent. The system will automatically use the specified models when the LLM integration is implemented.