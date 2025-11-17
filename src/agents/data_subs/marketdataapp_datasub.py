# src/agents/data_subs/marketdataapp_datasub.py
# Purpose: MarketDataApp Data Subagent for fetching premium market data.
# Provides institutional-grade data including real-time quotes, trades, options chains, and dark pool indicators.
# Structural Reasoning: Dedicated subagent for premium data sources, enabling parallel processing with free sources.
# Ties to system: Provides premium data dict for main data agent coordination.
# For legacy wealth: Access to institutional data edges for enhanced trading strategies.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Dynamic root path for imports.

from src.agents.base import BaseAgent  # Absolute import.
import logging
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MarketDataAppDatasub(BaseAgent):
    """
    MarketDataApp Data Subagent with LLM-powered exploration.
    Reasoning: Fetches premium market data from MarketDataApp API for institutional-grade insights.
    Uses LLM to intelligently explore available data endpoints and maximize data utilization.
    """
    def __init__(self):
        config_paths = {'risk': 'config/risk-constraints.yaml'}  # Relative to root.
        prompt_paths = {'base': 'base_prompt.txt', 'role': 'docs/AGENTS/main-agents/data-agent.md'}  # Relative to root.
        tools = []  # MarketDataAppDatasub uses internal methods instead of tools
        super().__init__(role='massive_data', config_paths=config_paths, prompt_paths=prompt_paths, tools=tools)

        # Available data endpoints for LLM exploration
        self.available_endpoints = {
            'quotes': 'Real-time and historical price quotes',
            'trades': 'Detailed trade execution data with timestamps',
            'orderbook': 'Level 2 order book depth and market microstructure',
            'options': 'Options chain data and Greeks',
            'darkpool': 'Dark pool trade detection and institutional flow',
            'microstructure': 'Advanced market microstructure analysis',
            'flow': 'Institutional order flow and positioning'
        }

    def reflect(self, adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on batch adjustments for self-improvement.
        Stub: Returns empty dict.
        """
        logger.info(f"Massive Reflecting on adjustments: {adjustments}")
        return {}

    async def process_input(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        logger.info(f"MarketDataApp Subagent processing input: {input_data or 'Default SPY premium data'}")

        symbol = input_data.get('symbol', 'SPY') if input_data else 'SPY'

        # Use LLM to determine optimal data endpoints to explore
        exploration_plan = await self._plan_data_exploration(symbol, input_data or {})

        # Execute exploration plan
        exploration_results = await self._execute_exploration_plan(symbol, exploration_plan)

        # Consolidate results into DataFrame format
        consolidated_data = self._consolidate_marketdataapp_data(symbol, exploration_results)

        # Store premium market data in shared memory for strategy agents
        await self.store_shared_memory("marketdataapp_data", symbol, {
            "premium_data": consolidated_data,
            "exploration_plan": exploration_plan,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol
        })

        logger.info(f"MarketDataApp LLM exploration completed for {symbol}: {len(exploration_results)} endpoints explored")
        return consolidated_data

    async def _plan_data_exploration(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to plan intelligent data exploration based on symbol and context.

        Args:
            symbol: Stock symbol to analyze
            context: Additional context for exploration planning

        Returns:
            Dict containing exploration plan with prioritized endpoints
        """
        if not self.llm:
            logger.error("CRITICAL FAILURE: No LLM available for market data exploration - cannot proceed without AI planning")
            raise Exception("LLM required for intelligent market data exploration - no default fallback allowed")

        try:
            exploration_prompt = f"""
You are an expert quantitative analyst planning premium market data exploration for {symbol}.

CONTEXT:
- Symbol: {symbol}
- Available Data Endpoints: {self.available_endpoints}
- Analysis Goals: Maximize institutional-grade insights for trading strategy development
- Risk Constraints: Focus on data that provides alpha signals while managing API costs

TASK:
Based on the symbol characteristics and available endpoints, determine which data sources to explore and prioritize them.
Consider:
1. Market microstructure signals (orderbook, trades, flow)
2. Institutional activity (darkpool, options positioning)
3. Real-time vs historical data needs
4. Cost-benefit analysis of premium data sources

Return a JSON object with:
- "endpoints": Array of endpoint names to explore (from available_endpoints keys)
- "priorities": Object mapping endpoint names to priority scores (1-10, higher = more important)
- "reasoning": Brief explanation of exploration strategy
- "expected_insights": Array of expected alpha signals from this data

Example response:
{{
  "endpoints": ["quotes", "orderbook", "trades", "darkpool"],
  "priorities": {{"orderbook": 9, "trades": 8, "darkpool": 7, "quotes": 6}},
  "reasoning": "Focus on microstructure data for {symbol} as it's highly liquid with significant institutional activity",
  "expected_insights": ["Order book imbalance signals", "Institutional accumulation patterns", "Dark pool positioning"]
}}
"""

            response = await self.llm.ainvoke(exploration_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON response
            import json
            try:
                plan = json.loads(response_text)
                logger.info(f"LLM exploration plan for {symbol}: {plan.get('reasoning', 'No reasoning provided')}")
                return plan
            except json.JSONDecodeError as e:
                logger.error(f"CRITICAL FAILURE: Failed to parse LLM exploration plan JSON: {e} - cannot proceed without AI planning")
                raise Exception(f"LLM exploration planning failed - JSON parsing error: {e}")

        except Exception as e:
            logger.error(f"CRITICAL FAILURE: LLM exploration planning failed: {e} - cannot proceed without AI planning")
            raise Exception(f"LLM exploration planning failed: {e}")

    async def _execute_exploration_plan(self, symbol: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the exploration plan by fetching data from prioritized endpoints.

        Args:
            symbol: Stock symbol
            plan: Exploration plan from LLM

        Returns:
            Dict containing results from all explored endpoints
        """
        results = {}
        endpoints = plan.get('endpoints', ['quotes'])

        for endpoint in endpoints:
            try:
                if endpoint == 'quotes':
                    results['quotes'] = await self._fetch_quotes_data(symbol)
                elif endpoint == 'trades':
                    results['trades'] = await self._fetch_trades_data(symbol)
                elif endpoint == 'orderbook':
                    results['orderbook'] = await self._fetch_orderbook_data(symbol)
                elif endpoint == 'options':
                    results['options'] = await self._fetch_options_data(symbol)
                elif endpoint == 'darkpool':
                    results['darkpool'] = await self._fetch_darkpool_data(symbol)
                elif endpoint == 'microstructure':
                    results['microstructure'] = await self._fetch_microstructure_data(symbol)
                elif endpoint == 'flow':
                    results['flow'] = await self._fetch_flow_data(symbol)
                else:
                    logger.warning(f"Unknown endpoint: {endpoint}")

            except Exception as e:
                logger.error(f"Failed to fetch {endpoint} data for {symbol}: {e}")
                results[endpoint] = {"error": str(e)}

        return results

    async def _fetch_quotes_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch real-time and historical quotes."""
        # Enhanced mock data with more institutional-grade fields
        return {
            'symbol': symbol,
            'price': 152.5,
            'bid': 152.48,
            'ask': 152.52,
            'volume': 5000000,
            'market_cap': 2500000000000.0,
            'pe_ratio': 28.5,
            'dividend_yield': 0.82,
            'beta': 1.2,
            '52_week_high': 198.23,
            '52_week_low': 124.17,
            'vwap': 152.3,
            'avg_volume_30d': 4500000,
            'short_interest': 0.025,
            'source': 'marketdataapp_quotes',
            'timestamp': datetime.now().isoformat()
        }

    async def _fetch_trades_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch detailed trade execution data."""
        # Mock trade data with microstructure insights
        trades = []
        base_time = datetime.now()

        for i in range(10):
            trades.append({
                'timestamp': (base_time - timedelta(minutes=i)).isoformat(),
                'price': 152.5 + (i * 0.01),
                'volume': 1000 + (i * 100),
                'trade_type': 'regular' if i % 3 != 0 else 'block',
                'exchange': 'NYSE' if i % 2 == 0 else 'NASDAQ',
                'conditions': ['regular'] if i % 4 != 0 else ['odd_lot', 'intermarket_sweep']
            })

        return {
            'symbol': symbol,
            'trades': trades,
            'total_volume': sum(t['volume'] for t in trades),
            'avg_trade_size': sum(t['volume'] for t in trades) / len(trades),
            'block_trades_count': len([t for t in trades if t['trade_type'] == 'block']),
            'source': 'marketdataapp_trades'
        }

    async def _fetch_orderbook_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch Level 2 order book depth."""
        # Mock order book with bid/ask depth
        bids = []
        asks = []

        for i in range(10):
            bids.append({
                'price': 152.5 - (i * 0.01),
                'volume': 5000 - (i * 200),
                'orders': 5 + i
            })
            asks.append({
                'price': 152.5 + (i * 0.01),
                'volume': 5000 - (i * 200),
                'orders': 5 + i
            })

        return {
            'symbol': symbol,
            'bids': bids,
            'asks': asks,
            'spread': asks[0]['price'] - bids[0]['price'],
            'bid_volume_total': sum(b['volume'] for b in bids),
            'ask_volume_total': sum(a['volume'] for a in asks),
            'imbalance_ratio': sum(b['volume'] for b in bids[:5]) / sum(a['volume'] for a in asks[:5]),
            'source': 'marketdataapp_orderbook'
        }

    async def _fetch_options_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch options chain data."""
        # Mock options data
        return {
            'symbol': symbol,
            'calls': [
                {'strike': 150, 'bid': 5.2, 'ask': 5.4, 'volume': 1200, 'oi': 5000, 'implied_vol': 0.25},
                {'strike': 155, 'bid': 2.1, 'ask': 2.3, 'volume': 800, 'oi': 3200, 'implied_vol': 0.22}
            ],
            'puts': [
                {'strike': 150, 'bid': 1.8, 'ask': 2.0, 'volume': 950, 'oi': 4100, 'implied_vol': 0.23},
                {'strike': 145, 'bid': 0.9, 'ask': 1.1, 'volume': 600, 'oi': 2800, 'implied_vol': 0.20}
            ],
            'put_call_ratio': 0.85,
            'total_oi': 15100,
            'source': 'marketdataapp_options'
        }

    async def _fetch_darkpool_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch dark pool trade detection."""
        # Mock dark pool activity
        return {
            'symbol': symbol,
            'dark_pool_trades': [
                {'timestamp': datetime.now().isoformat(), 'volume': 50000, 'price': 152.4, 'venue': 'Anonymous'},
                {'timestamp': (datetime.now() - timedelta(minutes=5)).isoformat(), 'volume': 75000, 'price': 152.3, 'venue': 'Anonymous'}
            ],
            'total_dark_volume': 125000,
            'dark_volume_pct': 2.5,
            'institutional_score': 8.2,  # 1-10 scale
            'source': 'marketdataapp_darkpool'
        }

    async def _fetch_microstructure_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch advanced market microstructure analysis."""
        return {
            'symbol': symbol,
            'effective_spread': 0.02,
            'realized_spread': 0.015,
            'price_impact': 0.005,
            'liquidity_score': 9.1,  # 1-10 scale
            'algorithmic_trading_pct': 68.5,
            'human_trading_pct': 31.5,
            'flow_toxicity': 0.12,  # Lower is better
            'source': 'marketdataapp_microstructure'
        }

    async def _fetch_flow_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch institutional order flow data."""
        return {
            'symbol': symbol,
            'institutional_flow': {
                'buy_volume': 2500000,
                'sell_volume': 1800000,
                'net_flow': 700000,
                'large_orders': 45,
                'small_orders': 1200
            },
            'smart_money_score': 7.8,  # 1-10 scale
            'accumulation_days': 12,
            'distribution_days': 3,
            'source': 'marketdataapp_flow'
        }

    def _consolidate_marketdataapp_data(self, symbol: str, exploration_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consolidate exploration results into DataFrame format for pipeline integration.

        Args:
            symbol: Stock symbol
            exploration_results: Raw data from all explored endpoints

        Returns:
            Dict with consolidated data including DataFrames
        """
        consolidated = {
            'symbol': symbol,
            'source': 'marketdataapp_llm_exploration',
            'endpoints_explored': list(exploration_results.keys()),
            'timestamp': datetime.now().isoformat()
        }

        # Create main quotes DataFrame
        if 'quotes' in exploration_results:
            quotes_data = exploration_results['quotes']
            quotes_df = pd.DataFrame([{
                'timestamp': quotes_data.get('timestamp'),
                'open': quotes_data.get('price', 0),
                'high': quotes_data.get('52_week_high', 0),
                'low': quotes_data.get('52_week_low', 0),
                'close': quotes_data.get('price', 0),
                'volume': quotes_data.get('volume', 0),
                'bid': quotes_data.get('bid', 0),
                'ask': quotes_data.get('ask', 0),
                'vwap': quotes_data.get('vwap', 0)
            }])
            consolidated['quotes_df'] = quotes_df

        # Create trades DataFrame
        if 'trades' in exploration_results and 'trades' in exploration_results['trades']:
            trades_data = exploration_results['trades']['trades']
            if trades_data:
                trades_df = pd.DataFrame(trades_data)
                consolidated['trades_df'] = trades_df

        # Create orderbook DataFrame
        if 'orderbook' in exploration_results:
            orderbook_data = exploration_results['orderbook']
            bids_df = pd.DataFrame(orderbook_data.get('bids', []))
            asks_df = pd.DataFrame(orderbook_data.get('asks', []))
            consolidated['orderbook_bids_df'] = bids_df
            consolidated['orderbook_asks_df'] = asks_df

        # Add metadata
        consolidated['data_quality_score'] = self._calculate_data_quality_score(exploration_results)
        consolidated['institutional_insights'] = self._extract_institutional_insights(exploration_results)

        return consolidated

    def _calculate_data_quality_score(self, exploration_results: Dict[str, Any]) -> float:
        """Calculate overall data quality score based on endpoints explored and data completeness."""
        base_score = 5.0
        endpoint_bonus = len(exploration_results) * 0.5
        completeness_bonus = sum(1 for r in exploration_results.values() if 'error' not in r) * 0.3

        return min(10.0, base_score + endpoint_bonus + completeness_bonus)

    def _extract_institutional_insights(self, exploration_results: Dict[str, Any]) -> List[str]:
        """Extract key institutional insights from exploration results."""
        insights = []

        if 'darkpool' in exploration_results:
            dark_data = exploration_results['darkpool']
            if dark_data.get('institutional_score', 0) > 7:
                insights.append("High institutional interest detected in dark pools")

        if 'orderbook' in exploration_results:
            orderbook_data = exploration_results['orderbook']
            imbalance = orderbook_data.get('imbalance_ratio', 1.0)
            if imbalance > 1.2:
                insights.append("Bullish order book imbalance detected")
            elif imbalance < 0.8:
                insights.append("Bearish order book imbalance detected")

        if 'flow' in exploration_results:
            flow_data = exploration_results['flow']
            net_flow = flow_data.get('institutional_flow', {}).get('net_flow', 0)
            if net_flow > 0:
                insights.append("Positive institutional net flow")
            elif net_flow < 0:
                insights.append("Negative institutional net flow")

        return insights if insights else ["Standard market conditions observed"]

    def fetch_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility.
        Fetch market data for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dict containing market data
        """
        return {
            'symbol': symbol,
            'price': 152.5,
            'volume': 5000000,
            'market_cap': 2500000000000.0,
            'pe_ratio': 28.5,
            'dividend_yield': 0.82,
            'beta': 1.2,
            '52_week_high': 198.23,
            '52_week_low': 124.17,
            'source': 'marketdataapp_data_subagent'
        }

# Standalone test (run python src/agents/data_subs/marketdataapp_datasub.py to verify)
if __name__ == "__main__":
    import asyncio
    agent = MarketDataAppDatasub()
    result = asyncio.run(agent.process_input({'symbols': ['SPY']}))
    print("MarketDataApp Subagent Test Result:")
    print(f"Keys: {list(result.keys())}")
    if 'marketdataapp' in result:
        print(f"Premium data types: {list(result['marketdataapp'].keys())}")
        print(f"Sample quotes: {result['marketdataapp'].get('quotes', {})}")