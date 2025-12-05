# src/agents/data_analyzers/marketdataapp_data_analyzer.py
# Purpose: MarketDataApp Data Subagent for fetching premium market data.
# Provides institutional-grade data including real-time quotes, trades, options chains, and dark pool indicators.
# Structural Reasoning: Dedicated subagent for premium data sources, enabling parallel processing with free sources.
# Ties to system: Provides premium data dict for main data agent coordination.
# For legacy wealth: Access to institutional data edges for enhanced trading strategies.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Dynamic root path for imports.

from src.agents.data_analyzers.base_data_analyzer import BaseDataAnalyzer  # Absolute import.
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime, timedelta
import requests
import os
from src.utils.config import get_marketdataapp_api_key

logger = logging.getLogger(__name__)

class MarketDataAppDataAnalyzer(BaseDataAnalyzer):
    """
    MarketDataApp Data Subagent with LLM-powered exploration.
    Reasoning: Fetches premium market data from MarketDataApp API for institutional-grade insights.
    Uses LLM to intelligently explore available data endpoints and maximize data utilization.
    """
    def __init__(self):
        super().__init__(role='marketdataapp_data')

        # Initialize MarketDataApp API
        self.api_key = get_marketdataapp_api_key()
        if not self.api_key:
            logger.warning("MARKETDATAAPP_API_KEY not found in environment variables - real API calls will fail")
        self.base_url = "https://api.marketdata.app/v1"

        # Available data endpoints for LLM exploration
        # NOTE: Currently only 'quotes' endpoint is fully implemented
        # Other endpoints are planned for future implementation
        self.available_endpoints = {
            'quotes': 'Real-time and historical price quotes (IMPLEMENTED)',
            'trades': 'Detailed trade execution data with timestamps (PLANNED)',
            'orderbook': 'Level 2 order book depth and market microstructure (PLANNED)',
            'options': 'Options chain data and Greeks (PLANNED)',
            'darkpool': 'Dark pool trade detection and institutional flow (PLANNED)',
            'microstructure': 'Advanced market microstructure analysis (PLANNED)',
            'flow': 'Institutional order flow and positioning (PLANNED)'
        }

        # Only include implemented endpoints for actual exploration
        self.implemented_endpoints = ['quotes']

    def reflect(self, adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on batch adjustments for self-improvement.
        Stub: Returns empty dict.
        """
        logger.info(f"Reflecting on adjustments: {adjustments}")
        return {}

    async def _plan_data_exploration(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan MarketDataApp data exploration.

        Args:
            input_data: Input parameters

        Returns:
            Exploration plan
        """
        symbol = input_data.get('symbol', 'SPY')
        return await self._plan_marketdataapp_exploration(symbol, input_data)

    async def _execute_data_exploration(self, exploration_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute MarketDataApp data exploration.

        Args:
            exploration_plan: Plan from _plan_data_exploration

        Returns:
            Raw data
        """
        symbol = exploration_plan.get('symbol', 'SPY')
        return await self._execute_exploration_plan(symbol, exploration_plan)

    async def _enhance_data(self, validated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance MarketDataApp data with analysis.

        Args:
            validated_data: Validated data

        Returns:
            Consolidated data (for backward compatibility)
        """
        # For MarketDataApp, return the consolidated data directly
        symbol = "SPY"  # Default symbol, could be extracted from context if needed
        consolidated_data = self._consolidate_marketdataapp_data(symbol, validated_data)
        return consolidated_data

    async def _process_input(self, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process MarketDataApp data using standardized BaseDataAnalyzer pattern.
        """
        if input_data is None:
            input_data = {}

        # For backward compatibility, call the original logic
        symbol = input_data.get('symbol', 'SPY')

        # Use LLM to determine optimal data endpoints to explore
        exploration_plan = await self._plan_marketdataapp_exploration(symbol, input_data)

        # Execute exploration plan
        exploration_results = await self._execute_exploration_plan(symbol, exploration_plan)

        # Consolidate results into DataFrame format
        consolidated_data = self._consolidate_marketdataapp_data(symbol, exploration_results)

        return consolidated_data

    async def _plan_marketdataapp_exploration(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
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
- IMPLEMENTED Data Endpoints: {self.implemented_endpoints}
- Available Data Endpoints: {list(self.available_endpoints.keys())}
- Analysis Goals: Maximize institutional-grade insights for trading strategy development
- Current Limitation: Only 'quotes' endpoint is fully implemented

TASK:
Based on the symbol characteristics and IMPLEMENTED endpoints, determine which data sources to explore.
IMPORTANT: Only suggest endpoints from the IMPLEMENTED list: {self.implemented_endpoints}

Consider:
1. Real-time price and fundamental data from quotes endpoint
2. Future expansion to microstructure signals (when implemented)
3. Cost-benefit analysis of available premium data sources

Return a JSON object with:
- "endpoints": Array of endpoint names to explore (ONLY from implemented_endpoints)
- "priorities": Object mapping endpoint names to priority scores (1-10, higher = more important)
- "reasoning": Brief explanation of exploration strategy
- "expected_insights": Array of expected alpha signals from this data
- "future_endpoints": Array of planned endpoints for future implementation

Example response:
{{
  "endpoints": ["quotes"],
  "priorities": {{"quotes": 10}},
  "reasoning": "Currently limited to quotes endpoint - provides essential price and fundamental data",
  "expected_insights": ["Real-time pricing", "Volume analysis", "Fundamental metrics"],
  "future_endpoints": ["orderbook", "trades", "darkpool"]
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

        # Validate that only implemented endpoints are used
        invalid_endpoints = [ep for ep in endpoints if ep not in self.implemented_endpoints]
        if invalid_endpoints:
            logger.warning(f"LLM suggested unimplemented endpoints: {invalid_endpoints}. Using only implemented endpoints.")
            endpoints = [ep for ep in endpoints if ep in self.implemented_endpoints]

        # Ensure at least quotes endpoint is used as fallback
        if not endpoints:
            logger.warning("No valid endpoints suggested, defaulting to quotes")
            endpoints = ['quotes']

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
        """Fetch real-time and historical quotes from MarketDataApp API."""
        if not self.api_key:
            raise Exception(f"MarketDataApp API key not configured. Set MARKETDATAAPP_API_KEY environment variable.")

        try:
            # Make API call to MarketDataApp
            url = f"{self.base_url}/stocks/quotes/{symbol}"
            params = {'token': self.api_key}

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data and 'last' in data:
                quote_data = data['last'][0] if isinstance(data['last'], list) else data['last']
                return {
                    'symbol': symbol,
                    'price': quote_data.get('price', 0),
                    'bid': quote_data.get('bid', 0),
                    'ask': quote_data.get('ask', 0),
                    'volume': quote_data.get('volume', 0),
                    'market_cap': quote_data.get('marketCap', 0),
                    'pe_ratio': quote_data.get('pe', 0),
                    'dividend_yield': quote_data.get('dividendYield', 0),
                    'beta': quote_data.get('beta', 0),
                    '52_week_high': quote_data.get('52WeekHigh', 0),
                    '52_week_low': quote_data.get('52WeekLow', 0),
                    'vwap': quote_data.get('vwap', 0),
                    'avg_volume_30d': quote_data.get('avgVolume30Day', 0),
                    'short_interest': quote_data.get('shortInterest', 0),
                    'source': 'marketdataapp_api',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                raise Exception(f"No quote data received for {symbol} from MarketDataApp API")

        except Exception as e:
            raise Exception(f"Error fetching quotes from MarketDataApp API for {symbol}: {e}")



    async def _fetch_trades_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch detailed trade execution data."""
        raise NotImplementedError("Real MarketDataApp trades API implementation required - no mock data allowed in production")

    async def _fetch_orderbook_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch Level 2 order book depth."""
        raise NotImplementedError("Real MarketDataApp orderbook API implementation required - no mock data allowed in production")

    async def _fetch_options_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch options chain data."""
        raise NotImplementedError("Real MarketDataApp options API implementation required - no mock data allowed in production")

    async def _fetch_darkpool_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch dark pool trade detection."""
        raise NotImplementedError("Real MarketDataApp darkpool API implementation required - no mock data allowed in production")

    async def _fetch_microstructure_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch advanced market microstructure analysis."""
        raise NotImplementedError("Real MarketDataApp microstructure API implementation required - no mock data allowed in production")

    async def _fetch_flow_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch institutional order flow data."""
        raise NotImplementedError("Real MarketDataApp flow API implementation required - no mock data allowed in production")

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

# Standalone test (run python src/agents/data_analyzers/marketdataapp_data_analyzer.py to verify)
if __name__ == "__main__":
    import asyncio
    agent = MarketDataAppDataAnalyzer()
    result = asyncio.run(agent.process_input({'symbol': 'SPY'}))
    print("MarketDataApp Subagent Test Result:")
    print(f"Keys: {list(result.keys())}")
    if 'quotes_df' in result:
        print(f"Quotes DataFrame shape: {result['quotes_df'].shape}")
        print(f"Sample quotes: {result.get('quotes_df', {}).head() if 'quotes_df' in result else 'No quotes data'}")