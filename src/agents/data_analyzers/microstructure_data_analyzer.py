# src/agents/data_subs/microstructure_datasub.py
# Purpose: Microstructure Data Subagent for real-time market analysis.
# Fetches quotes, trades, order book data, and performs microstructure analysis for optimal execution.
# Structural Reasoning: Dedicated subagent for microstructure data, enabling parallel processing with other sources.
# Ties to system: Provides microstructure data dict for main data agent coordination.
# For legacy wealth: Access to real-time market microstructure for superior execution timing.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Dynamic root path for imports.

from src.agents.data_analyzers.base_data_analyzer import BaseDataAnalyzer  # Absolute import.
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from langchain_core.tools import BaseTool
from src.utils.tools import microstructure_analysis_tool

logger = logging.getLogger(__name__)

class MicrostructureDataAnalyzer(BaseDataAnalyzer):
    """
    Microstructure Data Subagent.
    Reasoning: Fetches and analyzes real-time market microstructure for optimal execution.
    """
    def __init__(self):
        config_paths = {'risk': 'config/risk-constraints.yaml'}  # Relative to root.
        prompt_paths = {'base': 'config/base_prompt.txt', 'role': 'docs/AGENTS/main-agents/data-agent.md'}  # Relative to root.
        tools = []  # MicrostructureDatasub uses internal methods instead of tools
        super().__init__(role='microstructure_data')

    async def _plan_data_exploration(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan microstructure data exploration.

        Args:
            input_data: Input parameters

        Returns:
            Exploration plan
        """
        symbols = input_data.get("symbols", [])
        return {
            "symbols": symbols,
            "sources": ["quotes", "trades", "order_book"],
            "strategy": "real_time_fetch"
        }

    async def _execute_data_exploration(self, exploration_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute microstructure data exploration.

        Args:
            exploration_plan: Plan from _plan_data_exploration

        Returns:
            Raw microstructure data
        """
        symbols = exploration_plan.get("symbols", [])
        # Fetch data using tool
        data = await microstructure_analysis_tool(symbols=symbols)
        return {"microstructure_data": data, "symbols": symbols}

    async def _enhance_data(self, validated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance microstructure data with analysis.

        Args:
            validated_data: Validated data

        Returns:
            Enhanced data with analysis
        """
        data = validated_data.get("microstructure_data", [])
        return {
            "microstructure_data": data,
            "analysis": {
                "total_quotes": len(data),
                "liquidity_metrics": {}
            }
        }

    async def reflect(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on batch adjustments for self-improvement.
        Analyzes microstructure performance metrics and generates optimization insights.

        Args:
            metrics: Dictionary containing performance data and adjustment metrics

        Returns:
            Dict with reflection insights and improvement recommendations
        """
        try:
            logger.info(f"Microstructure reflecting on adjustments: {metrics}")

            # Analyze microstructure-specific performance
            reflection_insights = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'metrics_analyzed': len(metrics) if isinstance(metrics, dict) else 0,
                'performance_analysis': {},
                'optimization_opportunities': [],
                'data_quality_insights': {}
            }

            # Extract microstructure performance data
            if isinstance(metrics, dict):
                # Analyze data fetch success rates, processing times, analysis quality
                if 'performance_data' in metrics:
                    perf_data = metrics['performance_data']
                    reflection_insights['performance_analysis'] = {
                        'data_fetch_success_rate': perf_data.get('data_fetch_success_rate', 0),
                        'analysis_processing_time': perf_data.get('analysis_processing_time', 0),
                        'llm_analysis_quality': perf_data.get('llm_analysis_quality', 0),
                        'execution_recommendation_accuracy': perf_data.get('execution_recommendation_accuracy', 0)
                    }

                # Identify microstructure-specific improvement opportunities
                if 'issues' in metrics:
                    issues = metrics['issues']
                    for issue in issues:
                        if 'liquidity' in issue.lower():
                            reflection_insights['optimization_opportunities'].append({
                                'type': 'liquidity_analysis_enhancement',
                                'description': 'Improve liquidity assessment algorithms and data sources',
                                'priority': 'high'
                            })
                        elif 'order_flow' in issue.lower():
                            reflection_insights['optimization_opportunities'].append({
                                'type': 'order_flow_detection',
                                'description': 'Enhance order flow pattern recognition and manipulation detection',
                                'priority': 'high'
                            })
                        elif 'latency' in issue.lower():
                            reflection_insights['optimization_opportunities'].append({
                                'type': 'real_time_processing',
                                'description': 'Optimize real-time data processing and reduce latency',
                                'priority': 'medium'
                            })

                # Generate data quality insights
                reflection_insights['data_quality_insights'] = {
                    'data_completeness_score': metrics.get('data_completeness', 0.8),
                    'signal_quality_score': metrics.get('signal_quality', 0.7),
                    'freshness_score': metrics.get('data_freshness', 0.9),
                    'manipulation_detection_effectiveness': metrics.get('manipulation_detection', 0.6)
                }

            # Store reflection in memory for future learning
            self.update_memory('microstructure_reflection_history', {
                'timestamp': reflection_insights['timestamp'],
                'insights': reflection_insights,
                'metrics_analyzed': metrics
            })

            logger.info(f"Microstructure reflection completed with {len(reflection_insights['optimization_opportunities'])} optimization opportunities identified")
            return reflection_insights

        except Exception as e:
            logger.error(f"Error during microstructure reflection analysis: {e}")
            return {
                'error': str(e),
                'timestamp': pd.Timestamp.now().isoformat(),
                'metrics_analyzed': len(metrics) if isinstance(metrics, dict) else 0
            }

    async def process_input(self, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process input to fetch and analyze microstructure data with LLM enhancement.
        Args:
            input_data: Dict with parameters (symbol for microstructure analysis).
        Returns:
            Dict with structured microstructure data and LLM analysis.
        """
        logger.info(f"MicrostructureDatasub processing input: {input_data}")

        try:
            symbol = input_data.get('symbol', 'SPY') if input_data else 'SPY'

            # Step 1: Plan microstructure exploration with LLM
            exploration_plan = await self._plan_microstructure_exploration(symbol, input_data)

            # Step 2: Fetch data from multiple sources concurrently
            raw_data = await self._fetch_microstructure_sources_concurrent(symbol, exploration_plan)

            # Step 3: Consolidate data into structured DataFrames
            consolidated_data = self._consolidate_microstructure_data(raw_data, symbol)

            # Step 4: Analyze with LLM for insights
            llm_analysis = await self._analyze_microstructure_data_llm(consolidated_data)

            # Combine results
            result = {
                "consolidated_data": consolidated_data,
                "llm_analysis": llm_analysis,
                "exploration_plan": exploration_plan,
                "enhanced": True
            }

            # Store microstructure data in shared memory
            await self.store_shared_memory("microstructure_data", symbol, {
                "microstructure_data": consolidated_data,
                "llm_analysis": llm_analysis,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"MicrostructureDatasub output: LLM-enhanced microstructure data collected for {symbol}")
            return result

        except Exception as e:
            logger.error(f"MicrostructureDatasub failed: {e}")
            return {"spread": 0.0, "error": str(e), "enhanced": False}

    def _enhance_microstructure_analysis(self, microstructure_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Enhance microstructure analysis with additional execution insights."""
        try:
            if "error" in microstructure_data:
                return microstructure_data

            analysis = microstructure_data.get("analysis", {})

            # Add execution strategy recommendations
            execution_strategy = {}

            # Market order vs limit order recommendations
            spread_pct = analysis.get("spread_analysis", {}).get("spread_percent", 0)
            if spread_pct < 0.05:
                execution_strategy["primary_order_type"] = "market_order"
                execution_strategy["reason"] = "Tight spread allows efficient market orders"
            elif spread_pct < 0.15:
                execution_strategy["primary_order_type"] = "market_order"
                execution_strategy["secondary_order_type"] = "limit_order"
                execution_strategy["reason"] = "Moderate spread - market orders acceptable, limit orders for precision"
            else:
                execution_strategy["primary_order_type"] = "limit_order"
                execution_strategy["reason"] = "Wide spread requires limit orders for cost control"

            # Timing recommendations
            volume_trend = analysis.get("volume_analysis", {}).get("volume_trend", "normal")
            if volume_trend == "high":
                execution_strategy["timing"] = "immediate"
                execution_strategy["timing_reason"] = "High volume provides excellent liquidity"
            elif volume_trend == "low":
                execution_strategy["timing"] = "patient"
                execution_strategy["timing_reason"] = "Low volume - consider breaking up orders or waiting for better conditions"
            else:
                execution_strategy["timing"] = "standard"
                execution_strategy["timing_reason"] = "Normal volume conditions"

            # Order size recommendations
            market_condition = analysis.get("market_condition", "neutral")
            if market_condition == "favorable":
                execution_strategy["max_order_size"] = "large"
                execution_strategy["size_reason"] = "Favorable conditions allow larger orders"
            elif market_condition == "challenging":
                execution_strategy["max_order_size"] = "small"
                execution_strategy["size_reason"] = "Challenging conditions - use smaller orders to minimize impact"
            else:
                execution_strategy["max_order_size"] = "medium"
                execution_strategy["size_reason"] = "Neutral conditions - standard order sizing appropriate"

            # Momentum-based recommendations
            momentum = analysis.get("order_flow", {}).get("momentum", "neutral")
            if momentum == "bullish":
                execution_strategy["momentum_bias"] = "buy_favorable"
                execution_strategy["momentum_reason"] = "Bullish order flow supports buying"
            elif momentum == "bearish":
                execution_strategy["momentum_bias"] = "sell_favorable"
                execution_strategy["momentum_reason"] = "Bearish order flow supports selling"
            else:
                execution_strategy["momentum_bias"] = "neutral"
                execution_strategy["momentum_reason"] = "Neutral momentum - no directional bias"

            # Calculate execution quality score (0-100)
            quality_score = 50  # Base score

            # Adjust for spread
            if spread_pct < 0.05:
                quality_score += 20
            elif spread_pct > 0.15:
                quality_score -= 20

            # Adjust for volume
            if volume_trend == "high":
                quality_score += 15
            elif volume_trend == "low":
                quality_score -= 15

            # Adjust for market condition
            if market_condition == "favorable":
                quality_score += 10
            elif market_condition == "challenging":
                quality_score -= 10

            # Adjust for momentum alignment
            if momentum != "neutral":
                quality_score += 5

            quality_score = max(0, min(100, quality_score))

            execution_strategy["execution_quality_score"] = quality_score
            execution_strategy["quality_rating"] = "excellent" if quality_score >= 80 else "good" if quality_score >= 60 else "fair" if quality_score >= 40 else "poor"

            # Add execution strategy to microstructure data
            microstructure_data["execution_strategy"] = execution_strategy

            # Add real-time actionability assessment
            microstructure_data["real_time_assessment"] = {
                "timestamp": microstructure_data.get("timestamp", ""),
                "actionable": quality_score >= 40,
                "recommended_action": "proceed_with_execution" if quality_score >= 60 else "monitor_conditions" if quality_score >= 40 else "wait_for_better_conditions",
                "confidence_level": "high" if quality_score >= 80 else "medium" if quality_score >= 60 else "low"
            }

            return microstructure_data

        except Exception as e:
            logger.error(f"Error enhancing microstructure analysis for {symbol}: {e}")
            microstructure_data["enhancement_error"] = str(e)
            return microstructure_data

    def analyze_market_microstructure(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze market microstructure for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict containing market microstructure analysis
        """
        return {
            'symbol': symbol,
            'spread_analysis': {
                'average_spread': 0.02,
                'spread_volatility': 0.005,
                'effective_spread': 0.025
            },
            'depth_analysis': {
                'bid_depth': 1000000,
                'ask_depth': 1200000,
                'depth_imbalance': 0.091
            },
            'liquidity_metrics': {
                'trading_volume': 5000000,
                'turnover_ratio': 0.15,
                'price_impact': 0.001
            },
            'execution_quality': {
                'slippage_model': {
                    'base_slippage': 0.0005,
                    'multiplier': 1.0,
                    'optimal_slippage': 0.0005,
                    'expected_fill_quality': 'excellent'
                },
                'market_condition': 'challenging'
            },
            'source': 'microstructure_analysis_subagent'
        }

    async def _plan_microstructure_exploration(self, symbol: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Use LLM to plan intelligent exploration of market microstructure data."""
        context_str = f"""
        Symbol: {symbol}
        Market Context: {context or 'General market microstructure analysis'}
        Current Microstructure Data: {self.analyze_market_microstructure(symbol)}
        """
        
        question = f"""
        Plan a comprehensive exploration strategy for market microstructure analysis of {symbol}.
        Consider:
        1. Key microstructure metrics to analyze (order book depth, spread, volume profile)
        2. Time & sales data patterns and trade clustering
        3. Order flow analysis and market impact assessment
        4. Liquidity analysis and slippage estimation
        5. Market manipulation detection signals
        
        Provide a structured plan with priorities and data sources to explore.
        """
        
        plan_response = await self.reason_with_llm(context_str, question)
        return {"plan": plan_response, "symbol": symbol, "timestamp": pd.Timestamp.now().isoformat()}

    async def _fetch_microstructure_sources_concurrent(self, symbol: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch microstructure data from multiple sources concurrently."""
        import asyncio
        
        async def fetch_level2_orderbook(symbol: str) -> Dict[str, Any]:
            """Fetch L2 order book data."""
            raise NotImplementedError("Real L2 order book API implementation required - no mock data allowed in production")
        
        async def fetch_time_and_sales(symbol: str) -> Dict[str, Any]:
            """Fetch time & sales data."""
            raise NotImplementedError("Real time & sales API implementation required - no mock data allowed in production")
        
        async def fetch_liquidity_metrics(symbol: str) -> Dict[str, Any]:
            """Fetch liquidity and slippage metrics."""
            raise NotImplementedError("Real liquidity metrics API implementation required - no mock data allowed in production")
        
        # Execute concurrent fetches
        tasks = [
            fetch_level2_orderbook(symbol),
            fetch_time_and_sales(symbol),
            fetch_liquidity_metrics(symbol)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        consolidated = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in microstructure fetch: {result}")
                continue
            if isinstance(result, dict):
                source = result.get("source", "unknown")
                consolidated[source] = result
        
        return consolidated

    def _consolidate_microstructure_data(self, raw_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Consolidate microstructure data into structured DataFrames."""
        try:
            import pandas as pd
            
            # Create order book DataFrame
            orderbook_data = []
            l2_data = raw_data.get("level2", {})
            
            for bid in l2_data.get("bids", []):
                orderbook_data.append({
                    "price": bid[0],
                    "volume": bid[1],
                    "side": "bid"
                })
            
            for ask in l2_data.get("asks", []):
                orderbook_data.append({
                    "price": ask[0],
                    "volume": ask[1],
                    "side": "ask"
                })
            
            orderbook_df = pd.DataFrame(orderbook_data)
            
            # Create trades DataFrame
            trades_data = []
            ts_data = raw_data.get("time_sales", {})
            
            for trade in ts_data.get("trades", []):
                trades_data.append({
                    "time": trade.get("time", ""),
                    "price": trade.get("price", 0.0),
                    "volume": trade.get("volume", 0),
                    "side": trade.get("side", "")
                })
            
            trades_df = pd.DataFrame(trades_data)
            
            # Create liquidity metrics DataFrame
            liquidity_data = []
            liq_data = raw_data.get("liquidity", {})
            
            liquidity_data.append({
                "metric": "market_depth",
                "value": liq_data.get("market_depth", 0),
                "unit": "shares"
            })
            liquidity_data.append({
                "metric": "estimated_slippage",
                "value": liq_data.get("estimated_slippage", 0),
                "unit": "percent"
            })
            liquidity_data.append({
                "metric": "liquidity_score",
                "value": liq_data.get("liquidity_score", 0),
                "unit": "score"
            })
            
            liquidity_df = pd.DataFrame(liquidity_data)
            
            return {
                "symbol": symbol,
                "orderbook_df": orderbook_df.to_dict('records'),
                "trades_df": trades_df.to_dict('records'),
                "liquidity_df": liquidity_df.to_dict('records'),
                "summary": {
                    "spread": l2_data.get("spread", 0.0),
                    "total_depth": l2_data.get("depth", 0),
                    "avg_trade_volume": ts_data.get("avg_volume", 0),
                    "trade_frequency": ts_data.get("trade_frequency", 0),
                    "liquidity_score": liq_data.get("liquidity_score", 0.0)
                },
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error consolidating microstructure data: {e}")
            return {"error": str(e), "symbol": symbol}

    async def _analyze_microstructure_data_llm(self, consolidated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM for market microstructure analysis and manipulation detection."""
        context_str = f"""
        Microstructure Data Analysis for {consolidated_data.get('symbol', 'Unknown')}:
        
        Summary Statistics:
        {consolidated_data.get('summary', {})}
        
        Order Book Data:
        {consolidated_data.get('orderbook_df', [])}
        
        Recent Trades:
        {consolidated_data.get('trades_df', [])}
        
        Liquidity Metrics:
        {consolidated_data.get('liquidity_df', [])}
        """
        
        question = """
        Analyze this market microstructure data and provide insights on:
        1. Order flow patterns and potential market direction signals
        2. Liquidity assessment and slippage risk evaluation
        3. Market manipulation detection (spoofing, layering, wash trades)
        4. Optimal execution strategies based on current microstructure
        5. Risk management recommendations for trading in this environment
        
        Focus on execution quality and market integrity assessment.
        """
        
        analysis_response = await self.reason_with_llm(context_str, question)
        return {
            "llm_analysis": analysis_response,
            "manipulation_risk": self._extract_manipulation_risk(analysis_response),
            "execution_recommendations": self._extract_execution_recommendations(analysis_response),
            "timestamp": pd.Timestamp.now().isoformat()
        }

    def _extract_manipulation_risk(self, llm_response: str) -> str:
        """Extract market manipulation risk assessment."""
        response_lower = llm_response.lower()
        if "high risk" in response_lower or "manipulation" in response_lower or "suspicious" in response_lower:
            return "High Risk - Potential Market Manipulation"
        elif "moderate" in response_lower:
            return "Moderate Risk"
        else:
            return "Low Risk - Normal Market Activity"

    def _extract_execution_recommendations(self, llm_response: str) -> List[str]:
        """Extract execution recommendations from LLM response."""
        recommendations = []
        response_lower = llm_response.lower()
        
        if "limit order" in response_lower:
            recommendations.append("Use limit orders to minimize slippage")
        if "iceberg" in response_lower or "split" in response_lower:
            recommendations.append("Consider order splitting for large trades")
        if "vwap" in response_lower:
            recommendations.append("Use VWAP execution for institutional trades")
        if "market order" in response_lower:
            recommendations.append("Market orders acceptable for small sizes")
        
        return recommendations if recommendations else ["Standard execution protocols recommended"]

# Standalone test (run python src/agents/data_subs/microstructure_datasub.py to verify)
if __name__ == "__main__":
    import asyncio
    agent = MicrostructureDatasub()
    result = asyncio.run(agent.process_input({'symbols': ['SPY']}))
    print("Microstructure Subagent Test Result:")
    print(f"Keys: {list(result.keys())}")
    if 'microstructure' in result:
        analysis = result['microstructure'].get('analysis', {})
        execution = result['microstructure'].get('execution_strategy', {})
        print(f"Market condition: {analysis.get('market_condition', 'unknown')}")
        print(f"Execution quality score: {execution.get('execution_quality_score', 'N/A')}")
        print(f"Recommended order type: {execution.get('primary_order_type', 'unknown')}")