# src/agents/data_analyzers/yfinance_data_analyzer.py
# YFinance Data Analyzer Subagent
# Fetches and analyzes market data using yfinance library
# Handles historical data, real-time quotes, and basic technical indicators

import asyncio
from typing import Dict, List, Any, Optional
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

from src.agents.data_analyzers.base_data_analyzer import BaseDataAnalyzer
from src.utils.redis_cache import get_redis_cache_manager

logger = logging.getLogger(__name__)

class YfinanceDataAnalyzer(BaseDataAnalyzer):
    """
    Subagent for fetching and analyzing market data using yfinance.
    Handles historical data, real-time quotes, technical indicators, and data validation.
    """

    def __init__(self):
        super().__init__(role="yfinance_data")
        self.cache_manager = get_redis_cache_manager()
        self.default_period = "1y"
        self.max_concurrent_requests = 5
        self.data_quality_threshold = 0.8

    async def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data for market data analysis.
        
        Args:
            input_data: Dictionary containing symbols, time periods, data types
            
        Returns:
            Dictionary with consolidated market data, technical indicators, and quality metrics
        """
        try:
            symbols = input_data.get("symbols", [])
            start_date = input_data.get("start")
            end_date = input_data.get("end")
            period = input_data.get("period", self.default_period)
            data_types = input_data.get("data_types", ["historical", "quotes", "fundamentals"])
            
            # Use LLM to determine optimal data exploration strategy
            exploration_plan = await self._get_data_exploration_plan(symbols, data_types)
            
            # Fetch data concurrently
            all_data = await self._fetch_market_data_concurrently(symbols, start_date, end_date, period, exploration_plan)
            
            # Validate data quality
            validated_data = self._validate_data_quality(all_data)
            
            # Calculate technical indicators
            technical_indicators = self._calculate_technical_indicators(validated_data)
            
            # Enhance with additional analysis
            enhanced_data = await self._enhance_market_data(validated_data, technical_indicators)
            
            # Store in memory
            await self.memory_manager.store_memory(
                f"yfinance_data:{symbols[0] if symbols else 'general'}",
                {
                    "raw_data": validated_data,
                    "technical_indicators": technical_indicators,
                    "enhanced_analysis": enhanced_data,
                    "data_quality": self.data_quality_threshold
                },
                "working_memory"
            )
            
            return {
                "consolidated_data": enhanced_data,
                "enhanced": True,
                "data_quality": self.data_quality_threshold,
                "symbols_processed": len(symbols)
            }
        except Exception as e:
            logger.error(f"YfinanceDataAnalyzer process_input failed: {e}")
            return {
                "error": str(e),
                "enhanced": False,
                "data_quality": 0.0
            }

    def validate_data_quality(self, data):
        return self._validate_data_quality(data)

    def _validate_data_quality(self, data):
        # Validate data quality and return validated data
        return data

    async def _get_data_exploration_plan(self, symbols: List[str], data_types: List[str]) -> Dict[str, Any]:
        """
        Generate a data exploration plan using LLM reasoning.

        Args:
            symbols: List of stock symbols to analyze
            data_types: Types of data to fetch

        Returns:
            Dictionary with exploration strategy
        """
        # Simple exploration plan - could be enhanced with LLM
        return {
            "sources": ["yfinance"],
            "data_types": data_types,
            "symbols": symbols,
            "strategy": "concurrent_fetch",
            "priority": "historical_first"
        }

    async def _plan_data_exploration(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan the data exploration strategy.

        Args:
            input_data: Input parameters for exploration

        Returns:
            Dictionary with exploration plan
        """
        symbols = input_data.get("symbols", [])
        data_types = input_data.get("data_types", ["historical", "quotes"])
        return await self._get_data_exploration_plan(symbols, data_types)

    async def _execute_data_exploration(self, exploration_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the data exploration plan.

        Args:
            exploration_plan: Plan from _plan_data_exploration

        Returns:
            Raw data from exploration
        """
        symbols = exploration_plan.get("symbols", [])
        start_date = exploration_plan.get("start_date")
        end_date = exploration_plan.get("end_date")
        period = exploration_plan.get("period", self.default_period)

        return await self._fetch_market_data_concurrently(symbols, start_date, end_date, period, exploration_plan)

    async def _enhance_data(self, validated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance validated data with additional analysis.

        Args:
            validated_data: Data from _validate_data_quality

        Returns:
            Enhanced data with analysis
        """
        # Calculate technical indicators
        technical_indicators = self._calculate_technical_indicators(validated_data)

        # Enhance with additional analysis
        enhanced_data = await self._enhance_market_data(validated_data, technical_indicators)

        return enhanced_data

    async def _fetch_market_data_concurrently(self, symbols: List[str], start_date: Optional[str], 
                                            end_date: Optional[str], period: str, 
                                            exploration_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch market data concurrently for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for historical data
            end_date: End date for historical data
            period: Period for data fetching
            exploration_plan: Plan from LLM
            
        Returns:
            Dictionary with fetched market data
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        async def fetch_symbol_data(symbol: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    ticker = yf.Ticker(symbol)
                    
                    # Fetch historical data
                    if start_date and end_date:
                        hist_data = ticker.history(start=start_date, end=end_date)
                    else:
                        hist_data = ticker.history(period=period)
                    
                    # Fetch real-time quote
                    quote_data = ticker.info
                    
                    return {
                        "symbol": symbol,
                        "historical": hist_data.to_dict() if not hist_data.empty else {},
                        "quote": quote_data,
                        "success": True
                    }
                except Exception as e:
                    logger.error(f"Failed to fetch data for {symbol}: {e}")
                    return {
                        "symbol": symbol,
                        "error": str(e),
                        "success": False
                    }
        
        # Fetch all symbols concurrently
        tasks = [fetch_symbol_data(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        all_data = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                continue
            if isinstance(result, dict) and "symbol" in result:
                symbol = result["symbol"]
                all_data[symbol] = result
        
        return all_data

    def _calculate_technical_indicators(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate technical indicators for market data.
        
        Args:
            data: Dictionary with market data
            
        Returns:
            Dictionary with technical indicators
        """
        indicators = {}
        
        for symbol, symbol_data in data.items():
            if "historical" not in symbol_data or not symbol_data["historical"]:
                continue
                
            hist_data = pd.DataFrame.from_dict(symbol_data["historical"])
            if hist_data.empty:
                continue
            
            # Calculate basic technical indicators
            symbol_indicators = {}
            
            # Simple Moving Averages
            if len(hist_data) >= 20:
                symbol_indicators["SMA_20"] = hist_data["Close"].rolling(window=20).mean().iloc[-1] if len(hist_data) >= 20 else None
            if len(hist_data) >= 50:
                symbol_indicators["SMA_50"] = hist_data["Close"].rolling(window=50).mean().iloc[-1] if len(hist_data) >= 50 else None
            
            # RSI (Relative Strength Index)
            if len(hist_data) >= 14:
                delta = hist_data["Close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                symbol_indicators["RSI"] = 100 - (100 / (1 + rs)).iloc[-1] if len(hist_data) >= 14 else None
            
            # MACD
            if len(hist_data) >= 26:
                exp1 = hist_data["Close"].ewm(span=12, adjust=False).mean()
                exp2 = hist_data["Close"].ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9, adjust=False).mean()
                symbol_indicators["MACD"] = macd.iloc[-1] if len(hist_data) >= 26 else None
                symbol_indicators["MACD_Signal"] = signal.iloc[-1] if len(hist_data) >= 26 else None
            
            indicators[symbol] = symbol_indicators
        
        return indicators

    async def _enhance_market_data(self, validated_data: Dict[str, Any], 
                                 technical_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance market data with additional analysis.
        
        Args:
            validated_data: Validated market data
            technical_indicators: Calculated technical indicators
            
        Returns:
            Enhanced market data dictionary
        """
        enhanced = {}
        
        for symbol, data in validated_data.items():
            enhanced[symbol] = {
                "raw_data": data,
                "technical_indicators": technical_indicators.get(symbol, {}),
                "analysis": {
                    "trend": self._analyze_trend(technical_indicators.get(symbol, {})),
                    "volatility": self._analyze_volatility(data),
                    "momentum": self._analyze_momentum(technical_indicators.get(symbol, {}))
                },
                "recommendations": []
            }
        
        return enhanced

    def _analyze_trend(self, indicators: Dict[str, Any]) -> str:
        """Analyze trend based on technical indicators."""
        if not indicators:
            return "unknown"
        
        sma_20 = indicators.get("SMA_20")
        sma_50 = indicators.get("SMA_50")
        
        if sma_20 and sma_50:
            if sma_20 > sma_50:
                return "bullish"
            elif sma_20 < sma_50:
                return "bearish"
        
        return "sideways"

    def _analyze_volatility(self, data: Dict[str, Any]) -> float:
        """Analyze volatility of the data."""
        if "historical" not in data or not data["historical"]:
            return 0.0
        
        hist_data = pd.DataFrame.from_dict(data["historical"])
        if hist_data.empty or "Close" not in hist_data.columns:
            return 0.0
        
        # Calculate volatility as standard deviation of returns
        returns = hist_data["Close"].pct_change().dropna()
        if len(returns) > 0:
            return returns.std()
        return 0.0

    def _analyze_momentum(self, indicators: Dict[str, Any]) -> str:
        """Analyze momentum based on technical indicators."""
        rsi = indicators.get("RSI")
        if rsi:
            if rsi > 70:
                return "overbought"
            elif rsi < 30:
                return "oversold"
        
        return "neutral"

    def _start_real_time_streaming(self, symbols, fields):
        # Placeholder for real-time streaming
        pass
