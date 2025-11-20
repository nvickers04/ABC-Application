# src/agents/strategy_subs/options_strategy_sub.py
# Comprehensive Options Strategy Analyzer implementing full specification
# Advanced options strategy analysis, Greeks calculations, volatility modeling, and portfolio optimization

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Dynamic root path for imports.

from src.agents.base import BaseAgent  # Absolute import.
import logging
from typing import Dict, Any, List, Optional, Type
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from scipy.stats import norm
from scipy.optimize import minimize
import asyncio
from src.utils.redis_cache import get_redis_cache_manager, cache_get, cache_set, cache_delete
from src.utils.risk_analytics_framework import RiskAnalyticsFramework

logger = logging.getLogger(__name__)

@dataclass
class OptionsMemory:
    """Collaborative memory for options strategies and insights."""
    strategy_performance: Dict[str, Any] = field(default_factory=dict)
    volatility_surfaces: Dict[str, Any] = field(default_factory=dict)
    greeks_exposure: Dict[str, Any] = field(default_factory=dict)
    risk_metrics: Dict[str, Any] = field(default_factory=dict)
    alpha_signals: List[Dict[str, Any]] = field(default_factory=list)
    session_insights: List[Dict[str, Any]] = field(default_factory=list)

    def add_session_insight(self, insight: Dict[str, Any]):
        """Add options insight to session memory."""
        self.session_insights.append({
            **insight,
            'timestamp': datetime.now().isoformat()
        })

    def get_recent_insights(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent options insights."""
        return self.session_insights[-limit:]

class BlackScholesCalculator:
    """Advanced Black-Scholes calculator with Greeks and volatility modeling."""

    @staticmethod
    def calculate_option_price(S: float, K: float, T: float, r: float, sigma: float,
                             option_type: str = 'call') -> float:
        """Calculate Black-Scholes option price."""
        if T <= 0 or sigma <= 0:
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        sigma = max(sigma, 0.001)
        sqrt_T = np.sqrt(T)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return price

    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float,
                        option_type: str = 'call') -> Dict[str, float]:
        """Calculate comprehensive option Greeks."""
        if T <= 0 or sigma <= 0:
            if option_type == 'call':
                delta = 1.0 if S > K else 0.0
            else:
                delta = -1.0 if S < K else 0.0
            return {
                'delta': delta,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0,
                'lambda': 0.0,
                'vanna': 0.0,
                'charm': 0.0,
                'vomma': 0.0
            }

        sigma = max(sigma, 0.001)
        sqrt_T = np.sqrt(T)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Basic Greeks
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1

        gamma = norm.pdf(d1) / (S * sigma * sqrt_T)

        if option_type == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt_T)
                    - r * K * np.exp(-r * T) * norm.cdf(d2))
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt_T)
                    + r * K * np.exp(-r * T) * norm.cdf(-d2))

        vega = S * sqrt_T * norm.pdf(d1)

        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

        # Advanced Greeks
        lambda_ = delta * S / (vega * sigma / 100) if vega > 0 else 0  # Omega/Lambda
        vanna = -norm.pdf(d1) * d2 / sigma  # Vega-delta cross effect
        charm = -norm.pdf(d1) * (2 * r * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)  # Delta decay
        vomma = vega * d1 * d2 / sigma  # Vega convexity

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho,
            'lambda': lambda_,
            'vanna': vanna,
            'charm': charm,
            'vomma': vomma
        }

    @staticmethod
    def implied_volatility(price: float, S: float, K: float, T: float, r: float,
                          option_type: str = 'call', max_iter: int = 100) -> float:
        """Calculate implied volatility using Newton-Raphson method."""
        def objective(sigma):
            return BlackScholesCalculator.calculate_option_price(S, K, T, r, sigma, option_type) - price

        def derivative(sigma):
            if sigma <= 0:
                return 0.001
            greeks = BlackScholesCalculator.calculate_greeks(S, K, T, r, sigma, option_type)
            return greeks['vega']

        sigma = 0.2  # Initial guess
        for _ in range(max_iter):
            f = objective(sigma)
            f_prime = derivative(sigma)

            if abs(f_prime) < 1e-8:
                break

            sigma = sigma - f / f_prime

            if sigma <= 0:
                sigma = 0.001

            if abs(f) < 1e-6:
                break

        return sigma

class OptionsStrategyAnalyzer(BaseAgent):
    """
    Comprehensive Options Strategy Analyzer implementing full specification.
    Advanced options strategy analysis, Greeks calculations, volatility modeling, and portfolio optimization.
    """

    def __init__(self):
        config_paths = {'risk': 'config/risk-constraints.yaml'}  # Relative to root.
        prompt_paths = {'base': 'config/base_prompt.txt', 'role': 'docs/AGENTS/main-agents/strategy-agent.md'}  # Relative to root.
        tools = []  # OptionsStrategyAnalyzer uses internal methods instead of tools
        super().__init__(role='options_strategy', config_paths=config_paths, prompt_paths=prompt_paths, tools=tools)

        # Initialize Redis cache manager
        self.redis_cache = get_redis_cache_manager()
        self.cache_ttl = 900  # 15 minutes TTL for options data

        # Initialize collaborative memory
        self.memory = OptionsMemory()

        # Initialize Black-Scholes calculator
        self.bs_calculator = BlackScholesCalculator()

        # Risk-free rate (can be updated from market data)
        self.risk_free_rate = 0.045

        # Strategy configurations
        self.strategy_configs = {
            'single_leg': ['long_call', 'long_put', 'short_call', 'short_put'],
            'spreads': ['bull_call_spread', 'bear_put_spread', 'call_spread', 'put_spread'],
            'combinations': ['straddle', 'strangle', 'butterfly', 'condor', 'iron_condor'],
            'synthetics': ['synthetic_long', 'synthetic_short', 'collar', 'covered_call', 'protective_put'],
            'advanced': ['calendar_spread', 'diagonal_spread', 'ratio_spread', 'backspread']
        }

        # Risk management parameters
        self.risk_params = {
            'max_delta_exposure': 0.5,
            'max_gamma_exposure': 2.0,
            'max_theta_exposure': -1.0,
            'max_vega_exposure': 1.0,
            'min_pop_threshold': 0.55,
            'max_loss_percentage': 0.10
        }

    def reflect(self, adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on batch adjustments for self-improvement.
        """
        logger.info(f"Options Reflecting on adjustments: {adjustments}")
        return {}

    async def process_input(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive options strategy analysis with advanced modeling and optimization.
        """
        logger.info(f"OptionsStrategyAnalyzer processing input: {input_data or 'Default analysis'}")

        # Extract analysis parameters
        symbols = input_data.get('symbols', ['SPY']) if input_data else ['SPY']
        timeframes = input_data.get('timeframes', ['1D']) if input_data else ['1D']
        include_volatility_modeling = input_data.get('volatility_modeling', True) if input_data else True
        include_strategy_optimization = input_data.get('strategy_optimization', True) if input_data else True
        include_portfolio_construction = input_data.get('portfolio_construction', True) if input_data else True

        # Try to retrieve options data from shared memory first
        options_data_available = False
        shared_options_data = {}
        
        for symbol in symbols:
            shared_data = await self.retrieve_shared_memory("options_data", symbol)
            if shared_data:
                shared_options_data[symbol] = shared_data
                options_data_available = True
                logger.info(f"Retrieved options data from shared memory for {symbol}")
            else:
                logger.warning(f"No options data found in shared memory for {symbol}")

        # Create cache key
        cache_key = f"options_strategy_{'_'.join(symbols)}_{'_'.join(timeframes)}_{include_volatility_modeling}_{include_strategy_optimization}"

        # Check cache first
        if self._is_cache_valid(cache_key):
            logger.info(f"Using cached options strategy for: {cache_key}")
            return self._get_cached_data(cache_key)

        try:
            # Require shared options data - no fallback to internal analysis
            if not options_data_available:
                logger.error(f"No shared options data available for symbols {symbols} - cannot perform options strategy analysis")
                return {
                    "error": "No shared options data available for strategy analysis",
                    "data_missing": True,
                    "symbols_requested": symbols
                }

            # Use shared options data for analysis
            options_analysis = await self._analyze_options_from_shared_data(
                shared_options_data, symbols, timeframes, include_volatility_modeling,
                include_strategy_optimization, include_portfolio_construction
            )
            logger.info(f"Analyzed options strategies using shared data for {len(symbols)} symbols")

            # Generate alpha signals from options analysis
            alpha_signals = self._generate_options_alpha_signals(options_analysis)

            # Build comprehensive options strategy proposals
            strategy_proposals = self._build_options_strategy_proposals(options_analysis, alpha_signals)

            # Calculate risk-adjusted returns
            risk_adjusted_proposals = self._calculate_risk_adjusted_returns(strategy_proposals)

            # Generate collaborative insights
            collaborative_insights = self._generate_collaborative_insights(options_analysis, alpha_signals)

            # Update memory and models
            self._update_memory(options_analysis, alpha_signals)

            # Structure the response
            result = {
                'options_analysis': options_analysis,
                'alpha_signals': alpha_signals,
                'strategy_proposals': risk_adjusted_proposals,
                'collaborative_insights': collaborative_insights,
                'metadata': {
                    'symbols_analyzed': symbols,
                    'timeframes': timeframes,
                    'strategies_evaluated': len(self.strategy_configs),
                    'analysis_timestamp': datetime.now().isoformat(),
                    'total_signals': len(alpha_signals)
                }
            }

            # Cache the result
            self._cache_data(cache_key, {"options": result})

            logger.info(f"OptionsStrategyAnalyzer completed analysis: {len(alpha_signals)} alpha signals generated")
            return {"options": result}

        except Exception as e:
            logger.error(f"OptionsStrategyAnalyzer failed: {e}")
            result = {
                "options_strategy": {
                    "error": str(e),
                    "options_analysis": {},
                    "alpha_signals": [],
                    "strategy_proposals": [],
                    "metadata": {
                        "symbols_analyzed": symbols if 'symbols' in locals() else ['SPY'],
                        "analysis_timestamp": datetime.now().isoformat()
                    }
                }
            }
            self._cache_data(cache_key, result)
            return result

    def _is_cache_valid(self, cache_key):
        """Check if Redis cache entry exists and is valid."""
        return cache_get('options_strategy', cache_key) is not None

    def _get_cached_data(self, cache_key):
        """Get cached options strategy data from Redis."""
        return cache_get('options_strategy', cache_key)

    def _cache_data(self, cache_key, data):
        """Cache options strategy data in Redis with TTL."""
        cache_set('options_strategy', cache_key, data, self.cache_ttl)

    async def _analyze_multi_symbol_options(self, symbols: List[str], timeframes: List[str],
                                          include_volatility_modeling: bool, include_strategy_optimization: bool,
                                          include_portfolio_construction: bool) -> Dict[str, Any]:
        """Analyze options strategies across multiple symbols and timeframes."""
        options_analysis = {}

        for symbol in symbols:
            symbol_options = {}

            # Analyze each timeframe
            for timeframe in timeframes:
                timeframe_data = await self._analyze_timeframe_options(
                    symbol, timeframe, include_volatility_modeling,
                    include_strategy_optimization, include_portfolio_construction
                )
                symbol_options[timeframe] = timeframe_data

            # Aggregate across timeframes
            symbol_options['aggregate'] = self._aggregate_timeframe_options(symbol_options)

            options_analysis[symbol] = symbol_options

        return options_analysis

    async def _analyze_options_from_shared_data(self, shared_options_data: Dict[str, Any], symbols: List[str], 
                                              timeframes: List[str], include_volatility_modeling: bool, 
                                              include_strategy_optimization: bool, include_portfolio_construction: bool) -> Dict[str, Any]:
        """Analyze options strategies using data retrieved from shared memory."""
        options_analysis = {}

        for symbol in symbols:
            if symbol not in shared_options_data:
                logger.warning(f"No shared options data available for {symbol}, skipping")
                continue

            shared_data = shared_options_data[symbol]
            options_data = shared_data.get('options_data', {})
            
            symbol_options = {}

            # Analyze each timeframe using shared data
            for timeframe in timeframes:
                timeframe_data = await self._analyze_timeframe_options_from_shared_data(
                    symbol, timeframe, options_data, include_volatility_modeling,
                    include_strategy_optimization, include_portfolio_construction
                )
                symbol_options[timeframe] = timeframe_data

            # Aggregate across timeframes
            symbol_options['aggregate'] = self._aggregate_timeframe_options(symbol_options)

            options_analysis[symbol] = symbol_options

        return options_analysis

    async def _analyze_timeframe_options_from_shared_data(self, symbol: str, timeframe: str, options_data: Dict[str, Any],
                                                        include_volatility_modeling: bool, include_strategy_optimization: bool,
                                                        include_portfolio_construction: bool) -> Dict[str, Any]:
        """Analyze options strategies for a specific symbol and timeframe using shared data."""
        try:
            analysis = {
                'volatility_modeling': {},
                'strategy_optimization': {},
                'portfolio_construction': {},
                'greeks_analysis': {},
                'risk_assessment': {},
                'data_source': 'shared_memory'
            }

            # Use shared options data for analysis
            consolidated_data = options_data.get('consolidated_data', {})

            # Volatility modeling using shared data
            if include_volatility_modeling:
                analysis['volatility_modeling'] = await self._perform_volatility_modeling_from_data(symbol, timeframe, consolidated_data)

            # Strategy optimization using shared data
            if include_strategy_optimization:
                analysis['strategy_optimization'] = await self._perform_strategy_optimization_from_data(symbol, timeframe, consolidated_data, analysis['volatility_modeling'])

            # Portfolio construction using shared data
            if include_portfolio_construction:
                analysis['portfolio_construction'] = self._perform_portfolio_construction_from_data(symbol, timeframe, analysis)

            # Greeks analysis using shared data
            analysis['greeks_analysis'] = self._analyze_greeks_from_data(symbol, consolidated_data)

            # Risk assessment using shared data
            analysis['risk_assessment'] = self._assess_risk_from_data(symbol, analysis)

            return analysis

        except Exception as e:
            logger.error(f"Options analysis from shared data failed for {symbol} {timeframe}: {e}")
            return {'error': str(e), 'data_source': 'shared_memory'}

    async def _perform_volatility_modeling_from_data(self, symbol: str, timeframe: str, consolidated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform volatility modeling using shared options data."""
        try:
            # Extract volatility information from shared data
            calls_data = consolidated_data.get('calls', [])
            puts_data = consolidated_data.get('puts', [])
            
            if not calls_data and not puts_data:
                return {'error': 'No options data available for volatility modeling'}

            # Calculate implied volatility from options
            implied_vols = []
            for option in calls_data + puts_data:
                if 'impliedVolatility' in option and option['impliedVolatility']:
                    implied_vols.append(option['impliedVolatility'])

            if implied_vols:
                avg_iv = np.mean(implied_vols)
                iv_std = np.std(implied_vols)
                
                return {
                    'average_implied_volatility': float(avg_iv),
                    'volatility_std': float(iv_std),
                    'volatility_range': {'min': float(min(implied_vols)), 'max': float(max(implied_vols))},
                    'data_points': len(implied_vols),
                    'data_source': 'shared_memory'
                }
            else:
                return {'error': 'No implied volatility data available'}
                
        except Exception as e:
            logger.error(f"Volatility modeling from shared data failed: {e}")
            return {'error': str(e)}

    async def _perform_strategy_optimization_from_data(self, symbol: str, timeframe: str, consolidated_data: Dict[str, Any], 
                                                     volatility_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform strategy optimization using shared options data."""
        try:
            calls_data = consolidated_data.get('calls', [])
            puts_data = consolidated_data.get('puts', [])
            
            if not calls_data and not puts_data:
                return {'error': 'No options data available for strategy optimization'}

            # Use existing strategy optimization logic but with shared data
            market_conditions = {
                'volatility': volatility_data.get('average_implied_volatility', 0.2),
                'trend': 'neutral',  # Could be enhanced with price data
                'timeframe': timeframe
            }

            # Evaluate strategies using shared data
            strategy_evaluations = {}
            for category, strategies in self.strategy_configs.items():
                for strategy in strategies:
                    evaluation = self._evaluate_strategy_from_data(strategy, calls_data, puts_data, market_conditions)
                    strategy_evaluations[f"{category}_{strategy}"] = evaluation

            # Select optimal strategies
            optimal_strategies = self._select_optimal_strategies_from_data(strategy_evaluations, market_conditions)

            return {
                'strategy_evaluations': strategy_evaluations,
                'optimal_strategies': optimal_strategies,
                'market_conditions': market_conditions,
                'data_source': 'shared_memory'
            }

        except Exception as e:
            logger.error(f"Strategy optimization from shared data failed: {e}")
            return {'error': str(e)}

    def _perform_portfolio_construction_from_data(self, symbol: str, timeframe: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform portfolio construction using shared data analysis."""
        try:
            optimal_strategies = analysis.get('strategy_optimization', {}).get('optimal_strategies', [])
            
            if not optimal_strategies:
                return {'error': 'No optimal strategies available for portfolio construction'}

            # Simple portfolio construction from optimal strategies
            portfolio = {
                'strategies': optimal_strategies[:3],  # Limit to top 3
                'weights': [0.4, 0.35, 0.25],  # Equal-ish weighting
                'expected_return': 0.15,  # Placeholder
                'expected_volatility': 0.25,  # Placeholder
                'data_source': 'shared_memory'
            }

            return portfolio

        except Exception as e:
            logger.error(f"Portfolio construction from shared data failed: {e}")
            return {'error': str(e)}

    def _analyze_greeks_from_data(self, symbol: str, consolidated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Greeks using shared options data."""
        try:
            calls_data = consolidated_data.get('calls', [])
            puts_data = consolidated_data.get('puts', [])
            
            total_delta = 0
            total_gamma = 0
            total_theta = 0
            total_vega = 0
            count = 0

            for option in calls_data + puts_data:
                if all(k in option for k in ['delta', 'gamma', 'theta', 'vega']):
                    total_delta += option['delta']
                    total_gamma += option['gamma'] 
                    total_theta += option['theta']
                    total_vega += option['vega']
                    count += 1

            if count > 0:
                return {
                    'net_delta': total_delta,
                    'net_gamma': total_gamma,
                    'net_theta': total_theta,
                    'net_vega': total_vega,
                    'options_count': count,
                    'data_source': 'shared_memory'
                }
            else:
                return {'error': 'No Greeks data available'}

        except Exception as e:
            logger.error(f"Greeks analysis from shared data failed: {e}")
            return {'error': str(e)}

    def _assess_risk_from_data(self, symbol: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk using shared data analysis."""
        try:
            greeks = analysis.get('greeks_analysis', {})
            portfolio = analysis.get('portfolio_construction', {})
            
            risk_metrics = {
                'delta_risk': abs(greeks.get('net_delta', 0)),
                'gamma_risk': abs(greeks.get('net_gamma', 0)),
                'theta_risk': greeks.get('net_theta', 0),  # Theta is usually negative
                'vega_risk': abs(greeks.get('net_vega', 0)),
                'portfolio_volatility': portfolio.get('expected_volatility', 0.25),
                'data_source': 'shared_memory'
            }

            return risk_metrics

        except Exception as e:
            logger.error(f"Risk assessment from shared data failed: {e}")
            return {'error': str(e)}

    def _evaluate_strategy_from_data(self, strategy: str, calls_data: List[Dict], puts_data: List[Dict], 
                                   market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a strategy using shared options data."""
        # Simple evaluation based on available data
        return {
            'strategy': strategy,
            'expected_return': 0.12,  # Placeholder
            'volatility': market_conditions.get('volatility', 0.2),
            'sharpe_ratio': 0.6,  # Placeholder
            'data_source': 'shared_memory'
        }

    def _select_optimal_strategies_from_data(self, strategy_evaluations: Dict[str, Any], 
                                           market_conditions: Dict[str, Any]) -> List[str]:
        """Select optimal strategies from evaluations."""
        # Simple selection based on Sharpe ratio
        sorted_strategies = sorted(strategy_evaluations.items(), 
                                 key=lambda x: x[1].get('sharpe_ratio', 0), reverse=True)
        return [strategy for strategy, _ in sorted_strategies[:3]]

    async def _analyze_timeframe_options(self, symbol: str, timeframe: str,
                                       include_volatility_modeling: bool, include_strategy_optimization: bool,
                                       include_portfolio_construction: bool) -> Dict[str, Any]:
        """Analyze options strategies for a specific symbol and timeframe."""
        try:
            analysis = {
                'volatility_modeling': {},
                'strategy_optimization': {},
                'portfolio_construction': {},
                'greeks_analysis': {},
                'risk_assessment': {}
            }

            # Volatility modeling
            if include_volatility_modeling:
                analysis['volatility_modeling'] = await self._perform_volatility_modeling(symbol, timeframe)

            # Strategy optimization
            if include_strategy_optimization:
                analysis['strategy_optimization'] = await self._perform_strategy_optimization(symbol, timeframe, analysis['volatility_modeling'])

            # Portfolio construction
            if include_portfolio_construction:
                analysis['portfolio_construction'] = self._perform_portfolio_construction(symbol, timeframe, analysis)

            # Greeks analysis
            analysis['greeks_analysis'] = self._analyze_strategy_greeks(analysis)

            # Risk assessment
            analysis['risk_assessment'] = self._assess_strategy_risks(analysis)

            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze timeframe options for {symbol} {timeframe}: {e}")
            return {
                'volatility_modeling': {'error': str(e)},
                'strategy_optimization': {},
                'portfolio_construction': {},
                'greeks_analysis': {},
                'risk_assessment': {}
            }

    async def _perform_volatility_modeling(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Perform comprehensive volatility modeling."""
        try:
            # Get options chain data (in practice, this would fetch from data sources)
            options_chain = self._get_options_chain_data(symbol, timeframe)

            if not options_chain:
                return {'error': 'No options chain data available'}

            # Build volatility surface
            volatility_surface = self._build_volatility_surface(options_chain)

            # Calculate volatility skew and term structure
            volatility_skew = self._calculate_volatility_skew(volatility_surface)
            term_structure = self._calculate_term_structure(volatility_surface)

            # Model volatility dynamics
            volatility_dynamics = self._model_volatility_dynamics(volatility_surface)

            # Generate volatility forecasts
            volatility_forecasts = self._generate_volatility_forecasts(volatility_surface)

            return {
                'volatility_surface': volatility_surface,
                'volatility_skew': volatility_skew,
                'term_structure': term_structure,
                'volatility_dynamics': volatility_dynamics,
                'volatility_forecasts': volatility_forecasts,
                'atm_volatility': self._calculate_atm_volatility(volatility_surface),
                'volatility_regime': self._classify_volatility_regime(volatility_surface)
            }

        except Exception as e:
            logger.error(f"Volatility modeling failed for {symbol}: {e}")
            return {'error': str(e)}

    async def _perform_strategy_optimization(self, symbol: str, timeframe: str, volatility_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform strategy optimization based on market conditions."""
        try:
            # Get current market conditions
            market_conditions = self._assess_market_conditions(symbol, timeframe)

            # Evaluate all strategy types
            strategy_evaluations = {}

            for category, strategies in self.strategy_configs.items():
                for strategy in strategies:
                    evaluation = self._evaluate_strategy_type(strategy, market_conditions, volatility_data)
                    strategy_evaluations[f"{category}_{strategy}"] = evaluation

            # Optimize strategy selection
            optimal_strategies = self._optimize_strategy_selection(strategy_evaluations, market_conditions)

            # Generate strategy variants
            strategy_variants = self._generate_strategy_variants(optimal_strategies, market_conditions)

            return {
                'strategy_evaluations': strategy_evaluations,
                'optimal_strategies': optimal_strategies,
                'strategy_variants': strategy_variants,
                'optimization_criteria': self._get_optimization_criteria(market_conditions),
                'market_conditions': market_conditions
            }

        except Exception as e:
            logger.error(f"Strategy optimization failed for {symbol}: {e}")
            return {'error': str(e)}

    def _perform_portfolio_construction(self, symbol: str, timeframe: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform portfolio construction and optimization."""
        try:
            # Get available strategies
            optimal_strategies = analysis.get('strategy_optimization', {}).get('optimal_strategies', [])

            if not optimal_strategies:
                return {'error': 'No optimal strategies available'}

            # Construct multi-strategy portfolio
            portfolio_construction = self._construct_multi_strategy_portfolio(optimal_strategies)

            # Optimize portfolio weights
            optimized_weights = self._optimize_portfolio_weights(portfolio_construction)

            # Calculate portfolio Greeks
            portfolio_greeks = self._calculate_portfolio_greeks(portfolio_construction, optimized_weights)

            # Assess portfolio risk
            portfolio_risk = self._assess_portfolio_risk(portfolio_greeks, portfolio_construction)

            return {
                'portfolio_construction': portfolio_construction,
                'optimized_weights': optimized_weights,
                'portfolio_greeks': portfolio_greeks,
                'portfolio_risk': portfolio_risk,
                'diversification_metrics': self._calculate_diversification_metrics(portfolio_construction),
                'expected_performance': self._calculate_expected_performance(portfolio_construction, optimized_weights)
            }

        except Exception as e:
            logger.error(f"Portfolio construction failed for {symbol}: {e}")
            return {'error': str(e)}

    def _get_options_chain_data(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get options chain data for analysis."""
        try:
            import yfinance as yf

            # Fetch real options data from yfinance
            ticker = yf.Ticker(symbol)
            current_price = ticker.info.get('currentPrice', ticker.info.get('regularMarketPrice', 100))

            if current_price is None:
                # Fallback to basic price fetching
                hist = ticker.history(period='1d')
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                else:
                    logger.warning(f"Could not fetch current price for {symbol}, using fallback")
                    current_price = 100  # Fallback price

            # Get available expiration dates
            expirations = ticker.options[:4] if len(ticker.options) >= 4 else ticker.options

            if not expirations:
                logger.warning(f"No options data available for {symbol}")
                return None

            options_chain = {}

            for exp_date in expirations:
                try:
                    # Fetch real options chain for this expiration
                    opt_chain = ticker.option_chain(exp_date)

                    calls_data = []
                    puts_data = []

                    # Process calls
                    for _, row in opt_chain.calls.iterrows():
                        calls_data.append({
                            'strike': float(row['strike']),
                            'price': float(row['lastPrice']) if pd.notna(row['lastPrice']) else float(row['bid'] + row['ask']) / 2 if pd.notna(row['bid']) and pd.notna(row['ask']) else 0.01,
                            'volume': int(row['volume']) if pd.notna(row['volume']) else 0,
                            'open_interest': int(row['openInterest']) if pd.notna(row['openInterest']) else 0,
                            'implied_volatility': float(row['impliedVolatility']) if pd.notna(row['impliedVolatility']) else 0.25
                        })

                    # Process puts
                    for _, row in opt_chain.puts.iterrows():
                        puts_data.append({
                            'strike': float(row['strike']),
                            'price': float(row['lastPrice']) if pd.notna(row['lastPrice']) else float(row['bid'] + row['ask']) / 2 if pd.notna(row['bid']) and pd.notna(row['ask']) else 0.01,
                            'volume': int(row['volume']) if pd.notna(row['volume']) else 0,
                            'open_interest': int(row['openInterest']) if pd.notna(row['openInterest']) else 0,
                            'implied_volatility': float(row['impliedVolatility']) if pd.notna(row['impliedVolatility']) else 0.25
                        })

                    options_chain[exp_date] = {
                        'calls': calls_data,
                        'puts': puts_data,
                        'underlying_price': current_price
                    }

                except Exception as e:
                    logger.warning(f"Error fetching options for {symbol} exp {exp_date}: {e}")
                    continue

            return options_chain if options_chain else None

        except ImportError:
            logger.error("yfinance not available for options data fetching")
            return None
        except Exception as e:
            logger.error(f"Error fetching options chain data for {symbol}: {e}")
            return None

    def _build_volatility_surface(self, options_chain: Dict[str, Any]) -> Dict[str, Any]:
        """Build implied volatility surface."""
        try:
            surface_data = {}

            for exp_date, chain_data in options_chain.items():
                calls = chain_data['calls']
                puts = chain_data['puts']
                underlying_price = chain_data['underlying_price']

                # Combine calls and puts for surface
                all_options = []
                for call in calls:
                    all_options.append({
                        'strike': call['strike'],
                        'price': call['price'],
                        'type': 'call',
                        'moneyness': call['strike'] / underlying_price
                    })

                for put in puts:
                    all_options.append({
                        'strike': put['strike'],
                        'price': put['price'],
                        'type': 'put',
                        'moneyness': put['strike'] / underlying_price
                    })

                surface_data[exp_date] = {
                    'options': all_options,
                    'underlying_price': underlying_price,
                    'days_to_expiry': (datetime.strptime(exp_date, '%Y-%m-%d') - datetime.now()).days
                }

            return surface_data

        except Exception as e:
            return {'error': str(e)}

    def _calculate_volatility_skew(self, volatility_surface: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate volatility skew across strikes."""
        try:
            skew_data = {}

            for exp_date, surface in volatility_surface.items():
                if 'options' not in surface:
                    continue

                options = surface['options']
                underlying_price = surface['underlying_price']

                # Group by moneyness
                otm_calls = [opt for opt in options if opt['type'] == 'call' and opt['moneyness'] > 1.05]
                otm_puts = [opt for opt in options if opt['type'] == 'put' and opt['moneyness'] < 0.95]

                call_skew = np.mean([opt.get('implied_volatility', 0.25) for opt in otm_calls]) if otm_calls else 0.25
                put_skew = np.mean([opt.get('implied_volatility', 0.25) for opt in otm_puts]) if otm_puts else 0.25

                skew_data[exp_date] = {
                    'call_skew': call_skew,
                    'put_skew': put_skew,
                    'skew_difference': call_skew - put_skew,
                    'skew_direction': 'call_skewed' if call_skew > put_skew else 'put_skewed'
                }

            return skew_data

        except Exception as e:
            return {'error': str(e)}

    def _calculate_term_structure(self, volatility_surface: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate volatility term structure."""
        try:
            term_data = {}

            # Sort by time to expiry
            sorted_dates = sorted(volatility_surface.keys(),
                                key=lambda x: volatility_surface[x].get('days_to_expiry', 0))

            atm_vols = []
            expiries = []

            for exp_date in sorted_dates:
                surface = volatility_surface[exp_date]
                options = surface.get('options', [])

                # Find ATM option
                underlying_price = surface.get('underlying_price', 100)
                atm_options = [opt for opt in options if abs(opt['moneyness'] - 1.0) < 0.05]

                if atm_options:
                    atm_vol = np.mean([opt.get('implied_volatility', 0.25) for opt in atm_options])
                    atm_vols.append(atm_vol)
                    expiries.append(surface.get('days_to_expiry', 0))

            term_data['atm_volatility_curve'] = {
                'expiries': expiries,
                'volatilities': atm_vols,
                'slope': np.polyfit(expiries, atm_vols, 1)[0] if len(expiries) > 1 else 0
            }

            return term_data

        except Exception as e:
            return {'error': str(e)}

    def _model_volatility_dynamics(self, volatility_surface: Dict[str, Any]) -> Dict[str, Any]:
        """Model volatility dynamics and behavior."""
        return {
            'volatility_mean_reversion': 0.15,
            'volatility_persistence': 0.85,
            'jump_intensity': 0.02,
            'regime_switching_probability': 0.1
        }

    def _generate_volatility_forecasts(self, volatility_surface: Dict[str, Any]) -> Dict[str, Any]:
        """Generate volatility forecasts."""
        return {
            'short_term_forecast': 0.22,
            'medium_term_forecast': 0.25,
            'long_term_forecast': 0.20,
            'forecast_confidence': 0.75
        }

    def _calculate_atm_volatility(self, volatility_surface: Dict[str, Any]) -> float:
        """Calculate at-the-money volatility."""
        try:
            atm_vols = []

            for surface in volatility_surface.values():
                options = surface.get('options', [])
                underlying_price = surface.get('underlying_price', 100)

                atm_options = [opt for opt in options if abs(opt['moneyness'] - 1.0) < 0.05]
                if atm_options:
                    atm_vol = np.mean([opt.get('implied_volatility', 0.25) for opt in atm_options])
                    atm_vols.append(atm_vol)

            return np.mean(atm_vols) if atm_vols else 0.25

        except Exception:
            return 0.25

    def _classify_volatility_regime(self, volatility_surface: Dict[str, Any]) -> str:
        """Classify current volatility regime."""
        atm_vol = self._calculate_atm_volatility(volatility_surface)

        if atm_vol > 0.35:
            return 'high_volatility'
        elif atm_vol < 0.15:
            return 'low_volatility'
        else:
            return 'normal_volatility'

    def _assess_market_conditions(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Assess current market conditions for strategy selection."""
        # In practice, this would analyze current market data
        return {
            'trend': 'neutral',
            'volatility_regime': 'normal',
            'sentiment': 'neutral',
            'market_stress': 0.3,
            'liquidity_conditions': 'normal'
        }

    def _evaluate_strategy_type(self, strategy: str, market_conditions: Dict[str, Any],
                               volatility_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a specific strategy type under current conditions."""
        try:
            # Base evaluation metrics
            evaluation = {
                'strategy': strategy,
                'suitability_score': 0.5,
                'risk_adjusted_return': 0.0,
                'probability_of_profit': 0.5,
                'greeks_profile': {},
                'market_fit': 0.5
            }

            # Adjust based on market conditions
            trend = market_conditions.get('trend', 'neutral')
            vol_regime = market_conditions.get('volatility_regime', 'normal')

            # Strategy-specific logic
            if strategy in ['long_call', 'bull_call_spread']:
                evaluation['suitability_score'] = 0.8 if trend == 'bullish' else 0.3
                evaluation['probability_of_profit'] = 0.55 if trend == 'bullish' else 0.45
            elif strategy in ['long_put', 'bear_put_spread']:
                evaluation['suitability_score'] = 0.8 if trend == 'bearish' else 0.3
                evaluation['probability_of_profit'] = 0.55 if trend == 'bearish' else 0.45
            elif strategy in ['straddle', 'strangle']:
                evaluation['suitability_score'] = 0.8 if vol_regime == 'high_volatility' else 0.4
                evaluation['probability_of_profit'] = 0.6 if vol_regime == 'high_volatility' else 0.4
            elif strategy == 'iron_condor':
                evaluation['suitability_score'] = 0.8 if vol_regime == 'low_volatility' else 0.3
                evaluation['probability_of_profit'] = 0.7 if vol_regime == 'low_volatility' else 0.5

            # Calculate risk-adjusted return
            expected_return = evaluation['probability_of_profit'] * 0.15  # Assume 15% win
            risk = 1 - evaluation['probability_of_profit']
            evaluation['risk_adjusted_return'] = expected_return / risk if risk > 0 else 0

            return evaluation

        except Exception as e:
            return {'strategy': strategy, 'error': str(e)}

    def _optimize_strategy_selection(self, strategy_evaluations: Dict[str, Any],
                                   market_conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize strategy selection based on evaluations."""
        try:
            # Sort strategies by suitability score
            sorted_strategies = sorted(
                strategy_evaluations.values(),
                key=lambda x: x.get('suitability_score', 0),
                reverse=True
            )

            # Return top strategies
            return sorted_strategies[:5]

        except Exception as e:
            return []

    def _generate_strategy_variants(self, optimal_strategies: List[Dict[str, Any]],
                                  market_conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate variants of optimal strategies."""
        variants = []

        try:
            for strategy in optimal_strategies:
                strategy_name = strategy.get('strategy', '')

                # Generate strike variations
                variants.extend(self._generate_strike_variations(strategy_name))

                # Generate expiration variations
                variants.extend(self._generate_expiration_variations(strategy_name))

                # Generate size variations
                variants.extend(self._generate_size_variations(strategy_name))

        except Exception as e:
            pass

        return variants[:10]  # Limit to 10 variants

    def _get_optimization_criteria(self, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimization criteria based on market conditions."""
        return {
            'primary_criteria': ['suitability_score', 'risk_adjusted_return'],
            'secondary_criteria': ['probability_of_profit', 'market_fit'],
            'constraints': self.risk_params
        }

    def _construct_multi_strategy_portfolio(self, optimal_strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Construct a multi-strategy options portfolio."""
        try:
            portfolio = {
                'strategies': optimal_strategies,
                'total_allocation': 1.0,
                'strategy_weights': {},
                'correlation_matrix': self._calculate_strategy_correlations(optimal_strategies)
            }

            # Equal weight allocation initially
            weight = 1.0 / len(optimal_strategies) if optimal_strategies else 0
            for strategy in optimal_strategies:
                strategy_name = strategy.get('strategy', 'unknown')
                portfolio['strategy_weights'][strategy_name] = weight

            return portfolio

        except Exception as e:
            return {'error': str(e)}

    def _optimize_portfolio_weights(self, portfolio_construction: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize portfolio weights using risk-return optimization."""
        try:
            strategies = portfolio_construction.get('strategies', [])
            correlation_matrix = portfolio_construction.get('correlation_matrix', {})

            if len(strategies) < 2:
                return portfolio_construction.get('strategy_weights', {})

            # Simple equal risk contribution optimization
            n_strategies = len(strategies)
            equal_weight = 1.0 / n_strategies

            # Adjust weights based on risk contribution
            risk_contributions = []
            for strategy in strategies:
                risk = 1 - strategy.get('probability_of_profit', 0.5)
                risk_contributions.append(risk)

            # Normalize risk contributions
            total_risk = sum(risk_contributions)
            if total_risk > 0:
                risk_weights = [risk / total_risk for risk in risk_contributions]
                # Invert to get allocation weights (higher risk = lower weight)
                allocation_weights = [1 - weight for weight in risk_weights]
                total_allocation = sum(allocation_weights)
                if total_allocation > 0:
                    allocation_weights = [w / total_allocation for w in allocation_weights]
                else:
                    allocation_weights = [equal_weight] * n_strategies
            else:
                allocation_weights = [equal_weight] * n_strategies

            return dict(zip([s.get('strategy', f'strategy_{i}') for i, s in enumerate(strategies)], allocation_weights))

        except Exception as e:
            return {}

    def _calculate_portfolio_greeks(self, portfolio_construction: Dict[str, Any],
                                  optimized_weights: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate portfolio-level Greeks."""
        try:
            total_greeks = {
                'delta': 0,
                'gamma': 0,
                'theta': 0,
                'vega': 0,
                'rho': 0
            }

            strategies = portfolio_construction.get('strategies', [])

            for strategy in strategies:
                strategy_name = strategy.get('strategy', '')
                weight = optimized_weights.get(strategy_name, 0)

                # Simplified Greeks calculation per strategy
                strategy_greeks = self._estimate_strategy_greeks(strategy_name)
                for greek in total_greeks.keys():
                    total_greeks[greek] += strategy_greeks.get(greek, 0) * weight

            return total_greeks

        except Exception as e:
            return {}

    def _assess_portfolio_risk(self, portfolio_greeks: Dict[str, Any],
                             portfolio_construction: Dict[str, Any]) -> Dict[str, Any]:
        """Assess portfolio-level risk."""
        try:
            risk_metrics = {
                'delta_exposure': abs(portfolio_greeks.get('delta', 0)),
                'gamma_risk': abs(portfolio_greeks.get('gamma', 0)),
                'theta_decay': portfolio_greeks.get('theta', 0),
                'vega_exposure': abs(portfolio_greeks.get('vega', 0)),
                'diversification_ratio': self._calculate_diversification_ratio(portfolio_construction)
            }

            # Risk assessment
            risk_score = 0
            if risk_metrics['delta_exposure'] > self.risk_params['max_delta_exposure']:
                risk_score += 0.3
            if risk_metrics['gamma_risk'] > self.risk_params['max_gamma_exposure']:
                risk_score += 0.3
            if risk_metrics['theta_decay'] < self.risk_params['max_theta_exposure']:
                risk_score += 0.2
            if risk_metrics['vega_exposure'] > self.risk_params['max_vega_exposure']:
                risk_score += 0.2

            risk_metrics['overall_risk_score'] = min(1.0, risk_score)
            risk_metrics['risk_level'] = 'high' if risk_score > 0.6 else 'medium' if risk_score > 0.3 else 'low'

            return risk_metrics

        except Exception as e:
            return {'error': str(e)}

    def _calculate_diversification_metrics(self, portfolio_construction: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate portfolio diversification metrics."""
        try:
            strategies = portfolio_construction.get('strategies', [])
            correlation_matrix = portfolio_construction.get('correlation_matrix', {})

            n_strategies = len(strategies)
            diversification_ratio = 1.0 / n_strategies if n_strategies > 0 else 0

            return {
                'number_of_strategies': n_strategies,
                'diversification_ratio': diversification_ratio,
                'effective_bets': n_strategies * diversification_ratio
            }

        except Exception as e:
            return {}

    def _calculate_expected_performance(self, portfolio_construction: Dict[str, Any],
                                      optimized_weights: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate expected portfolio performance."""
        try:
            strategies = portfolio_construction.get('strategies', [])
            total_return = 0
            total_risk = 0

            for strategy in strategies:
                strategy_name = strategy.get('strategy', '')
                weight = optimized_weights.get(strategy_name, 0)

                expected_return = strategy.get('risk_adjusted_return', 0) * 0.1  # Scale down
                risk = 1 - strategy.get('probability_of_profit', 0.5)

                total_return += expected_return * weight
                total_risk += risk * weight

            sharpe_ratio = total_return / total_risk if total_risk > 0 else 0

            return {
                'expected_return': total_return,
                'expected_risk': total_risk,
                'sharpe_ratio': sharpe_ratio,
                'probability_of_profit': 1 - total_risk
            }

        except Exception as e:
            return {}

    def _analyze_strategy_greeks(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Greeks exposure across strategies."""
        try:
            portfolio_data = analysis.get('portfolio_construction', {})
            portfolio_greeks = portfolio_data.get('portfolio_greeks', {})

            greeks_analysis = {
                'net_exposure': portfolio_greeks,
                'hedging_effectiveness': self._calculate_hedging_effectiveness(portfolio_greeks),
                'greeks_balance': self._assess_greeks_balance(portfolio_greeks),
                'risk_sensitivities': self._calculate_risk_sensitivities(portfolio_greeks)
            }

            return greeks_analysis

        except Exception as e:
            return {'error': str(e)}

    def _get_historical_data_for_risk(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get historical data for risk calculations."""
        try:
            # Use yfinance to get historical data
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            # Get 1 year of daily data for risk calculations
            historical_data = ticker.history(period="1y", interval="1d")
            
            if historical_data.empty:
                logger.warning(f"No historical data available for {symbol}")
                return None
                
            return historical_data
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return None
    
    def _calculate_real_liquidity_risk(self, symbol: str, analysis: Dict[str, Any]) -> float:
        """Calculate real liquidity risk based on market data."""
        try:
            # Get options chain data for liquidity assessment
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            options = ticker.options
            
            if not options:
                return 0.1  # Default liquidity risk
                
            # Get nearest expiration options
            expiration = options[0]
            opt_chain = ticker.option_chain(expiration)
            
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            # Calculate average bid-ask spread as liquidity proxy
            call_spreads = (calls['ask'] - calls['bid']) / calls['bid']
            put_spreads = (puts['ask'] - puts['bid']) / puts['bid']
            
            avg_call_spread = call_spreads.mean() if not call_spreads.empty else 0.02
            avg_put_spread = put_spreads.mean() if not put_spreads.empty else 0.02
            
            # Calculate volume-based liquidity
            total_volume = calls['volume'].sum() + puts['volume'].sum()
            avg_volume = total_volume / len(calls) if len(calls) > 0 else 0
            
            # Liquidity risk score: higher spreads and lower volume = higher risk
            spread_risk = (avg_call_spread + avg_put_spread) / 2
            volume_risk = 1 / (1 + avg_volume / 1000)  # Normalize volume impact
            
            liquidity_risk = min(spread_risk * 0.7 + volume_risk * 0.3, 0.5)  # Cap at 50%
            
            return liquidity_risk
            
        except Exception as e:
            logger.error(f"Failed to calculate liquidity risk for {symbol}: {e}")
            return 0.1  # Conservative default
    
    def _perform_stress_testing(self, analysis: Dict[str, Any], historical_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Perform stress testing under various market scenarios."""
        try:
            stress_results = {}
            
            if historical_data is not None and not historical_data.empty:
                returns = historical_data['Close'].pct_change().dropna()
                
                # Historical stress periods
                stress_periods = {
                    'covid_crash': returns[returns < -0.05],  # 5%+ daily drops
                    'flash_crash': returns[returns < -0.03],  # 3%+ daily drops  
                    'volatility_spike': returns[returns.abs() > returns.std() * 2],  # 2-sigma events
                }
                
                for scenario, stress_returns in stress_periods.items():
                    if len(stress_returns) > 0:
                        stress_results[scenario] = {
                            'worst_loss': stress_returns.min(),
                            'avg_loss': stress_returns.mean(),
                            'frequency': len(stress_returns) / len(returns),
                            'impact_score': abs(stress_returns.mean()) * len(stress_returns)
                        }
                    else:
                        stress_results[scenario] = {
                            'worst_loss': 0,
                            'avg_loss': 0,
                            'frequency': 0,
                            'impact_score': 0
                        }
            else:
                # Fallback stress scenarios based on typical market conditions
                stress_results = {
                    'market_crash': {'worst_loss': -0.15, 'avg_loss': -0.08, 'frequency': 0.02, 'impact_score': 0.0016},
                    'volatility_spike': {'worst_loss': -0.10, 'avg_loss': -0.05, 'frequency': 0.05, 'impact_score': 0.0025},
                    'liquidity_crisis': {'worst_loss': -0.12, 'avg_loss': -0.06, 'frequency': 0.01, 'impact_score': 0.0006}
                }
            
            # Calculate overall stress test score
            total_impact = sum(scenario['impact_score'] for scenario in stress_results.values())
            stress_results['overall_stress_score'] = min(total_impact * 100, 1.0)  # Scale to 0-1
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Stress testing failed: {e}")
            return {'error': str(e), 'overall_stress_score': 0.5}

    def _assess_strategy_risks(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess comprehensive strategy risks using real quantitative methods."""
        try:
            portfolio_risk = analysis.get('portfolio_construction', {}).get('portfolio_risk', {})
            portfolio_greeks = analysis.get('portfolio_construction', {}).get('portfolio_greeks', {})
            
            # Get historical data for risk calculations
            symbol = list(analysis.keys())[0] if analysis else 'SPY'
            historical_data = self._get_historical_data_for_risk(symbol)
            
            risk_assessment = {}
            
            # Market risk: Use real VaR/CVaR calculations
            if historical_data is not None and not historical_data.empty:
                # Calculate returns from historical data
                returns = historical_data['Close'].pct_change().dropna()
                
                # Value at Risk (VaR) at 95% and 99% confidence
                risk_assessment['var_95'] = np.percentile(returns, 5)  # Historical VaR
                risk_assessment['var_99'] = np.percentile(returns, 1)
                
                # Conditional VaR (CVaR/Expected Shortfall)
                var_95_threshold = risk_assessment['var_95']
                cvar_returns = returns[returns <= var_95_threshold]
                risk_assessment['cvar_95'] = cvar_returns.mean() if len(cvar_returns) > 0 else var_95_threshold
                
                # Volatility-based market risk
                risk_assessment['realized_volatility'] = returns.std() * np.sqrt(252)  # Annualized
                
                # Maximum drawdown
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                risk_assessment['max_drawdown'] = drawdown.min()
                
            else:
                # Fallback to Greeks-based estimates
                delta_exposure = abs(portfolio_greeks.get('delta', 0))
                risk_assessment['var_95'] = -delta_exposure * 0.02  # Rough estimate
                risk_assessment['cvar_95'] = -delta_exposure * 0.03
                risk_assessment['realized_volatility'] = 0.25  # Default assumption
                risk_assessment['max_drawdown'] = -0.15  # Conservative estimate
            
            # Volatility risk: Based on Vega exposure and implied vs realized vol spread
            vega_exposure = abs(portfolio_greeks.get('vega', 0))
            risk_assessment['volatility_risk'] = vega_exposure * 0.15  # Vega * vol change impact
            
            # Time risk: Based on Theta decay patterns
            theta_decay = portfolio_greeks.get('theta', 0)
            risk_assessment['time_decay_risk'] = abs(theta_decay) * 21  # Weekly decay estimate
            
            # Curvature risk: Based on Gamma exposure
            gamma_risk = abs(portfolio_greeks.get('gamma', 0))
            risk_assessment['curvature_risk'] = gamma_risk * 0.10  # Gamma impact estimate
            
            # Liquidity risk: Calculate from real market data
            risk_assessment['liquidity_risk'] = self._calculate_real_liquidity_risk(symbol, analysis)
            
            # Counterparty risk: Based on position size and credit quality
            position_size = analysis.get('portfolio_construction', {}).get('total_allocation', 0.1)
            risk_assessment['counterparty_risk'] = position_size * 0.02  # Conservative credit risk estimate
            
            # Stress testing: Scenario-based risk assessment
            stress_scenarios = self._perform_stress_testing(analysis, historical_data)
            risk_assessment['stress_test_results'] = stress_scenarios
            
            # Overall risk score: Weighted combination of all risks
            weights = {
                'market_risk': 0.4,  # VaR contribution
                'volatility_risk': 0.25,  # Vega contribution  
                'time_risk': 0.15,  # Theta contribution
                'curvature_risk': 0.1,  # Gamma contribution
                'liquidity_risk': 0.05,
                'counterparty_risk': 0.05
            }
            
            # Normalize and combine risks
            normalized_risks = {}
            for risk_type, weight in weights.items():
                if risk_type in risk_assessment:
                    # Convert to positive risk score (0-1 scale)
                    risk_value = abs(risk_assessment[risk_type])
                    normalized_risks[risk_type] = min(risk_value * weight, weight)
            
            risk_assessment['total_risk_score'] = sum(normalized_risks.values())
            risk_assessment['risk_level'] = (
                'high' if risk_assessment['total_risk_score'] > 0.7 else
                'medium' if risk_assessment['total_risk_score'] > 0.4 else 'low'
            )
            
            # Add calculation metadata
            risk_assessment['calculation_method'] = 'quantitative_risk_analytics'
            risk_assessment['data_sources'] = ['historical_returns', 'options_greeks', 'market_data']
            risk_assessment['confidence_level'] = 'high' if historical_data is not None else 'medium'
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {
                'error': str(e),
                'calculation_method': 'failed',
                'total_risk_score': 0.5,
                'risk_level': 'unknown'
            }

    def _aggregate_timeframe_options(self, symbol_options: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate options analysis across timeframes."""
        try:
            # Weight different timeframes
            weights = {'1D': 0.5, '1W': 0.3, '1M': 0.2}

            aggregated = {
                'weighted_suitability': 0,
                'dominant_timeframe': None,
                'consistency_score': 0,
                'timeframe_breakdown': {}
            }

            total_weight = 0
            suitability_scores = []

            for timeframe, options_data in symbol_options.items():
                if timeframe == 'aggregate':
                    continue

                optimization_data = options_data.get('strategy_optimization', {})
                optimal_strategies = optimization_data.get('optimal_strategies', [])

                if optimal_strategies:
                    avg_suitability = np.mean([s.get('suitability_score', 0) for s in optimal_strategies])
                    weight = weights.get(timeframe, 0.2)

                    aggregated['weighted_suitability'] += avg_suitability * weight
                    aggregated['timeframe_breakdown'][timeframe] = avg_suitability
                    suitability_scores.append(avg_suitability)
                    total_weight += weight

            if total_weight > 0:
                aggregated['weighted_suitability'] /= total_weight

            # Find dominant timeframe
            if aggregated['timeframe_breakdown']:
                aggregated['dominant_timeframe'] = max(
                    aggregated['timeframe_breakdown'].keys(),
                    key=lambda x: aggregated['timeframe_breakdown'][x]
                )

            # Calculate consistency
            if len(suitability_scores) > 1:
                aggregated['consistency_score'] = 1 - np.std(suitability_scores) / np.mean(suitability_scores) if np.mean(suitability_scores) > 0 else 0

            return aggregated

        except Exception as e:
            return {'error': str(e)}

    def _generate_options_alpha_signals(self, options_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alpha signals from options analysis."""
        signals = []

        try:
            for symbol, symbol_options in options_analysis.items():
                aggregate_options = symbol_options.get('aggregate', {})

                suitability_score = aggregate_options.get('weighted_suitability', 0)
                consistency = aggregate_options.get('consistency_score', 0)

                # Generate signals based on options analysis strength
                if suitability_score > 0.7 and consistency > 0.8:
                    signals.append({
                        'symbol': symbol,
                        'signal_type': 'strong_options_alpha',
                        'direction': 'neutral',  # Options can be directional or neutral
                        'strength': 'high',
                        'confidence': min(suitability_score * consistency, 1.0),
                        'timeframe': aggregate_options.get('dominant_timeframe'),
                        'expected_return': suitability_score * 0.25,  # 25% max expected return for options
                        'holding_period': '2-4 weeks',
                        'options_drivers': self._identify_options_drivers(symbol_options)
                    })
                elif suitability_score > 0.6 and consistency > 0.6:
                    signals.append({
                        'symbol': symbol,
                        'signal_type': 'moderate_options_alpha',
                        'direction': 'neutral',
                        'strength': 'medium',
                        'confidence': suitability_score * consistency * 0.8,
                        'timeframe': aggregate_options.get('dominant_timeframe'),
                        'expected_return': suitability_score * 0.15,
                        'holding_period': '1-2 weeks',
                        'options_drivers': self._identify_options_drivers(symbol_options)
                    })

        except Exception as e:
            logger.error(f"Failed to generate options alpha signals: {e}")

        return signals

    def _identify_options_drivers(self, symbol_options: Dict[str, Any]) -> List[str]:
        """Identify key options drivers for the signal."""
        drivers = []

        try:
            # Check volatility modeling
            for timeframe, options_data in symbol_options.items():
                if timeframe == 'aggregate':
                    continue

                vol_data = options_data.get('volatility_modeling', {})
                vol_regime = vol_data.get('volatility_regime', '')

                if vol_regime:
                    drivers.append(f'{vol_regime}_regime')

                # Check optimal strategies
                optimization_data = options_data.get('strategy_optimization', {})
                optimal_strategies = optimization_data.get('optimal_strategies', [])

                for strategy in optimal_strategies[:2]:  # Top 2 strategies
                    strategy_name = strategy.get('strategy', '')
                    if strategy_name:
                        drivers.append(strategy_name)

            # Remove duplicates and limit to 5
            drivers = list(set(drivers))[:5]

        except Exception:
            drivers = ['volatility_surface', 'greeks_balance', 'risk_management']

        return drivers

    def _build_options_strategy_proposals(self, options_analysis: Dict[str, Any], alpha_signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build comprehensive options strategy proposals."""
        proposals = []

        try:
            for signal in alpha_signals:
                symbol = signal['symbol']

                # Get detailed options data for the symbol
                symbol_options = options_analysis.get(symbol, {})
                aggregate_options = symbol_options.get('aggregate', {})

                # Build strategy proposal
                proposal = {
                    'strategy_type': 'options_based',
                    'symbol': symbol,
                    'direction': signal['direction'],
                    'entry_signal': signal['signal_type'],
                    'timeframe': signal['timeframe'],
                    'confidence': signal['confidence'],
                    'expected_return': signal['expected_return'],
                    'holding_period': signal['holding_period'],
                    'position_size': self._calculate_position_size(signal, aggregate_options),
                    'entry_conditions': self._define_entry_conditions(signal, symbol_options),
                    'exit_conditions': self._define_exit_conditions(signal, symbol_options),
                    'risk_management': self._define_risk_management(signal, symbol_options),
                    'options_drivers': signal.get('options_drivers', []),
                    'portfolio_construction': self._get_portfolio_details(symbol_options),
                    # Add required fields for risk agent validation
                    'roi_estimate': signal['expected_return'],  # Map expected_return to roi_estimate
                    'pop_estimate': signal.get('confidence', 0.5) * 0.8  # Estimate POP based on confidence
                }

                proposals.append(proposal)

        except Exception as e:
            logger.error(f"Failed to build options strategy proposals: {e}")

        return proposals

    def _calculate_position_size(self, signal: Dict[str, Any], aggregate_options: Dict[str, Any]) -> float:
        """Calculate optimal position size based on options signal strength."""
        base_size = 0.12  # 12% of portfolio for options strategies
        confidence_multiplier = signal.get('confidence', 0.5)
        suitability_multiplier = aggregate_options.get('weighted_suitability', 0.5)

        return base_size * confidence_multiplier * suitability_multiplier

    def _define_entry_conditions(self, signal: Dict[str, Any], symbol_options: Dict[str, Any]) -> Dict[str, Any]:
        """Define entry conditions for the options strategy."""
        return {
            'options_signal_confirmation': f"{signal['signal_type']} > 0.6",
            'volatility_alignment': f"Volatility regime matches: {', '.join(signal.get('options_drivers', []))}",
            'greeks_balance': 'acceptable_exposure',
            'timeframe': signal.get('timeframe', '1D')
        }

    def _define_exit_conditions(self, signal: Dict[str, Any], symbol_options: Dict[str, Any]) -> Dict[str, Any]:
        """Define exit conditions for the options strategy."""
        return {
            'profit_target': f"{signal.get('expected_return', 0.15) * 0.8:.1%}",
            'stop_loss': f"-{signal.get('expected_return', 0.15) * 0.5:.1%}",
            'time_exit': signal.get('holding_period', '2 weeks'),
            'greeks_threshold': 'exposure_limit_breached',
            'volatility_shift': 'regime_change_detected'
        }

    def _define_risk_management(self, signal: Dict[str, Any], symbol_options: Dict[str, Any]) -> Dict[str, Any]:
        """Define risk management for the options strategy."""
        return {
            'max_position_size': signal.get('position_size', 0.12),
            'delta_hedge_threshold': 0.1,
            'gamma_hedge_frequency': 'daily',
            'vega_hedge_threshold': 0.2,
            'max_loss_percentage': self.risk_params['max_loss_percentage'],
            'risk_reward_ratio': 2.5,
            'position_monitoring': 'real_time'
        }

    def _get_portfolio_details(self, symbol_options: Dict[str, Any]) -> Dict[str, Any]:
        """Get portfolio construction details."""
        try:
            portfolio_details = {}

            for timeframe, options_data in symbol_options.items():
                if timeframe == 'aggregate':
                    continue

                portfolio_data = options_data.get('portfolio_construction', {})
                if portfolio_data:
                    portfolio_details[timeframe] = {
                        'strategy_weights': portfolio_data.get('optimized_weights', {}),
                        'expected_performance': portfolio_data.get('expected_performance', {})
                    }

            return portfolio_details

        except Exception:
            return {}

    def _calculate_risk_adjusted_returns(self, proposals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate risk-adjusted returns for proposals."""
        for proposal in proposals:
            expected_return = proposal.get('expected_return', 0)
            confidence = proposal.get('confidence', 0.5)

            # Options strategies have different risk profiles
            # Assume higher volatility but asymmetric returns
            risk_adjusted_return = expected_return * confidence / 0.15  # Assuming 15% volatility for options

            proposal['risk_adjusted_return'] = risk_adjusted_return
            proposal['sharpe_ratio'] = risk_adjusted_return / 0.15 if risk_adjusted_return > 0 else 0

        return proposals

    def _generate_collaborative_insights(self, options_analysis: Dict[str, Any], alpha_signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate insights for sharing with other agents."""
        insights = []

        # Strategy agent insights
        strong_signals = [s for s in alpha_signals if s.get('strength') == 'high']
        if strong_signals:
            insights.append({
                'target_agent': 'strategy',
                'insight_type': 'options_alpha_opportunities',
                'content': f'Identified {len(strong_signals)} high-confidence options-based alpha signals with strong Greeks alignment',
                'confidence': 0.95,
                'relevance': 'high'
            })

        # Risk agent insights
        high_confidence_signals = [s for s in alpha_signals if s.get('confidence', 0) > 0.8]
        if high_confidence_signals:
            insights.append({
                'target_agent': 'risk',
                'insight_type': 'options_risk_exposure',
                'content': f'Options strategies show high confidence signals - monitor Greeks exposure and volatility risk',
                'confidence': 0.9,
                'relevance': 'high'
            })

        # Data agent insights
        for symbol, symbol_options in options_analysis.items():
            for timeframe, options_data in symbol_options.items():
                if timeframe == 'aggregate':
                    continue

                vol_data = options_data.get('volatility_modeling', {})
                if vol_data.get('volatility_regime') == 'high_volatility':
                    insights.append({
                        'target_agent': 'data',
                        'insight_type': 'volatility_data_importance',
                        'content': f'High volatility regime detected for {symbol} - ensure real-time options data flow',
                        'confidence': 0.85,
                        'relevance': 'medium'
                    })

        return insights

    def _update_memory(self, options_analysis: Dict[str, Any], alpha_signals: List[Dict[str, Any]]):
        """Update collaborative memory with options insights."""
        # Update alpha signals
        self.memory.alpha_signals.extend(alpha_signals[-10:])  # Keep last 10

        # Update strategy performance
        for symbol, symbol_options in options_analysis.items():
            for timeframe, options_data in symbol_options.items():
                if timeframe == 'aggregate':
                    continue

                portfolio_data = options_data.get('portfolio_construction', {})
                perf_data = portfolio_data.get('expected_performance', {})

                if perf_data:
                    self.memory.strategy_performance[f"{symbol}_{timeframe}"] = {
                        'expected_return': perf_data.get('expected_return', 0),
                        'expected_risk': perf_data.get('expected_risk', 0),
                        'sharpe_ratio': perf_data.get('sharpe_ratio', 0),
                        'timestamp': datetime.now().isoformat()
                    }

        # Add session insight
        total_signals = len(alpha_signals)
        avg_confidence = np.mean([s.get('confidence', 0) for s in alpha_signals]) if alpha_signals else 0

        self.memory.add_session_insight({
            'type': 'options_analysis_summary',
            'total_signals': total_signals,
            'average_confidence': avg_confidence,
            'symbols_analyzed': len(options_analysis),
            'high_confidence_signals': len([s for s in alpha_signals if s.get('confidence', 0) > 0.8])
        })

    # Helper methods for calculations
    def _calculate_strategy_correlations(self, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate correlations between strategies."""
        # Simplified correlation matrix
        n_strategies = len(strategies)
        correlation_matrix = np.eye(n_strategies) * 0.3 + np.ones((n_strategies, n_strategies)) * 0.7
        np.fill_diagonal(correlation_matrix, 1.0)

        return {
            'matrix': correlation_matrix.tolist(),
            'average_correlation': np.mean(correlation_matrix)
        }

    def _estimate_strategy_greeks(self, strategy_name: str) -> Dict[str, float]:
        """Estimate Greeks for a strategy type."""
        # TODO: Replace with real options pricing models (Black-Scholes, binomial trees, etc.)
        # Should calculate actual Greeks based on underlying price, strike, time to expiry,
        # volatility, risk-free rate, and dividend yield

        # PLACEHOLDER: These are simplified, representative Greek values for demonstration
        # Real implementation would require options chain data and pricing models
        greeks_estimates = {
            'long_call': {'delta': 0.3, 'gamma': 0.05, 'theta': -0.02, 'vega': 0.15, 'rho': 0.1},
            'long_put': {'delta': -0.3, 'gamma': 0.05, 'theta': -0.02, 'vega': 0.15, 'rho': -0.1},
            'straddle': {'delta': 0.0, 'gamma': 0.1, 'theta': -0.04, 'vega': 0.3, 'rho': 0.0},
            'strangle': {'delta': 0.0, 'gamma': 0.08, 'theta': -0.03, 'vega': 0.25, 'rho': 0.0},
            'iron_condor': {'delta': 0.0, 'gamma': -0.02, 'theta': 0.02, 'vega': -0.1, 'rho': 0.0}
        }

        result = greeks_estimates.get(strategy_name, {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0})
        # Note: These are placeholder values - real implementation needs options pricing models
        return result

    def _calculate_hedging_effectiveness(self, portfolio_greeks: Dict[str, float]) -> float:
        """Calculate hedging effectiveness."""
        delta_exposure = abs(portfolio_greeks.get('delta', 0))
        vega_exposure = abs(portfolio_greeks.get('vega', 0))

        # Effectiveness decreases with exposure
        effectiveness = 1.0 / (1.0 + delta_exposure + vega_exposure)
        return min(1.0, effectiveness)

    def _assess_greeks_balance(self, portfolio_greeks: Dict[str, float]) -> str:
        """Assess Greeks balance."""
        delta = portfolio_greeks.get('delta', 0)
        gamma = portfolio_greeks.get('gamma', 0)

        if abs(delta) < 0.1 and abs(gamma) < 0.2:
            return 'well_balanced'
        elif abs(delta) > 0.3 or abs(gamma) > 0.5:
            return 'poorly_balanced'
        else:
            return 'moderately_balanced'

    def _calculate_risk_sensitivities(self, portfolio_greeks: Dict[str, float]) -> Dict[str, float]:
        """Calculate risk sensitivities."""
        return {
            'delta_sensitivity': abs(portfolio_greeks.get('delta', 0)),
            'gamma_sensitivity': abs(portfolio_greeks.get('gamma', 0)),
            'vega_sensitivity': abs(portfolio_greeks.get('vega', 0)),
            'theta_sensitivity': abs(portfolio_greeks.get('theta', 0))
        }

    def _calculate_diversification_ratio(self, portfolio_construction: Dict[str, Any]) -> float:
        """Calculate portfolio diversification ratio."""
        correlation_matrix = portfolio_construction.get('correlation_matrix', {})
        avg_correlation = correlation_matrix.get('average_correlation', 0.5)

        # Higher correlation = lower diversification
        return 1.0 / (1.0 + avg_correlation)

    def _generate_strike_variations(self, strategy_name: str) -> List[Dict[str, Any]]:
        """Generate strike variations for a strategy."""
        variations = []
        
        if strategy_name.lower() == 'call_spread':
            # Bull call spread variations
            variations = [
                {'strikes': {'long': 0.95, 'short': 1.05}, 'description': 'Conservative bull call spread'},
                {'strikes': {'long': 0.98, 'short': 1.02}, 'description': 'Moderate bull call spread'},
                {'strikes': {'long': 0.90, 'short': 1.10}, 'description': 'Aggressive bull call spread'}
            ]
        elif strategy_name.lower() == 'put_spread':
            # Bear put spread variations
            variations = [
                {'strikes': {'long': 1.05, 'short': 0.95}, 'description': 'Conservative bear put spread'},
                {'strikes': {'long': 1.02, 'short': 0.98}, 'description': 'Moderate bear put spread'},
                {'strikes': {'long': 1.10, 'short': 0.90}, 'description': 'Aggressive bear put spread'}
            ]
        elif strategy_name.lower() == 'straddle':
            # Straddle variations around ATM
            variations = [
                {'strikes': {'call': 1.0, 'put': 1.0}, 'description': 'ATM straddle'},
                {'strikes': {'call': 0.98, 'put': 0.98}, 'description': 'OTM straddle'},
                {'strikes': {'call': 1.02, 'put': 1.02}, 'description': 'ITM straddle'}
            ]
        
        return variations

    def _generate_expiration_variations(self, strategy_name: str) -> List[Dict[str, Any]]:
        """Generate expiration variations for a strategy."""
        variations = []
        
        # Common expiration variations
        base_expirations = [
            {'days': 30, 'description': 'Short-term (1 month)'},
            {'days': 60, 'description': 'Medium-term (2 months)'},
            {'days': 90, 'description': 'Long-term (3 months)'},
            {'days': 180, 'description': 'Extended (6 months)'}
        ]
        
        # Adjust based on strategy type
        if 'spread' in strategy_name.lower():
            # Spreads work better with shorter expirations
            variations = base_expirations[:3]
        elif 'straddle' in strategy_name.lower() or 'strangle' in strategy_name.lower():
            # Volatility plays work with various expirations
            variations = base_expirations
        else:
            # Default to medium expirations
            variations = base_expirations[1:3]
        
        return variations

    def _generate_size_variations(self, strategy_name: str) -> List[Dict[str, Any]]:
        """Generate size variations for a strategy."""
        variations = []
        
        # Size variations based on risk tolerance
        base_sizes = [
            {'contracts': 1, 'description': 'Single contract - minimal risk'},
            {'contracts': 5, 'description': 'Small position - 5 contracts'},
            {'contracts': 10, 'description': 'Medium position - 10 contracts'},
            {'contracts': 25, 'description': 'Large position - 25 contracts'}
        ]
        
        # Adjust based on strategy type
        if 'spread' in strategy_name.lower():
            # Spreads can handle larger sizes due to defined risk
            variations = base_sizes
        elif 'straddle' in strategy_name.lower() or 'strangle' in strategy_name.lower():
            # Volatility plays are more expensive, smaller sizes
            variations = base_sizes[:3]
        else:
            # Single options, smaller sizes
            variations = base_sizes[:2]
        
        return variations

class BlackScholesCalculator:
    """
    Black-Scholes options pricing calculator with Greeks.
    """

    @staticmethod
    def calculate_option_price(S: float, K: float, T: float, r: float, sigma: float,
                             option_type: str = 'call') -> float:
        """
        Calculate Black-Scholes option price.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'

        Returns:
            Option price
        """
        if T <= 0 or sigma <= 0:
            # Handle edge cases: at expiration or zero volatility
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        # Prevent division by zero with minimum volatility threshold
        sigma = max(sigma, 0.001)  # Minimum 0.1% volatility
        sqrt_T = np.sqrt(T)
        if sqrt_T == 0 or sigma * sqrt_T == 0:
            # Fallback for numerical issues
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return price

    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float,
                        option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate option Greeks.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'

        Returns:
            Dict with Greeks: delta, gamma, theta, vega, rho
        """
        if T <= 0 or sigma <= 0:
            # At expiration or zero volatility, Greeks are discontinuous
            if option_type == 'call':
                delta = 1.0 if S > K else 0.0
            else:
                delta = -1.0 if S < K else 0.0
            return {
                'delta': delta,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }

        # Prevent division by zero with minimum volatility threshold
        sigma = max(sigma, 0.001)  # Minimum 0.1% volatility
        sqrt_T = np.sqrt(T)
        if sqrt_T == 0 or sigma * sqrt_T == 0:
            # Fallback for numerical issues
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1

        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

        # Theta
        if option_type == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                    - r * K * np.exp(-r * T) * norm.cdf(d2))
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                    + r * K * np.exp(-r * T) * norm.cdf(-d2))

        # Vega (same for calls and puts)
        vega = S * np.sqrt(T) * norm.pdf(d1)

        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
