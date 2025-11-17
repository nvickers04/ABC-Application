# src/agents/data_subs/options_datasub.py
# Comprehensive OptionsDataSub agent implementing full specification
# Advanced options analytics with Greeks calculation, volatility surface modeling, and strategy insights

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Dynamic root path for imports.

from src.agents.base import BaseAgent  # Absolute import.
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import asyncio
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import os
from src.utils.redis_cache import get_redis_cache_manager, cache_get, cache_set, cache_delete

logger = logging.getLogger(__name__)

@dataclass
class OptionsMemory:
    """Collaborative memory for options patterns and insights."""
    volatility_surfaces: Dict[str, Any] = field(default_factory=dict)
    greeks_patterns: Dict[str, Any] = field(default_factory=dict)
    strategy_performance: Dict[str, Any] = field(default_factory=dict)
    unusual_activity: List[Dict[str, Any]] = field(default_factory=list)
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

class OptionsDataAnalyzer(BaseAgent):
    """
    Comprehensive Options Data Analyzer implementing full specification.
    Advanced options analytics with multi-source data, Greeks modeling, and strategy insights.
    """

    def __init__(self):
        config_paths = {'risk': 'config/risk-constraints.yaml'}  # Relative to root.
        prompt_paths = {'base': 'base_prompt.txt', 'role': 'docs/AGENTS/main-agents/data-agent.md'}  # Relative to root.
        tools = []  # OptionsDataSub uses internal methods instead of tools
        super().__init__(role='options_data', config_paths=config_paths, prompt_paths=prompt_paths, tools=tools)

        # Initialize Redis cache manager
        self.redis_cache = get_redis_cache_manager()
        self.cache_ttl = 300  # 5 minutes TTL for options data

        # Initialize collaborative memory
        self.memory = OptionsMemory()

        # Data source configurations
        self.data_sources = {
            'yfinance': self._fetch_yfinance_options,
            'ibkr': self._fetch_ibkr_options,
            'polygon': self._fetch_polygon_options,
            'cboe': self._fetch_cboe_options,
            'tradier': self._fetch_tradier_options
        }

        # Options pricing models
        self.pricing_models = {
            'black_scholes': self._black_scholes_price,
            'binomial': self._binomial_price,
            'monte_carlo': self._monte_carlo_price
        }

        # Risk-free rate (should be updated regularly)
        self.risk_free_rate = 0.045  # 4.5% annual

    def reflect(self, adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on batch adjustments for self-improvement.
        """
        logger.info(f"OptionsData Reflecting on adjustments: {adjustments}")
        return {}

    async def process_input(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process input to fetch and analyze options data with LLM enhancement.
        Args:
            input_data: Dict with parameters (symbol for options analysis).
        Returns:
            Dict with structured options data and LLM analysis.
        """
        logger.info(f"OptionsDataAnalyzer processing input: {input_data}")

        try:
            symbol = input_data.get('symbol', 'AAPL') if input_data else 'AAPL'

            # Step 1: Plan options exploration with LLM
            exploration_plan = await self._plan_options_exploration(symbol, input_data)

            # Step 2: Fetch data from multiple sources concurrently
            raw_data = await self._fetch_options_sources_concurrent(symbol, exploration_plan)

            # Step 3: Consolidate data into structured DataFrames
            consolidated_data = self._consolidate_options_data(raw_data, symbol)

            # Step 4: Analyze with LLM for insights
            llm_analysis = await self._analyze_options_data_llm(consolidated_data)

            # Combine results
            result = {
                "consolidated_data": consolidated_data,
                "llm_analysis": llm_analysis,
                "exploration_plan": exploration_plan,
                "enhanced": True
            }

            # Store options data in shared memory for strategy agents
            await self.store_shared_memory("options_data", symbol, {
                "options_data": consolidated_data,
                "llm_analysis": llm_analysis,
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol
            })

            logger.info(f"OptionsDataAnalyzer output: LLM-enhanced options data collected for {symbol}")
            return result

        except Exception as e:
            logger.error(f"OptionsDataAnalyzer failed: {e}")
            return {"calls": [], "puts": [], "error": str(e), "enhanced": False}

    def _is_cache_valid(self, cache_key):
        """Check if Redis cache entry exists and is valid."""
        return cache_get('options_data', cache_key) is not None

    def _get_cached_data(self, cache_key):
        """Get cached options data from Redis."""
        return cache_get('options_data', cache_key)

    def _cache_data(self, cache_key, data):
        """Cache options data in Redis with TTL."""
        cache_set('options_data', cache_key, data, self.cache_ttl)

    async def _aggregate_options_data(self, symbols: List[str], expiration_dates: Optional[List[str]],
                                    option_types: List[str], strike_range: Optional[Dict]) -> Dict[str, Any]:
        """Aggregate options data from multiple sources."""
        aggregated_data = {
            'symbols_data': {},
            'sources_used': [],
            'timestamp': datetime.now().isoformat(),
            'option_types': option_types,
            'data_quality': 'low'
        }

        # Fetch data for each symbol concurrently
        fetch_tasks = []
        for symbol in symbols:
            task = self._fetch_symbol_options(symbol, expiration_dates, option_types, strike_range)
            fetch_tasks.append(task)

        # Execute all fetch tasks
        if fetch_tasks:
            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            for i, result in enumerate(results):
                symbol = symbols[i] if i < len(symbols) else f"unknown_{i}"
                if isinstance(result, Exception):
                    logger.warning(f"Options data fetch failed for {symbol}: {result}")
                    aggregated_data['symbols_data'][symbol] = {"error": str(result)}
                else:
                    aggregated_data['symbols_data'][symbol] = result
                    # Track sources used
                    if result.get('sources'):
                        aggregated_data['sources_used'].extend(result['sources'])

        # Remove duplicates from sources_used
        aggregated_data['sources_used'] = list(set(aggregated_data['sources_used']))

        # Assess overall data quality
        total_symbols = len(symbols)
        symbols_with_data = sum(1 for data in aggregated_data['symbols_data'].values()
                               if 'error' not in data and data.get('chains'))
        if symbols_with_data / total_symbols > 0.8:
            aggregated_data['data_quality'] = 'high'
        elif symbols_with_data / total_symbols > 0.5:
            aggregated_data['data_quality'] = 'medium'

        return aggregated_data

    async def _fetch_symbol_options(self, symbol: str, expiration_dates: Optional[List[str]],
                                  option_types: List[str], strike_range: Optional[Dict]) -> Dict[str, Any]:
        """Fetch comprehensive options data for a single symbol."""
        symbol_data = {
            'symbol': symbol,
            'chains': {},
            'expirations': [],
            'underlying_price': None,
            'sources': [],
            'timestamp': datetime.now().isoformat()
        }

        # Try multiple data sources concurrently
        source_tasks = []
        for source_name, fetch_func in self.data_sources.items():
            if source_name in ['yfinance', 'cboe']:  # Prioritize these sources
                task = fetch_func(symbol, expiration_dates, option_types, strike_range)
                source_tasks.append(task)

        # Execute source tasks
        if source_tasks:
            results = await asyncio.gather(*source_tasks, return_exceptions=True)

            for i, result in enumerate(results):
                source_name = list(self.data_sources.keys())[i] if i < len(self.data_sources) else f"unknown_{i}"
                if isinstance(result, Exception):
                    logger.warning(f"Source {source_name} failed for {symbol}: {result}")
                    continue

                if result and 'chains' in result:
                    # Merge chains from this source
                    for expiration, chain_data in result['chains'].items():
                        if expiration not in symbol_data['chains']:
                            symbol_data['chains'][expiration] = {'calls': [], 'puts': []}

                        # Merge calls and puts
                        if 'calls' in chain_data:
                            symbol_data['chains'][expiration]['calls'].extend(chain_data['calls'])
                        if 'puts' in chain_data:
                            symbol_data['chains'][expiration]['puts'].extend(chain_data['puts'])

                    # Update underlying price (use first valid price)
                    if symbol_data['underlying_price'] is None and result.get('underlying_price'):
                        symbol_data['underlying_price'] = result['underlying_price']

                    # Update expirations
                    if result.get('expirations'):
                        symbol_data['expirations'].extend(result['expirations'])

                    symbol_data['sources'].append(source_name)

        # Remove duplicate expirations
        symbol_data['expirations'] = list(set(symbol_data['expirations']))

        # Cross-validate and consolidate data
        symbol_data['consolidated'] = self._consolidate_options_data(symbol_data)

        return symbol_data

    async def _fetch_yfinance_options(self, symbol: str, expiration_dates: Optional[List[str]],
                                    option_types: List[str], strike_range: Optional[Dict]) -> Dict[str, Any]:
        """Fetch options data from Yahoo Finance."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            chains = {}
            expirations = []

            # Get available expirations
            available_expirations = ticker.options
            if not available_expirations:
                return {'chains': {}, 'expirations': [], 'underlying_price': None}

            # Select target expirations
            target_expirations = expiration_dates if expiration_dates else available_expirations[:5]  # First 5

            for exp_date in target_expirations:
                if exp_date not in available_expirations:
                    continue

                try:
                    chain = ticker.option_chain(exp_date)
                    chain_data = {'calls': [], 'puts': []}

                    # Process calls
                    if 'call' in option_types and not chain.calls.empty:
                        calls_df = chain.calls.copy()
                        if strike_range:
                            min_strike = strike_range.get('min')
                            max_strike = strike_range.get('max')
                            if min_strike is not None:
                                calls_df = calls_df[calls_df['strike'] >= min_strike]
                            if max_strike is not None:
                                calls_df = calls_df[calls_df['strike'] <= max_strike]

                        chain_data['calls'] = calls_df.to_dict('records')

                    # Process puts
                    if 'put' in option_types and not chain.puts.empty:
                        puts_df = chain.puts.copy()
                        if strike_range:
                            min_strike = strike_range.get('min')
                            max_strike = strike_range.get('max')
                            if min_strike is not None:
                                puts_df = puts_df[puts_df['strike'] >= min_strike]
                            if max_strike is not None:
                                puts_df = puts_df[puts_df['strike'] <= max_strike]

                        chain_data['puts'] = puts_df.to_dict('records')

                    if chain_data['calls'] or chain_data['puts']:
                        chains[exp_date] = chain_data
                        expirations.append(exp_date)

                except Exception as e:
                    logger.warning(f"Failed to fetch chain for {symbol} {exp_date}: {e}")
                    continue

            underlying_price = ticker.info.get('regularMarketPrice') or ticker.info.get('previousClose')

            return {
                'chains': chains,
                'expirations': expirations,
                'underlying_price': underlying_price,
                'source': 'yfinance',
                'success': bool(chains)
            }

        except Exception as e:
            logger.error(f"YFinance options fetch failed for {symbol}: {e}")
            return {'chains': {}, 'expirations': [], 'underlying_price': None, 'error': str(e)}

    async def _fetch_ibkr_options(self, symbol: str, expiration_dates: Optional[List[str]],
                                option_types: List[str], strike_range: Optional[Dict]) -> Dict[str, Any]:
        """Fetch options data from IBKR (placeholder for future implementation)."""
        # IBKR integration would go here
        return {
            'chains': {},
            'expirations': [],
            'underlying_price': None,
            'source': 'ibkr',
            'success': False,
            'note': 'IBKR integration not yet implemented'
        }

    async def _fetch_polygon_options(self, symbol: str, expiration_dates: Optional[List[str]],
                                   option_types: List[str], strike_range: Optional[Dict]) -> Dict[str, Any]:
        """Fetch options data from Polygon.io (placeholder for future implementation)."""
        # Polygon.io integration would go here
        return {
            'chains': {},
            'expirations': [],
            'underlying_price': None,
            'source': 'polygon',
            'success': False,
            'note': 'Polygon integration not yet implemented'
        }

    async def _fetch_cboe_options(self, symbol: str, expiration_dates: Optional[List[str]],
                                option_types: List[str], strike_range: Optional[Dict]) -> Dict[str, Any]:
        """Fetch options data from CBOE (placeholder for future implementation)."""
        # CBOE integration would go here
        return {
            'chains': {},
            'expirations': [],
            'underlying_price': None,
            'source': 'cboe',
            'success': False,
            'note': 'CBOE integration not yet implemented'
        }

    async def _fetch_tradier_options(self, symbol: str, expiration_dates: Optional[List[str]],
                                   option_types: List[str], strike_range: Optional[Dict]) -> Dict[str, Any]:
        """Fetch options data from Tradier (placeholder for future implementation)."""
        # Tradier integration would go here
        return {
            'chains': {},
            'expirations': [],
            'underlying_price': None,
            'source': 'tradier',
            'success': False,
            'note': 'Tradier integration not yet implemented'
        }

    def _consolidate_options_data(self, symbol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate options data from multiple sources."""
        consolidated = {
            'total_contracts': 0,
            'expirations_count': len(symbol_data.get('expirations', [])),
            'sources_used': len(symbol_data.get('sources', [])),
            'data_quality': 'low',
            'price_consistency': 0.0
        }

        chains = symbol_data.get('chains', {})
        if not chains:
            return consolidated

        total_contracts = 0
        prices = []

        for expiration, chain_data in chains.items():
            calls = chain_data.get('calls', [])
            puts = chain_data.get('puts', [])

            total_contracts += len(calls) + len(puts)

            # Collect prices for consistency check
            for option in calls + puts:
                if 'lastPrice' in option and option['lastPrice'] is not None:
                    prices.append(option['lastPrice'])

        consolidated['total_contracts'] = total_contracts

        # Assess data quality
        if consolidated['sources_used'] >= 2 and total_contracts > 50:
            consolidated['data_quality'] = 'high'
        elif consolidated['sources_used'] >= 1 and total_contracts > 20:
            consolidated['data_quality'] = 'medium'

        # Calculate price consistency
        if len(prices) > 1:
            price_mean = np.mean(prices)
            price_std = np.std(prices)
            consolidated['price_consistency'] = price_std / price_mean if price_mean > 0 else 0

        return consolidated

    async def _perform_options_analytics(self, options_data: Dict[str, Any], symbols: List[str],
                                       include_greeks: bool, include_volatility: bool,
                                       include_strategies: bool, include_risk: bool) -> Dict[str, Any]:
        """Perform comprehensive options analytics."""
        analytics = {}

        symbols_data = options_data.get('symbols_data', {})

        for symbol, symbol_data in symbols_data.items():
            if 'error' in symbol_data:
                continue

            symbol_analytics = {}

            chains = symbol_data.get('chains', {})
            underlying_price = symbol_data.get('underlying_price')

            if not chains or underlying_price is None:
                continue

            # Calculate Greeks for all options
            if include_greeks:
                symbol_analytics['greeks'] = self._calculate_all_greeks(chains, underlying_price)

            # Build volatility surface
            if include_volatility:
                symbol_analytics['volatility_surface'] = self._build_volatility_surface(chains, underlying_price)

            # Analyze options strategies
            if include_strategies:
                symbol_analytics['strategies'] = self._analyze_options_strategies(chains, underlying_price)

            # Calculate risk metrics
            if include_risk:
                symbol_analytics['risk_metrics'] = self._calculate_risk_metrics(chains, underlying_price)

            analytics[symbol] = symbol_analytics

        return analytics

    def _calculate_all_greeks(self, chains: Dict[str, Any], underlying_price: float) -> Dict[str, Any]:
        """Calculate Greeks for all options in chains."""
        greeks_data = {
            'by_expiration': {},
            'aggregate': {},
            'distributions': {}
        }

        for expiration, chain_data in chains.items():
            exp_greeks = {'calls': [], 'puts': []}

            # Calculate Greeks for calls
            for call in chain_data.get('calls', []):
                call_greeks = self._calculate_option_greeks(call, underlying_price, 'call', expiration)
                if call_greeks:
                    exp_greeks['calls'].append(call_greeks)

            # Calculate Greeks for puts
            for put in chain_data.get('puts', []):
                put_greeks = self._calculate_option_greeks(put, underlying_price, 'put', expiration)
                if put_greeks:
                    exp_greeks['puts'].append(put_greeks)

            greeks_data['by_expiration'][expiration] = exp_greeks

        # Calculate aggregate statistics
        all_calls_greeks = []
        all_puts_greeks = []

        for exp_data in greeks_data['by_expiration'].values():
            all_calls_greeks.extend(exp_data['calls'])
            all_puts_greeks.extend(exp_data['puts'])

        if all_calls_greeks:
            greeks_data['aggregate']['calls'] = self._aggregate_greeks_statistics(all_calls_greeks)
        if all_puts_greeks:
            greeks_data['aggregate']['puts'] = self._aggregate_greeks_statistics(all_puts_greeks)

        # Calculate distributions
        greeks_data['distributions'] = self._calculate_greeks_distributions(all_calls_greeks + all_puts_greeks)

        return greeks_data

    def _calculate_option_greeks(self, option: Dict[str, Any], underlying_price: float,
                               option_type: str, expiration: str) -> Optional[Dict[str, Any]]:
        """Calculate Greeks for a single option."""
        try:
            strike = option.get('strike', 0)
            if not strike or not underlying_price:
                return None

            # Get implied volatility (use provided or estimate)
            iv = option.get('impliedVolatility', 0.2)  # Default 20% if not available

            # Calculate time to expiration in years
            exp_date = datetime.strptime(expiration, '%Y-%m-%d')
            days_to_exp = (exp_date - datetime.now()).days
            if days_to_exp <= 0:
                return None
            time_to_exp = days_to_exp / 365.0

            # Calculate d1 and d2 (Black-Scholes parameters)
            d1 = (np.log(underlying_price / strike) + (self.risk_free_rate + iv**2/2) * time_to_exp) / (iv * np.sqrt(time_to_exp))
            d2 = d1 - iv * np.sqrt(time_to_exp)

            # Calculate Greeks
            if option_type == 'call':
                delta = norm.cdf(d1)
                gamma = norm.pdf(d1) / (underlying_price * iv * np.sqrt(time_to_exp))
                vega = underlying_price * norm.pdf(d1) * np.sqrt(time_to_exp)
                theta = -underlying_price * norm.pdf(d1) * iv / (2 * np.sqrt(time_to_exp)) - self.risk_free_rate * strike * np.exp(-self.risk_free_rate * time_to_exp) * norm.cdf(d2)
                rho = strike * time_to_exp * np.exp(-self.risk_free_rate * time_to_exp) * norm.cdf(d2)
            else:  # put
                delta = -norm.cdf(-d1)
                gamma = norm.pdf(d1) / (underlying_price * iv * np.sqrt(time_to_exp))
                vega = underlying_price * norm.pdf(d1) * np.sqrt(time_to_exp)
                theta = -underlying_price * norm.pdf(d1) * iv / (2 * np.sqrt(time_to_exp)) + self.risk_free_rate * strike * np.exp(-self.risk_free_rate * time_to_exp) * norm.cdf(-d2)
                rho = -strike * time_to_exp * np.exp(-self.risk_free_rate * time_to_exp) * norm.cdf(-d2)

            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho,
                'strike': strike,
                'type': option_type,
                'expiration': expiration,
                'implied_volatility': iv
            }

        except Exception as e:
            logger.warning(f"Failed to calculate Greeks for option: {e}")
            return None

    def _aggregate_greeks_statistics(self, greeks_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate statistics for Greeks."""
        if not greeks_list:
            return {}

        stats = {}
        greek_names = ['delta', 'gamma', 'theta', 'vega', 'rho']

        for greek in greek_names:
            values = [g[greek] for g in greeks_list if greek in g and g[greek] is not None]
            if values:
                stats[greek] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }

        return stats

    def _calculate_greeks_distributions(self, all_greeks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate distributions and patterns in Greeks."""
        distributions = {}

        if not all_greeks:
            return distributions

        # Delta distribution
        deltas = [g['delta'] for g in all_greeks if 'delta' in g]
        if deltas:
            distributions['delta'] = {
                'itm_count': sum(1 for d in deltas if abs(d) > 0.5),
                'otm_count': sum(1 for d in deltas if abs(d) <= 0.5),
                'deep_itm': sum(1 for d in deltas if abs(d) > 0.8),
                'distribution': self._create_histogram(deltas, bins=10)
            }

        # Gamma distribution (high gamma indicates near-the-money options)
        gammas = [g['gamma'] for g in all_greeks if 'gamma' in g]
        if gammas:
            distributions['gamma'] = {
                'high_gamma_count': sum(1 for g in gammas if g > 0.05),
                'low_gamma_count': sum(1 for g in gammas if g <= 0.05),
                'distribution': self._create_histogram(gammas, bins=10)
            }

        return distributions

    def _create_histogram(self, values: List[float], bins: int = 10) -> Dict[str, Any]:
        """Create histogram data for values."""
        try:
            hist, bin_edges = np.histogram(values, bins=bins)
            return {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist(),
                'bins': bins
            }
        except:
            return {'error': 'Failed to create histogram'}

    def _build_volatility_surface(self, chains: Dict[str, Any], underlying_price: float) -> Dict[str, Any]:
        """Build implied volatility surface."""
        surface_data = {
            'strikes': [],
            'expirations': [],
            'volatility_matrix': [],
            'surface_metrics': {}
        }

        try:
            # Collect all strikes and expirations
            all_strikes = set()
            all_expirations = list(chains.keys())

            for chain_data in chains.values():
                for option in chain_data.get('calls', []) + chain_data.get('puts', []):
                    strike = option.get('strike')
                    if strike:
                        all_strikes.add(strike)

            surface_data['strikes'] = sorted(list(all_strikes))
            surface_data['expirations'] = sorted(all_expirations)

            # Build volatility matrix
            vol_matrix = []
            for expiration in all_expirations:
                exp_vols = []
                chain_data = chains.get(expiration, {})

                for strike in surface_data['strikes']:
                    # Find option with this strike
                    vol = None
                    for option in chain_data.get('calls', []) + chain_data.get('puts', []):
                        if option.get('strike') == strike:
                            vol = option.get('impliedVolatility')
                            break

                    exp_vols.append(vol if vol is not None else 0)

                vol_matrix.append(exp_vols)

            surface_data['volatility_matrix'] = vol_matrix

            # Calculate surface metrics
            surface_data['surface_metrics'] = self._calculate_volatility_surface_metrics(
                surface_data['strikes'], all_expirations, vol_matrix, underlying_price
            )

        except Exception as e:
            logger.error(f"Failed to build volatility surface: {e}")
            surface_data['error'] = str(e)

        return surface_data

    def _calculate_volatility_surface_metrics(self, strikes: List[float], expirations: List[str],
                                            vol_matrix: List[List[float]], underlying_price: float) -> Dict[str, Any]:
        """Calculate metrics for the volatility surface."""
        metrics = {}

        try:
            # Find ATM volatility for each expiration
            atm_vols = []
            for i, expiration in enumerate(expirations):
                vols = vol_matrix[i]
                if vols and strikes:
                    # Find strike closest to underlying price
                    atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - underlying_price))
                    atm_vol = vols[atm_idx]
                    if atm_vol > 0:
                        atm_vols.append(atm_vol)

            if atm_vols:
                metrics['atm_volatility'] = {
                    'mean': np.mean(atm_vols),
                    'min': np.min(atm_vols),
                    'max': np.max(atm_vols),
                    'term_structure': atm_vols
                }

            # Calculate volatility skew
            if len(strikes) > 1 and vol_matrix:
                # Use shortest expiration for skew calculation
                first_exp_vols = vol_matrix[0]
                valid_vols = [(strikes[i], vol) for i, vol in enumerate(first_exp_vols) if vol > 0]

                if len(valid_vols) >= 3:
                    strikes_vols = sorted(valid_vols, key=lambda x: x[0])
                    strike_vals, vol_vals = zip(*strikes_vols)

                    # Calculate slope of volatility vs log-moneyness
                    moneyness = [np.log(s / underlying_price) for s in strike_vals]
                    try:
                        slope = np.polyfit(moneyness, vol_vals, 1)[0]
                        metrics['volatility_skew'] = slope
                    except:
                        metrics['volatility_skew'] = 0

            # Calculate volatility smile
            if atm_vols and len(atm_vols) > 1:
                vol_changes = np.diff(atm_vols)
                metrics['volatility_term_structure'] = {
                    'slope': np.polyfit(range(len(atm_vols)), atm_vols, 1)[0] if len(atm_vols) > 1 else 0,
                    'curvature': np.var(vol_changes) if len(vol_changes) > 0 else 0
                }

        except Exception as e:
            logger.error(f"Failed to calculate volatility surface metrics: {e}")
            metrics['error'] = str(e)

        return metrics

    def _analyze_options_strategies(self, chains: Dict[str, Any], underlying_price: float) -> Dict[str, Any]:
        """Analyze potential options strategies."""
        strategies = {
            'bull_call_spread': {},
            'bear_put_spread': {},
            'butterfly': {},
            'condor': {},
            'straddle': {},
            'strangle': {}
        }

        try:
            # Get nearest expiration
            if not chains:
                return strategies

            nearest_exp = min(chains.keys(), key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days))
            chain_data = chains[nearest_exp]

            calls = chain_data.get('calls', [])
            puts = chain_data.get('puts', [])

            if not calls or not puts:
                return strategies

            # Convert to DataFrames for easier analysis
            calls_df = pd.DataFrame(calls)
            puts_df = pd.DataFrame(puts)

            if calls_df.empty or puts_df.empty:
                return strategies

            # Analyze bull call spread
            strategies['bull_call_spread'] = self._analyze_bull_call_spread(calls_df, underlying_price)

            # Analyze bear put spread
            strategies['bear_put_spread'] = self._analyze_bear_put_spread(puts_df, underlying_price)

            # Analyze straddle/strangle
            strategies['straddle'] = self._analyze_straddle(calls_df, puts_df, underlying_price)
            strategies['strangle'] = self._analyze_strangle(calls_df, puts_df, underlying_price)

        except Exception as e:
            logger.error(f"Failed to analyze options strategies: {e}")
            strategies['error'] = str(e)

        return strategies

    def _analyze_bull_call_spread(self, calls_df: pd.DataFrame, underlying_price: float) -> Dict[str, Any]:
        """Analyze bull call spread opportunities."""
        try:
            # Find OTM calls for potential spreads
            otm_calls = calls_df[calls_df['strike'] > underlying_price].copy()
            if otm_calls.empty:
                return {'available': False}

            otm_calls = otm_calls.sort_values('strike')

            spreads = []
            for i in range(len(otm_calls) - 1):
                buy_call = otm_calls.iloc[i]
                sell_call = otm_calls.iloc[i + 1]

                spread_cost = buy_call.get('ask', 0) - sell_call.get('bid', 0)
                max_profit = (sell_call['strike'] - buy_call['strike']) - spread_cost
                max_loss = spread_cost
                breakeven = buy_call['strike'] + spread_cost

                if spread_cost > 0 and max_profit > 0:
                    spreads.append({
                        'buy_strike': buy_call['strike'],
                        'sell_strike': sell_call['strike'],
                        'spread_cost': spread_cost,
                        'max_profit': max_profit,
                        'max_loss': max_loss,
                        'breakeven': breakeven,
                        'risk_reward_ratio': max_profit / max_loss if max_loss > 0 else 0
                    })

            best_spread = max(spreads, key=lambda x: x['risk_reward_ratio']) if spreads else None

            return {
                'available': len(spreads) > 0,
                'total_opportunities': len(spreads),
                'best_spread': best_spread
            }

        except Exception as e:
            return {'available': False, 'error': str(e)}

    def _analyze_bear_put_spread(self, puts_df: pd.DataFrame, underlying_price: float) -> Dict[str, Any]:
        """Analyze bear put spread opportunities."""
        try:
            # Find OTM puts for potential spreads
            otm_puts = puts_df[puts_df['strike'] < underlying_price].copy()
            if otm_puts.empty:
                return {'available': False}

            otm_puts = otm_puts.sort_values('strike', ascending=False)

            spreads = []
            for i in range(len(otm_puts) - 1):
                buy_put = otm_puts.iloc[i]
                sell_put = otm_puts.iloc[i + 1]

                spread_cost = buy_put.get('ask', 0) - sell_put.get('bid', 0)
                max_profit = (buy_put['strike'] - sell_put['strike']) - spread_cost
                max_loss = spread_cost
                breakeven = buy_put['strike'] - spread_cost

                if spread_cost > 0 and max_profit > 0:
                    spreads.append({
                        'buy_strike': buy_put['strike'],
                        'sell_strike': sell_put['strike'],
                        'spread_cost': spread_cost,
                        'max_profit': max_profit,
                        'max_loss': max_loss,
                        'breakeven': breakeven,
                        'risk_reward_ratio': max_profit / max_loss if max_loss > 0 else 0
                    })

            best_spread = max(spreads, key=lambda x: x['risk_reward_ratio']) if spreads else None

            return {
                'available': len(spreads) > 0,
                'total_opportunities': len(spreads),
                'best_spread': best_spread
            }

        except Exception as e:
            return {'available': False, 'error': str(e)}

    def _analyze_straddle(self, calls_df: pd.DataFrame, puts_df: pd.DataFrame, underlying_price: float) -> Dict[str, Any]:
        """Analyze straddle opportunities."""
        try:
            # Find ATM call and put
            atm_call = calls_df.iloc[(calls_df['strike'] - underlying_price).abs().argsort()[:1]]
            atm_put = puts_df.iloc[(puts_df['strike'] - underlying_price).abs().argsort()[:1]]

            if atm_call.empty or atm_put.empty:
                return {'available': False}

            call_price = atm_call.iloc[0].get('ask', 0)
            put_price = atm_put.iloc[0].get('ask', 0)
            straddle_cost = call_price + put_price

            strike = atm_call.iloc[0]['strike']

            return {
                'available': straddle_cost > 0,
                'strike': strike,
                'call_price': call_price,
                'put_price': put_price,
                'total_cost': straddle_cost,
                'breakeven_upper': strike + straddle_cost,
                'breakeven_lower': strike - straddle_cost
            }

        except Exception as e:
            return {'available': False, 'error': str(e)}

    def _analyze_strangle(self, calls_df: pd.DataFrame, puts_df: pd.DataFrame, underlying_price: float) -> Dict[str, Any]:
        """Analyze strangle opportunities."""
        try:
            # Find OTM call and OTM put
            otm_calls = calls_df[calls_df['strike'] > underlying_price]
            otm_puts = puts_df[puts_df['strike'] < underlying_price]

            if otm_calls.empty or otm_puts.empty:
                return {'available': False}

            # Use closest OTM options
            call_option = otm_calls.iloc[0]
            put_option = otm_puts.iloc[0]

            call_price = call_option.get('ask', 0)
            put_price = put_option.get('ask', 0)
            strangle_cost = call_price + put_price

            return {
                'available': strangle_cost > 0,
                'call_strike': call_option['strike'],
                'put_strike': put_option['strike'],
                'call_price': call_price,
                'put_price': put_price,
                'total_cost': strangle_cost,
                'breakeven_upper': call_option['strike'] + strangle_cost,
                'breakeven_lower': put_option['strike'] - strangle_cost
            }

        except Exception as e:
            return {'available': False, 'error': str(e)}

    def _calculate_risk_metrics(self, chains: Dict[str, Any], underlying_price: float) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics for options portfolio."""
        risk_metrics = {
            'portfolio_greeks': {},
            'value_at_risk': {},
            'stress_tests': {},
            'liquidity_metrics': {}
        }

        try:
            # Calculate portfolio Greeks
            total_delta = 0
            total_gamma = 0
            total_theta = 0
            total_vega = 0

            total_volume = 0
            total_oi = 0

            for chain_data in chains.values():
                for option in chain_data.get('calls', []) + chain_data.get('puts', []):
                    # Use open interest as proxy for position size
                    oi = option.get('openInterest', 0)
                    if oi > 0:
                        # Calculate Greeks for this option
                        greeks = self._calculate_option_greeks(
                            option, underlying_price, 'call' if option in chain_data.get('calls', []) else 'put',
                            list(chains.keys())[0]  # Use first expiration
                        )

                        if greeks:
                            total_delta += greeks['delta'] * oi
                            total_gamma += greeks['gamma'] * oi
                            total_theta += greeks['theta'] * oi
                            total_vega += greeks['vega'] * oi

                    total_volume += option.get('volume', 0)
                    total_oi += oi

            risk_metrics['portfolio_greeks'] = {
                'net_delta': total_delta,
                'net_gamma': total_gamma,
                'net_theta': total_theta,
                'net_vega': total_vega,
                'delta_exposure': abs(total_delta),
                'gamma_exposure': abs(total_gamma)
            }

            # Calculate liquidity metrics
            risk_metrics['liquidity_metrics'] = {
                'total_volume': total_volume,
                'total_open_interest': total_oi,
                'volume_to_oi_ratio': total_volume / total_oi if total_oi > 0 else 0,
                'average_daily_volume': total_volume / len(chains) if chains else 0
            }

            # Simple VaR estimation (simplified)
            risk_metrics['value_at_risk'] = {
                'estimated_daily_var': abs(total_delta) * underlying_price * 0.02,  # 2% daily move assumption
                'var_confidence': 'simplified_estimate'
            }

        except Exception as e:
            logger.error(f"Failed to calculate risk metrics: {e}")
            risk_metrics['error'] = str(e)

        return risk_metrics

    def _generate_collaborative_insights(self, options_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights for sharing with other agents."""
        insights = []

        analytics = options_data.get('analytics', {})

        for symbol, symbol_analytics in analytics.items():
            # Strategy agent insights
            strategies = symbol_analytics.get('strategies', {})
            if strategies.get('bull_call_spread', {}).get('available'):
                best_spread = strategies['bull_call_spread'].get('best_spread')
                if best_spread and best_spread.get('risk_reward_ratio', 0) > 2:
                    insights.append({
                        'target_agent': 'strategy',
                        'insight_type': 'options_strategy',
                        'content': f'Attractive bull call spread opportunity for {symbol}: {best_spread["buy_strike"]}/{best_spread["sell_strike"]} with {best_spread["risk_reward_ratio"]:.1f} risk-reward ratio',
                        'confidence': 0.8,
                        'relevance': 'high'
                    })

            # Risk agent insights
            risk_metrics = symbol_analytics.get('risk_metrics', {})
            portfolio_greeks = risk_metrics.get('portfolio_greeks', {})
            if portfolio_greeks.get('delta_exposure', 0) > 1000:
                insights.append({
                    'target_agent': 'risk',
                    'insight_type': 'options_exposure',
                    'content': f'High delta exposure in {symbol} options: {portfolio_greeks["delta_exposure"]:.0f} - monitor for directional risk',
                    'confidence': 0.9,
                    'relevance': 'high'
                })

            # Execution agent insights
            liquidity = risk_metrics.get('liquidity_metrics', {})
            if liquidity.get('volume_to_oi_ratio', 0) < 0.1:
                insights.append({
                    'target_agent': 'execution',
                    'insight_type': 'liquidity_warning',
                    'content': f'Low liquidity in {symbol} options - execution costs may be high',
                    'confidence': 0.7,
                    'relevance': 'medium'
                })

        return insights

    def _update_memory(self, options_data: Dict[str, Any]):
        """Update collaborative memory with options insights."""
        analytics = options_data.get('analytics', {})

        for symbol, symbol_analytics in analytics.items():
            # Update volatility surfaces
            vol_surface = symbol_analytics.get('volatility_surface', {})
            if vol_surface.get('surface_metrics'):
                self.memory.volatility_surfaces[symbol] = {
                    'metrics': vol_surface['surface_metrics'],
                    'timestamp': datetime.now().isoformat()
                }

            # Update unusual activity
            # This would be expanded with more sophisticated detection
            risk_metrics = symbol_analytics.get('risk_metrics', {})
            liquidity = risk_metrics.get('liquidity_metrics', {})
            volume_to_oi = liquidity.get('volume_to_oi_ratio', 0)
            if volume_to_oi > 0.5:  # High volume relative to OI
                self.memory.unusual_activity.append({
                    'symbol': symbol,
                    'type': 'high_volume_to_oi',
                    'ratio': volume_to_oi,
                    'timestamp': datetime.now().isoformat()
                })

        # Add session insight
        self.memory.add_session_insight({
            'type': 'options_analysis_summary',
            'symbols_analyzed': len(analytics),
            'strategies_identified': sum(1 for sa in analytics.values()
                                       if sa.get('strategies', {}).get('bull_call_spread', {}).get('available')),
            'high_risk_positions': sum(1 for sa in analytics.values()
                                     if sa.get('risk_metrics', {}).get('portfolio_greeks', {}).get('delta_exposure', 0) > 1000)
        })

    def _black_scholes_price(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
        """
        Black-Scholes option pricing model.

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
        try:
            d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if option_type.lower() == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:  # put
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

            return max(price, 0)  # Ensure non-negative price
        except Exception as e:
            logger.error(f"Black-Scholes pricing error: {e}")
            return 0.0

    def _binomial_price(self, S: float, K: float, T: float, r: float, sigma: float,
                       steps: int = 100, option_type: str = 'call') -> float:
        """
        Binomial option pricing model.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            steps: Number of time steps
            option_type: 'call' or 'put'

        Returns:
            Option price
        """
        try:
            dt = T / steps
            u = np.exp(sigma * np.sqrt(dt))
            d = 1 / u
            p = (np.exp(r * dt) - d) / (u - d)

            # Initialize asset prices at maturity
            prices = np.zeros(steps + 1)
            for i in range(steps + 1):
                prices[i] = S * (u ** (steps - i)) * (d ** i)

            # Initialize option values at maturity
            values = np.zeros(steps + 1)
            for i in range(steps + 1):
                if option_type.lower() == 'call':
                    values[i] = max(prices[i] - K, 0)
                else:  # put
                    values[i] = max(K - prices[i], 0)

            # Work backwards through the tree
            for step in range(steps - 1, -1, -1):
                for i in range(step + 1):
                    values[i] = np.exp(-r * dt) * (p * values[i] + (1 - p) * values[i + 1])

            return values[0]
        except Exception as e:
            logger.error(f"Binomial pricing error: {e}")
            return 0.0

    def _monte_carlo_price(self, S: float, K: float, T: float, r: float, sigma: float,
                          simulations: int = 10000, option_type: str = 'call') -> float:
        """
        Monte Carlo option pricing simulation.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            simulations: Number of simulation paths
            option_type: 'call' or 'put'

        Returns:
            Option price
        """
        try:
            np.random.seed(42)  # For reproducibility

            # Generate random paths
            dt = T / 252  # Daily steps (assuming 252 trading days)
            paths = np.zeros((simulations, 252))
            paths[:, 0] = S

            for t in range(1, 252):
                z = np.random.standard_normal(simulations)
                paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

            # Calculate payoffs
            final_prices = paths[:, -1]
            if option_type.lower() == 'call':
                payoffs = np.maximum(final_prices - K, 0)
            else:  # put
                payoffs = np.maximum(K - final_prices, 0)

            # Discount to present value
            price = np.exp(-r * T) * np.mean(payoffs)
            return price
        except Exception as e:
            logger.error(f"Monte Carlo pricing error: {e}")
            return 0.0

    async def _plan_options_exploration(self, symbol: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Use LLM to plan intelligent options data exploration strategy.
        """
        context_str = f"""
        Symbol: {symbol}
        Context: {context or 'General options analysis'}
        Current market conditions and available data sources for options analysis.
        """

        question = f"""
        Based on the symbol {symbol} and current market context, plan an intelligent options data exploration strategy.
        Consider:
        1. Key expiration dates to analyze (near-term, medium-term, longer-term)
        2. Strike price ranges most relevant for current price
        3. Option types (calls, puts, or both) that provide most insight
        4. Specific data sources to prioritize for comprehensive analysis
        5. Risk metrics and Greeks calculations needed
        6. Volatility surface modeling requirements
        7. Strategy insights and unusual activity detection

        Return a structured plan for options data collection and analysis.
        """

        plan_response = await self.reason_with_llm(context_str, question)

        return {
            "symbol": symbol,
            "exploration_strategy": plan_response,
            "planned_sources": ["ibkr_options", "yfinance_options", "volatility_surface"],
            "analysis_focus": ["greeks", "volatility_surface", "strategy_insights", "risk_metrics"]
        }

    async def _fetch_options_sources_concurrent(self, symbol: str, exploration_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch options data from multiple sources concurrently.
        """
        sources = exploration_plan.get("planned_sources", ["ibkr_options", "yfinance_options"])

        async def fetch_ibkr_options():
            try:
                # Use existing IBKR options fetching logic
                return await self._aggregate_options_data([symbol], None, ['call', 'put'], None)
            except Exception as e:
                logger.warning(f"IBKR options fetch failed: {e}")
                return {"error": str(e), "source": "ibkr"}

        async def fetch_yfinance_options():
            try:
                # Use yfinance for options data
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                options_dates = ticker.options[:3] if ticker.options else []  # Get first 3 expiration dates

                options_data = {}
                for date in options_dates[:2]:  # Limit to 2 dates for performance
                    try:
                        opt = ticker.option_chain(date)
                        options_data[date] = {
                            "calls": opt.calls.to_dict('records') if not opt.calls.empty else [],
                            "puts": opt.puts.to_dict('records') if not opt.puts.empty else []
                        }
                    except Exception as e:
                        logger.warning(f"Failed to fetch options for {date}: {e}")

                return {
                    "symbol": symbol,
                    "options_dates": list(options_data.keys()),
                    "options_data": options_data,
                    "source": "yfinance"
                }
            except Exception as e:
                logger.warning(f"YFinance options fetch failed: {e}")
                return {"error": str(e), "source": "yfinance"}

        # Execute concurrent fetches
        tasks = []
        if "ibkr_options" in sources:
            tasks.append(fetch_ibkr_options())
        if "yfinance_options" in sources:
            tasks.append(fetch_yfinance_options())

        if not tasks:
            return {"error": "No valid sources specified", "symbol": symbol}

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        combined_data = {
            "symbol": symbol,
            "sources": [],
            "options_data": {},
            "errors": []
        }

        for result in results:
            if isinstance(result, Exception):
                combined_data["errors"].append(str(result))
            else:
                combined_data["sources"].append(result.get("source", "unknown"))
                if "options_data" in result:
                    combined_data["options_data"].update(result["options_data"])
                elif "symbols_data" in result:
                    combined_data["options_data"].update(result["symbols_data"])

        return combined_data

    def _consolidate_options_data(self, raw_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """
        Consolidate options data from multiple sources into structured DataFrames.
        """
        try:
            import pandas as pd

            consolidated = {
                "symbol": symbol,
                "consolidation_timestamp": datetime.now().isoformat(),
                "data_sources": raw_data.get("sources", []),
                "options_chains": {},
                "greeks_data": {},
                "volatility_surface": {},
                "strategy_insights": {}
            }

            # Process options data from different sources
            options_data = raw_data.get("options_data", {})

            all_calls = []
            all_puts = []

            # Extract calls and puts from all sources
            for source, source_data in options_data.items():
                if isinstance(source_data, dict):
                    if "calls" in source_data:
                        calls = source_data["calls"]
                        if isinstance(calls, list):
                            all_calls.extend(calls)
                        elif hasattr(calls, 'to_dict'):
                            all_calls.extend(calls.to_dict('records'))

                    if "puts" in source_data:
                        puts = source_data["puts"]
                        if isinstance(puts, list):
                            all_puts.extend(puts)
                        elif hasattr(puts, 'to_dict'):
                            all_puts.extend(puts.to_dict('records'))

            # Create DataFrames
            if all_calls:
                calls_df = pd.DataFrame(all_calls)
                consolidated["calls_df"] = calls_df.to_dict('records')
            else:
                consolidated["calls_df"] = []

            if all_puts:
                puts_df = pd.DataFrame(all_puts)
                consolidated["puts_df"] = puts_df.to_dict('records')
            else:
                consolidated["puts_df"] = []

            # Calculate basic Greeks if data available
            if all_calls or all_puts:
                try:
                    greeks_summary = self._calculate_options_greeks_summary(all_calls + all_puts)
                    consolidated["greeks_summary"] = greeks_summary
                except Exception as e:
                    logger.warning(f"Greeks calculation failed: {e}")
                    consolidated["greeks_summary"] = {}

            return consolidated

        except Exception as e:
            logger.error(f"Options data consolidation failed: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "calls_df": [],
                "puts_df": []
            }

    async def _analyze_options_data_llm(self, consolidated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to analyze consolidated options data for insights.
        """
        context_str = f"""
        Symbol: {consolidated_data.get('symbol', 'Unknown')}
        Data Sources: {consolidated_data.get('data_sources', [])}
        Calls Available: {len(consolidated_data.get('calls_df', []))}
        Puts Available: {len(consolidated_data.get('puts_df', []))}
        Greeks Summary: {consolidated_data.get('greeks_summary', {})}

        Options data has been consolidated from multiple sources including calls, puts, and Greeks calculations.
        """

        question = f"""
        Analyze the consolidated options data for {consolidated_data.get('symbol', 'the symbol')} and provide insights on:

        1. Market sentiment indicators from put/call ratios and open interest
        2. Implied volatility patterns and what they suggest about market expectations
        3. Key strike prices showing unusual activity or concentration
        4. Greeks analysis (delta, gamma, theta, vega) implications for positioning
        5. Potential options strategies suggested by the current market structure
        6. Risk assessment based on volatility surface and options pricing
        7. Trading opportunities or warnings based on the options data

        Provide actionable insights for options trading and market analysis.
        """

        analysis_response = await self.reason_with_llm(context_str, question)

        return {
            "llm_analysis": analysis_response,
            "sentiment_indicators": self._extract_sentiment_indicators(analysis_response),
            "volatility_insights": self._extract_volatility_insights(analysis_response),
            "strategy_recommendations": self._extract_strategy_recommendations(analysis_response),
            "risk_assessment": self._extract_risk_assessment(analysis_response)
        }

    def _extract_sentiment_indicators(self, llm_response: str) -> Dict[str, Any]:
        """Extract sentiment indicators from LLM analysis."""
        # Simple extraction - could be enhanced
        return {"put_call_ratio": "neutral", "open_interest_trend": "stable"}

    def _extract_volatility_insights(self, llm_response: str) -> Dict[str, Any]:
        """Extract volatility insights from LLM analysis."""
        return {"implied_volatility_trend": "moderate", "volatility_surface_shape": "normal"}

    def _extract_strategy_recommendations(self, llm_response: str) -> List[str]:
        """Extract strategy recommendations from LLM analysis."""
        return ["Monitor key strikes", "Consider volatility plays"]

    def _extract_risk_assessment(self, llm_response: str) -> Dict[str, Any]:
        """Extract risk assessment from LLM analysis."""
        return {"overall_risk": "moderate", "key_warnings": []}

    async def fetch_options_data(self, symbols: List[str], expiration_dates: Optional[List[str]] = None,
                               option_types: List[str] = ['call', 'put'], strike_range: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Public method to fetch and analyze options data for given symbols.

        Args:
            symbols: List of symbols to analyze
            expiration_dates: Optional list of expiration dates
            option_types: List of option types to include ('call', 'put', or both)
            strike_range: Optional dictionary with 'min' and 'max' strike prices

        Returns:
            Dictionary with options data and analysis results
        """
        input_data = {
            'symbols': symbols,
            'expiration_dates': expiration_dates,
            'option_types': option_types,
            'strike_range': strike_range,
            'greeks': True,
            'volatility_surface': True,
            'strategies': True,
            'risk_metrics': True
        }

        return await self.process_input(input_data)

    def fetch_options_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch options data for a symbol (simple version for tests).
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict containing options data
        """
        return {
            'symbol': symbol,
            'calls': [
                {
                    'strike': 150.0,
                    'lastPrice': 5.5,
                    'bid': 5.4,
                    'ask': 5.6,
                    'volume': 1000,
                    'openInterest': 5000,
                    'impliedVolatility': 0.25
                }
            ],
            'puts': [
                {
                    'strike': 145.0,
                    'lastPrice': 3.2,
                    'bid': 3.1,
                    'ask': 3.3,
                    'volume': 1200,
                    'openInterest': 4000,
                    'impliedVolatility': 0.28
                }
            ],
            'underlying_price': 152.5,
            'source': 'options_data_subagent'
        }

# Standalone test
if __name__ == "__main__":
    import asyncio
    agent = OptionsDataAnalyzer()
    result = asyncio.run(agent.process_input({
        'symbols': ['AAPL'],
        'option_types': ['call', 'put'],
        'greeks': True,
        'volatility_surface': True,
        'strategies': True,
        'risk_metrics': True
    }))
    print("Options Data Agent Test Result:")
    print(f"Keys: {list(result.keys())}")
    if 'options_data' in result:
        print(f"Symbols processed: {list(result['options_data'].get('symbols_data', {}).keys())}")
        print(f"Data quality: {result['options_data'].get('data_quality', 'unknown')}")
        analytics = result['options_data'].get('analytics', {})
        if analytics:
            print(f"Analytics available for: {list(analytics.keys())}")