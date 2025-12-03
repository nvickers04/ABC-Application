#!/usr/bin/env python3
"""
Volatility Calculator for ABC-Application

Provides proper volatility calculations for risk management and position sizing.
Implements various volatility measures including historical volatility, realized volatility,
and volatility-adjusted position sizing.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class VolatilityMethod(Enum):
    """Volatility calculation methods"""
    CLOSE_TO_CLOSE = "close_to_close"  # Standard historical volatility
    PARKINSON = "parkinson"  # Range-based volatility
    GARMAN_KLASS = "garman_klass"  # OHLC-based volatility
    ROGERS_SATCHELL = "rogers_satchell"  # OHLC with drift
    YANG_ZHANG = "yang_zhang"  # Overnight and intraday components


@dataclass
class VolatilityResult:
    """Result of volatility calculation"""
    symbol: str
    method: VolatilityMethod
    volatility: float  # Annualized volatility (decimal)
    daily_volatility: float  # Daily volatility (decimal)
    confidence_interval: Tuple[float, float]  # 95% confidence bounds
    data_points: int
    calculation_date: datetime
    metadata: Dict[str, Any]


class VolatilityCalculator:
    """
    Advanced volatility calculator for risk management

    Supports multiple volatility calculation methods and provides
    confidence intervals and statistical measures.
    """

    def __init__(self, trading_days_per_year: int = 252):
        self.trading_days_per_year = trading_days_per_year
        self.cache = {}  # Simple cache for recent calculations

    def calculate_volatility(self, symbol: str, price_data: List[Dict[str, Any]],
                           method: VolatilityMethod = VolatilityMethod.CLOSE_TO_CLOSE,
                           window_days: int = 30) -> Optional[VolatilityResult]:
        """
        Calculate volatility using specified method

        Args:
            symbol: Trading symbol
            price_data: List of OHLC price data (newest first)
            method: Volatility calculation method
            window_days: Number of days to use for calculation

        Returns:
            VolatilityResult or None if calculation fails
        """
        try:
            if not price_data or len(price_data) < 2:
                logger.warning(f"Insufficient price data for {symbol}")
                return None

            # Convert to DataFrame for easier manipulation
            df = self._prepare_price_dataframe(price_data)

            if len(df) < 2:
                logger.warning(f"Insufficient valid price data for {symbol}")
                return None

            # Limit to window_days if we have more data
            if len(df) > window_days:
                df = df.tail(window_days)

            # Calculate volatility based on method
            if method == VolatilityMethod.CLOSE_TO_CLOSE:
                volatility = self._calculate_close_to_close_volatility(df)
            elif method == VolatilityMethod.PARKINSON:
                volatility = self._calculate_parkinson_volatility(df)
            elif method == VolatilityMethod.GARMAN_KLASS:
                volatility = self._calculate_garman_klass_volatility(df)
            elif method == VolatilityMethod.ROGERS_SATCHELL:
                volatility = self._calculate_rogers_satchell_volatility(df)
            elif method == VolatilityMethod.YANG_ZHANG:
                volatility = self._calculate_yang_zhang_volatility(df)
            else:
                logger.error(f"Unknown volatility method: {method}")
                return None

            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(df, volatility)

            # Annualize the volatility
            daily_vol = volatility
            annualized_vol = volatility * np.sqrt(self.trading_days_per_year)

            result = VolatilityResult(
                symbol=symbol,
                method=method,
                volatility=annualized_vol,
                daily_volatility=daily_vol,
                confidence_interval=confidence_interval,
                data_points=len(df),
                calculation_date=datetime.now(),
                metadata={
                    'window_days': window_days,
                    'method_details': method.value,
                    'data_quality': self._assess_data_quality(df)
                }
            )

            # Cache result
            cache_key = f"{symbol}_{method.value}_{window_days}"
            self.cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"Volatility calculation failed for {symbol}: {e}")
            return None

    def _prepare_price_dataframe(self, price_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare price data into a clean DataFrame"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(price_data)

            # Ensure we have required columns
            required_cols = ['close']
            optional_cols = ['open', 'high', 'low']

            if 'close' not in df.columns:
                # Try alternative column names
                if 'price' in df.columns:
                    df['close'] = df['price']
                else:
                    raise ValueError("No close price data found")

            # Fill missing OHLC data with close price if needed
            for col in optional_cols:
                if col not in df.columns:
                    df[col] = df['close']

            # Convert to numeric and handle missing values
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows with missing close prices
            df = df.dropna(subset=['close'])

            # Sort by date if available (assuming data is in reverse chronological order)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')

            return df

        except Exception as e:
            logger.error(f"Failed to prepare price DataFrame: {e}")
            return pd.DataFrame()

    def _calculate_close_to_close_volatility(self, df: pd.DataFrame) -> float:
        """Calculate standard close-to-close historical volatility"""
        try:
            # Calculate log returns
            returns = np.log(df['close'] / df['close'].shift(1))
            returns = returns.dropna()

            if len(returns) < 2:
                return 0.02  # Default 2% daily volatility

            # Calculate standard deviation of returns
            volatility = returns.std()

            return max(volatility, 0.001)  # Minimum volatility floor

        except Exception as e:
            logger.warning(f"Close-to-close volatility calculation failed: {e}")
            return 0.02

    def _calculate_parkinson_volatility(self, df: pd.DataFrame) -> float:
        """Calculate Parkinson range-based volatility"""
        try:
            # Parkinson volatility: sqrt(1/(4*N*ln(2)) * sum(ln(H/L)^2))
            if 'high' not in df.columns or 'low' not in df.columns:
                return self._calculate_close_to_close_volatility(df)

            # Calculate log of high/low ratio squared
            log_hl = np.log(df['high'] / df['low']) ** 2
            log_hl = log_hl.dropna()

            if len(log_hl) == 0:
                return self._calculate_close_to_close_volatility(df)

            # Parkinson formula
            parkinson_vol = np.sqrt((1 / (4 * len(log_hl) * np.log(2))) * log_hl.sum())

            return max(parkinson_vol, 0.001)

        except Exception as e:
            logger.warning(f"Parkinson volatility calculation failed: {e}")
            return self._calculate_close_to_close_volatility(df)

    def _calculate_garman_klass_volatility(self, df: pd.DataFrame) -> float:
        """Calculate Garman-Klass OHLC volatility"""
        try:
            if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                return self._calculate_close_to_close_volatility(df)

            # Garman-Klass formula components
            log_hl = np.log(df['high'] / df['low']) ** 2
            log_co = np.log(df['close'] / df['open']) ** 2

            # Combined calculation
            n = len(df)
            if n == 0:
                return 0.02

            gk_vol = np.sqrt((0.5 * log_hl.sum() - (2 * np.log(2) - 1) * log_co.sum()) / n)

            return max(gk_vol, 0.001)

        except Exception as e:
            logger.warning(f"Garman-Klass volatility calculation failed: {e}")
            return self._calculate_close_to_close_volatility(df)

    def _calculate_rogers_satchell_volatility(self, df: pd.DataFrame) -> float:
        """Calculate Rogers-Satchell volatility (with drift)"""
        try:
            if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                return self._calculate_close_to_close_volatility(df)

            # Rogers-Satchell components
            log_ho = np.log(df['high'] / df['open'])
            log_lo = np.log(df['low'] / df['open'])
            log_co = np.log(df['close'] / df['open'])

            # RS formula
            rs_components = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
            rs_vol = np.sqrt(rs_components.sum() / len(df))

            return max(rs_vol, 0.001)

        except Exception as e:
            logger.warning(f"Rogers-Satchell volatility calculation failed: {e}")
            return self._calculate_close_to_close_volatility(df)

    def _calculate_yang_zhang_volatility(self, df: pd.DataFrame) -> float:
        """Calculate Yang-Zhang volatility (overnight + intraday)"""
        try:
            if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                return self._calculate_close_to_close_volatility(df)

            # Yang-Zhang combines overnight and intraday volatility
            # Simplified version - in practice this requires open-to-open data
            close_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
            open_returns = np.log(df['open'] / df['close'].shift(1)).dropna()

            if len(close_returns) < 2 or len(open_returns) < 2:
                return self._calculate_close_to_close_volatility(df)

            # Estimate overnight and intraday components
            overnight_vol = open_returns.std()
            intraday_vol = self._calculate_rogers_satchell_volatility(df)

            # Yang-Zhang combination (simplified)
            total_vol = np.sqrt(overnight_vol**2 + intraday_vol**2)

            return max(total_vol, 0.001)

        except Exception as e:
            logger.warning(f"Yang-Zhang volatility calculation failed: {e}")
            return self._calculate_close_to_close_volatility(df)

    def _calculate_confidence_interval(self, df: pd.DataFrame, volatility: float) -> Tuple[float, float]:
        """Calculate 95% confidence interval for volatility estimate"""
        try:
            n = len(df)
            if n < 2:
                return (volatility * 0.8, volatility * 1.2)

            # Use t-distribution approximation for confidence interval
            # For large n, approximately normal
            z_score = 1.96  # 95% confidence
            standard_error = volatility / np.sqrt(n)

            lower_bound = max(volatility - z_score * standard_error, 0.001)
            upper_bound = volatility + z_score * standard_error

            return (lower_bound, upper_bound)

        except Exception as e:
            logger.warning(f"Confidence interval calculation failed: {e}")
            return (volatility * 0.8, volatility * 1.2)

    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess the quality of input data"""
        quality = {
            'total_points': len(df),
            'missing_data': {},
            'data_completeness': 1.0
        }

        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col in df.columns:
                missing = df[col].isna().sum()
                quality['missing_data'][col] = missing
                if missing > 0:
                    quality['data_completeness'] *= (len(df) - missing) / len(df)

        return quality

    def get_cached_volatility(self, symbol: str, method: VolatilityMethod,
                            window_days: int) -> Optional[VolatilityResult]:
        """Get cached volatility result if available and recent"""
        cache_key = f"{symbol}_{method.value}_{window_days}"
        result = self.cache.get(cache_key)

        if result:
            # Check if cache is still valid (within last hour)
            if (datetime.now() - result.calculation_date).total_seconds() < 3600:
                return result
            else:
                # Remove stale cache
                del self.cache[cache_key]

        return None

    def clear_cache(self):
        """Clear the volatility calculation cache"""
        self.cache.clear()


# Global instance
_volatility_calculator: Optional[VolatilityCalculator] = None


def get_volatility_calculator() -> VolatilityCalculator:
    """Get singleton volatility calculator instance"""
    global _volatility_calculator
    if _volatility_calculator is None:
        _volatility_calculator = VolatilityCalculator()
    return _volatility_calculator


# Convenience functions
def calculate_symbol_volatility(symbol: str, price_data: List[Dict[str, Any]],
                              method: VolatilityMethod = VolatilityMethod.CLOSE_TO_CLOSE,
                              window_days: int = 30) -> Optional[VolatilityResult]:
    """Convenience function to calculate volatility for a symbol"""
    calculator = get_volatility_calculator()
    return calculator.calculate_volatility(symbol, price_data, method, window_days)


def get_volatility_adjusted_position_size(account_value: float, volatility: float,
                                        base_risk_pct: float = 0.005,
                                        max_position_pct: float = 0.1) -> float:
    """
    Calculate volatility-adjusted position size

    Args:
        account_value: Total account value
        volatility: Annualized volatility (decimal)
        base_risk_pct: Base risk percentage per position
        max_position_pct: Maximum position size as % of account

    Returns:
        Maximum position value
    """
    try:
        # Adjust risk based on volatility
        # Higher volatility = lower position size
        vol_adjustment = max(0.1, 1.0 - (volatility - 0.2))  # Reduce size for vol > 20%

        adjusted_risk_pct = base_risk_pct * vol_adjustment
        max_position_value = account_value * min(adjusted_risk_pct, max_position_pct)

        return max_position_value

    except Exception as e:
        logger.warning(f"Position size calculation failed: {e}")
        return account_value * base_risk_pct