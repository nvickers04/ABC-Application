# src/utils/timing_optimizer.py

"""
TimingOptimizer for liquidity-based trade execution timing.
Delays entries 15-30 min in low-liquidity windows (open/close) and responds to VIX >18.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from typing import Dict, Any, Optional
from datetime import datetime, time, timedelta
import asyncio

logger = logging.getLogger(__name__)

class TimingOptimizer:
    """
    Optimizes trade execution timing based on liquidity and market conditions.
    Delays entries during low-liquidity periods and responds to high volatility.
    """

    def __init__(self):
        # Low-liquidity time windows (market open/close)
        self.low_liquidity_windows = [
            (time(9, 30), time(10, 0)),   # Market open
            (time(15, 30), time(16, 0))  # Market close
        ]

        # VIX thresholds for timing adjustments
        self.vix_thresholds = {
            'normal': 18,      # Below 18: normal timing
            'elevated': 25,    # 18-25: delay 15 min
            'high': 30,        # 25-30: delay 30 min
            'extreme': 35      # Above 35: delay 60 min or avoid
        }

        # Default delay durations
        self.delay_durations = {
            'low_liquidity': timedelta(minutes=15),
            'elevated_volatility': timedelta(minutes=15),
            'high_volatility': timedelta(minutes=30),
            'extreme_volatility': timedelta(minutes=60)
        }

        logger.info("TimingOptimizer initialized")

    async def optimize_execution_timing(self, symbol: str, order_details: Dict[str, Any],
                                      market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize execution timing based on current market conditions.

        Args:
            symbol: Trading symbol
            order_details: Order details (quantity, action, etc.)
            market_data: Current market data including VIX

        Returns:
            Dict with timing recommendation and delay information
        """
        try:
            logger.info(f"Optimizing execution timing for {symbol}")

            # Get current time and VIX level
            current_time = datetime.now().time()
            vix_level = market_data.get('VIX', {}).get('price', 18.0)

            # Check if current time is in low-liquidity window
            in_low_liquidity_window = self._is_low_liquidity_window(current_time)

            # Determine volatility regime
            volatility_regime = self._determine_volatility_regime(vix_level)

            # Calculate recommended delay
            delay_recommendation = self._calculate_delay(
                in_low_liquidity_window, volatility_regime, current_time
            )

            # Check if execution should be delayed or avoided
            should_execute_now = self._should_execute_immediately(delay_recommendation, volatility_regime)

            timing_optimization = {
                'symbol': symbol,
                'current_time': current_time.isoformat(),
                'vix_level': vix_level,
                'volatility_regime': volatility_regime,
                'in_low_liquidity_window': in_low_liquidity_window,
                'recommended_delay': delay_recommendation['delay_minutes'],
                'delay_reason': delay_recommendation['reason'],
                'should_execute_now': should_execute_now,
                'optimal_execution_time': self._calculate_optimal_time(current_time, delay_recommendation),
                'liquidity_score': self._calculate_liquidity_score(current_time, vix_level),
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"Timing optimization for {symbol}: delay {delay_recommendation['delay_minutes']} min, "
                       f"execute_now={should_execute_now}")

            return timing_optimization

        except Exception as e:
            logger.error(f"Error optimizing execution timing: {e}")
            # Default to immediate execution on error
            return {
                'symbol': symbol,
                'error': str(e),
                'should_execute_now': True,
                'recommended_delay': 0,
                'delay_reason': 'timing_optimization_error'
            }

    def _is_low_liquidity_window(self, current_time: time) -> bool:
        """
        Check if current time falls within low-liquidity windows.
        """
        for start_time, end_time in self.low_liquidity_windows:
            if start_time <= current_time <= end_time:
                return True
        return False

    def _determine_volatility_regime(self, vix_level: float) -> str:
        """
        Determine volatility regime based on VIX level.
        """
        if vix_level >= self.vix_thresholds['extreme']:
            return 'extreme'
        elif vix_level >= self.vix_thresholds['high']:
            return 'high'
        elif vix_level >= self.vix_thresholds['elevated']:
            return 'elevated'
        else:
            return 'normal'

    def _calculate_delay(self, in_low_liquidity: bool, volatility_regime: str,
                        current_time: time) -> Dict[str, Any]:
        """
        Calculate recommended execution delay based on conditions.
        """
        delay_minutes = 0
        reasons = []

        # Low liquidity delay
        if in_low_liquidity:
            delay_minutes = max(delay_minutes, self.delay_durations['low_liquidity'].seconds // 60)
            reasons.append('low_liquidity_window')

        # Volatility-based delay
        if volatility_regime == 'extreme':
            delay_minutes = max(delay_minutes, self.delay_durations['extreme_volatility'].seconds // 60)
            reasons.append('extreme_volatility')
        elif volatility_regime == 'high':
            delay_minutes = max(delay_minutes, self.delay_durations['high_volatility'].seconds // 60)
            reasons.append('high_volatility')
        elif volatility_regime == 'elevated':
            delay_minutes = max(delay_minutes, self.delay_durations['elevated_volatility'].seconds // 60)
            reasons.append('elevated_volatility')

        # Cap maximum delay at 60 minutes
        delay_minutes = min(delay_minutes, 60)

        return {
            'delay_minutes': delay_minutes,
            'reason': ', '.join(reasons) if reasons else 'optimal_conditions'
        }

    def _should_execute_immediately(self, delay_recommendation: Dict[str, Any],
                                  volatility_regime: str) -> bool:
        """
        Determine if order should execute immediately or be delayed.
        """
        # Execute immediately if no delay recommended
        if delay_recommendation['delay_minutes'] == 0:
            return True

        # In extreme volatility, may need to avoid execution entirely
        if volatility_regime == 'extreme' and delay_recommendation['delay_minutes'] >= 60:
            return False  # Suggest avoiding execution

        # Otherwise, delay execution
        return False

    def _calculate_optimal_time(self, current_time: time, delay_recommendation: Dict[str, Any]) -> Optional[str]:
        """
        Calculate the optimal execution time after delay.
        """
        if delay_recommendation['delay_minutes'] == 0:
            return None

        current_datetime = datetime.combine(datetime.today(), current_time)
        optimal_datetime = current_datetime + timedelta(minutes=delay_recommendation['delay_minutes'])

        # Check if optimal time is still within market hours
        market_close = time(16, 0)  # 4:00 PM ET
        if optimal_datetime.time() > market_close:
            return None  # Cannot execute today

        return optimal_datetime.time().isoformat()

    def _calculate_liquidity_score(self, current_time: time, vix_level: float) -> float:
        """
        Calculate liquidity score from 0-1 (higher is better liquidity).
        """
        try:
            # Base score starts at 1.0 (good liquidity)
            liquidity_score = 1.0

            # Reduce score during low liquidity windows
            if self._is_low_liquidity_window(current_time):
                liquidity_score -= 0.3  # 30% reduction

            # Reduce score based on VIX level
            if vix_level > self.vix_thresholds['extreme']:
                liquidity_score -= 0.4  # 40% reduction in extreme vol
            elif vix_level > self.vix_thresholds['high']:
                liquidity_score -= 0.3  # 30% reduction in high vol
            elif vix_level > self.vix_thresholds['elevated']:
                liquidity_score -= 0.2  # 20% reduction in elevated vol

            # Ensure score stays within bounds
            liquidity_score = max(0.0, min(1.0, liquidity_score))

            return liquidity_score

        except Exception as e:
            logger.warning(f"Error calculating liquidity score: {e}")
            return 0.5  # Neutral score on error

    async def get_timing_status(self, symbol: str) -> Dict[str, Any]:
        """
        Get current timing status for a symbol.
        """
        try:
            # This would integrate with market data to get current conditions
            # For now, return basic status
            current_time = datetime.now().time()

            return {
                'symbol': symbol,
                'current_time': current_time.isoformat(),
                'in_low_liquidity_window': self._is_low_liquidity_window(current_time),
                'market_hours': self._is_market_hours(current_time),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting timing status: {e}")
            return {'error': str(e)}

    def _is_market_hours(self, current_time: time) -> bool:
        """
        Check if current time is within regular market hours.
        """
        market_open = time(9, 30)  # 9:30 AM ET
        market_close = time(16, 0)  # 4:00 PM ET

        return market_open <= current_time <= market_close

    async def update_vix_response(self, vix_level: float) -> Dict[str, Any]:
        """
        Update timing behavior based on VIX level changes.
        """
        try:
            volatility_regime = self._determine_volatility_regime(vix_level)

            # Adjust delay durations based on VIX
            if volatility_regime == 'extreme':
                self.delay_durations['extreme_volatility'] = timedelta(minutes=60)
            elif volatility_regime == 'high':
                self.delay_durations['high_volatility'] = timedelta(minutes=30)
            elif volatility_regime == 'elevated':
                self.delay_durations['elevated_volatility'] = timedelta(minutes=15)
            else:
                self.delay_durations['elevated_volatility'] = timedelta(minutes=15)

            logger.info(f"Updated VIX response: regime={volatility_regime}, vix={vix_level}")

            return {
                'volatility_regime': volatility_regime,
                'vix_level': vix_level,
                'updated_delays': {
                    key: value.seconds // 60 for key, value in self.delay_durations.items()
                },
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error updating VIX response: {e}")
            return {'error': str(e)}