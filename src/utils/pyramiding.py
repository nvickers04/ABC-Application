# src/utils/pyramiding.py
# Purpose: Implements dynamic pyramiding logic for position scaling and risk management.
# Provides intelligent position sizing based on market conditions, volatility, and profit/loss levels.
# Structural Reasoning: Centralized pyramiding logic for consistent risk management across strategies.
# For legacy wealth: Enables compounding gains while managing drawdown risk.

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class PyramidingEngine:
    """
    Dynamic pyramiding engine for intelligent position scaling.
    """

    def __init__(self, max_tiers: int = 5, base_risk_pct: float = 0.02):
        """
        Initialize pyramiding engine.

        Args:
            max_tiers: Maximum number of pyramiding tiers
            base_risk_pct: Base risk percentage per position (2%)
        """
        self.max_tiers = max_tiers
        self.base_risk_pct = base_risk_pct

        # Pyramiding parameters
        self.volatility_multipliers = {
            'low': 1.2,      # Scale up in low volatility
            'normal': 1.0,   # Normal scaling
            'high': 0.7      # Scale down in high volatility
        }

        self.trend_strength_multipliers = {
            'weak': 0.8,     # Conservative in weak trends
            'moderate': 1.0, # Normal in moderate trends
            'strong': 1.4    # Aggressive in strong trends
        }

        # Learning adaptation memory
        self.learning_adaptations = {
            'tier_adjustments': {},
            'scaling_adjustments': {},
            'volatility_adjustments': {},
            'applied_directives': []
        }

    def calculate_pyramiding_plan(self,
                                current_price: float,
                                entry_price: float,
                                volatility: float,
                                trend_strength: float,
                                current_pnl_pct: float,
                                max_drawdown_pct: float,
                                portfolio_value: float) -> Dict[str, Any]:
        """
        Calculate dynamic pyramiding plan based on market conditions.

        Args:
            current_price: Current market price
            entry_price: Original entry price
            volatility: Current volatility (0-1 scale)
            trend_strength: Trend strength (0-1 scale)
            current_pnl_pct: Current profit/loss percentage
            max_drawdown_pct: Maximum allowed drawdown
            portfolio_value: Current portfolio value

        Returns:
            Dict with pyramiding plan including tiers, scaling, and triggers
        """

        # Determine volatility regime
        if volatility < 0.15:
            vol_regime = 'low'
        elif volatility > 0.30:
            vol_regime = 'high'
        else:
            vol_regime = 'normal'

        # Determine trend strength
        if trend_strength < 0.3:
            trend_regime = 'weak'
        elif trend_strength > 0.7:
            trend_regime = 'strong'
        else:
            trend_regime = 'moderate'

        # Calculate base position size
        base_position_size = portfolio_value * self.base_risk_pct

        # Apply volatility and trend adjustments
        vol_multiplier = self.volatility_multipliers[vol_regime]
        trend_multiplier = self.trend_strength_multipliers[trend_regime]

        adjusted_base_size = base_position_size * vol_multiplier * trend_multiplier

        # Calculate pyramiding tiers based on profit levels
        tiers = self._calculate_tiers(current_pnl_pct, vol_regime, trend_regime)

        # Calculate scaling factors for each tier
        scaling_factors = self._calculate_scaling_factors(tiers, current_pnl_pct)

        # Calculate price triggers for each tier
        price_triggers = self._calculate_price_triggers(
            entry_price, current_pnl_pct, tiers, trend_regime
        )

        # Calculate risk management stops
        stops = self.calculate_stops(
            entry_price, current_price, max_drawdown_pct, vol_regime
        )

        pyramiding_plan = {
            'base_position_size': adjusted_base_size,
            'tiers': tiers,
            'scaling_factors': scaling_factors,
            'price_triggers': price_triggers,
            'stops': stops,
            'volatility_regime': vol_regime,
            'trend_regime': trend_regime,
            'total_exposure_limit': adjusted_base_size * sum(scaling_factors),
            'risk_adjusted': True
        }

        logger.info(f"Generated pyramiding plan: {tiers} tiers, total exposure: ${pyramiding_plan['total_exposure_limit']:.2f}")
        return pyramiding_plan

    def _calculate_tiers(self, current_pnl_pct: float, vol_regime: str, trend_regime: str) -> int:
        """
        Calculate number of pyramiding tiers based on current conditions.
        """
        base_tiers = 3  # Default

        # Adjust based on profit level
        if current_pnl_pct > 0.15:  # 15% profit
            base_tiers += 1
        elif current_pnl_pct > 0.25:  # 25% profit
            base_tiers += 2

        # Adjust based on market conditions
        if vol_regime == 'low' and trend_regime == 'strong':
            base_tiers += 1  # More aggressive in favorable conditions
        elif vol_regime == 'high' or trend_regime == 'weak':
            base_tiers -= 1  # More conservative in adverse conditions

        # Ensure within bounds
        return max(1, min(base_tiers, self.max_tiers))

    def _calculate_scaling_factors(self, tiers: int, current_pnl_pct: float) -> List[float]:
        """
        Calculate position scaling factors for each tier.
        """
        factors = [1.0]  # Base position

        # Progressive scaling based on profit
        profit_multiplier = 1.0 + (current_pnl_pct * 0.5)  # Scale up with profits

        for i in range(1, tiers):
            # Each subsequent tier is larger, but with diminishing returns
            scale_factor = profit_multiplier * (1.0 + (i * 0.3))
            factors.append(min(scale_factor, 3.0))  # Cap at 3x base size

        return factors

    def _calculate_price_triggers(self, entry_price: float, current_pnl_pct: float,
                                tiers: int, trend_regime: str) -> List[float]:
        """
        Calculate price levels that trigger each pyramiding tier.
        """
        triggers = [entry_price]  # Initial entry

        # Base trigger increment
        if trend_regime == 'strong':
            trigger_step = 0.05  # 5% steps in strong trends
        elif trend_regime == 'moderate':
            trigger_step = 0.08  # 8% steps in moderate trends
        else:
            trigger_step = 0.12  # 12% steps in weak trends (more conservative)

        for i in range(1, tiers):
            # Progressive triggers based on current profit
            profit_adjustment = current_pnl_pct * 0.3  # Adjust based on current gains
            trigger_price = entry_price * (1 + (i * trigger_step) + profit_adjustment)
            triggers.append(trigger_price)

        return triggers

    def calculate_stops(self, entry_price: float, current_price: float,
                        max_drawdown_pct: float, vol_regime: str) -> Dict[str, float]:
        """
        Calculate risk management stops.
        """
        # Base stop based on volatility
        if vol_regime == 'high':
            stop_pct = max_drawdown_pct * 0.7  # Tighter stops in high vol
        elif vol_regime == 'low':
            stop_pct = max_drawdown_pct * 1.2  # Looser stops in low vol
        else:
            stop_pct = max_drawdown_pct

        # Calculate stop price
        if current_price > entry_price:  # Profitable position
            stop_price = current_price * (1 - stop_pct)
        else:  # Losing position
            stop_price = entry_price * (1 - stop_pct)

        # Trailing stop (moves up with profits)
        trailing_stop_pct = stop_pct * 0.8
        trailing_stop = current_price * (1 - trailing_stop_pct)

        return {
            'initial_stop': stop_price,
            'trailing_stop': trailing_stop,
            'max_drawdown_stop': entry_price * (1 - max_drawdown_pct),
            'volatility_adjusted': True
        }

    def should_add_to_position(self, current_price: float, last_tier_price: float,
                             current_pnl_pct: float, volatility: float) -> bool:
        """
        Determine if conditions are right to add to position (pyramid).
        """
        # Must be profitable
        if current_pnl_pct <= 0:
            return False

        # Price must have moved favorably since last tier
        price_improvement = (current_price - last_tier_price) / last_tier_price

        # Adjust threshold based on volatility
        if volatility > 0.25:  # High volatility
            required_improvement = 0.08  # 8%
        elif volatility < 0.15:  # Low volatility
            required_improvement = 0.05  # 5%
        else:
            required_improvement = 0.06  # 6%

        # Must have sufficient profit buffer
        min_profit_buffer = 0.03 + (volatility * 0.1)  # 3% + vol adjustment

        conditions_met = (
            price_improvement >= required_improvement and
            current_pnl_pct >= min_profit_buffer
        )

        if conditions_met:
            logger.info(f"Pyramiding conditions met: price improvement {price_improvement:.1%}, "
                       f"current PnL {current_pnl_pct:.1%}")

        return conditions_met

    def calculate_take_profit_levels(self, entry_price: float, current_price: float,
                                   tiers: int, risk_reward_ratio: float = 2.0) -> List[float]:
        """
        Calculate take profit levels for partial exits.
        """
        profit_targets = []

        # Base profit target (2:1 reward-to-risk)
        base_target = entry_price * (1 + (abs(current_price - entry_price) / entry_price) * risk_reward_ratio)

        profit_targets.append(base_target)

        # Additional targets for pyramided positions
        for i in range(1, tiers):
            # Scale targets higher for additional tiers
            scaled_target = base_target * (1 + (i * 0.15))  # 15% higher each tier
            profit_targets.append(scaled_target)

        return profit_targets

    def apply_learning_directives(self, directives: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply learning directives to adapt pyramiding parameters.

        Args:
            directives: List of learning directives from LearningAgent

        Returns:
            Dict with applied changes and adaptation summary
        """
        applied_changes = []
        rejected_directives = []

        for directive in directives:
            directive_name = directive.get('refinement', '')
            value = directive.get('value', 1.0)
            reason = directive.get('reason', 'No reason provided')

            try:
                if directive_name.startswith('pyramiding_'):
                    change_applied = self._apply_pyramiding_directive(directive_name, value, reason)
                    if change_applied:
                        applied_changes.append({
                            'directive': directive_name,
                            'value': value,
                            'reason': reason,
                            'timestamp': pd.Timestamp.now()
                        })
                    else:
                        rejected_directives.append(directive)
                else:
                    rejected_directives.append(directive)  # Not a pyramiding directive

            except Exception as e:
                logger.warning(f"Failed to apply directive {directive_name}: {e}")
                rejected_directives.append(directive)

        # Store applied directives for tracking
        self.learning_adaptations['applied_directives'].extend(applied_changes)

        # Keep only last 20 applied directives
        if len(self.learning_adaptations['applied_directives']) > 20:
            self.learning_adaptations['applied_directives'] = self.learning_adaptations['applied_directives'][-20:]

        adaptation_summary = {
            'applied_changes': len(applied_changes),
            'rejected_directives': len(rejected_directives),
            'total_directives': len(directives),
            'learning_active': True
        }

        logger.info(f"Applied {len(applied_changes)} learning directives to pyramiding engine")
        return adaptation_summary

    def _apply_pyramiding_directive(self, directive_name: str, value: float, reason: str) -> bool:
        """
        Apply a specific pyramiding learning directive.

        Args:
            directive_name: Name of the directive
            value: Directive value (multiplier)
            reason: Reason for the directive

        Returns:
            True if directive was applied successfully
        """
        if 'tier_boost' in directive_name:
            # Increase maximum tiers
            old_max = self.max_tiers
            self.max_tiers = int(self.max_tiers * value)
            self.learning_adaptations['tier_adjustments']['max_tiers'] = self.max_tiers
            logger.info(f"Tier boost: {old_max} -> {self.max_tiers} ({reason})")
            return True

        elif 'conservative_tiers' in directive_name:
            # Decrease maximum tiers for safety
            old_max = self.max_tiers
            self.max_tiers = max(2, int(self.max_tiers * value))  # Minimum 2 tiers
            self.learning_adaptations['tier_adjustments']['max_tiers'] = self.max_tiers
            logger.info(f"Conservative tiers: {old_max} -> {self.max_tiers} ({reason})")
            return True

        elif 'aggressive_scaling' in directive_name:
            # Increase scaling factors
            for regime in self.volatility_multipliers:
                old_value = self.volatility_multipliers[regime]
                self.volatility_multipliers[regime] *= value
                self.learning_adaptations['scaling_adjustments'][f'vol_{regime}'] = self.volatility_multipliers[regime]
                logger.info(f"Aggressive scaling {regime}: {old_value:.2f} -> {self.volatility_multipliers[regime]:.2f}")
            return True

        elif 'conservative_scaling' in directive_name:
            # Decrease scaling factors for safety
            for regime in self.volatility_multipliers:
                old_value = self.volatility_multipliers[regime]
                self.volatility_multipliers[regime] *= value
                self.learning_adaptations['scaling_adjustments'][f'vol_{regime}'] = self.volatility_multipliers[regime]
                logger.info(f"Conservative scaling {regime}: {old_value:.2f} -> {self.volatility_multipliers[regime]:.2f}")
            return True

        elif directive_name.startswith('pyramiding_vol_') and '_boost' in directive_name:
            # Boost specific volatility regime
            regime = directive_name.split('_')[2]  # Extract regime name
            if regime in self.volatility_multipliers:
                old_value = self.volatility_multipliers[regime]
                self.volatility_multipliers[regime] *= value
                self.learning_adaptations['volatility_adjustments'][regime] = self.volatility_multipliers[regime]
                logger.info(f"Volatility boost {regime}: {old_value:.2f} -> {self.volatility_multipliers[regime]:.2f} ({reason})")
                return True

        elif directive_name.startswith('pyramiding_vol_') and '_conserve' in directive_name:
            # Conservative adjustment for specific volatility regime
            regime = directive_name.split('_')[2]  # Extract regime name
            if regime in self.volatility_multipliers:
                old_value = self.volatility_multipliers[regime]
                self.volatility_multipliers[regime] *= value
                self.learning_adaptations['volatility_adjustments'][regime] = self.volatility_multipliers[regime]
                logger.info(f"Volatility conserve {regime}: {old_value:.2f} -> {self.volatility_multipliers[regime]:.2f} ({reason})")
                return True

        return False  # Directive not recognized

    def get_learning_status(self) -> Dict[str, Any]:
        """
        Get current learning adaptation status.

        Returns:
            Dict with current learning state and adaptations
        """
        return {
            'learning_active': True,
            'current_max_tiers': self.max_tiers,
            'current_volatility_multipliers': self.volatility_multipliers.copy(),
            'adaptations_applied': len(self.learning_adaptations['applied_directives']),
            'recent_directives': self.learning_adaptations['applied_directives'][-5:],  # Last 5
            'tier_adjustments': self.learning_adaptations['tier_adjustments'].copy(),
            'scaling_adjustments': self.learning_adaptations['scaling_adjustments'].copy(),
            'volatility_adjustments': self.learning_adaptations['volatility_adjustments'].copy()
        }

    def reset_learning_adaptations(self) -> None:
        """
        Reset all learning adaptations to baseline parameters.
        Useful for testing or when learning goes wrong.
        """
        # Reset to baseline parameters
        self.max_tiers = 5
        self.volatility_multipliers = {
            'low': 1.2,
            'normal': 1.0,
            'high': 0.7
        }

        # Clear adaptation memory
        self.learning_adaptations = {
            'tier_adjustments': {},
            'scaling_adjustments': {},
            'volatility_adjustments': {},
            'applied_directives': []
        }

        logger.info("Pyramiding engine learning adaptations reset to baseline")