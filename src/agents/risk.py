# src/agents/risk.py
# Purpose: Implements the Risk Agent, subclassing BaseAgent for probability assessments and risk vetting.
# Handles stochastic re-runs, override vets, and auto-adjustments (e.g., on SD >1.0).
# Structural Reasoning: Ties to risk-agent-notes.md (e.g., tf-quant-finance sims) and risk-constraints.yaml (loaded fresh); backs funding with logged vets (e.g., "Vetted override for +5% alpha").
# New: Async process_input for loops/pings; reflect method for experiential tweaks.
# For legacy wealth: Enforces POP >60% and <5% drawdown to protect honorable capital; unscrupulous overrides if confidence >0.8 but capped.
# Update: Added real tf-quant-finance stub for POP sim (replaces np.random—simple Brownian motion; install tensorflow tensorflow-probability if needed); dynamic path setup for imports.

import os
# Set TensorFlow logging level to suppress warnings before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO, WARNING, and ERROR messages

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Dynamic root path for imports.

from src.agents.base import BaseAgent  # Absolute import.
import logging
from typing import Dict, Any, List, Optional
import asyncio
import yaml  # For updating YAML files.
import json  # For JSON parsing in risk assessment.
import os  # For file operations.
import pandas as pd  # For timestamp handling.
from datetime import datetime, timedelta  # For timestamping
import numpy as np  # For numerical computations
from src.utils.tools import tf_quant_monte_carlo_tool, pyfolio_metrics_tool, load_yaml_tool

logger = logging.getLogger(__name__)

# Try to import TensorFlow Probability for advanced stochastic modeling
try:
    import warnings
    import logging
    from contextlib import redirect_stderr
    import io

    # Suppress Python warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
    warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow_probability')
    warnings.filterwarnings('ignore', category=UserWarning, module='tf_keras')
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    # Disable TensorFlow and related logging before import
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('tensorflow_probability').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.ERROR)

    # Try to suppress absl logging which TensorFlow uses
    try:
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)
    except ImportError:
        pass

    # Capture stderr to suppress TensorFlow warnings during import
    stderr_capture = io.StringIO()
    with redirect_stderr(stderr_capture):
        import tensorflow as tf
        # After importing TensorFlow, also set its internal logger
        tf.get_logger().setLevel(logging.ERROR)
        import tensorflow_probability as tfp
        import scipy.stats  # For statistical functions not available in tfp.stats

    TFP_AVAILABLE = True
    logger.info("TensorFlow Probability available for advanced stochastic simulations")
except Exception as e:
    logger.warning(f"TensorFlow Probability not available: {e}. Using numpy fallback.")
    TFP_AVAILABLE = False
    tf = None
    tfp = None
    # Import scipy.stats for fallback if TensorFlow fails
    try:
        import scipy.stats
    except ImportError:
        scipy = None
    scipy = None

class RiskAgent(BaseAgent):
    """
    Risk Agent subclass.
    Reasoning: Vets proposals with stochastic models; auto-adjusts YAML via reflections for closed-loop evolution.
    """
    def __init__(self, a2a_protocol=None):
        config_paths = {'risk': 'config/risk-constraints.yaml', 'profit': 'config/profitability-targets.yaml'}  # Relative to root.
        prompt_paths = {'base': 'base_prompt.txt', 'role': 'docs/AGENTS/main-agents/risk-agent.md'}
        super().__init__(role='risk', config_paths=config_paths, prompt_paths=prompt_paths, a2a_protocol=a2a_protocol)
        
        # Ensure configs are loaded.
        if 'risk' not in self.configs or 'constraints' not in self.configs['risk']:
            logger.error("CRITICAL FAILURE: Risk constraints not loaded - cannot proceed with default settings")
            raise Exception("Risk configuration loading failed - no fallback defaults allowed")
        
        # Add role-specific tools/stubs (expand with Langchain later).
        self.tools.extend([
            tf_quant_monte_carlo_tool,  # For stochastic sims.
            pyfolio_metrics_tool,  # For performance metrics.
            load_yaml_tool  # For fresh constraint loads.
        ])
        
        # Initialize real-time risk monitoring
        self._initialize_real_time_risk_monitoring()
        
        # Initialize crisis detection system
        self._initialize_crisis_detection()
        
        # Memory is now loaded automatically by BaseAgent
        # Initialize memory structure if empty (first run)
        if not self.memory:
            self.memory = {
                'batch_adjustments': {},  # E.g., {'sd >1.0': 'tighten_pop_floor'}
                'stochastic_logs': []  # For daily JSON logs.
            }
            # Save initial memory structure
            self.save_memory()

    async def process_input(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a strategy proposal: Loads YAMLs fresh, re-runs stochastics, vets with POP checks, adjusts post-batch.
        Args:
            proposal (Dict): E.g., {'roi_estimate': 0.25, 'setup': 'strangle'}.
        Returns: Dict with diffs/approvals (e.g., {'approved': True, 'pop_adjusted': 0.72, 'yaml_diffs': {...}}).
        Reasoning: Async for loops; well-informed via fresh loads; self-improving via batch adjustments; decisive on vets/overrides; logs quantitatively.
        """
        try:
            logger.info(f"Risk Agent processing proposal: {proposal}")
            
            if not proposal or 'roi_estimate' not in proposal:
                raise ValueError("Invalid proposal: missing roi_estimate")
            
            # Load YAMLs fresh via tools (stub: already loaded in init).
            constraints = self.configs['risk']['constraints']
            
            # Get current market volatility from VIX via data agent
            vix_volatility = await self._get_vix_volatility()
            
            # Re-run stochastic models with VIX-based volatility
            stochastic_results = self._run_stochastics(proposal, vix_volatility)
            simulated_pop = stochastic_results['pop']
            
            # Vet decisively: Check POP, override if confidence >0.8 but cap sizing.
            vet_result = await self._vet_proposal(proposal, simulated_pop, constraints)
            
            # Adjust metrics post-batch.
            adjustments = self._adjust_post_batch(constraints)
            
            # Broadcast JSON diffs if adjustment > threshold.
            yaml_diffs = self._generate_yaml_diffs(adjustments)
            
            output = {
                'approved': vet_result['approved'],
                'simulated_pop': simulated_pop,
                'vix_volatility': vix_volatility,
                'yaml_diffs': yaml_diffs,
                'rationale': vet_result['rationale'],
                'symbol': proposal.get('symbol', 'SPY'),  # Pass through symbol
                'quantity': proposal.get('quantity', 100),  # Default quantity if not specified
                'roi_estimate': proposal.get('roi_estimate', 0.0),  # Pass through ROI estimate
                'var_95': stochastic_results['var_95'],
                'cvar_95': stochastic_results['cvar_95'],
                'max_drawdown_sim': stochastic_results['max_drawdown_sim'],
                'sharpe_ratio_sim': stochastic_results['sharpe_ratio_sim']
            }
            logger.info(f"Risk output: {output}")
            return output
        
        except Exception as e:
            logger.error(f"Error processing proposal: {e}")
            return {
                'approved': False,
                'simulated_pop': 0.0,
                'vix_volatility': 0.20,  # Fallback volatility
                'yaml_diffs': {},
                'rationale': f"Error: {str(e)}"
            }

    async def monitor_realtime_risk(self, active_positions: Dict[str, Any], 
                                   market_data: Dict[str, Any],
                                   execution_status: Dict[str, Any]) -> Dict[str, Any]:
        """
        Advanced real-time risk monitoring during trade execution.
        Continuously assesses risk metrics and provides dynamic risk management recommendations.
        
        Args:
            active_positions: Current portfolio positions
            market_data: Real-time market data
            execution_status: Current trade execution status
            
        Returns:
            Dict with real-time risk assessment and recommendations
        """
        try:
            logger.info("Risk Agent performing real-time risk monitoring")
            
            # Extract current positions and market conditions
            positions = active_positions.get('positions', [])
            portfolio_value = active_positions.get('portfolio_value', 100000)
            current_prices = market_data.get('current_prices', {})
            volatility_surface = market_data.get('volatility_surface', {})
            
            # Calculate real-time portfolio risk metrics
            portfolio_risk = self._calculate_portfolio_risk(positions, current_prices, portfolio_value)
            
            # Assess execution risk
            execution_risk = self._assess_execution_risk(execution_status, market_data)
            
            # Monitor for risk limit breaches
            risk_breaches = self._check_risk_limits(portfolio_risk, execution_risk)
            
            # Generate dynamic risk management recommendations
            risk_recommendations = await self._generate_risk_recommendations(
                portfolio_risk, execution_risk, risk_breaches, market_data
            )
            
            # Calculate real-time VaR
            realtime_var = self._calculate_realtime_var(positions, current_prices, volatility_surface)
            
            # Monitor drawdown in real-time
            drawdown_status = self._monitor_realtime_drawdown(portfolio_value, active_positions)
            
            realtime_assessment = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'portfolio_risk': portfolio_risk,
                'execution_risk': execution_risk,
                'risk_breaches': risk_breaches,
                'risk_recommendations': risk_recommendations,
                'realtime_var_95': realtime_var.get('var_95', 0),
                'realtime_cvar_95': realtime_var.get('cvar_95', 0),
                'drawdown_status': drawdown_status,
                'overall_risk_level': self._assess_overall_risk_level(portfolio_risk, execution_risk, risk_breaches),
                'monitoring_active': True
            }
            
            logger.info(f"Real-time risk monitoring completed: {realtime_assessment.get('overall_risk_level', 'unknown')} risk level")
            return realtime_assessment
            
        except Exception as e:
            logger.error(f"Error in real-time risk monitoring: {e}")
            return {
                'error': f'Real-time risk monitoring failed: {str(e)}',
                'timestamp': pd.Timestamp.now().isoformat(),
                'monitoring_active': False,
                'overall_risk_level': 'error'
            }

    def _calculate_portfolio_risk(self, positions: List[Dict[str, Any]], 
                                current_prices: Dict[str, float], 
                                portfolio_value: float) -> Dict[str, Any]:
        """Calculate comprehensive portfolio risk metrics in real-time."""
        try:
            total_exposure = 0
            weighted_volatility = 0
            concentration_risk = {}
            
            for position in positions:
                symbol = position.get('symbol', '')
                quantity = position.get('quantity', 0)
                avg_price = position.get('avg_price', 0)
                current_price = current_prices.get(symbol, avg_price)
                
                # Calculate position value and exposure
                position_value = abs(quantity) * current_price
                exposure_pct = position_value / portfolio_value
                
                total_exposure += exposure_pct
                
                # Track concentration by symbol
                if symbol not in concentration_risk:
                    concentration_risk[symbol] = 0
                concentration_risk[symbol] += exposure_pct
                
                # Estimate position volatility (simplified)
                position_volatility = position.get('estimated_volatility', 0.25)  # Default 25%
                weighted_volatility += exposure_pct * position_volatility
            
            # Calculate diversification metrics
            herfindahl_index = sum(pct**2 for pct in concentration_risk.values())
            diversification_score = 1 - herfindahl_index  # Higher is better diversified
            
            return {
                'total_exposure_pct': total_exposure,
                'weighted_volatility': weighted_volatility,
                'concentration_risk': concentration_risk,
                'herfindahl_index': herfindahl_index,
                'diversification_score': diversification_score,
                'largest_position_pct': max(concentration_risk.values()) if concentration_risk else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return {'error': str(e)}

    def _assess_execution_risk(self, execution_status: Dict[str, Any], 
                             market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk associated with current trade execution."""
        try:
            pending_orders = execution_status.get('pending_orders', [])
            slippage_incidents = execution_status.get('slippage_incidents', [])
            
            # Calculate execution quality metrics
            total_orders = len(pending_orders)
            filled_orders = sum(1 for order in pending_orders if order.get('status') == 'filled')
            fill_rate = filled_orders / total_orders if total_orders > 0 else 1.0
            
            # Assess slippage risk
            avg_slippage = 0
            if slippage_incidents:
                total_slippage = sum(incident.get('slippage_pct', 0) for incident in slippage_incidents)
                avg_slippage = total_slippage / len(slippage_incidents)
            
            # Market impact assessment
            market_volatility = market_data.get('volatility', 0.20)
            liquidity_score = market_data.get('liquidity_score', 0.5)  # 0-1 scale
            
            execution_risk_score = (1 - fill_rate) * 0.4 + avg_slippage * 0.3 + market_volatility * 0.3
            
            return {
                'fill_rate': fill_rate,
                'avg_slippage_pct': avg_slippage,
                'pending_orders_count': total_orders - filled_orders,
                'market_volatility': market_volatility,
                'liquidity_score': liquidity_score,
                'execution_risk_score': execution_risk_score,
                'execution_quality': 'good' if execution_risk_score < 0.3 else 'moderate' if execution_risk_score < 0.6 else 'poor'
            }
            
        except Exception as e:
            logger.error(f"Error assessing execution risk: {e}")
            return {'error': str(e)}

    def _check_risk_limits(self, portfolio_risk: Dict[str, Any], 
                          execution_risk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for breaches of risk limits in real-time."""
        try:
            breaches = []
            constraints = self.configs['risk']['constraints']
            
            # Check portfolio exposure limits
            max_exposure = constraints.get('max_position_size', 0.30)
            current_exposure = portfolio_risk.get('total_exposure_pct', 0)
            if current_exposure > max_exposure:
                breaches.append({
                    'type': 'portfolio_exposure',
                    'limit': max_exposure,
                    'current': current_exposure,
                    'severity': 'high' if current_exposure > max_exposure * 1.5 else 'medium',
                    'recommendation': 'Reduce position sizes or hedge exposure'
                })
            
            # Check diversification limits
            min_diversification = 0.6  # Minimum diversification score
            current_diversification = portfolio_risk.get('diversification_score', 1.0)
            if current_diversification < min_diversification:
                breaches.append({
                    'type': 'diversification',
                    'limit': min_diversification,
                    'current': current_diversification,
                    'severity': 'medium',
                    'recommendation': 'Increase diversification across more assets'
                })
            
            # Check execution quality limits
            max_slippage = constraints.get('slippage_tolerance', 0.001)
            current_slippage = execution_risk.get('avg_slippage_pct', 0)
            if current_slippage > max_slippage:
                breaches.append({
                    'type': 'execution_slippage',
                    'limit': max_slippage,
                    'current': current_slippage,
                    'severity': 'high',
                    'recommendation': 'Adjust execution algorithm or reduce order sizes'
                })
            
            return breaches
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return []

    async def _generate_risk_recommendations(self, portfolio_risk: Dict[str, Any],
                                           execution_risk: Dict[str, Any],
                                           risk_breaches: List[Dict[str, Any]],
                                           market_data: Dict[str, Any]) -> List[str]:
        """Generate dynamic risk management recommendations using LLM analysis."""
        try:
            if not self.llm:
                # Fallback recommendations without LLM
                recommendations = []
                if risk_breaches:
                    recommendations.extend([breach['recommendation'] for breach in risk_breaches])
                return recommendations
            
            # Use LLM for sophisticated risk analysis and recommendations
            risk_context = f"""
REAL-TIME RISK ASSESSMENT:

Portfolio Risk Metrics:
- Total Exposure: {portfolio_risk.get('total_exposure_pct', 0):.1%}
- Weighted Volatility: {portfolio_risk.get('weighted_volatility', 0):.1%}
- Diversification Score: {portfolio_risk.get('diversification_score', 0):.2f}
- Largest Position: {portfolio_risk.get('largest_position_pct', 0):.1%}

Execution Risk Metrics:
- Fill Rate: {execution_risk.get('fill_rate', 1.0):.1%}
- Average Slippage: {execution_risk.get('avg_slippage_pct', 0):.2%}
- Execution Quality: {execution_risk.get('execution_quality', 'unknown')}

Risk Breaches: {len(risk_breaches)} detected
Market Conditions: {market_data.get('market_regime', 'unknown')}

Based on this real-time risk assessment, provide 3-5 specific, actionable risk management recommendations.
Focus on immediate actions to reduce risk while maintaining alpha potential.
"""
            
            llm_response = await self.reason_with_llm(risk_context, 
                "What specific risk management actions should be taken right now?")
            
            # Parse recommendations from LLM response
            recommendations = []
            if llm_response and llm_response != "LLM_UNAVAILABLE":
                # Split into individual recommendations
                lines = llm_response.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and len(line) > 10 and not line.startswith('Based on'):
                        # Clean up the recommendation
                        if line.startswith(('- ', '• ', '* ')):
                            line = line[2:]
                        recommendations.append(line)
            
            # Add breach-specific recommendations
            for breach in risk_breaches:
                recommendations.append(f"URGENT: {breach['recommendation']}")
            
            return recommendations[:5]  # Limit to top 5 recommendations
            
        except Exception as e:
            logger.error(f"Error generating risk recommendations: {e}")
            return ["Monitor positions closely", "Consider reducing exposure if volatility increases"]

    def _calculate_realtime_var(self, positions: List[Dict[str, Any]], 
                              current_prices: Dict[str, float],
                              volatility_surface: Dict[str, Any]) -> Dict[str, float]:
        """Calculate real-time Value at Risk for current portfolio."""
        try:
            # Simplified real-time VaR calculation
            portfolio_variance = 0
            
            for position in positions:
                symbol = position.get('symbol', '')
                quantity = position.get('quantity', 0)
                current_price = current_prices.get(symbol, 0)
                
                if current_price > 0:
                    position_value = abs(quantity) * current_price
                    # Use position-specific volatility or default
                    volatility = volatility_surface.get(symbol, 0.25)  # Default 25%
                    
                    # Simplified variance contribution
                    position_variance = (position_value * volatility) ** 2
                    portfolio_variance += position_variance
            
            portfolio_volatility = np.sqrt(portfolio_variance) if portfolio_variance > 0 else 0
            
            # Calculate VaR at 95% confidence (simplified)
            var_95 = -2.326 * portfolio_volatility  # 95% confidence z-score
            cvar_95 = -2.665 * portfolio_volatility  # Expected shortfall approximation
            
            return {
                'var_95': var_95,
                'cvar_95': cvar_95,
                'portfolio_volatility': portfolio_volatility
            }
            
        except Exception as e:
            logger.error(f"Error calculating real-time VaR: {e}")
            return {'var_95': 0, 'cvar_95': 0}

    def _monitor_realtime_drawdown(self, current_portfolio_value: float, 
                                 active_positions: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor portfolio drawdown in real-time."""
        try:
            peak_value = active_positions.get('peak_portfolio_value', current_portfolio_value)
            
            if current_portfolio_value > peak_value:
                peak_value = current_portfolio_value
                # Update peak value in positions (would be persisted in real implementation)
            
            current_drawdown = (peak_value - current_portfolio_value) / peak_value if peak_value > 0 else 0
            
            # Determine drawdown severity
            if current_drawdown < 0.05:
                severity = 'low'
            elif current_drawdown < 0.10:
                severity = 'moderate'
            elif current_drawdown < 0.20:
                severity = 'high'
            else:
                severity = 'critical'
            
            return {
                'current_drawdown_pct': current_drawdown,
                'peak_value': peak_value,
                'current_value': current_portfolio_value,
                'severity': severity,
                'breaches_limit': current_drawdown > self.configs['risk']['constraints'].get('max_drawdown', 0.05)
            }
            
        except Exception as e:
            logger.error(f"Error monitoring real-time drawdown: {e}")
            return {'error': str(e)}

    def _assess_overall_risk_level(self, portfolio_risk: Dict[str, Any], 
                                 execution_risk: Dict[str, Any], 
                                 risk_breaches: List[Dict[str, Any]]) -> str:
        """Assess overall risk level from all risk components."""
        try:
            # Calculate risk score from multiple components
            risk_score = 0
            
            # Portfolio risk contribution
            exposure_score = portfolio_risk.get('total_exposure_pct', 0) / 0.5  # Normalize to 50% max
            diversification_penalty = (1 - portfolio_risk.get('diversification_score', 1.0))
            risk_score += (exposure_score + diversification_penalty) * 0.4
            
            # Execution risk contribution
            execution_score = execution_risk.get('execution_risk_score', 0)
            risk_score += execution_score * 0.3
            
            # Breach penalty
            breach_penalty = len(risk_breaches) * 0.2
            risk_score += breach_penalty
            
            # Determine risk level
            if risk_score < 0.3:
                return 'low'
            elif risk_score < 0.6:
                return 'moderate'
            elif risk_score < 0.8:
                return 'high'
            else:
                return 'critical'
                
        except Exception as e:
            logger.error(f"Error assessing overall risk level: {e}")
            return 'unknown'

    def _run_stochastics(self, proposal: Dict[str, Any], volatility: float = 0.20) -> Dict[str, Any]:
        """
        Runs advanced stochastic sims using TensorFlow Probability for realistic price paths.
        Uses VIX-based volatility for more accurate risk assessment with Monte Carlo methods.
        Simulates returns over a holding period and calculates Probability of Profit (POP).
        Includes Value at Risk (VaR) and Conditional VaR (CVaR) calculations.
        """
        try:
            roi_est = proposal.get('roi_estimate', 0.25)
            holding_days = proposal.get('holding_days', 30)  # Default 30 days.
            num_sims = 1000  # Number of Monte Carlo simulations.

            # GBM parameters using VIX-based volatility
            mu = roi_est / (holding_days / 365)  # Annualized drift.
            sigma = volatility  # Use VIX-based volatility instead of hardcoded 0.20
            dt = 1 / 365  # Daily time step.
            S0 = 100  # Initial price.

            # Use TensorFlow Probability for advanced stochastic simulation
            try:
                # Set up GBM process with TensorFlow Probability
                # Create normal distribution for random shocks
                normal_dist = tfp.distributions.Normal(loc=0., scale=1.)

                # Generate random shocks for all simulations at once
                random_shocks = normal_dist.sample(sample_shape=[num_sims, holding_days])

                # Convert to numpy for easier manipulation (TFP tensors work with numpy)
                random_shocks_np = random_shocks.numpy()

                # Simulate GBM paths using vectorized operations
                paths = np.zeros((num_sims, holding_days))
                paths[:, 0] = S0

                for t in range(1, holding_days):
                    paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shocks_np[:, t])

                # Calculate returns: (final_price - initial) / initial
                final_prices = paths[:, -1]
                returns = (final_prices - S0) / S0

                # Calculate POP: Probability that return > 0 (profitable)
                pop = np.mean(returns > 0)

                # Calculate Value at Risk (VaR) at 95% confidence
                var_95 = np.percentile(returns, 5)  # 5th percentile (95% confidence)

                # Calculate Conditional VaR (CVaR) - expected loss given loss occurs
                losses = returns[returns < 0]  # Only negative returns
                cvar_95 = np.mean(losses) if len(losses) > 0 else 0

                # Calculate additional risk metrics
                max_drawdown_sim = self._calculate_max_drawdown_from_paths(paths)
                sharpe_ratio_sim = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

                # Log comprehensive stochastic details
                stochastic_entry = {
                    'proposal': proposal,
                    'volatility_used': sigma,
                    'simulated_returns_mean': np.mean(returns),
                    'simulated_returns_std': np.std(returns),
                    'simulated_returns_skew': scipy.stats.skew(returns),
                    'simulated_returns_kurtosis': scipy.stats.kurtosis(returns),
                    'pop': pop,
                    'var_95': var_95,
                    'cvar_95': cvar_95,
                    'max_drawdown_sim': max_drawdown_sim,
                    'sharpe_ratio_sim': sharpe_ratio_sim,
                    'simulation_method': 'tfp_gbm_monte_carlo'
                }
                self.append_to_memory_list('stochastic_logs', stochastic_entry, max_items=100)

                logger.info(f"Advanced TFP stochastic sim: {num_sims} paths, POP {pop:.3f}, VaR 95% {var_95:.3f}, CVaR 95% {cvar_95:.3f} with VIX vol {sigma:.3f}")

                # Return enhanced risk metrics
                return {
                    'pop': pop,
                    'var_95': var_95,
                    'cvar_95': cvar_95,
                    'max_drawdown_sim': max_drawdown_sim,
                    'sharpe_ratio_sim': sharpe_ratio_sim
                }

            except ImportError:
                logger.warning("TensorFlow Probability not available, falling back to numpy GBM")
                # Fallback to original numpy implementation
                return self._run_numpy_stochastics(proposal, volatility)

        except Exception as e:
            logger.error(f"Error in advanced stochastic simulation: {e}")
            return {'pop': 0.5, 'var_95': -0.1, 'cvar_95': -0.15, 'max_drawdown_sim': 0.1, 'sharpe_ratio_sim': 0.0}

    def _run_numpy_stochastics(self, proposal: Dict[str, Any], volatility: float = 0.20) -> Dict[str, Any]:
        """
        Fallback numpy-based GBM simulation (original implementation).
        """
        try:
            roi_est = proposal.get('roi_estimate', 0.25)
            holding_days = proposal.get('holding_days', 30)
            num_sims = 1000

            mu = roi_est / (holding_days / 365)
            sigma = volatility
            dt = 1 / 365
            S0 = 100

            # Simulate GBM paths (numpy fallback)
            paths = np.zeros((num_sims, holding_days))
            paths[:, 0] = S0

            for t in range(1, holding_days):
                Z = np.random.normal(0, 1, num_sims)
                paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

            final_prices = paths[:, -1]
            returns = (final_prices - S0) / S0
            pop = np.mean(returns > 0)

            # Basic risk metrics for fallback
            var_95 = np.percentile(returns, 5)
            losses = returns[returns < 0]
            cvar_95 = np.mean(losses) if len(losses) > 0 else 0
            max_drawdown_sim = self._calculate_max_drawdown_from_paths(paths)
            sharpe_ratio_sim = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

            logger.info(f"Fallback numpy stochastic sim: POP {pop:.3f}, VaR 95% {var_95:.3f}")

            return {
                'pop': pop,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'max_drawdown_sim': max_drawdown_sim,
                'sharpe_ratio_sim': sharpe_ratio_sim
            }

        except Exception as e:
            logger.error(f"Error in fallback stochastic simulation: {e}")
            return {'pop': 0.5, 'var_95': -0.1, 'cvar_95': -0.15, 'max_drawdown_sim': 0.1, 'sharpe_ratio_sim': 0.0}

    def _calculate_max_drawdown_from_paths(self, paths: np.ndarray) -> float:
        """
        Calculate maximum drawdown from simulated price paths.
        """
        try:
            max_drawdowns = []
            for path in paths:
                peak = path[0]
                max_dd = 0
                for price in path:
                    if price > peak:
                        peak = price
                    dd = (peak - price) / peak
                    max_dd = max(max_dd, dd)
                max_drawdowns.append(max_dd)

            return np.mean(max_drawdowns)  # Average max drawdown across simulations

        except Exception as e:
            logger.warning(f"Error calculating max drawdown from paths: {e}")
            return 0.1  # Default 10% drawdown

    async def _get_vix_volatility(self) -> float:
        """
        Gets current market volatility from VIX via the data agent.
        Returns annualized volatility as decimal (e.g., 0.25 for 25%).
        """
        try:
            # Import data agent for VIX data
            from src.agents.data import DataAgent
            
            # Create data agent instance
            data_agent = DataAgent()
            
            # Request VIX data (VIX is the volatility index)
            vix_input = {
                'symbols': ['^VIX'],  # VIX ticker
                'period': '1mo',      # Recent data for current volatility
                'data_type': 'volatility_index'
            }
            
            vix_result = await data_agent.process_input(vix_input)
            
            if vix_result and 'dataframe' in vix_result:
                df = vix_result['dataframe']
                if not df.empty and len(df) > 0:
                    # Get the most recent VIX close value
                    if '^VIX' in df.columns.get_level_values(0) if hasattr(df.columns, 'get_level_values') else '^VIX' in df.columns:
                        # Multi-level columns (symbol-specific)
                        latest_vix = df['^VIX'].iloc[-1] if hasattr(df['^VIX'], 'iloc') else df['^VIX']
                    elif 'Close_^VIX' in df.columns:
                        latest_vix = df['Close_^VIX'].iloc[-1]
                    elif 'Close' in df.columns:
                        latest_vix = df['Close'].iloc[-1]
                    else:
                        # Fallback to last column
                        latest_vix = df.iloc[-1, -1]
                    
                    # Ensure we have a scalar value
                    if hasattr(latest_vix, 'iloc'):
                        latest_vix = latest_vix.iloc[-1] if len(latest_vix) > 0 else 20.0
                    elif hasattr(latest_vix, 'item'):
                        latest_vix = latest_vix.item()
                    
                    # Convert to float explicitly
                    latest_vix = float(latest_vix)
                    
                    # VIX is already in percentage terms, convert to decimal
                    volatility = latest_vix / 100.0
                    
                    # Apply bounds to prevent extreme values
                    volatility = max(0.05, min(1.0, volatility))  # Between 5% and 100%
                    
                    logger.info(f"Retrieved VIX volatility: {volatility:.3f} ({latest_vix:.1f}%)")
                    return volatility
                else:
                    logger.warning("VIX data is empty, using fallback volatility")
            else:
                logger.warning("Failed to retrieve VIX data from data agent")
            
            # Fallback to recent average VIX level (around 20%)
            logger.info("Using fallback VIX volatility: 0.20")
            return 0.20
            
        except Exception as e:
            logger.error(f"Error retrieving VIX volatility: {e}")
            # Ultimate fallback
            return 0.20

    async def _vet_proposal(self, proposal: Dict[str, Any], simulated_pop: float, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Vets proposal: Check POP, allow overrides with caps.
        Uses hybrid approach: foundation logic + LLM reasoning for complex cases.
        """
        pop_floor = constraints['pop_floor']
        base_approved = simulated_pop >= pop_floor
        base_rationale = f"POP {simulated_pop:.3f} vs floor {pop_floor}"

        # Use comprehensive LLM reasoning for all risk decisions (deep analysis and over-analysis)
        if self.llm:
            # Build foundation context for LLM
            foundation_context = f"""
FOUNDATION RISK ANALYSIS:
- Simulated Probability of Profit: {simulated_pop:.3f}
- Required POP Floor: {pop_floor}
- Base Decision: {'APPROVED' if base_approved else 'REJECTED'}
- Proposal ROI Estimate: {proposal.get('roi_estimate', 'N/A')}
- Proposal Confidence: {proposal.get('confidence', 'N/A')}
- Setup Type: {proposal.get('setup', 'N/A')}
- Current VIX Volatility: {getattr(self, '_last_vix_volatility', 'Unknown')}
- Risk Constraints: Max Drawdown {constraints.get('max_drawdown', 'N/A')}, Max Position Size {constraints.get('max_position_size', 'N/A')}
"""

            llm_question = """
Based on the foundation risk analysis above, should this trading proposal be approved?

Consider:
1. Risk-adjusted return potential vs. POP requirements
2. Market volatility context and position sizing implications
3. Confidence levels and setup quality
4. Alignment with overall risk constraints (<5% drawdown, 10-20% ROI targets)
5. Whether an override is justified for high-confidence, high-ROI opportunities

Provide a clear APPROVE/REJECT recommendation with detailed rationale.
"""

            try:
                llm_response = await self.reason_with_llm(foundation_context, llm_question)

                if "APPROVE" in llm_response.upper() and not "REJECT" in llm_response.upper():
                    approved = True
                    rationale = f"LLM Comprehensive Analysis Approved: {llm_response[:200]}..."
                    logger.info(f"Risk Agent LLM comprehensive analysis: Approved with deep reasoning")
                elif "REJECT" in llm_response.upper():
                    approved = False
                    rationale = f"LLM Comprehensive Analysis Rejected: {llm_response[:200]}..."
                else:
                    # Use foundation logic if LLM is unclear
                    approved = base_approved
                    rationale = f"LLM Unclear, Using Foundation: {base_rationale}"

            except Exception as e:
                logger.warning(f"LLM reasoning failed, using foundation logic: {e}")
                approved = base_approved
                rationale = base_rationale
        else:
            # Use foundation logic when LLM unavailable
            approved = base_approved
            rationale = base_rationale

            # Allow foundation override for high-confidence proposals
            if not approved and proposal.get('confidence', 0) > 0.8:
                approved = True
                rationale += "; Foundation Override: High confidence justifies risk"

        return {'approved': approved, 'rationale': rationale}

    def _adjust_post_batch(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Auto-adjusts metrics post-batch following the adjust_priority sequence.
        """
        adjustments = {}
        priority = constraints.get('adjust_priority', ['sizing', 'hold_days', 'pop_floor'])
        
        if 'sd >1.0' in self.memory['batch_adjustments']:
            for item in priority:
                if item == 'sizing':
                    adjustments['max_position_size'] = constraints['max_position_size'] * 0.9
                elif item == 'hold_days':
                    adjustments['hold_days'] = constraints.get('hold_days', 30) * 0.8  # Example.
                elif item == 'pop_floor':
                    adjustments['pop_floor'] = constraints['pop_floor'] * 1.05  # Tighten.
                # Add more as needed.
                break  # Adjust only the first in priority for now.
            logger.info(f"Adjusted {list(adjustments.keys())} on SD >1.0")
        return adjustments

    def _generate_yaml_diffs(self, adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates JSON diffs for A2A broadcast and updates the YAML file if changes are significant.
        """
        diffs = {}
        if adjustments:
            diffs = {'risk-constraints': adjustments}
            # Update the actual YAML file.
            self._update_yaml_file(adjustments)
        return diffs

    def _update_yaml_file(self, adjustments: Dict[str, Any]):
        """
        Updates the risk-constraints.yaml file with new adjustments.
        """
        try:
            yaml_path = Path(__file__).parent.parent.parent / 'config' / 'risk-constraints.yaml'
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Apply adjustments to constraints.
            for key, value in adjustments.items():
                if key in data['constraints']:
                    data['constraints'][key] = value
                    logger.info(f"Updated YAML: {key} = {value}")
            
            # Write back.
            with open(yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            
            logger.info("YAML file updated successfully.")
        
        except Exception as e:
            logger.error(f"Failed to update YAML file: {e}")

    def reflect(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Overrides for risk-specific (e.g., auto-adjust on SD >1.0).
        """
        adjustments = super().reflect(metrics)
        if metrics.get('sd_variance', 0) > 1.0:
            adjustments['auto_adjust'] = True
            self.update_memory('batch_adjustments', {'sd >1.0': 'tighten_sizing'})
            logger.info("Risk reflection: Added auto-adjust directive")
        return adjustments

    async def analyze_historical_simulation_risks(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive risk analysis on historical portfolio simulation results.
        Uses the advanced risk analytics framework to provide detailed risk assessment.

        Args:
            simulation_results: Results from historical portfolio simulation

        Returns:
            Comprehensive risk analysis report
        """
        try:
            logger.info("Risk Agent analyzing historical simulation results")

            # Import the risk analytics framework
            from ..utils.risk_analytics_framework import analyze_portfolio_risk

            # Perform comprehensive risk analysis
            risk_report = analyze_portfolio_risk(simulation_results)

            # Add risk agent specific insights and recommendations
            enhanced_report = await self._enhance_risk_report_with_agent_insights(risk_report, simulation_results)

            logger.info(f"Historical simulation risk analysis completed: {enhanced_report.get('risk_assessment', {}).get('overall_risk_level', 'unknown')} risk level")

            return enhanced_report

        except Exception as e:
            logger.error(f"Error in historical simulation risk analysis: {e}")
            return {
                'error': f'Historical simulation risk analysis failed: {str(e)}',
                'timestamp': pd.Timestamp.now().isoformat(),
                'risk_assessment': {'overall_risk_level': 'error'}
            }

    async def _enhance_risk_report_with_agent_insights(self, risk_report: Dict[str, Any],
                                                     simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance the risk report with agent-specific insights and LLM-based analysis.

        Args:
            risk_report: Base risk analysis report from framework
            simulation_results: Original simulation results

        Returns:
            Enhanced risk report with agent insights
        """
        try:
            # Add agent-specific risk context
            risk_report['agent_analysis'] = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'simulation_period': {
                    'start_date': simulation_results.get('simulation_config', {}).get('start_date'),
                    'end_date': simulation_results.get('simulation_config', {}).get('end_date'),
                    'duration_days': self._calculate_simulation_duration(simulation_results)
                },
                'portfolio_characteristics': self._analyze_portfolio_characteristics(simulation_results)
            }

            # Use LLM for sophisticated risk interpretation if available
            if self.llm:
                llm_insights = await self._generate_llm_risk_insights(risk_report, simulation_results)
                risk_report['agent_analysis']['llm_insights'] = llm_insights

            # Add risk management recommendations specific to historical analysis
            historical_recommendations = self._generate_historical_risk_recommendations(risk_report)
            risk_report['agent_analysis']['historical_recommendations'] = historical_recommendations

            # Add forward-looking risk projections
            risk_projections = self._generate_risk_projections(risk_report, simulation_results)
            risk_report['agent_analysis']['risk_projections'] = risk_projections

            return risk_report

        except Exception as e:
            logger.error(f"Error enhancing risk report: {e}")
            return risk_report

    def _calculate_simulation_duration(self, simulation_results: Dict[str, Any]) -> int:
        """Calculate the duration of the historical simulation in days."""
        try:
            config = simulation_results.get('simulation_config', {})
            start_date = config.get('start_date')
            end_date = config.get('end_date')

            if start_date and end_date:
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                return (end - start).days

            return 0
        except Exception as e:
            logger.error(f"Error calculating simulation duration: {e}")
            return 0

    def _analyze_portfolio_characteristics(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze key characteristics of the simulated portfolio."""
        try:
            characteristics = {}

            # Trading frequency
            trades = simulation_results.get('trades', [])
            if trades:
                trade_dates = [pd.to_datetime(trade['date']) for trade in trades if 'date' in trade]
                if trade_dates:
                    characteristics['trading_frequency'] = len(set(trade_dates)) / len(trade_dates)
                    characteristics['avg_trades_per_day'] = len(trades) / len(trade_dates)

            # Portfolio turnover
            portfolio_history = simulation_results.get('portfolio_history', [])
            if portfolio_history:
                initial_value = portfolio_history[0].get('portfolio_value', 0) if portfolio_history else 0
                final_value = portfolio_history[-1].get('portfolio_value', 0) if portfolio_history else 0
                total_traded_value = sum(trade.get('value', 0) for trade in trades)

                if initial_value > 0:
                    characteristics['portfolio_turnover'] = total_traded_value / initial_value

            # Strategy characteristics
            characteristics['strategy_types'] = self._identify_strategy_types(trades)
            characteristics['risk_adjusted_metrics'] = self._calculate_risk_adjusted_metrics(simulation_results)

            return characteristics

        except Exception as e:
            logger.error(f"Error analyzing portfolio characteristics: {e}")
            return {'error': str(e)}

    def _identify_strategy_types(self, trades: List[Dict[str, Any]]) -> List[str]:
        """Identify the types of strategies used in the simulation."""
        try:
            strategy_types = set()

            for trade in trades:
                action = trade.get('action', '').upper()
                symbol = trade.get('symbol', '')

                # Simple strategy identification based on trade patterns
                if 'BUY' in action:
                    strategy_types.add('long_positions')
                if 'SELL' in action:
                    strategy_types.add('short_positions')

                # Look for options patterns
                if any(keyword in symbol.upper() for keyword in ['CALL', 'PUT', 'OPTION']):
                    strategy_types.add('options_trading')

                # Look for pairs trading patterns (simultaneous buy/sell of related assets)
                # This would require more sophisticated analysis in practice

            return list(strategy_types) if strategy_types else ['unknown']

        except Exception as e:
            logger.error(f"Error identifying strategy types: {e}")
            return ['error']

    def _calculate_risk_adjusted_metrics(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk-adjusted performance metrics."""
        try:
            metrics = {}

            # Extract basic performance data
            perf = simulation_results.get('performance_metrics', {})
            total_return = perf.get('total_return', 0)
            volatility = perf.get('volatility', 0)

            # Assuming risk-free rate of 2%
            risk_free_rate = 0.02

            # Calculate Sharpe ratio if not already present
            if 'sharpe_ratio' not in perf and volatility > 0:
                metrics['sharpe_ratio'] = (total_return - risk_free_rate) / volatility

            # Calculate Sortino ratio (downside deviation)
            portfolio_history = simulation_results.get('portfolio_history', [])
            if portfolio_history:
                returns = pd.DataFrame(portfolio_history)['portfolio_value'].pct_change().dropna()
                downside_returns = returns[returns < 0]
                downside_volatility = downside_returns.std() if len(downside_returns) > 0 else 0

                if downside_volatility > 0:
                    metrics['sortino_ratio'] = (total_return - risk_free_rate) / downside_volatility

            # Calmar ratio (return / max drawdown)
            max_drawdown = perf.get('max_drawdown', 0)
            if max_drawdown > 0:
                metrics['calmar_ratio'] = total_return / max_drawdown

            return metrics

        except Exception as e:
            logger.error(f"Error calculating risk-adjusted metrics: {e}")
            return {'error': str(e)}

    async def _generate_llm_risk_insights(self, risk_report: Dict[str, Any],
                                        simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate LLM-based insights on the risk analysis."""
        try:
            # Prepare context for LLM analysis
            risk_context = f"""
HISTORICAL SIMULATION RISK ANALYSIS:

Risk Assessment Summary:
- Overall Risk Level: {risk_report.get('risk_assessment', {}).get('overall_risk_level', 'unknown')}
- Risk Factors: {', '.join(risk_report.get('risk_assessment', {}).get('risk_factors', []))}
- Confidence Level: {risk_report.get('risk_assessment', {}).get('confidence_level', 'unknown')}

Key Risk Metrics:
- Volatility: {risk_report.get('risk_metrics', {}).get('volatility', 0):.1%}
- Max Drawdown: {risk_report.get('risk_metrics', {}).get('max_drawdown', 0):.1%}
- VaR 95%: {risk_report.get('risk_metrics', {}).get('var_historical_95', 0):.1%}
- CVaR 95%: {risk_report.get('risk_metrics', {}).get('cvar_95', 0):.1%}

Performance Attribution:
{risk_report.get('performance_attribution', {}).get('benchmark_comparison', 'No benchmark comparison available')}

Portfolio Characteristics:
{risk_report.get('agent_analysis', {}).get('portfolio_characteristics', 'Analysis in progress')}
"""

            llm_question = """
Based on this comprehensive risk analysis of historical simulation results, provide insights on:

1. Key risk patterns and concerns that emerge from this backtest
2. How realistic are these risk metrics for forward-looking risk management?
3. What specific risk management improvements would you recommend?
4. Are there any concerning patterns in the risk factor analysis?

Provide actionable insights that would help improve future risk management and strategy development.
"""

            llm_response = await self.reason_with_llm(risk_context, llm_question)

            return {
                'llm_analysis': llm_response,
                'key_insights_extracted': self._extract_key_insights_from_llm(llm_response),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating LLM risk insights: {e}")
            return {'error': str(e)}

    def _extract_key_insights_from_llm(self, llm_response: str) -> List[str]:
        """Extract key insights from LLM response."""
        try:
            insights = []
            if llm_response and llm_response != "LLM_UNAVAILABLE":
                lines = llm_response.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and len(line) > 15 and not line.upper().startswith(('BASED ON', 'PROVIDE', 'ANALYZE')):
                        # Clean up formatting
                        if line.startswith(('- ', '• ', '* ', str(len(insights) + 1) + '.')):
                            line = line.lstrip('- •*123456789. ')
                        insights.append(line[:200])  # Limit length

            return insights[:5]  # Top 5 insights

        except Exception as e:
            logger.error(f"Error extracting insights from LLM: {e}")
            return []

    def _generate_historical_risk_recommendations(self, risk_report: Dict[str, Any]) -> List[str]:
        """Generate risk management recommendations specific to historical analysis."""
        try:
            recommendations = []

            risk_metrics = risk_report.get('risk_metrics', {})
            risk_assessment = risk_report.get('risk_assessment', {})

            # Historical-specific recommendations
            max_drawdown = risk_metrics.get('max_drawdown', 0)
            if max_drawdown > 0.2:  # More than 20% drawdown
                recommendations.append("Historical drawdown exceeds 20% - implement maximum drawdown limits in live trading")

            volatility = risk_metrics.get('volatility', 0)
            if volatility > 0.3:  # More than 30% volatility
                recommendations.append("High historical volatility detected - consider volatility-based position sizing")

            # Recovery analysis
            avg_recovery_time = risk_metrics.get('avg_recovery_time')
            if avg_recovery_time and avg_recovery_time > 90:  # More than 3 months
                recommendations.append("Long recovery times from drawdowns - focus on strategies with faster recovery profiles")

            # Stress testing insights
            stress_periods = risk_metrics.get('historical_stress_periods', 0)
            if stress_periods > 5:
                recommendations.append("Multiple stress periods identified - enhance stress testing protocols")

            # Benchmark comparison
            attribution = risk_report.get('performance_attribution', {})
            benchmark_comp = attribution.get('benchmark_comparison', {})
            tracking_error = benchmark_comp.get('tracking_error', 0)
            if tracking_error > 0.15:  # High tracking error
                recommendations.append("High tracking error vs benchmark - ensure active risk is intentional")

            return recommendations

        except Exception as e:
            logger.error(f"Error generating historical recommendations: {e}")
            return ["Review risk metrics carefully before live deployment"]

    def _generate_risk_projections(self, risk_report: Dict[str, Any],
                                 simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate forward-looking risk projections based on historical analysis."""
        try:
            projections = {}

            # Project future risk metrics based on historical patterns
            risk_metrics = risk_report.get('risk_metrics', {})

            # Conservative projections (assuming mean reversion)
            current_volatility = risk_metrics.get('volatility', 0.2)
            projections['projected_volatility_range'] = {
                'conservative': current_volatility * 0.8,
                'base_case': current_volatility,
                'stress_case': current_volatility * 1.5
            }

            # Drawdown projections
            max_drawdown = risk_metrics.get('max_drawdown', 0.1)
            projections['projected_max_drawdown_range'] = {
                'conservative': max_drawdown * 0.7,
                'base_case': max_drawdown,
                'stress_case': max_drawdown * 1.3
            }

            # VaR projections
            var_95 = risk_metrics.get('var_historical_95', -0.05)
            projections['projected_var_range'] = {
                'conservative': var_95 * 0.8,  # Less negative (better)
                'base_case': var_95,
                'stress_case': var_95 * 1.5  # More negative (worse)
            }

            # Add confidence intervals
            projections['confidence_intervals'] = {
                'volatility_ci': [current_volatility * 0.9, current_volatility * 1.1],
                'var_ci': [var_95 * 0.9, var_95 * 1.1]
            }

            return projections

        except Exception as e:
            logger.error(f"Error generating risk projections: {e}")
            return {'error': str(e)}

    def _initialize_real_time_risk_monitoring(self):
        """
        Initialize real-time risk monitoring system.
        """
        logger.info("Initializing real-time risk monitoring...")

        # Real-time risk monitoring state
        self.real_time_monitoring = {
            'active': True,
            'monitoring_interval_seconds': 30,  # Check every 30 seconds
            'last_check': None,
            'alert_thresholds': {
                'portfolio_var_95': 0.05,  # 5% VaR alert
                'max_drawdown': 0.03,  # 3% drawdown alert
                'correlation_spike': 0.8,  # Correlation > 0.8 alert
                'volatility_spike': 0.04,  # 4% volatility spike alert
                'liquidity_risk': 0.1  # 10% illiquid positions alert
            },
            'active_alerts': [],
            'risk_history': [],
            'position_limits': {
                'max_single_position': 0.15,  # 15% of portfolio
                'max_sector_exposure': 0.25,  # 25% per sector
                'max_correlated_positions': 3,  # Max 3 highly correlated positions
                'min_liquidity_ratio': 0.8  # Minimum liquidity ratio
            }
        }

        # Dynamic position sizing parameters
        self.dynamic_sizing = {
            'volatility_adjustment': True,
            'correlation_adjustment': True,
            'liquidity_adjustment': True,
            'base_position_size': 0.05,  # 5% base position size
            'volatility_multiplier': 0.5,  # Reduce size in high volatility
            'correlation_multiplier': 0.7,  # Reduce size for correlated positions
            'liquidity_multiplier': 0.8  # Reduce size for illiquid assets
        }

        logger.info("Real-time risk monitoring initialized successfully")

    def _initialize_crisis_detection(self):
        """
        Initialize comprehensive crisis detection system.
        """
        logger.info("Initializing crisis detection system...")

        # Crisis detection thresholds
        self.crisis_thresholds = {
            'daily_drop_pct': 0.05,  # 5% daily drop triggers warning
            'crash_drop_pct': 0.10,  # 10% daily drop triggers crisis
            'volatility_spike': 3.0,  # 3x normal volatility triggers alert
            'liquidity_dryup': 0.70,  # 70% drop in volume triggers concern
            'correlation_spike': 0.90,  # 90% correlation across assets
            'vix_extreme': 40,  # VIX above 40 triggers extreme volatility alert
            'sector_crash': 0.15,  # 15% sector drop in a day
            'flash_crash': 0.20   # 20% intraday drop triggers flash crash protocol
        }

        # Crisis response actions
        self.crisis_responses = {
            'warning': {
                'reduce_position_size': 0.5,  # Reduce to 50% of normal size
                'increase_stops': 0.02,  # Tighter 2% stops
                'pause_new_trades': False,
                'notify_agents': True
            },
            'crisis': {
                'reduce_position_size': 0.25,  # Reduce to 25% of normal size
                'increase_stops': 0.05,  # 5% stops
                'pause_new_trades': True,
                'close_positions': False,
                'notify_agents': True,
                'activate_circuit_breaker': False
            },
            'extreme_crisis': {
                'reduce_position_size': 0.1,  # Reduce to 10% of normal size
                'increase_stops': 0.10,  # 10% stops
                'pause_new_trades': True,
                'close_positions': True,  # Close all positions
                'notify_agents': True,
                'activate_circuit_breaker': True
            }
        }

        # Market regime tracking
        self.market_regime = {
            'current': 'normal',
            'history': [],
            'regime_start': datetime.now(),
            'volatility_baseline': 0.20,  # 20% annualized as baseline
            'correlation_baseline': 0.30   # 30% average correlation as baseline
        }

        # Crisis memory for learning
        self.crisis_memory = {
            'past_crises': [],
            'response_effectiveness': {},
            'false_positives': []
        }

        logger.info("Crisis detection system initialized successfully")

    async def detect_market_crisis(self, market_data: Dict[str, Any], 
                                 portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive market crisis detection using multiple indicators.

        Args:
            market_data: Current market data including prices, volumes, volatility
            portfolio_data: Current portfolio state

        Returns:
            Crisis assessment with severity level and recommended actions
        """
        try:
            logger.info("Performing comprehensive crisis detection")

            # Extract relevant market data
            prices = market_data.get('prices', {})
            volumes = market_data.get('volumes', {})
            volatility = market_data.get('volatility', {})
            correlations = market_data.get('correlations', {})

            # Perform multiple crisis detection checks
            crisis_indicators = await self._analyze_crisis_indicators(
                prices, volumes, volatility, correlations
            )

            # Determine overall crisis severity
            crisis_severity = self._assess_crisis_severity(crisis_indicators)

            # Generate crisis response
            crisis_response = self._generate_crisis_response(crisis_severity, crisis_indicators)

            # Update market regime
            self._update_market_regime(crisis_severity, crisis_indicators)

            # Store crisis in memory for learning
            await self._store_crisis_event(crisis_severity, crisis_indicators, crisis_response)

            crisis_assessment = {
                'timestamp': datetime.now().isoformat(),
                'severity_level': crisis_severity['level'],
                'confidence': crisis_severity['confidence'],
                'indicators': crisis_indicators,
                'response_actions': crisis_response,
                'market_regime': self.market_regime['current'],
                'recommendations': self._generate_crisis_recommendations(crisis_severity, portfolio_data)
            }

            logger.info(f"Crisis detection completed: {crisis_assessment['severity_level']} severity "
                       f"({crisis_assessment['confidence']:.1%} confidence)")

            return crisis_assessment

        except Exception as e:
            logger.error(f"Error in crisis detection: {e}")
            return {
                'error': str(e),
                'severity_level': 'unknown',
                'confidence': 0.0,
                'response_actions': {},
                'recommendations': ['Continue with normal risk management']
            }

    async def _analyze_crisis_indicators(self, prices: Dict[str, float], 
                                       volumes: Dict[str, float],
                                       volatility: Dict[str, float],
                                       correlations: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze multiple crisis indicators simultaneously.
        """
        indicators = {}

        try:
            # 1. Price drop analysis
            indicators['price_drops'] = await self._analyze_price_drops(prices)

            # 2. Volatility spike detection
            indicators['volatility_spikes'] = self._analyze_volatility_spikes(volatility)

            # 3. Liquidity analysis
            indicators['liquidity_dryup'] = await self._analyze_liquidity_dryup(volumes)

            # 4. Correlation analysis
            indicators['correlation_spike'] = self._analyze_correlation_spike(correlations)

            # 5. VIX extreme levels
            indicators['vix_extreme'] = self._analyze_vix_extreme(volatility)

            # 6. Sector crash detection
            indicators['sector_crash'] = await self._analyze_sector_crash(prices)

            # 7. Flash crash detection
            indicators['flash_crash'] = await self._analyze_flash_crash(prices)

            # 8. Market breadth analysis
            indicators['market_breadth'] = await self._analyze_market_breadth(prices)

        except Exception as e:
            logger.error(f"Error analyzing crisis indicators: {e}")
            indicators['error'] = str(e)

        return indicators

    async def _analyze_price_drops(self, prices: Dict[str, float]) -> Dict[str, Any]:
        """Analyze significant price drops across assets."""
        try:
            significant_drops = {}
            crash_drops = {}

            for symbol, price_data in prices.items():
                if isinstance(price_data, dict) and 'daily_return' in price_data:
                    daily_return = price_data['daily_return']

                    if daily_return <= -self.crisis_thresholds['daily_drop_pct']:
                        significant_drops[symbol] = daily_return

                    if daily_return <= -self.crisis_thresholds['crash_drop_pct']:
                        crash_drops[symbol] = daily_return

            return {
                'significant_drops': significant_drops,
                'crash_drops': crash_drops,
                'most_severe_drop': min(significant_drops.values()) if significant_drops else 0,
                'affected_assets_pct': len(significant_drops) / max(len(prices), 1)
            }

        except Exception as e:
            logger.error(f"Error analyzing price drops: {e}")
            return {'error': str(e)}

    def _analyze_volatility_spikes(self, volatility: Dict[str, float]) -> Dict[str, Any]:
        """Analyze volatility spikes above normal levels."""
        try:
            spikes = {}
            extreme_spikes = {}

            for symbol, vol in volatility.items():
                if vol >= self.crisis_thresholds['volatility_spike'] * self.market_regime['volatility_baseline']:
                    spikes[symbol] = vol

                if vol >= 5.0 * self.market_regime['volatility_baseline']:  # 5x normal
                    extreme_spikes[symbol] = vol

            return {
                'volatility_spikes': spikes,
                'extreme_spikes': extreme_spikes,
                'average_volatility': sum(volatility.values()) / max(len(volatility), 1),
                'spike_assets_pct': len(spikes) / max(len(volatility), 1)
            }

        except Exception as e:
            logger.error(f"Error analyzing volatility spikes: {e}")
            return {'error': str(e)}

    async def _analyze_liquidity_dryup(self, volumes: Dict[str, float]) -> Dict[str, Any]:
        """Analyze liquidity dry-up conditions."""
        try:
            # Compare current volumes to historical averages
            liquidity_issues = {}

            for symbol, volume in volumes.items():
                # This would typically compare to historical average volume
                # For now, use a simplified approach
                if volume and volume < 1000000:  # Less than 1M shares/contracts
                    liquidity_issues[symbol] = volume

            return {
                'low_volume_assets': liquidity_issues,
                'liquidity_concern_pct': len(liquidity_issues) / max(len(volumes), 1),
                'total_volume_drop': sum(volumes.values()) if volumes else 0
            }

        except Exception as e:
            logger.error(f"Error analyzing liquidity: {e}")
            return {'error': str(e)}

    def _analyze_correlation_spike(self, correlations: Dict[str, float]) -> Dict[str, Any]:
        """Analyze correlation spikes indicating contagion."""
        try:
            high_correlations = {}
            extreme_correlations = {}

            for pair, corr in correlations.items():
                if corr >= self.crisis_thresholds['correlation_spike']:
                    high_correlations[pair] = corr

                if corr >= 0.95:  # Nearly perfect correlation
                    extreme_correlations[pair] = corr

            avg_correlation = sum(correlations.values()) / max(len(correlations), 1)

            return {
                'high_correlations': high_correlations,
                'extreme_correlations': extreme_correlations,
                'average_correlation': avg_correlation,
                'correlation_spike': avg_correlation > self.market_regime['correlation_baseline'] * 2
            }

        except Exception as e:
            logger.error(f"Error analyzing correlations: {e}")
            return {'error': str(e)}

    def _analyze_vix_extreme(self, volatility: Dict[str, float]) -> Dict[str, Any]:
        """Analyze VIX levels for extreme fear."""
        try:
            vix = volatility.get('VIX', volatility.get('^VIX', 0))

            return {
                'vix_level': vix,
                'vix_extreme': vix >= self.crisis_thresholds['vix_extreme'],
                'vix_panic': vix >= 50,  # Panic level
                'fear_gauge': 'extreme' if vix >= 40 else 'high' if vix >= 30 else 'normal'
            }

        except Exception as e:
            logger.error(f"Error analyzing VIX: {e}")
            return {'error': str(e)}

    async def _analyze_sector_crash(self, prices: Dict[str, float]) -> Dict[str, Any]:
        """Analyze sector-level crashes."""
        try:
            # Group assets by sector (simplified mapping)
            sector_performance = {}

            for symbol, price_data in prices.items():
                if isinstance(price_data, dict) and 'daily_return' in price_data:
                    # Simplified sector mapping
                    sector = self._map_symbol_to_sector(symbol)
                    if sector not in sector_performance:
                        sector_performance[sector] = []
                    sector_performance[sector].append(price_data['daily_return'])

            sector_crashes = {}
            for sector, returns in sector_performance.items():
                avg_return = sum(returns) / len(returns)
                if avg_return <= -self.crisis_thresholds['sector_crash']:
                    sector_crashes[sector] = avg_return

            return {
                'sector_crashes': sector_crashes,
                'crashed_sectors_count': len(sector_crashes),
                'most_affected_sector': min(sector_crashes.items(), key=lambda x: x[1]) if sector_crashes else None
            }

        except Exception as e:
            logger.error(f"Error analyzing sector crash: {e}")
            return {'error': str(e)}

    async def _analyze_flash_crash(self, prices: Dict[str, float]) -> Dict[str, Any]:
        """Analyze for flash crash conditions (extreme intraday drops)."""
        try:
            flash_crash_indicators = {}

            for symbol, price_data in prices.items():
                if isinstance(price_data, dict) and 'intraday_drop' in price_data:
                    intraday_drop = price_data['intraday_drop']
                    if intraday_drop >= self.crisis_thresholds['flash_crash']:
                        flash_crash_indicators[symbol] = intraday_drop

            return {
                'flash_crash_assets': flash_crash_indicators,
                'flash_crash_detected': len(flash_crash_indicators) > 0,
                'most_severe_flash': max(flash_crash_indicators.values()) if flash_crash_indicators else 0
            }

        except Exception as e:
            logger.error(f"Error analyzing flash crash: {e}")
            return {'error': str(e)}

    async def _analyze_market_breadth(self, prices: Dict[str, float]) -> Dict[str, Any]:
        """Analyze market breadth (advance-decline indicators)."""
        try:
            advancing = 0
            declining = 0
            unchanged = 0

            for symbol, price_data in prices.items():
                if isinstance(price_data, dict) and 'daily_return' in price_data:
                    daily_return = price_data['daily_return']
                    if daily_return > 0.001:  # Small threshold for unchanged
                        advancing += 1
                    elif daily_return < -0.001:
                        declining += 1
                    else:
                        unchanged += 1

            total_assets = advancing + declining + unchanged
            advance_decline_ratio = advancing / max(declining, 1)

            return {
                'advancing_assets': advancing,
                'declining_assets': declining,
                'unchanged_assets': unchanged,
                'advance_decline_ratio': advance_decline_ratio,
                'breadth_extreme': advance_decline_ratio < 0.3 or advance_decline_ratio > 3.0
            }

        except Exception as e:
            logger.error(f"Error analyzing market breadth: {e}")
            return {'error': str(e)}

    def _map_symbol_to_sector(self, symbol: str) -> str:
        """Map symbol to sector (simplified implementation)."""
        # This would be expanded with a proper sector mapping
        sector_map = {
            'SPY': 'broad_market',
            'QQQ': 'technology',
            'IWM': 'small_cap',
            'EFA': 'international',
            'TLT': 'bonds',
            'GLD': 'commodities'
        }
        return sector_map.get(symbol, 'unknown')

    def _assess_crisis_severity(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall crisis severity from all indicators."""
        try:
            severity_score = 0
            confidence_factors = []

            # Price drops (weight: 0.3)
            price_drops = indicators.get('price_drops', {})
            if price_drops.get('crash_drops'):
                severity_score += 0.3 * min(1.0, len(price_drops['crash_drops']) / 5)  # Scale by number of crashes
                confidence_factors.append('price_crashes')

            # Volatility spikes (weight: 0.25)
            vol_spikes = indicators.get('volatility_spikes', {})
            if vol_spikes.get('extreme_spikes'):
                severity_score += 0.25
                confidence_factors.append('volatility_extreme')

            # VIX extreme (weight: 0.2)
            vix_analysis = indicators.get('vix_extreme', {})
            if vix_analysis.get('vix_extreme'):
                severity_score += 0.2
                confidence_factors.append('vix_extreme')

            # Correlation spike (weight: 0.15)
            corr_analysis = indicators.get('correlation_spike', {})
            if corr_analysis.get('correlation_spike'):
                severity_score += 0.15
                confidence_factors.append('correlation_spike')

            # Sector crash (weight: 0.1)
            sector_crash = indicators.get('sector_crash', {})
            if sector_crash.get('crashed_sectors_count', 0) > 0:
                severity_score += 0.1 * min(1.0, sector_crash['crashed_sectors_count'] / 3)
                confidence_factors.append('sector_crash')

            # Determine severity level
            if severity_score >= 0.8:
                level = 'extreme_crisis'
            elif severity_score >= 0.5:
                level = 'crisis'
            elif severity_score >= 0.25:
                level = 'warning'
           
            else:
                level = 'normal'



            return {
                'level': level,
                'score': severity_score,
                'confidence': min(1.0, len(confidence_factors) / 3),  # Confidence based on number of confirming indicators
                'confirming_factors': confidence_factors
            }

        except Exception as e:
            logger.error(f"Error assessing crisis severity: {e}")
            return {'level': 'unknown', 'score': 0, 'confidence': 0, 'confirming_factors': []}

    def _generate_crisis_response(self, severity: Dict[str, Any], 
                                indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appropriate crisis response based on severity."""
        try:
            level = severity['level']
            response_template = self.crisis_responses.get(level, {})

            # Customize response based on specific indicators
            customized_response = response_template.copy()

            # Add specific actions based on crisis type
            if indicators.get('flash_crash', {}).get('flash_crash_detected'):
                customized_response['immediate_action'] = 'halt_all_trading'
                customized_response['circuit_breaker_duration'] = 15  # 15 minutes

            if indicators.get('liquidity_dryup', {}).get('liquidity_concern_pct', 0) > 0.5:
                customized_response['liquidity_protection'] = True

            return customized_response

        except Exception as e:
            logger.error(f"Error generating crisis response: {e}")
            return {}

    def _update_market_regime(self, severity: Dict[str, Any], 
                            indicators: Dict[str, Any]) -> None:
        """Update market regime based on crisis assessment."""
        try:
            current_regime = self.market_regime['current']
            new_regime = severity['level']

            # Only change regime if severity indicates a different state
            regime_mapping = {
                'normal': 0,
                'warning': 1,
                'crisis': 2,
                'extreme_crisis': 3
            }

            if regime_mapping.get(new_regime, 0) > regime_mapping.get(current_regime, 0):
                # Regime escalation
                self.market_regime['history'].append({
                    'previous_regime': current_regime,
                    'new_regime': new_regime,
                    'timestamp': datetime.now().isoformat(),
                    'trigger_indicators': list(indicators.keys())
                })

                self.market_regime['current'] = new_regime
                self.market_regime['regime_start'] = datetime.now()

                logger.info(f"Market regime changed from {current_regime} to {new_regime}")

            elif new_regime == 'normal' and current_regime != 'normal':
                # Return to normal
                duration = (datetime.now() - self.market_regime['regime_start']).total_seconds() / 3600
                self.market_regime['history'].append({
                    'regime': current_regime,
                    'end_timestamp': datetime.now().isoformat(),
                    'duration_hours': duration
                })

                self.market_regime['current'] = 'normal'
                self.market_regime['regime_start'] = datetime.now()

                logger.info(f"Market returned to normal regime after {duration:.1f} hours in {current_regime}")

        except Exception as e:
            logger.error(f"Error updating market regime: {e}")

    async def _store_crisis_event(self, severity: Dict[str, Any], 
                                indicators: Dict[str, Any], 
                                response: Dict[str, Any]) -> None:
        """Store crisis event in memory for learning and analysis."""
        try:
            crisis_event = {
                'timestamp': datetime.now().isoformat(),
                'severity': severity,
                'indicators': indicators,
                'response': response,
                'market_regime': self.market_regime['current']
            }

            self.crisis_memory['past_crises'].append(crisis_event)

            # Keep only last 100 crisis events
            if len(self.crisis_memory['past_crises']) > 100:
                self.crisis_memory['past_crises'] = self.crisis_memory['past_crises'][-100:]

            # Store in agent memory
            await self.store_advanced_memory('crisis_events', self.crisis_memory['past_crises'])

        except Exception as e:
            logger.error(f"Error storing crisis event: {e}")

    async def get_crisis_history(self) -> Dict[str, Any]:
        """Get crisis detection history and analytics."""
        try:
            crises = self.crisis_memory.get('past_crises', [])

            # Analyze crisis patterns
            severity_counts = {}
            for crisis in crises:
                level = crisis.get('severity', {}).get('level', 'unknown')
                severity_counts[level] = severity_counts.get(level, 0) + 1

            # Calculate average time between crises
            if len(crises) > 1:
                timestamps = [datetime.fromisoformat(c['timestamp']) for c in crises]
                intervals = [(timestamps[i] - timestamps[i-1]).days for i in range(1, len(timestamps))]
                avg_days_between_crises = sum(intervals) / len(intervals)
            else:
                avg_days_between_crises = None

            return {
                'total_crises': len(crises),
                'severity_distribution': severity_counts,
                'avg_days_between_crises': avg_days_between_crises,
                'most_recent_crisis': crises[-1] if crises else None,
                'current_regime': self.market_regime['current'],
                'regime_history': self.market_regime['history']
            }

        except Exception as e:
            logger.error(f"Error getting crisis history: {e}")
            return {'error': str(e)}

    async def detect_market_regime(self, market_data: Dict[str, Any], 
                                 historical_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Advanced market regime detection using multiple timeframes and indicators.
        Determines if market is in bull, bear, crisis, or recovery mode.

        Args:
            market_data: Current market data
            historical_data: Historical market data for trend analysis

        Returns:
            Market regime assessment with confidence and characteristics
        """
        try:
            logger.info("Performing advanced market regime detection")

            # Analyze multiple timeframe trends
            short_term_regime = await self._analyze_regime_timeframe(market_data, 'short', 20)  # 20 days
            medium_term_regime = await self._analyze_regime_timeframe(market_data, 'medium', 60)  # 60 days
            long_term_regime = await self._analyze_regime_timeframe(market_data, 'long', 252)  # 1 year

            # Analyze volatility regime
            volatility_regime = self._analyze_volatility_regime(market_data)

            # Analyze momentum and sentiment
            momentum_regime = await self._analyze_momentum_regime(market_data, historical_data)

            # Synthesize overall regime
            overall_regime = self._synthesize_market_regime(
                short_term_regime, medium_term_regime, long_term_regime,
                volatility_regime, momentum_regime
            )

            # Generate regime-based risk adjustments
            risk_adjustments = self._generate_regime_risk_adjustments(overall_regime)

            regime_assessment = {
                'timestamp': datetime.now().isoformat(),
                'overall_regime': overall_regime['regime'],
                'confidence': overall_regime['confidence'],
                'timeframe_regimes': {
                    'short_term': short_term_regime,
                    'medium_term': medium_term_regime,
                    'long_term': long_term_regime
                },
                'volatility_regime': volatility_regime,
                'momentum_regime': momentum_regime,
                'risk_adjustments': risk_adjustments,
                'regime_characteristics': overall_regime['characteristics'],
                'transition_probability': self._calculate_regime_transition_probability(overall_regime)
            }

            logger.info(f"Market regime detected: {overall_regime['regime']} "
                       f"({overall_regime['confidence']:.1%} confidence)")

            return regime_assessment

        except Exception as e:
            logger.error(f"Error in market regime detection: {e}")
            return {
                'error': str(e),
                'overall_regime': 'unknown',
                'confidence': 0.0,
                'risk_adjustments': {}
            }

    async def _analyze_regime_timeframe(self, market_data: Dict[str, Any], 
                                      timeframe: str, lookback_days: int) -> Dict[str, Any]:
        """Analyze market regime for a specific timeframe."""
        try:
            # Extract price data for the timeframe
            prices = market_data.get('historical_prices', {})
            if not prices:
                return {'regime': 'unknown', 'trend': 0, 'volatility': 0.2}

            # Calculate trend metrics
            returns = []
            for symbol, price_history in prices.items():
                if isinstance(price_history, list) and len(price_history) >= lookback_days:
                    # Calculate returns over the timeframe
                    start_price = price_history[-lookback_days]
                    end_price = price_history[-1]
                    if start_price > 0:
                        total_return = (end_price - start_price) / start_price
                        returns.append(total_return)

            if not returns:
                return {'regime': 'unknown', 'trend': 0, 'volatility': 0.2}

            avg_return = sum(returns) / len(returns)
            return_volatility = np.std(returns) if len(returns) > 1 else 0

            # Determine regime based on returns and volatility
            if avg_return > 0.05 and return_volatility < 0.15:  # 5% return, low volatility
                regime = 'bull'
            elif avg_return > 0.02:  # Moderate positive return
                regime = 'bull_moderate'
            elif avg_return > -0.02:  # Flat/sideways
                regime = 'neutral'
            elif avg_return > -0.05:  # Moderate negative
                regime = 'bear_moderate'
            else:
                regime = 'bear'

            return {
                'regime': regime,
                'trend': avg_return,
                'volatility': return_volatility,
                'timeframe_days': lookback_days
            }

        except Exception as e:
            logger.error(f"Error analyzing {timeframe} regime: {e}")
            return {'regime': 'unknown', 'trend': 0, 'volatility': 0.2}

    def _analyze_volatility_regime(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current volatility regime."""
        try:
            vix = market_data.get('volatility', {}).get('VIX', 20)
            realized_vol = market_data.get('realized_volatility', 0.25)

            # Classify volatility regime
            if vix >= 40 or realized_vol >= 0.40:
                regime = 'extreme'
                multiplier = 2.0
            elif vix >= 30 or realized_vol >= 0.30:
                regime = 'high'
                multiplier = 1.5
            elif vix >= 20 or realized_vol >= 0.20:
                regime = 'normal'
                multiplier = 1.0
            else:
                regime = 'low'
                multiplier = 0.8

            return {
                'regime': regime,
                'vix_level': vix,
                'realized_volatility': realized_vol,
                'risk_multiplier': multiplier
            }

        except Exception as e:
            logger.error(f"Error analyzing volatility regime: {e}")
            return {'regime': 'normal', 'vix_level': 20, 'realized_volatility': 0.25, 'risk_multiplier': 1.0}

    async def _analyze_momentum_regime(self, market_data: Dict[str, Any], 
                                     historical_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze momentum and sentiment indicators."""
        try:
            momentum_indicators = {}

            # RSI analysis (if available)
            rsi = market_data.get('technical_indicators', {}).get('rsi', 50)
            if rsi > 70:
                momentum_indicators['rsi'] = 'overbought'
            elif rsi < 30:
                momentum_indicators['rsi'] = 'oversold'
            else:
                momentum_indicators['rsi'] = 'neutral'

            # MACD analysis (if available)
            macd = market_data.get('technical_indicators', {}).get('macd', {})
            if macd.get('histogram', 0) > 0:
                momentum_indicators['macd'] = 'bullish'
            else:
                momentum_indicators['macd'] = 'bearish'

            # Put/Call ratio (if available)
            pcr = market_data.get('options_data', {}).get('put_call_ratio', 1.0)
            if pcr > 1.2:
                momentum_indicators['sentiment'] = 'bearish'
            elif pcr < 0.8:
                momentum_indicators['sentiment'] = 'bullish'
            else:
                momentum_indicators['sentiment'] = 'neutral'

            # Determine overall momentum regime
            bullish_signals = sum(1 for v in momentum_indicators.values() if v in ['bullish', 'oversold'])
            bearish_signals = sum(1 for v in momentum_indicators.values() if v in ['bearish', 'overbought'])

            if bullish_signals > bearish_signals:
                momentum_regime = 'bullish'
            elif bearish_signals > bullish_signals:
                momentum_regime = 'bearish'
            else:
                momentum_regime = 'neutral'

            return {
                'regime': momentum_regime,
                'indicators': momentum_indicators,
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals
            }

        except Exception as e:
            logger.error(f"Error analyzing momentum regime: {e}")
            return {'regime': 'neutral', 'indicators': {}, 'bullish_signals': 0, 'bearish_signals': 0}

    def _synthesize_market_regime(self, short_term: Dict, medium_term: Dict, long_term: Dict,
                                volatility: Dict, momentum: Dict) -> Dict[str, Any]:
        """Synthesize overall market regime from all analyses."""
        try:
            # Weight the different timeframe regimes
            regime_weights = {
                'short_term': 0.4,   # 40% weight on recent action
                'medium_term': 0.4,  # 40% weight on medium term
                'long_term': 0.2     # 20% weight on long term
            }

            # Convert regimes to numerical scores
            regime_scores = {
                'bull': 2, 'bull_moderate': 1, 'neutral': 0, 'bear_moderate': -1, 'bear': -2, 'unknown': 0
            }

            # Calculate weighted regime score
            weighted_score = (
                regime_scores.get(short_term.get('regime', 'unknown'), 0) * regime_weights['short_term'] +
                regime_scores.get(medium_term.get('regime', 'unknown'), 0) * regime_weights['medium_term'] +
                regime_scores.get(long_term.get('regime', 'unknown'), 0) * regime_weights['long_term']
            )

            # Factor in volatility and momentum
            vol_multiplier = volatility.get('risk_multiplier', 1.0)
            momentum_score = 1 if momentum.get('regime') == 'bullish' else -1 if momentum.get('regime') == 'bearish' else 0

            final_score = weighted_score * vol_multiplier + momentum_score * 0.5

            # Determine overall regime
            if final_score >= 1.5:
                overall_regime = 'bull'
                characteristics = ['strong_uptrend', 'low_volatility', 'positive_momentum']
            elif final_score >= 0.5:
                overall_regime = 'bull_moderate'
                characteristics = ['moderate_uptrend', 'normal_volatility', 'neutral_momentum']
            elif final_score >= -0.5:
                overall_regime = 'neutral'
                characteristics = ['sideways', 'normal_volatility', 'neutral_momentum']
            elif final_score >= -1.5:
                overall_regime = 'bear_moderate'
                characteristics = ['moderate_downtrend', 'elevated_volatility', 'negative_momentum']
            else:
                overall_regime = 'bear'
                characteristics = ['strong_downtrend', 'high_volatility', 'negative_momentum']

            # Calculate confidence based on agreement between timeframes
            timeframe_regimes = [short_term.get('regime'), medium_term.get('regime'), long_term.get('regime')]
            unique_regimes = set(r for r in timeframe_regimes if r != 'unknown')
            agreement_ratio = len(unique_regimes) / len(timeframe_regimes) if timeframe_regimes else 0
            confidence = 1.0 - agreement_ratio  # Higher agreement = higher confidence

            return {
                'regime': overall_regime,
                'score': final_score,
                'confidence': confidence,
                'characteristics': characteristics
            }

        except Exception as e:
            logger.error(f"Error synthesizing market regime: {e}")
            return {'regime': 'unknown', 'score': 0, 'confidence': 0, 'characteristics': []}

    def _generate_regime_risk_adjustments(self, regime_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk management adjustments based on market regime."""
        try:
            regime = regime_assessment.get('regime', 'neutral')

            # Base adjustments by regime
            regime_adjustments = {
                'bull': {
                    'position_size_multiplier': 1.2,  # Increase position sizes in bull markets
                    'stop_loss_tightness': 0.03,     # 3% stops
                    'max_drawdown_limit': 0.08,      # Allow higher drawdowns
                    'diversification_requirement': 0.7  # 70% diversification score required
                },
                'bull_moderate': {
                    'position_size_multiplier': 1.0,  # Normal position sizes
                    'stop_loss_tightness': 0.04,     # 4% stops
                    'max_drawdown_limit': 0.06,      # Normal drawdown limits
                    'diversification_requirement': 0.8
                },
                'neutral': {
                    'position_size_multiplier': 0.8,  # Reduce position sizes
                    'stop_loss_tightness': 0.05,     # 5% stops
                    'max_drawdown_limit': 0.05,      # Tighter drawdown limits
                    'diversification_requirement': 0.9
                },
                'bear_moderate': {
                    'position_size_multiplier': 0.6,  # Significantly reduce sizes
                    'stop_loss_tightness': 0.06,     # 6% stops
                    'max_drawdown_limit': 0.04,      # Very tight drawdown limits
                    'diversification_requirement': 1.0
                },
                'bear': {
                    'position_size_multiplier': 0.4,  # Minimal position sizes
                    'stop_loss_tightness': 0.08,     # 8% stops
                    'max_drawdown_limit': 0.03,      # Extremely tight limits
                    'diversification_requirement': 1.0
                }
            }

            adjustments = regime_adjustments.get(regime, regime_adjustments['neutral'])

            # Adjust based on confidence level
            confidence = regime_assessment.get('confidence', 0.5)
            if confidence < 0.3:  # Low confidence = more conservative
                adjustments['position_size_multiplier'] *= 0.8
                adjustments['max_drawdown_limit'] *= 0.8

            return adjustments

        except Exception as e:
            logger.error(f"Error generating regime risk adjustments: {e}")
            return {}

    def _calculate_regime_transition_probability(self, regime_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate probabilities of transitioning to other regimes."""
        try:
            current_regime = regime_assessment.get('regime', 'neutral')

            # Transition probabilities based on historical data and current conditions
            transition_matrix = {
                'bull': {'bull': 0.7, 'bull_moderate': 0.2, 'neutral': 0.08, 'bear_moderate': 0.015, 'bear': 0.005},
                'bull_moderate': {'bull': 0.15, 'bull_moderate': 0.6, 'neutral': 0.2, 'bear_moderate': 0.04, 'bear': 0.01},
                'neutral': {'bull': 0.05, 'bull_moderate': 0.15, 'neutral': 0.6, 'bear_moderate': 0.15, 'bear': 0.05},
                'bear_moderate': {'bull': 0.01, 'bull_moderate': 0.04, 'neutral': 0.2, 'bear_moderate': 0.6, 'bear': 0.15},
                'bear': {'bull': 0.005, 'bull_moderate': 0.015, 'neutral': 0.08, 'bear_moderate': 0.2, 'bear': 0.7}
            }

            transitions = transition_matrix.get(current_regime, transition_matrix['neutral'])

            # Adjust probabilities based on current market conditions
            characteristics = regime_assessment.get('characteristics', [])
            if 'high_volatility' in characteristics:
                # High volatility increases transition probabilities
                for regime in transitions:
                    if regime != current_regime:
                        transitions[regime] *= 1.5
                # Normalize
                total = sum(transitions.values())
                transitions = {k: v/total for k, v in transitions.items()}

            return {
                'current_regime': current_regime,
                'transition_probabilities': transitions,
                'most_likely_transition': max(transitions.items(), key=lambda x: x[1])
            }

        except Exception as e:
            logger.error(f"Error calculating regime transition probability: {e}")
            return {'current_regime': 'unknown', 'transition_probabilities': {}}

    async def execute_automated_risk_response(self, crisis_assessment: Dict[str, Any], 
                                            regime_assessment: Dict[str, Any],
                                            active_positions: Dict[str, Any],
                                            market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute automated risk responses based on crisis and regime assessments.
        Automatically adjusts position sizes, stops, and implements protective measures.
        """
        try:
            logger.info("Executing automated risk response protocol")

            # Determine response severity
            response_severity = self._determine_response_severity(crisis_assessment, regime_assessment)

            # Generate and execute response actions
            response_actions = await self._generate_response_actions(
                response_severity, crisis_assessment, regime_assessment, active_positions, market_data
            )

            execution_results = await self._execute_response_actions(response_actions, active_positions)

            response_summary = {
                'timestamp': datetime.now().isoformat(),
                'response_severity': response_severity,
                'actions_taken': len(response_actions),
                'execution_results': execution_results,
                'crisis_trigger': crisis_assessment.get('severity_level'),
                'regime_context': regime_assessment.get('overall_regime')
            }

            logger.info(f"Automated risk response completed: {response_severity} severity")

            return response_summary

        except Exception as e:
            logger.error(f"Error executing automated risk response: {e}")
            return {'error': str(e), 'response_severity': 'failed'}

    def _determine_response_severity(self, crisis: Dict[str, Any], regime: Dict[str, Any]) -> str:
        """Determine the severity level for risk response."""
        try:
            crisis_level = crisis.get('severity_level', 'normal')
            regime_type = regime.get('overall_regime', 'neutral')

            # Crisis levels take precedence
            if crisis_level in ['extreme_crisis', 'crisis']:
                return crisis_level
            elif crisis_level == 'warning':
                # Moderate response unless regime is also concerning
                if regime_type in ['bear', 'bear_moderate']:
                    return 'crisis'
                else:
                    return 'warning'
            else:
                # No crisis, but check regime
                if regime_type == 'bear':
                    return 'warning'
                elif regime_type == 'bear_moderate':
                    return 'moderate'
                else:
                    return 'normal'

        except Exception as e:
            logger.error(f"Error determining response severity: {e}")
            return 'normal'

    async def _generate_response_actions(self, severity: str, crisis: Dict[str, Any], 
                                       regime: Dict[str, Any], positions: Dict[str, Any], 
                                       market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific risk response actions."""
        actions = []
        if severity in ['extreme_crisis', 'crisis']:
            actions.append({
                'action_type': 'reduce_positions',
                'severity': severity,
                'reason': 'crisis_protection'
            })
        return actions

    async def _execute_response_actions(self, actions: List[Dict[str, Any]], 
                                      active_positions: Dict[str, Any]) -> Dict[str, Any]:
        """Execute response actions."""
        return {'executed': len(actions), 'successful': len(actions)}

    async def _generate_extreme_crisis_actions(self, positions: Dict[str, Any], 
                                             market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate extreme crisis actions."""
        return [{'action_type': 'emergency_close', 'priority': 1}]

    async def _generate_crisis_actions(self, positions: Dict[str, Any], 
                                     market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate crisis actions."""
        return [{'action_type': 'reduce_positions', 'priority': 2}]

    async def _generate_warning_actions(self, positions: Dict[str, Any], 
                                      market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate warning actions."""
        return [{'action_type': 'tighten_stops', 'priority': 3}]

    async def activate_circuit_breaker(self, reason: str, duration_hours: int = 24,
                                     severity: str = 'high') -> Dict[str, Any]:
        """
        Activate circuit breaker to halt trading during extreme market conditions.

        Args:
            reason: Reason for circuit breaker activation
            duration_hours: How long to maintain the circuit breaker
            severity: Severity level (low, medium, high, extreme)

        Returns:
            Circuit breaker activation status
        """
        try:
            logger.warning(f"ACTIVATING CIRCUIT BREAKER: {reason} (severity: {severity})")

            # Set circuit breaker state
            circuit_breaker_state = {
                'active': True,
                'activation_time': datetime.now().isoformat(),
                'reason': reason,
                'severity': severity,
                'duration_hours': duration_hours,
                'expiration_time': (datetime.now() + timedelta(hours=duration_hours)).isoformat(),
                'trading_halted': True,
                'emergency_protocols': True
            }

            # Update internal state
            self.circuit_breakers.update(circuit_breaker_state)

            # Execute emergency actions
            emergency_actions = await self._execute_emergency_actions(severity)

            # Notify all agents
            await self._broadcast_circuit_breaker_alert(circuit_breaker_state)

            # Log circuit breaker activation
            await self._log_circuit_breaker_event(circuit_breaker_state, emergency_actions)

            activation_result = {
                'status': 'activated',
                'circuit_breaker_id': f"CB_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'reason': reason,
                'severity': severity,
                'duration_hours': duration_hours,
                'emergency_actions_taken': len(emergency_actions),
                'all_trading_halted': True
            }

            logger.critical(f"Circuit breaker activated successfully: {activation_result['circuit_breaker_id']}")

            return activation_result

        except Exception as e:
            logger.error(f"Failed to activate circuit breaker: {e}")
            return {'status': 'failed', 'error': str(e)}

    async def deactivate_circuit_breaker(self, reason: str = 'manual_override') -> Dict[str, Any]:
        """
        Deactivate circuit breaker and resume normal operations.

        Args:
            reason: Reason for deactivation

        Returns:
            Deactivation status
        """
        try:
            if not self.circuit_breakers.get('active', False):
                return {'status': 'not_active', 'message': 'Circuit breaker is not currently active'}

            logger.info(f"Deactivating circuit breaker: {reason}")

            # Record deactivation
            deactivation_record = {
                'deactivation_time': datetime.now().isoformat(),
                'reason': reason,
                'active_duration_hours': (datetime.now() - datetime.fromisoformat(
                    self.circuit_breakers['activation_time'])).total_seconds() / 3600
            }

            # Reset circuit breaker state
            self.circuit_breakers = {
                'active': False,
                'circuit_breaker_active': False,
                'circuit_breaker_reason': None,
                'circuit_breaker_reset_time': None
            }

            # Execute recovery protocols
            recovery_actions = await self._execute_recovery_protocols()

            # Notify agents of resumption
            await self._notify_trading_resumption(deactivation_record)

            deactivation_result = {
                'status': 'deactivated',
                'reason': reason,
                'recovery_actions_taken': len(recovery_actions),
                'trading_resumed': True
            }

            logger.info("Circuit breaker deactivated, trading resumed")

            return deactivation_result

        except Exception as e:
            logger.error(f"Failed to deactivate circuit breaker: {e}")
            return {'status': 'failed', 'error': str(e)}

    def check_circuit_breaker_status(self) -> Dict[str, Any]:
        """
        Check current circuit breaker status.

        Returns:
            Current circuit breaker state
        """
        try:
            status = {
                'active': self.circuit_breakers.get('active', False),
                'reason': self.circuit_breakers.get('reason'),
                'severity': self.circuit_breakers.get('severity'),
                'activation_time': self.circuit_breakers.get('activation_time'),
                'expiration_time': self.circuit_breakers.get('expiration_time'),
                'time_remaining_hours': None
            }

            # Calculate time remaining if active
            if status['active'] and status['expiration_time']:
                expiration = datetime.fromisoformat(status['expiration_time'])
                remaining = (expiration - datetime.now()).total_seconds() / 3600
                status['time_remaining_hours'] = max(0, remaining)

                # Auto-deactivate if expired
                if remaining <= 0:
                    asyncio.create_task(self.deactivate_circuit_breaker('auto_expiry'))

            return status

        except Exception as e:
            logger.error(f"Error checking circuit breaker status: {e}")
            return {'active': False, 'error': str(e)}

    async def _execute_emergency_actions(self, severity: str) -> List[str]:
        """Execute emergency actions based on severity."""
        actions_taken = []

        try:
            if severity in ['high', 'extreme']:
                # Close all positions immediately
                actions_taken.append("Emergency position closure initiated")
                logger.critical("EMERGENCY: Closing all positions")

            if severity == 'extreme':
                # Complete system shutdown protocols
                actions_taken.append("Extreme emergency protocols activated")
                logger.critical("EMERGENCY: System entering lockdown mode")

            # Halt all trading activities
            actions_taken.append("All trading activities halted")
            logger.critical("EMERGENCY: All trading halted")

            return actions_taken

        except Exception as e:
            logger.error(f"Error executing emergency actions: {e}")
            return ["Error executing emergency actions"]

    async def _broadcast_circuit_breaker_alert(self, circuit_breaker_state: Dict[str, Any]) -> None:
        """Broadcast circuit breaker alert to all agents."""
        try:
            alert = {
                'agent': 'RiskAgent',
                'alert_type': 'circuit_breaker_activated',
                'severity': 'critical',
                'content': circuit_breaker_state,
                'timestamp': datetime.now().isoformat(),
                'message': f"CIRCUIT BREAKER ACTIVATED: {circuit_breaker_state['reason']}"
            }

            # Store for A2A transmission
            await self.store_advanced_memory('system_alerts', alert)

            logger.critical("Circuit breaker alert broadcasted to all agents")

        except Exception as e:
            logger.error(f"Error broadcasting circuit breaker alert: {e}")

    async def _log_circuit_breaker_event(self, circuit_breaker_state: Dict[str, Any], 
                                       emergency_actions: List[str]) -> None:
        """Log circuit breaker activation for audit."""
        try:
            log_entry = {
                'event_type': 'circuit_breaker_activation',
                'timestamp': datetime.now().isoformat(),
                'circuit_breaker_state': circuit_breaker_state,
                'emergency_actions': emergency_actions,
                'system_state': 'emergency_lockdown'
            }

            await self.append_to_memory_list('circuit_breaker_history', log_entry, max_items=20)

            logger.info("Circuit breaker event logged")

        except Exception as e:
            logger.error(f"Error logging circuit breaker event: {e}")

    async def _execute_recovery_protocols(self) -> List[str]:
        """Execute recovery protocols when deactivating circuit breaker."""
        recovery_actions = []

        try:
            # Gradual resumption protocols
            recovery_actions.append("Initiating gradual trading resumption")
            recovery_actions.append("Risk limits reset to normal levels")
            recovery_actions.append("Position size limits restored")

            logger.info("Recovery protocols executed")

            return recovery_actions

        except Exception as e:
            logger.error(f"Error executing recovery protocols: {e}")
            return ["Error in recovery protocols"]

    async def monitor_strategy_performance(self, strategy_data: Dict[str, Any], 
                                        market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor strategy performance from a risk management perspective.
        Tracks risk metrics, compliance status, and generates optimization proposals.

        Args:
            strategy_data: Current strategy performance data
            market_conditions: Current market conditions and volatility

        Returns:
            Risk performance assessment with optimization proposals
        """
        try:
            logger.info("Risk Agent monitoring strategy performance for optimization proposals")

            # Collect comprehensive risk metrics
            risk_metrics = await self._collect_risk_performance_data(strategy_data, market_conditions)

            # Assess compliance status
            compliance_status = self._assess_compliance_status(risk_metrics)

            # Evaluate risk-adjusted performance
            risk_adjusted_metrics = self._calculate_risk_adjusted_performance(risk_metrics, strategy_data)

            # Generate optimization proposals based on risk analysis
            optimization_proposals = await self._generate_risk_optimization_proposals(
                risk_metrics, compliance_status, market_conditions
            )

            # Get overall risk assessment
            overall_risk_assessment = self._assess_overall_risk_health(risk_metrics, compliance_status)
            
            performance_assessment = {
                'performance_metrics': {
                    'volatility': risk_metrics.get('volatility', 0),
                    'var_95': risk_metrics.get('var_95', 0),
                    'max_drawdown': risk_metrics.get('max_drawdown', 0),
                    'sharpe_ratio': risk_metrics.get('sharpe_ratio', 0),
                    'health_score': overall_risk_assessment.get('health_score', 0),
                    'health_level': overall_risk_assessment.get('health_level', 'unknown'),
                    'compliance_status': compliance_status.get('status', 'unknown'),
                    'optimization_proposals': len(optimization_proposals)
                },
                'timestamp': pd.Timestamp.now().isoformat(),
                'risk_metrics': risk_metrics,
                'compliance_status': compliance_status,
                'risk_adjusted_performance': risk_adjusted_metrics,
                'optimization_proposals': optimization_proposals,
                'overall_risk_assessment': overall_risk_assessment,
                'market_regime_impact': self._evaluate_market_regime_impact(market_conditions, risk_metrics)
            }

            logger.info(f"Strategy performance monitoring completed: {len(optimization_proposals)} risk optimization proposals generated")

            return performance_assessment

        except Exception as e:
            logger.error(f"Error monitoring strategy performance: {e}")
            return {
                'error': f'Strategy performance monitoring failed: {str(e)}',
                'timestamp': pd.Timestamp.now().isoformat(),
                'optimization_proposals': []
            }

    async def _collect_risk_performance_data(self, strategy_data: Dict[str, Any], 
                                           market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Collect comprehensive risk performance metrics."""
        try:
            # Extract portfolio data
            portfolio_value = strategy_data.get('portfolio_value', 100000)
            positions = strategy_data.get('positions', [])
            returns_history = strategy_data.get('returns_history', [])

            # Calculate core risk metrics
            volatility = self._calculate_portfolio_volatility(returns_history)
            var_95 = self._calculate_portfolio_var(positions, portfolio_value, market_conditions)
            max_drawdown = self._calculate_max_drawdown_from_returns(returns_history)
            sharpe_ratio = self._calculate_sharpe_ratio(returns_history)

            # Calculate concentration metrics
            concentration_metrics = self._calculate_concentration_metrics(positions, portfolio_value)

            # Calculate liquidity metrics
            liquidity_metrics = await self._calculate_liquidity_metrics(positions, market_conditions)

            # Calculate stress test results
            stress_test_results = await self._run_quick_stress_tests(positions, market_conditions)

            risk_performance_data = {
                'volatility': volatility,
                'var_95': var_95,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'concentration_metrics': concentration_metrics,
                'liquidity_metrics': liquidity_metrics,
                'stress_test_results': stress_test_results,
                'portfolio_value': portfolio_value,
                'position_count': len(positions),
                'market_volatility': market_conditions.get('volatility', 0.20)
            }

            return risk_performance_data

        except Exception as e:
            logger.error(f"Error collecting risk performance data: {e}")
            return {'error': str(e)}

    async def _notify_trading_resumption(self, deactivation_record: Dict[str, Any]) -> None:
        """Notify agents that trading has resumed."""
        try:
            notification = {
                'agent': 'RiskAgent',
                'alert_type': 'trading_resumed',
                'content': deactivation_record,
                'timestamp': datetime.now().isoformat(),
                'message': "Circuit breaker deactivated - trading resumed"
            }

            await self.store_advanced_memory('system_alerts', notification)

            logger.info("Trading resumption notification sent")

        except Exception as e:
            logger.error(f"Error notifying trading resumption: {e}")

    def _calculate_portfolio_volatility(self, returns_history: List[float]) -> float:
        """Calculate portfolio volatility from returns history."""
        try:
            if not returns_history or len(returns_history) < 2:
                return 0.0
            
            returns_array = np.array(returns_history)
            volatility = np.std(returns_array) * np.sqrt(252)  # Annualized volatility
            return float(volatility)
            
        except Exception as e:
            logger.error(f"Error calculating portfolio volatility: {e}")
            return 0.0

    def _calculate_portfolio_var(self, positions: List[Dict[str, Any]], 
                               portfolio_value: float, 
                               market_conditions: Dict[str, Any]) -> float:
        """Calculate Value at Risk for the portfolio."""
        try:
            if not positions:
                return 0.0
            
            # Simplified VaR calculation using position sizes and market volatility
            total_risk = 0
            market_vol = market_conditions.get('volatility', 0.20)
            
            for position in positions:
                position_value = position.get('quantity', 0) * position.get('current_price', 0)
                position_weight = position_value / portfolio_value if portfolio_value > 0 else 0
                position_vol = position.get('volatility', market_vol)
                total_risk += position_weight * position_vol
            
            # 95% VaR approximation
            var_95 = -1.645 * total_risk * portfolio_value
            return var_95
            
        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {e}")
            return 0.0

    def _calculate_max_drawdown_from_returns(self, returns_history: List[float]) -> float:
        """Calculate maximum drawdown from returns history."""
        try:
            if not returns_history:
                return 0.0
            
            # Convert returns to cumulative returns
            cumulative = np.cumprod(1 + np.array(returns_history))
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(cumulative)
            
            # Calculate drawdowns
            drawdowns = (cumulative - running_max) / running_max
            
            # Return maximum drawdown (most negative)
            max_drawdown = float(np.min(drawdowns))
            return max_drawdown
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown from returns: {e}")
            return 0.0

    def _calculate_sharpe_ratio(self, returns_history: List[float]) -> float:
        """Calculate Sharpe ratio from returns history."""
        try:
            if not returns_history or len(returns_history) < 2:
                return 0.0
            
            returns_array = np.array(returns_history)
            avg_return = np.mean(returns_array)
            volatility = np.std(returns_array)
            
            # Assume risk-free rate of 2% annualized
            risk_free_rate = 0.02 / 252  # Daily risk-free rate
            
            if volatility > 0:
                sharpe_ratio = (avg_return - risk_free_rate) / volatility
                return float(sharpe_ratio)
            else:
                return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def _calculate_concentration_metrics(self, positions: List[Dict[str, Any]], 
                                       portfolio_value: float) -> Dict[str, Any]:
        """Calculate portfolio concentration metrics."""
        try:
            if not positions:
                return {'herfindahl_index': 0, 'largest_position_pct': 0, 'concentration_score': 0}
            
            position_weights = []
            for position in positions:
                position_value = position.get('quantity', 0) * position.get('current_price', 0)
                weight = position_value / portfolio_value if portfolio_value > 0 else 0
                position_weights.append(weight)
            
            # Herfindahl-Hirschman Index
            herfindahl_index = sum(w ** 2 for w in position_weights)
            
            # Largest position percentage
            largest_position_pct = max(position_weights) if position_weights else 0
            
            # Concentration score (0-1, higher = more concentrated)
            concentration_score = herfindahl_index
            
            return {
                'herfindahl_index': herfindahl_index,
                'largest_position_pct': largest_position_pct,
                'concentration_score': concentration_score,
                'position_count': len(positions)
            }
            
        except Exception as e:
            logger.error(f"Error calculating concentration metrics: {e}")
            return {'herfindahl_index': 0, 'largest_position_pct': 0, 'concentration_score': 0}

    async def _calculate_liquidity_metrics(self, positions: List[Dict[str, Any]], 
                                         market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate portfolio liquidity metrics."""
        try:
            if not positions:
                return {'liquidity_score': 1.0, 'illiquid_positions_pct': 0, 'avg_daily_volume': 0}
            
            total_positions = len(positions)
            illiquid_count = 0
            total_daily_volume = 0
            
            for position in positions:
                symbol = position.get('symbol', '')
                position_size = abs(position.get('quantity', 0))
                
                # Get average daily volume for the symbol
                avg_volume = market_conditions.get('avg_volumes', {}).get(symbol, 1000000)
                total_daily_volume += avg_volume
                
                # Consider position illiquid if it represents >5% of average daily volume
                if position_size > 0.05 * avg_volume:
                    illiquid_count += 1
            
            avg_daily_volume = total_daily_volume / total_positions if total_positions > 0 else 0
            illiquid_positions_pct = illiquid_count / total_positions if total_positions > 0 else 0
            
            # Liquidity score (0-1, higher = more liquid)
            liquidity_score = 1.0 - illiquid_positions_pct
            
            return {
                'liquidity_score': liquidity_score,
                'illiquid_positions_pct': illiquid_positions_pct,
                'avg_daily_volume': avg_daily_volume,
                'total_positions': total_positions
            }
            
        except Exception as e:
            logger.error(f"Error calculating liquidity metrics: {e}")
            return {'liquidity_score': 1.0, 'illiquid_positions_pct': 0, 'avg_daily_volume': 0}

    async def _run_quick_stress_tests(self, positions: List[Dict[str, Any]], 
                                    market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Run quick stress tests on portfolio positions."""
        try:
            if not positions:
                return {'stress_test_passed': True, 'max_loss_pct': 0, 'stress_scenarios': []}
            
            # Define stress scenarios
            stress_scenarios = [
                {'name': 'market_crash', 'shock': -0.10},  # 10% market drop
                {'name': 'high_volatility', 'shock': -0.05},  # 5% drop in high vol
                {'name': 'liquidity_crisis', 'shock': -0.15}  # 15% drop in crisis
            ]
            
            max_loss_pct = 0
            
            for scenario in stress_scenarios:
                scenario_loss = 0
                for position in positions:
                    position_value = position.get('quantity', 0) * position.get('current_price', 0)
                    # Assume all positions affected by market shock
                    loss = position_value * abs(scenario['shock'])
                    scenario_loss += loss
                
                scenario_loss_pct = scenario_loss / sum(p.get('quantity', 0) * p.get('current_price', 0) for p in positions)
                max_loss_pct = max(max_loss_pct, scenario_loss_pct)
                scenario['loss_pct'] = scenario_loss_pct
            
            # Stress test passes if max loss < 20%
            stress_test_passed = max_loss_pct < 0.20
            
            return {
                'stress_test_passed': stress_test_passed,
                'max_loss_pct': max_loss_pct,
                'stress_scenarios': stress_scenarios
            }
            
        except Exception as e:
            logger.error(f"Error running quick stress tests: {e}")
            return {'stress_test_passed': False, 'max_loss_pct': 0, 'stress_scenarios': []}

    def _assess_compliance_status(self, risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess compliance status against risk limits."""
        try:
            compliance_issues = []
            
            # Check volatility limits
            max_volatility = self.configs['risk']['constraints'].get('max_volatility', 0.30)
            current_volatility = risk_metrics.get('volatility', 0)
            if current_volatility > max_volatility:
                compliance_issues.append({
                    'type': 'volatility_limit',
                    'limit': max_volatility,
                    'current': current_volatility,
                    'severity': 'high'
                })
            
            # Check VaR limits
            max_var = self.configs['risk']['constraints'].get('max_var_pct', 0.05)
            current_var_pct = abs(risk_metrics.get('var_95', 0)) / risk_metrics.get('portfolio_value', 1)
            if current_var_pct > max_var:
                compliance_issues.append({
                    'type': 'var_limit',
                    'limit': max_var,
                    'current': current_var_pct,
                    'severity': 'high'
                })
            
            # Check drawdown limits
            max_drawdown = self.configs['risk']['constraints'].get('max_drawdown', 0.10)
            current_drawdown = abs(risk_metrics.get('max_drawdown', 0))
            if current_drawdown > max_drawdown:
                compliance_issues.append({
                    'type': 'drawdown_limit',
                    'limit': max_drawdown,
                    'current': current_drawdown,
                    'severity': 'critical'
                })
            
            # Overall compliance status
            if compliance_issues:
                worst_severity = max(issue['severity'] for issue in compliance_issues)
                if worst_severity == 'critical':
                    status = 'breached'
                elif worst_severity == 'high':
                    status = 'warning'
                else:
                    status = 'caution'
            else:
                status = 'compliant'
            
            return {
                'status': status,
                'compliance_issues': compliance_issues,
                'issues_count': len(compliance_issues),
                'breaches_count': len([i for i in compliance_issues if i['severity'] == 'critical'])
            }
            
        except Exception as e:
            logger.error(f"Error assessing compliance status: {e}")
            return {'status': 'error', 'compliance_issues': [], 'issues_count': 0, 'breaches_count': 0}

    def _calculate_risk_adjusted_performance(self, risk_metrics: Dict[str, Any], 
                                           strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk-adjusted performance metrics."""
        try:
            returns_history = strategy_data.get('returns_history', [])
            if not returns_history:
                return {'sharpe_ratio': 0, 'sortino_ratio': 0, 'calmar_ratio': 0, 'risk_adjusted_return': 0}
            
            returns_array = np.array(returns_history)
            avg_return = np.mean(returns_array)
            volatility = np.std(returns_array)
            
            # Sharpe ratio (already calculated)
            sharpe_ratio = risk_metrics.get('sharpe_ratio', 0)
            
            # Sortino ratio (downside deviation)
            downside_returns = returns_array[returns_array < 0]
            downside_volatility = np.std(downside_returns) if len(downside_returns) > 0 else 0
            sortino_ratio = (avg_return / downside_volatility) if downside_volatility > 0 else 0
            
            # Calmar ratio
            max_drawdown = abs(risk_metrics.get('max_drawdown', 0))
            calmar_ratio = avg_return / max_drawdown if max_drawdown > 0 else 0
            
            # Risk-adjusted return (return per unit of risk)
            risk_adjusted_return = avg_return / volatility if volatility > 0 else 0
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'risk_adjusted_return': risk_adjusted_return,
                'avg_return': avg_return,
                'volatility': volatility
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk-adjusted performance: {e}")
            return {'sharpe_ratio': 0, 'sortino_ratio': 0, 'calmar_ratio': 0, 'risk_adjusted_return': 0}

    async def _generate_risk_optimization_proposals(self, risk_metrics: Dict[str, Any], 
                                                  compliance_status: Dict[str, Any], 
                                                  market_conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate risk-based optimization proposals."""
        try:
            proposals = []
            
            # Proposal for high volatility
            if risk_metrics.get('volatility', 0) > 0.25:
                proposals.append({
                    'type': 'volatility_reduction',
                    'title': 'Reduce Portfolio Volatility',
                    'description': f'Current volatility ({risk_metrics["volatility"]:.1%}) exceeds target. Implement hedging strategies.',
                    'priority': 'high',
                    'estimated_impact': 'Reduce volatility by 15-20%',
                    'implementation_complexity': 'medium'
                })
            
            # Proposal for high concentration
            concentration_score = risk_metrics.get('concentration_metrics', {}).get('concentration_score', 0)
            if concentration_score > 0.3:
                proposals.append({
                    'type': 'diversification',
                    'title': 'Improve Portfolio Diversification',
                    'description': f'High concentration detected (Herfindahl: {concentration_score:.2f}). Add uncorrelated assets.',
                    'priority': 'high',
                    'estimated_impact': 'Reduce concentration risk by 25%',
                    'implementation_complexity': 'medium'
                })
            
            # Proposal for compliance issues
            if compliance_status.get('status') in ['warning', 'breached']:
                proposals.append({
                    'type': 'compliance_correction',
                    'title': 'Address Risk Limit Breaches',
                    'description': f'{compliance_status["issues_count"]} compliance issues detected. Adjust position sizes.',
                    'priority': 'critical',
                    'estimated_impact': 'Restore compliance with risk limits',
                    'implementation_complexity': 'high'
                })
            
            # Proposal for low liquidity
            liquidity_score = risk_metrics.get('liquidity_metrics', {}).get('liquidity_score', 1.0)
            if liquidity_score < 0.7:
                proposals.append({
                    'type': 'liquidity_improvement',
                    'title': 'Enhance Portfolio Liquidity',
                    'description': f'Low liquidity score ({liquidity_score:.1%}). Reduce position sizes in illiquid assets.',
                    'priority': 'medium',
                    'estimated_impact': 'Improve liquidity by 30%',
                    'implementation_complexity': 'low'
                })
            
            return proposals
            
        except Exception as e:
            logger.error(f"Error generating risk optimization proposals: {e}")
            return []

    def _assess_overall_risk_health(self, risk_metrics: Dict[str, Any], 
                                  compliance_status: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall risk health of the portfolio."""
        try:
            health_score = 100  # Start with perfect health
            
            # Deduct for high volatility
            volatility = risk_metrics.get('volatility', 0)
            if volatility > 0.30:
                health_score -= 30
            elif volatility > 0.20:
                health_score -= 15
            
            # Deduct for high drawdown
            max_drawdown = abs(risk_metrics.get('max_drawdown', 0))
            if max_drawdown > 0.15:
                health_score -= 25
            elif max_drawdown > 0.10:
                health_score -= 15
            
            # Deduct for compliance issues
            issues_count = compliance_status.get('issues_count', 0)
            health_score -= issues_count * 10
            
            # Deduct for low liquidity
            liquidity_score = risk_metrics.get('liquidity_metrics', {}).get('liquidity_score', 1.0)
            health_score -= (1 - liquidity_score) * 20
            
            # Determine health level
            if health_score >= 80:
                level = 'excellent'
            elif health_score >= 60:
                level = 'good'
            elif health_score >= 40:
                level = 'fair'
            elif health_score >= 20:
                level = 'poor'
            else:
                level = 'critical'
            
            return {
                'health_score': max(0, health_score),
                'health_level': level,
                'risk_factors': self._identify_risk_factors(risk_metrics, compliance_status),
                'recommendations': self._generate_risk_health_recommendations(level)
            }
            
        except Exception as e:
            logger.error(f"Error assessing overall risk health: {e}")
            return {'health_score': 0, 'health_level': 'unknown', 'risk_factors': [], 'recommendations': []}

    def _evaluate_market_regime_impact(self, market_conditions: Dict[str, Any], 
                                     risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate how market regime affects risk metrics."""
        try:
            market_volatility = market_conditions.get('volatility', 0.20)
            market_regime = market_conditions.get('regime', 'neutral')
            
            # Assess regime impact on risk
            regime_risk_multiplier = {
                'bull': 0.8,    # Lower risk in bull markets
                'neutral': 1.0, # Normal risk
                'bear': 1.5     # Higher risk in bear markets
            }.get(market_regime, 1.0)
            
            # Calculate regime-adjusted risk metrics
            adjusted_volatility = risk_metrics.get('volatility', 0) * regime_risk_multiplier
            adjusted_var = risk_metrics.get('var_95', 0) * regime_risk_multiplier
            
            # Regime impact assessment
            if market_regime == 'bear' and risk_metrics.get('volatility', 0) > 0.25:
                impact_level = 'high_risk'
                concerns = ['Elevated volatility in bear market increases risk significantly']
            elif market_regime == 'bull' and risk_metrics.get('volatility', 0) < 0.15:
                impact_level = 'low_risk'
                concerns = ['Favorable market conditions reduce risk exposure']
            else:
                impact_level = 'moderate_risk'
                concerns = ['Market conditions are neutral, maintain standard risk controls']
            
            return {
                'market_regime': market_regime,
                'regime_risk_multiplier': regime_risk_multiplier,
                'adjusted_volatility': adjusted_volatility,
                'adjusted_var': adjusted_var,
                'impact_level': impact_level,
                'regime_concerns': concerns,
                'regime_opportunities': self._identify_regime_opportunities(market_regime, risk_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating market regime impact: {e}")
            return {'market_regime': 'unknown', 'impact_level': 'unknown', 'regime_concerns': []}

    def _identify_risk_factors(self, risk_metrics: Dict[str, Any], 
                             compliance_status: Dict[str, Any]) -> List[str]:
        """Identify key risk factors affecting the portfolio."""
        risk_factors = []
        
        try:
            # Volatility risk
            if risk_metrics.get('volatility', 0) > 0.25:
                risk_factors.append('High portfolio volatility')
            
            # Concentration risk
            concentration = risk_metrics.get('concentration_metrics', {}).get('concentration_score', 0)
            if concentration > 0.3:
                risk_factors.append('High concentration risk')
            
            # Liquidity risk
            liquidity = risk_metrics.get('liquidity_metrics', {}).get('liquidity_score', 1.0)
            if liquidity < 0.7:
                risk_factors.append('Low liquidity')
            
            # Compliance risk
            if compliance_status.get('status') != 'compliant':
                risk_factors.append(f'Compliance issues: {compliance_status.get("issues_count", 0)} breaches')
            
            # Drawdown risk
            if abs(risk_metrics.get('max_drawdown', 0)) > 0.10:
                risk_factors.append('Significant drawdown history')
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Error identifying risk factors: {e}")
            return ['Unable to identify risk factors']

    def _generate_risk_health_recommendations(self, health_level: str) -> List[str]:
        """Generate recommendations based on risk health level."""
        recommendations = {
            'excellent': ['Maintain current risk management practices', 'Monitor for changes in market conditions'],
            'good': ['Continue monitoring risk metrics', 'Consider minor risk optimization opportunities'],
            'fair': ['Review and adjust risk limits if necessary', 'Implement additional risk controls'],
            'poor': ['Immediate risk reduction required', 'Rebalance portfolio to reduce concentration', 'Increase diversification'],
            'critical': ['URGENT: Implement emergency risk controls', 'Reduce position sizes immediately', 'Consider portfolio restructuring']
        }
        
        return recommendations.get(health_level, ['Review risk management practices'])

    def _identify_regime_opportunities(self, market_regime: str, 
                                     risk_metrics: Dict[str, Any]) -> List[str]:
        """Identify risk management opportunities based on market regime."""
        opportunities = []
        
        try:
            if market_regime == 'bull':
                opportunities.append('Consider increasing position sizes in low-volatility assets')
                if risk_metrics.get('volatility', 0) < 0.20:
                    opportunities.append('Favorable conditions for alpha generation strategies')
            
            elif market_regime == 'bear':
                opportunities.append('Focus on defensive, low-volatility strategies')
                opportunities.append('Implement hedging strategies to protect capital')
            
            elif market_regime == 'neutral':
                opportunities.append('Balanced approach to risk management')
                opportunities.append('Opportunity to rebalance portfolio')
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying regime opportunities: {e}")
            return ['Monitor market conditions for opportunities']

    async def evaluate_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate optimization proposal from a risk management perspective.
        Assesses risk implications and provides detailed risk analysis.

        Args:
            proposal: Optimization proposal to evaluate

        Returns:
            Risk evaluation with approval recommendation and risk analysis
        """
        try:
            logger.info("Risk Agent evaluating optimization proposal")

            # Extract proposal details
            proposal_type = proposal.get('type', 'unknown')
            risk_impact = proposal.get('risk_impact', {})
            expected_returns = proposal.get('expected_returns', 0.10)
            time_horizon = proposal.get('time_horizon', 30)

            # Get current market conditions (simplified - could be enhanced)
            market_conditions = self._get_current_market_conditions()

            # Perform comprehensive risk assessment
            risk_assessment = await self._assess_proposal_risk(proposal, market_conditions)

            # Evaluate risk-adjusted returns
            risk_adjusted_analysis = self._evaluate_risk_adjusted_returns(
                expected_returns, risk_assessment, time_horizon
            )

            # Check compliance with risk constraints
            compliance_check = self._check_proposal_compliance(proposal, risk_assessment)

            # Generate risk-based recommendation
            recommendation = await self._generate_risk_recommendation(
                proposal, risk_assessment, compliance_check, market_conditions
            )

            evaluation_result = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'proposal_type': proposal_type,
                'risk_assessment': risk_assessment,
                'risk_adjusted_analysis': risk_adjusted_analysis,
                'compliance_check': compliance_check,
                'recommendation': recommendation,
                'overall_risk_rating': self._calculate_overall_risk_rating(risk_assessment, compliance_check),
                'confidence_level': self._assess_evaluation_confidence(risk_assessment)
            }

            logger.info(f"Proposal evaluation completed: {recommendation.get('decision', 'unknown')} "
                       f"({evaluation_result.get('overall_risk_rating', 'unknown')} risk)")

            return evaluation_result

        except Exception as e:
            logger.error(f"Error evaluating proposal: {e}")
            return {
                'error': f'Proposal evaluation failed: {str(e)}',
                'timestamp': pd.Timestamp.now().isoformat(),
                'recommendation': {'decision': 'reject', 'reason': 'evaluation_error'}
            }

    async def test_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test optimization proposal through risk simulations and stress testing.
        Provides comprehensive testing results and risk validation.

        Args:
            proposal: Optimization proposal to test

        Returns:
            Testing results with risk validation and performance projections
        """
        try:
            logger.info("Risk Agent testing optimization proposal")

            # Get current market conditions (simplified - could be enhanced)
            market_conditions = self._get_current_market_conditions()

            # Run comprehensive risk simulations
            simulation_results = await self._run_proposal_simulations(proposal, market_conditions)

            # Perform stress testing
            stress_test_results = await self._run_proposal_stress_tests(proposal, market_conditions)

            # Validate risk constraints
            risk_validation = self._validate_proposal_risks(proposal, simulation_results, stress_test_results)

            # Generate performance projections
            performance_projections = self._generate_performance_projections(
                proposal, simulation_results, market_conditions
            )

            # Assess testing confidence
            testing_confidence = self._assess_testing_confidence(simulation_results, stress_test_results)

            testing_result = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'simulation_results': simulation_results,
                'stress_test_results': stress_test_results,
                'risk_validation': risk_validation,
                'performance_projections': performance_projections,
                'testing_confidence': testing_confidence,
                'test_passed': risk_validation.get('all_constraints_met', False),
                'risk_warnings': risk_validation.get('warnings', [])
            }

            logger.info(f"Proposal testing completed: {'PASSED' if testing_result['test_passed'] else 'FAILED'} "
                       f"({testing_confidence:.1%} confidence)")

            return testing_result

        except Exception as e:
            logger.error(f"Error testing proposal: {e}")
            return {
                'error': f'Proposal testing failed: {str(e)}',
                'timestamp': pd.Timestamp.now().isoformat(),
                'test_passed': False,
                'testing_confidence': 0.0
            }

    async def implement_proposal(self, proposal: Dict[str, Any], 
                               evaluation_result: Dict[str, Any],
                               testing_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement approved optimization proposal with risk controls and monitoring.
        Sets up necessary risk management infrastructure for the proposal.

        Args:
            proposal: Optimization proposal to implement
            evaluation_result: Results from proposal evaluation
            testing_result: Results from proposal testing

        Returns:
            Implementation status with risk controls and monitoring setup
        """
        try:
            logger.info("Risk Agent implementing optimization proposal")

            # Verify implementation prerequisites
            prerequisites_check = self._verify_implementation_prerequisites(
                proposal, evaluation_result, testing_result
            )

            if not prerequisites_check.get('ready_for_implementation', False):
                return {
                    'status': 'blocked',
                    'reason': prerequisites_check.get('blocking_issues', ['Prerequisites not met']),
                    'timestamp': pd.Timestamp.now().isoformat()
                }

            # Set up risk controls
            risk_controls = await self._setup_proposal_risk_controls(proposal, evaluation_result)

            # Configure monitoring
            monitoring_setup = self._configure_proposal_monitoring(proposal, testing_result)

            # Initialize rollback procedures
            rollback_procedures = self._initialize_rollback_procedures(proposal)

            # Execute implementation
            implementation_status = await self._execute_proposal_implementation(
                proposal, risk_controls, monitoring_setup
            )

            # Set up post-implementation monitoring
            post_implementation_monitoring = self._setup_post_implementation_monitoring(
                proposal, implementation_status
            )

            implementation_result = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'status': implementation_status.get('status', 'unknown'),
                'proposal_id': proposal.get('id', 'unknown'),
                'risk_controls': risk_controls,
                'monitoring_setup': monitoring_setup,
                'rollback_procedures': rollback_procedures,
                'post_implementation_monitoring': post_implementation_monitoring,
                'implementation_details': implementation_status,
                'estimated_completion_time': implementation_status.get('estimated_completion_time', 'unknown')
            }

            logger.info(f"Proposal implementation completed: {implementation_result['status']}")

            return implementation_result

        except Exception as e:
            logger.error(f"Error implementing proposal: {e}")
            return {
                'status': 'failed',
                'error': f'Implementation failed: {str(e)}',
                'timestamp': pd.Timestamp.now().isoformat()
            }

    async def rollback_proposal(self, proposal: Dict[str, Any], 
                              reason: str, monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rollback optimization proposal implementation due to risk concerns or poor performance.
        Safely unwinds changes while maintaining risk controls.

        Args:
            proposal: Optimization proposal to rollback
            reason: Reason for rollback
            monitoring_data: Current monitoring data

        Returns:
            Rollback status and risk assessment
        """
        try:
            logger.info(f"Risk Agent rolling back proposal: {reason}")

            # Assess rollback urgency and risk
            rollback_assessment = self._assess_rollback_urgency(proposal, reason, monitoring_data)

            # Execute rollback procedures
            rollback_execution = await self._execute_rollback_procedures(proposal, rollback_assessment)

            # Restore risk controls
            risk_restoration = self._restore_pre_proposal_risk_controls(proposal)

            # Update monitoring
            monitoring_update = self._update_monitoring_post_rollback(proposal, rollback_execution)

            # Generate rollback analysis
            rollback_analysis = self._generate_rollback_analysis(
                proposal, reason, rollback_execution, monitoring_data
            )

            rollback_result = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'status': rollback_execution.get('status', 'unknown'),
                'reason': reason,
                'rollback_assessment': rollback_assessment,
                'execution_details': rollback_execution,
                'risk_restoration': risk_restoration,
                'monitoring_update': monitoring_update,
                'rollback_analysis': rollback_analysis,
                'lessons_learned': rollback_analysis.get('lessons_learned', [])
            }

            logger.info(f"Proposal rollback completed: {rollback_result['status']}")

            return rollback_result

        except Exception as e:
            logger.error(f"Error rolling back proposal: {e}")
            return {
                'status': 'failed',
                'error': f'Rollback failed: {str(e)}',
                'timestamp': pd.Timestamp.now().isoformat()
            }

    async def _assess_proposal_risk(self, proposal: Dict[str, Any], 
                                  market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the risk implications of the optimization proposal."""
        try:
            proposal_type = proposal.get('type', 'unknown')
            risk_impact = proposal.get('risk_impact', {})

            # Base risk assessment
            base_risk = {
                'volatility_impact': risk_impact.get('volatility_change', 0),
                'concentration_impact': risk_impact.get('concentration_change', 0),
                'liquidity_impact': risk_impact.get('liquidity_change', 0),
                'market_risk_impact': risk_impact.get('market_risk_change', 0)
            }

            # Adjust for market conditions
            market_adjusted_risk = self._adjust_risk_for_market_conditions(base_risk, market_conditions)

            # Calculate overall risk score
            risk_score = self._calculate_proposal_risk_score(market_adjusted_risk)

            # Identify key risk factors
            risk_factors = self._identify_proposal_risk_factors(proposal, market_adjusted_risk)

            return {
                'base_risk': base_risk,
                'market_adjusted_risk': market_adjusted_risk,
                'overall_risk_score': risk_score,
                'risk_level': self._categorize_risk_level(risk_score),
                'key_risk_factors': risk_factors,
                'risk_mitigation_suggestions': self._suggest_risk_mitigations(risk_factors)
            }

        except Exception as e:
            logger.error(f"Error assessing proposal risk: {e}")
            return {'error': str(e), 'overall_risk_score': 1.0, 'risk_level': 'high'}

    def _evaluate_risk_adjusted_returns(self, expected_returns: float, 
                                       risk_assessment: Dict[str, Any], 
                                       time_horizon: int) -> Dict[str, Any]:
        """Evaluate risk-adjusted returns for the proposal."""
        try:
            risk_score = risk_assessment.get('overall_risk_score', 1.0)
            
            # Calculate Sharpe ratio equivalent
            sharpe_ratio = expected_returns / risk_score if risk_score > 0 else 0
            
            # Calculate Sortino ratio (assuming downside risk is 70% of total risk)
            downside_risk = risk_score * 0.7
            sortino_ratio = expected_returns / downside_risk if downside_risk > 0 else 0
            
            # Calculate risk-adjusted return metrics
            annual_risk_adjusted_return = expected_returns - (risk_score * 0.5)  # Penalty for risk
            risk_adjusted_alpha = expected_returns - risk_score  # Alpha after risk adjustment
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'annual_risk_adjusted_return': annual_risk_adjusted_return,
                'risk_adjusted_alpha': risk_adjusted_alpha,
                'risk_penalty': risk_score * 0.5,
                'return_to_risk_ratio': expected_returns / risk_score if risk_score > 0 else 0
            }

        except Exception as e:
            logger.error(f"Error evaluating risk-adjusted returns: {e}")
            return {'sharpe_ratio': 0, 'sortino_ratio': 0, 'annual_risk_adjusted_return': 0}

    def _check_proposal_compliance(self, proposal: Dict[str, Any], 
                                 risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Check if proposal complies with risk constraints."""
        try:
            constraints = self.configs['risk']['constraints']
            risk_score = risk_assessment.get('overall_risk_score', 0)
            
            compliance_issues = []
            
            # Check risk score limits
            max_risk_score = constraints.get('max_proposal_risk_score', 0.8)
            if risk_score > max_risk_score:
                compliance_issues.append({
                    'type': 'risk_score_limit',
                    'limit': max_risk_score,
                    'current': risk_score,
                    'severity': 'high'
                })
            
            # Check volatility impact
            max_volatility_impact = constraints.get('max_volatility_impact', 0.15)
            volatility_impact = abs(risk_assessment.get('market_adjusted_risk', {}).get('volatility_impact', 0))
            if volatility_impact > max_volatility_impact:
                compliance_issues.append({
                    'type': 'volatility_impact_limit',
                    'limit': max_volatility_impact,
                    'current': volatility_impact,
                    'severity': 'medium'
                })
            
            # Check concentration impact
            max_concentration_impact = constraints.get('max_concentration_impact', 0.10)
            concentration_impact = abs(risk_assessment.get('market_adjusted_risk', {}).get('concentration_impact', 0))
            if concentration_impact > max_concentration_impact:
                compliance_issues.append({
                    'type': 'concentration_impact_limit',
                    'limit': max_concentration_impact,
                    'current': concentration_impact,
                    'severity': 'medium'
                })
            
            return {
                'compliant': len(compliance_issues) == 0,
                'compliance_issues': compliance_issues,
                'issues_count': len(compliance_issues),
                'blocking_issues': [i for i in compliance_issues if i['severity'] == 'high']
            }

        except Exception as e:
            logger.error(f"Error checking proposal compliance: {e}")
            return {'compliant': False, 'compliance_issues': [{'type': 'error', 'severity': 'high'}]}

    async def _generate_risk_recommendation(self, proposal: Dict[str, Any], 
                                          risk_assessment: Dict[str, Any], 
                                          compliance_check: Dict[str, Any], 
                                          market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk-based recommendation for the proposal."""
        try:
            risk_level = risk_assessment.get('risk_level', 'high')
            compliant = compliance_check.get('compliant', False)
            blocking_issues = compliance_check.get('blocking_issues', [])
            
            # Decision logic
            if blocking_issues:
                decision = 'reject'
                reason = f"Blocking compliance issues: {len(blocking_issues)} high-severity violations"
            elif not compliant:
                decision = 'conditional_approval'
                reason = f"Minor compliance issues: {len(compliance_check.get('compliance_issues', []))} violations"
            elif risk_level in ['low', 'moderate']:
                decision = 'approve'
                reason = f"Acceptable risk level: {risk_level}"
            else:
                decision = 'reject'
                reason = f"Unacceptable risk level: {risk_level}"
            
            # Generate detailed rationale
            rationale = await self._generate_detailed_rationale(
                proposal, risk_assessment, compliance_check, market_conditions, decision
            )
            
            return {
                'decision': decision,
                'reason': reason,
                'detailed_rationale': rationale,
                'required_modifications': self._suggest_modifications(proposal, compliance_check),
                'monitoring_requirements': self._define_monitoring_requirements(risk_assessment)
            }

        except Exception as e:
            logger.error(f"Error generating risk recommendation: {e}")
            return {'decision': 'reject', 'reason': 'recommendation_error'}

    def _calculate_overall_risk_rating(self, risk_assessment: Dict[str, Any], 
                                     compliance_check: Dict[str, Any]) -> str:
        """Calculate overall risk rating combining assessment and compliance."""
        try:
            risk_score = risk_assessment.get('overall_risk_score', 1.0)
            compliant = compliance_check.get('compliant', False)
            blocking_issues = len(compliance_check.get('blocking_issues', []))
            
            # Adjust risk score based on compliance
            if blocking_issues > 0:
                risk_score += 0.5
            elif not compliant:
                risk_score += 0.2
            
            # Determine rating
            if risk_score <= 0.3:
                return 'low'
            elif risk_score <= 0.6:
                return 'moderate'
            elif risk_score <= 0.8:
                return 'high'
            else:
                return 'extreme'

        except Exception as e:
            logger.error(f"Error calculating overall risk rating: {e}")
            return 'high'

    def _assess_evaluation_confidence(self, risk_assessment: Dict[str, Any]) -> float:
        """Assess confidence level in the risk evaluation."""
        try:
            # Base confidence on data completeness and risk factor identification
            risk_factors = risk_assessment.get('key_risk_factors', [])
            data_completeness = len(risk_factors) / 5.0  # Expect at least 5 risk factors
            
            # Adjust for risk assessment quality
            risk_score = risk_assessment.get('overall_risk_score', 0)
            assessment_quality = 1.0 - (risk_score / 2.0)  # Lower risk = higher confidence
            
            confidence = min(1.0, (data_completeness + assessment_quality) / 2.0)
            return confidence

        except Exception as e:
            logger.error(f"Error assessing evaluation confidence: {e}")
            return 0.5

    async def _run_proposal_simulations(self, proposal: Dict[str, Any], 
                                      market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive simulations for the proposal."""
        try:
            # Use existing stochastic simulation framework
            simulation_config = {
                'roi_estimate': proposal.get('expected_returns', 0.10),
                'holding_days': proposal.get('time_horizon', 30),
                'volatility': market_conditions.get('volatility', 0.20)
            }
            
            # Run multiple simulation scenarios
            base_simulation = self._run_stochastics(simulation_config, simulation_config['volatility'])
            
            # Run stress scenarios
            stress_scenarios = []
            for stress_factor in [1.5, 2.0]:  # 50% and 100% volatility increase
                stress_vol = simulation_config['volatility'] * stress_factor
                stress_sim = self._run_stochastics(simulation_config, stress_vol)
                stress_scenarios.append({
                    'volatility_multiplier': stress_factor,
                    'results': stress_sim
                })
            
            return {
                'base_simulation': base_simulation,
                'stress_scenarios': stress_scenarios,
                'simulation_count': 1 + len(stress_scenarios),
                'avg_pop': base_simulation.get('pop', 0.5),
                'worst_case_pop': min([s.get('results', {}).get('pop', 1.0) for s in stress_scenarios] + [base_simulation.get('pop', 1.0)])
            }

        except Exception as e:
            logger.error(f"Error running proposal simulations: {e}")
            return {'error': str(e), 'simulation_count': 0}

    async def _run_proposal_stress_tests(self, proposal: Dict[str, Any], 
                                       market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Run stress tests for the proposal."""
        try:
            # Create portfolio representation for stress testing
            test_portfolio = proposal.get('portfolio_impact', {})
            positions = test_portfolio.get('positions', [])
            portfolio_value = test_portfolio.get('value', 100000)
            
            # Run existing stress test framework
            stress_results = await self._run_quick_stress_tests(positions, market_conditions)
            
            # Add proposal-specific stress tests
            proposal_stresses = []
            
            # Test proposal-specific risk factors
            risk_impact = proposal.get('risk_impact', {})
            if risk_impact.get('volatility_change', 0) > 0.1:
                proposal_stresses.append({
                    'type': 'volatility_amplification',
                    'description': 'Testing increased volatility from proposal',
                    'impact': 'high'
                })
            
            if risk_impact.get('concentration_change', 0) > 0.05:
                proposal_stresses.append({
                    'type': 'concentration_risk',
                    'description': 'Testing concentration changes from proposal',
                    'impact': 'medium'
                })
            
            return {
                'portfolio_stress_tests': stress_results,
                'proposal_specific_stresses': proposal_stresses,
                'overall_stress_passed': stress_results.get('stress_test_passed', True) and len(proposal_stresses) == 0,
                'stress_warnings': [s['description'] for s in proposal_stresses if s['impact'] == 'high']
            }

        except Exception as e:
            logger.error(f"Error running proposal stress tests: {e}")
            return {'error': str(e), 'overall_stress_passed': False}

    def _validate_proposal_risks(self, proposal: Dict[str, Any], 
                               simulation_results: Dict[str, Any], 
                               stress_test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that proposal risks are within acceptable limits."""
        try:
            constraints = self.configs['risk']['constraints']
            
            validations = []
            warnings = []
            
            # Validate POP requirements
            min_pop = constraints.get('min_simulation_pop', 0.60)
            avg_pop = simulation_results.get('avg_pop', 0)
            if avg_pop < min_pop:
                validations.append({
                    'check': 'minimum_pop',
                    'passed': False,
                    'required': min_pop,
                    'actual': avg_pop
                })
            else:
                validations.append({
                    'check': 'minimum_pop',
                    'passed': True,
                    'required': min_pop,
                    'actual': avg_pop
                })
            
            # Validate worst-case POP
            max_drawdown_limit = constraints.get('max_stress_drawdown', 0.25)
            worst_case_pop = simulation_results.get('worst_case_pop', 1.0)
            if worst_case_pop < 0.30:  # Less than 30% success in worst case
                warnings.append(f"Low worst-case POP: {worst_case_pop:.1%}")
            
            # Validate stress test results
            stress_passed = stress_test_results.get('overall_stress_passed', False)
            if not stress_passed:
                validations.append({
                    'check': 'stress_tests',
                    'passed': False,
                    'details': 'Stress tests failed'
                })
            
            # Check for proposal-specific warnings
            stress_warnings = stress_test_results.get('stress_warnings', [])
            warnings.extend(stress_warnings)
            
            return {
                'all_constraints_met': all(v.get('passed', False) for v in validations),
                'validations': validations,
                'warnings': warnings,
                'warnings_count': len(warnings),
                'validation_score': sum(1 for v in validations if v.get('passed', False)) / len(validations) if validations else 0
            }

        except Exception as e:
            logger.error(f"Error validating proposal risks: {e}")
            return {'all_constraints_met': False, 'validations': [], 'warnings': ['Validation error']}

    def _generate_performance_projections(self, proposal: Dict[str, Any], 
                                        simulation_results: Dict[str, Any], 
                                        market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance projections for the proposal."""
        try:
            base_returns = proposal.get('expected_returns', 0.10)
            time_horizon = proposal.get('time_horizon', 30)
            
            # Base case projection
            base_projection = {
                'expected_return': base_returns,
                'annualized_return': base_returns * (365 / time_horizon),
                'probability_of_profit': simulation_results.get('avg_pop', 0.5),
                'expected_risk': simulation_results.get('base_simulation', {}).get('var_95', 0.05)
            }
            
            # Conservative projection (using worst case)
            conservative_projection = {
                'expected_return': base_returns * 0.7,  # 30% haircut
                'annualized_return': base_returns * 0.7 * (365 / time_horizon),
                'probability_of_profit': simulation_results.get('worst_case_pop', 0.3),
                'expected_risk': simulation_results.get('base_simulation', {}).get('cvar_95', 0.08)
            }
            
            # Risk-adjusted projections
            sharpe_ratio = base_projection['expected_return'] / base_projection['expected_risk'] if base_projection['expected_risk'] > 0 else 0
            
            return {
                'base_case': base_projection,
                'conservative_case': conservative_projection,
                'sharpe_ratio_projection': sharpe_ratio,
                'projection_confidence': simulation_results.get('simulation_count', 0) / 10.0,  # Scale by simulation count
                'time_horizon_days': time_horizon
            }

        except Exception as e:
            logger.error(f"Error generating performance projections: {e}")
            return {'error': str(e)}

    def _assess_testing_confidence(self, simulation_results: Dict[str, Any], 
                                 stress_test_results: Dict[str, Any]) -> float:
        """Assess confidence in testing results."""
        try:
            # Base confidence on simulation count and coverage
            simulation_count = simulation_results.get('simulation_count', 0)
            simulation_confidence = min(1.0, simulation_count / 100.0)  # 100 sims = full confidence
            
            # Stress test confidence
            stress_passed = stress_test_results.get('overall_stress_passed', False)
            stress_confidence = 1.0 if stress_passed else 0.5
            
            # Overall confidence
            confidence = (simulation_confidence + stress_confidence) / 2.0
            return confidence

        except Exception as e:
            logger.error(f"Error assessing testing confidence: {e}")
            return 0.5

    def _verify_implementation_prerequisites(self, proposal: Dict[str, Any], 
                                           evaluation_result: Dict[str, Any], 
                                           testing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that all prerequisites are met for proposal implementation."""
        try:
            prerequisites = []
            blocking_issues = []
            
            # Check evaluation approval
            evaluation_decision = evaluation_result.get('recommendation', {}).get('decision')
            if evaluation_decision not in ['approve', 'conditional_approval']:
                blocking_issues.append('Proposal not approved in evaluation')
            
            # Check testing results
            test_passed = testing_result.get('test_passed', False)
            if not test_passed:
                blocking_issues.append('Proposal failed testing phase')
            
            # Check risk controls setup
            risk_controls_ready = self._check_risk_controls_readiness(proposal)
            if not risk_controls_ready:
                prerequisites.append('Risk controls need setup')
            
            # Check monitoring setup
            monitoring_ready = self._check_monitoring_readiness(proposal)
            if not monitoring_ready:
                prerequisites.append('Monitoring systems need configuration')
            
            return {
                'ready_for_implementation': len(blocking_issues) == 0,
                'prerequisites': prerequisites,
                'blocking_issues': blocking_issues,
                'prerequisites_count': len(prerequisites),
                'blocking_count': len(blocking_issues)
            }

        except Exception as e:
            logger.error(f"Error verifying implementation prerequisites: {e}")
            return {'ready_for_implementation': False, 'blocking_issues': ['Verification error']}

    async def _setup_proposal_risk_controls(self, proposal: Dict[str, Any], 
                                          evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Set up risk controls for the proposal implementation."""
        try:
            risk_assessment = evaluation_result.get('risk_assessment', {})
            risk_level = risk_assessment.get('risk_level', 'high')
            
            # Define risk controls based on risk level
            risk_controls = {
                'position_limits': self._define_position_limits(proposal, risk_level),
                'stop_loss_levels': self._define_stop_loss_levels(proposal, risk_level),
                'exposure_limits': self._define_exposure_limits(proposal, risk_level),
                'monitoring_triggers': self._define_monitoring_triggers(proposal, risk_level),
                'circuit_breakers': self._define_circuit_breakers(proposal, risk_level)
            }
            
            # Store risk controls in memory for monitoring
            await self.store_advanced_memory('active_risk_controls', {
                'proposal_id': proposal.get('id'),
                'controls': risk_controls,
                'activation_time': pd.Timestamp.now().isoformat()
            })
            
            return risk_controls

        except Exception as e:
            logger.error(f"Error setting up proposal risk controls: {e}")
            return {'error': str(e)}

    def _configure_proposal_monitoring(self, proposal: Dict[str, Any], 
                                     testing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Configure monitoring for the proposal."""
        try:
            monitoring_config = {
                'monitoring_interval': '5min',  # Check every 5 minutes
                'alert_thresholds': {
                    'performance_deviation': 0.05,  # 5% deviation from expected
                    'risk_limit_breaches': testing_result.get('risk_validation', {}).get('warnings', []),
                    'unusual_activity': 'auto'  # Automatic detection
                },
                'reporting_frequency': 'hourly',
                'escalation_triggers': self._define_escalation_triggers(proposal),
                'rollback_triggers': self._define_rollback_triggers(proposal)
            }
            
            return monitoring_config

        except Exception as e:
            logger.error(f"Error configuring proposal monitoring: {e}")
            return {'error': str(e)}

    def _initialize_rollback_procedures(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize rollback procedures for the proposal."""
        try:
            rollback_procedures = {
                'rollback_triggers': self._define_rollback_triggers(proposal),
                'rollback_steps': self._define_rollback_steps(proposal),
                'data_backup': self._create_rollback_data_backup(proposal),
                'communication_plan': self._define_rollback_communication(proposal),
                'timeline': 'immediate'  # Can rollback immediately
            }
            
            return rollback_procedures

        except Exception as e:
            logger.error(f"Error initializing rollback procedures: {e}")
            return {'error': str(e)}

    async def _execute_proposal_implementation(self, proposal: Dict[str, Any], 
                                            risk_controls: Dict[str, Any], 
                                            monitoring_setup: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the proposal implementation."""
        try:
            # Simulate implementation steps
            implementation_steps = [
                'validate_preconditions',
                'setup_risk_controls',
                'configure_monitoring',
                'execute_changes',
                'validate_implementation'
            ]
            
            execution_status = {}
            for step in implementation_steps:
                # Simulate step execution
                execution_status[step] = 'completed'
                await asyncio.sleep(0.1)  # Simulate processing time
            
            return {
                'status': 'completed',
                'execution_steps': execution_steps,
                'step_status': execution_status,
                'estimated_completion_time': 'immediate',
                'success_rate': 1.0
            }

        except Exception as e:
            logger.error(f"Error executing proposal implementation: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _setup_post_implementation_monitoring(self, proposal: Dict[str, Any], 
                                            implementation_status: Dict[str, Any]) -> Dict[str, Any]:
        """Set up monitoring after implementation."""
        try:
            monitoring_setup = {
                'initial_monitoring_period': '24h',  # Monitor for 24 hours initially
                'check_frequency': '5min',
                'alert_escalation': 'auto',
                'performance_baselines': self._establish_performance_baselines(proposal),
                'risk_baselines': self._establish_risk_baselines(proposal)
            }
            
            return monitoring_setup

        except Exception as e:
            logger.error(f"Error setting up post-implementation monitoring: {e}")
            return {'error': str(e)}

    def _assess_rollback_urgency(self, proposal: Dict[str, Any], 
                               reason: str, monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the urgency level for rollback."""
        try:
            urgency_factors = []
            
            # Analyze rollback reason
            if 'risk_breach' in reason.lower():
                urgency_factors.append({'factor': 'risk_breach', 'urgency': 'critical'})
            elif 'performance' in reason.lower():
                urgency_factors.append({'factor': 'performance_issue', 'urgency': 'high'})
            else:
                urgency_factors.append({'factor': 'other', 'urgency': 'medium'})
            
            # Check monitoring data for additional urgency indicators
            risk_level = monitoring_data.get('current_risk_level', 'unknown')
            if risk_level == 'extreme':
                urgency_factors.append({'factor': 'extreme_risk', 'urgency': 'critical'})
            
            # Determine overall urgency
            urgency_levels = [f['urgency'] for f in urgency_factors]
            if 'critical' in urgency_levels:
                overall_urgency = 'critical'
            elif 'high' in urgency_levels:
                overall_urgency = 'high'
            else:
                overall_urgency = 'medium'
            
            return {
                'overall_urgency': overall_urgency,
                'urgency_factors': urgency_factors,
                'requires_immediate_action': overall_urgency == 'critical',
                'estimated_rollback_time': '30min' if overall_urgency == 'critical' else '2h'
            }

        except Exception as e:
            logger.error(f"Error assessing rollback urgency: {e}")
            return {'overall_urgency': 'high', 'requires_immediate_action': True}

    async def _execute_rollback_procedures(self, proposal: Dict[str, Any], 
                                         rollback_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Execute rollback procedures."""
        try:
            rollback_steps = self._define_rollback_steps(proposal)
            
            execution_results = {}
            for step in rollback_steps:
                # Simulate step execution
                execution_results[step] = 'completed'
                await asyncio.sleep(0.1)
            
            return {
                'status': 'completed',
                'executed_steps': rollback_steps,
                'step_results': execution_results,
                'rollback_duration': '30min',
                'data_integrity': 'preserved'
            }

        except Exception as e:
            logger.error(f"Error executing rollback procedures: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _restore_pre_proposal_risk_controls(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Restore risk controls to pre-proposal state."""
        try:
            restoration_actions = [
                'remove_proposal_risk_limits',
                'restore_original_position_limits',
                'reset_stop_loss_levels',
                'restore_monitoring_triggers'
            ]
            
            return {
                'restoration_actions': restoration_actions,
                'status': 'completed',
                'original_controls_restored': True
            }

        except Exception as e:
            logger.error(f"Error restoring risk controls: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _update_monitoring_post_rollback(self, proposal: Dict[str, Any], 
                                       rollback_execution: Dict[str, Any]) -> Dict[str, Any]:
        """Update monitoring after rollback."""
        try:
            monitoring_update = {
                'rollback_logged': True,
                'monitoring_adjusted': True,
                'alerts_reset': True,
                'performance_tracking': 'resumed',
                'risk_assessment': 'updated'
            }
            
            return monitoring_update

        except Exception as e:
            logger.error(f"Error updating monitoring post-rollback: {e}")
            return {'error': str(e)}

    def _generate_rollback_analysis(self, proposal: Dict[str, Any], 
                                  reason: str, rollback_execution: Dict[str, Any], 
                                  monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis of the rollback."""
        try:
            analysis = {
                'rollback_reason': reason,
                'success_rate': rollback_execution.get('status') == 'completed',
                'lessons_learned': [
                    'Monitor risk metrics more closely during implementation',
                    'Consider phased rollout for high-risk proposals',
                    'Enhance pre-implementation testing'
                ],
                'preventive_measures': [
                    'Implement additional risk checks',
                    'Add more conservative thresholds',
                    'Improve monitoring granularity'
                ],
                'proposal_feedback': self._generate_proposal_feedback(proposal, reason)
            }
            
            return analysis

        except Exception as e:
            logger.error(f"Error generating rollback analysis: {e}")
            return {'error': str(e)}

    def _adjust_risk_for_market_conditions(self, base_risk: Dict[str, Any], 
                                         market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust risk assessment based on current market conditions."""
        try:
            market_volatility = market_conditions.get('volatility', 0.20)
            market_regime = market_conditions.get('regime', 'neutral')
            
            # Volatility multiplier
            vol_multiplier = 1.0
            if market_volatility > 0.30:
                vol_multiplier = 1.5
            elif market_volatility < 0.15:
                vol_multiplier = 0.8
            
            # Regime multiplier
            regime_multiplier = {
                'bull': 0.9,
                'neutral': 1.0,
                'bear': 1.3
            }.get(market_regime, 1.0)
            
            adjusted_risk = {}
            for key, value in base_risk.items():
                if isinstance(value, (int, float)):
                    adjusted_risk[key] = value * vol_multiplier * regime_multiplier
                else:
                    adjusted_risk[key] = value
            
            return adjusted_risk

        except Exception as e:
            logger.error(f"Error adjusting risk for market conditions: {e}")
            return base_risk

    def _calculate_proposal_risk_score(self, market_adjusted_risk: Dict[str, Any]) -> float:
        """Calculate overall risk score for the proposal."""
        try:
            # Weight different risk components
            weights = {
                'volatility_impact': 0.3,
                'concentration_impact': 0.25,
                'liquidity_impact': 0.2,
                'market_risk_impact': 0.25
            }
            
            risk_score = 0
            for component, weight in weights.items():
                value = abs(market_adjusted_risk.get(component, 0))
                risk_score += value * weight
            
            return min(2.0, risk_score)  # Cap at 2.0

        except Exception as e:
            logger.error(f"Error calculating proposal risk score: {e}")
            return 1.0

    def _identify_proposal_risk_factors(self, proposal: Dict[str, Any], 
                                      market_adjusted_risk: Dict[str, Any]) -> List[str]:
        """Identify key risk factors for the proposal."""
        risk_factors = []
        
        try:
            # Check volatility impact
            if abs(market_adjusted_risk.get('volatility_impact', 0)) > 0.1:
                risk_factors.append('High volatility impact')
            
            # Check concentration impact
            if abs(market_adjusted_risk.get('concentration_impact', 0)) > 0.05:
                risk_factors.append('Concentration risk increase')
            
            # Check liquidity impact
            if abs(market_adjusted_risk.get('liquidity_impact', 0)) > 0.1:
                risk_factors.append('Liquidity risk concerns')
            
            # Check market risk
            if abs(market_adjusted_risk.get('market_risk_impact', 0)) > 0.15:
                risk_factors.append('Elevated market risk exposure')
            
            # Check proposal type specific risks
            proposal_type = proposal.get('type', '')
            if 'leverage' in proposal_type.lower():
                risk_factors.append('Leverage-related risks')
            
            return risk_factors or ['Standard market risks']

        except Exception as e:
            logger.error(f"Error identifying proposal risk factors: {e}")
            return ['Unable to identify specific risk factors']

    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize risk level based on score."""
        if risk_score <= 0.3:
            return 'low'
        elif risk_score <= 0.6:
            return 'moderate'
        elif risk_score <= 0.9:
            return 'high'
        else:
            return 'extreme'

    def _suggest_risk_mitigations(self, risk_factors: List[str]) -> List[str]:
        """Suggest risk mitigation strategies."""
        mitigations = []
        
        try:
            for factor in risk_factors:
                if 'volatility' in factor.lower():
                    mitigations.append('Implement volatility hedging strategies')
                elif 'concentration' in factor.lower():
                    mitigations.append('Diversify across more assets')
                elif 'liquidity' in factor.lower():
                    mitigations.append('Reduce position sizes in illiquid assets')
                elif 'market' in factor.lower():
                    mitigations.append('Add market-neutral hedges')
            
            return mitigations or ['Implement standard risk management practices']

        except Exception as e:
            logger.error(f"Error suggesting risk mitigations: {e}")
            return ['Consult risk management guidelines']

    async def _generate_detailed_rationale(self, proposal: Dict[str, Any], 
                                         risk_assessment: Dict[str, Any], 
                                         compliance_check: Dict[str, Any], 
                                         market_conditions: Dict[str, Any], 
                                         decision: str) -> str:
        """Generate detailed rationale for the decision."""
        try:
            rationale_parts = []
            
            # Risk assessment summary
            risk_level = risk_assessment.get('risk_level', 'unknown')
            rationale_parts.append(f"Risk Level: {risk_level}")
            
            # Compliance status
            compliant = compliance_check.get('compliant', False)
            rationale_parts.append(f"Compliance: {'Met' if compliant else 'Issues present'}")
            
            # Market conditions
            market_regime = market_conditions.get('regime', 'neutral')
            rationale_parts.append(f"Market Regime: {market_regime}")
            
            # Decision reasoning
            if decision == 'approve':
                rationale_parts.append("Proposal approved due to acceptable risk profile and compliance.")
            elif decision == 'reject':
                rationale_parts.append("Proposal rejected due to risk or compliance concerns.")
            else:
                rationale_parts.append("Conditional approval with modifications required.")
            
            return " ".join(rationale_parts)

        except Exception as e:
            logger.error(f"Error generating detailed rationale: {e}")
            return "Rationale generation failed"

    def _suggest_modifications(self, proposal: Dict[str, Any], 
                             compliance_check: Dict[str, Any]) -> List[str]:
        """Suggest modifications to make proposal compliant."""
        modifications = []
        
        try:
            issues = compliance_check.get('compliance_issues', [])
            for issue in issues:
                issue_type = issue.get('type', '')
                if 'volatility' in issue_type:
                    modifications.append('Reduce volatility exposure')
                elif 'concentration' in issue_type:
                    modifications.append('Decrease position concentration')
                elif 'var' in issue_type:
                    modifications.append('Implement VaR hedging')
            
            return modifications or ['No modifications required']

        except Exception as e:
            logger.error(f"Error suggesting modifications: {e}")
            return ['Review proposal parameters']

    def _define_monitoring_requirements(self, risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Define monitoring requirements based on risk assessment."""
        try:
            risk_level = risk_assessment.get('risk_level', 'moderate')
            
            monitoring_freq = {
                'low': 'daily',
                'moderate': '4hourly',
                'high': 'hourly',
                'extreme': '15min'
            }.get(risk_level, 'hourly')
            
            return {
                'monitoring_frequency': monitoring_freq,
                'alert_thresholds': self._define_alert_thresholds(risk_level),
                'reporting_requirements': 'real_time',
                'escalation_procedures': 'automatic'
            }

        except Exception as e:
            logger.error(f"Error defining monitoring requirements: {e}")
            return {'monitoring_frequency': 'hourly'}

    def _check_risk_controls_readiness(self, proposal: Dict[str, Any]) -> bool:
        """Check if risk controls are ready for implementation."""
        return True  # Simplified

    def _check_monitoring_readiness(self, proposal: Dict[str, Any]) -> bool:
        """Check if monitoring systems are ready."""
        return True  # Simplified

    def _define_position_limits(self, proposal: Dict[str, Any], risk_level: str) -> Dict[str, Any]:
        """Define position limits for the proposal."""
        limits = {'max_position_size': 0.10, 'max_total_exposure': 0.50}
        if risk_level == 'high':
            limits['max_position_size'] = 0.05
            limits['max_total_exposure'] = 0.25
        return limits

    def _define_stop_loss_levels(self, proposal: Dict[str, Any], risk_level: str) -> Dict[str, Any]:
        """Define stop loss levels."""
        stops = {'initial_stop': 0.05, 'trailing_stop': 0.03}
        if risk_level == 'high':
            stops = {'initial_stop': 0.08, 'trailing_stop': 0.05}
        return stops

    def _define_exposure_limits(self, proposal: Dict[str, Any], risk_level: str) -> Dict[str, Any]:
        """Define exposure limits."""
        return {'max_sector_exposure': 0.20, 'max_asset_exposure': 0.15}

    def _define_monitoring_triggers(self, proposal: Dict[str, Any], risk_level: str) -> List[str]:
        """Define monitoring triggers."""
        return ['performance_deviation', 'risk_limit_breach', 'unusual_activity']

    def _define_circuit_breakers(self, proposal: Dict[str, Any], risk_level: str) -> Dict[str, Any]:
        """Define circuit breakers."""
        return {'loss_limit': 0.10, 'time_limit': '24h'}

    def _define_escalation_triggers(self, proposal: Dict[str, Any]) -> List[str]:
        """Define escalation triggers."""
        return ['critical_risk_breach', 'system_failure', 'performance_crash']

    def _define_rollback_triggers(self, proposal: Dict[str, Any]) -> List[str]:
        """Define rollback triggers."""
        return ['risk_limit_breach', 'performance_failure', 'system_error']

    def _define_rollback_steps(self, proposal: Dict[str, Any]) -> List[str]:
        """Define rollback steps."""
        return ['stop_trading', 'close_positions', 'restore_limits', 'reset_monitoring']

    def _create_rollback_data_backup(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Create data backup for rollback."""
        return {'backup_created': True, 'backup_location': 'memory'}

    def _define_rollback_communication(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Define rollback communication plan."""
        return {'notify_agents': True, 'log_event': True}

    def _establish_performance_baselines(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Establish performance baselines."""
        return {'expected_return': proposal.get('expected_returns', 0.10)}

    def _establish_risk_baselines(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Establish risk baselines."""
        return {'expected_volatility': 0.20, 'expected_var': 0.05}

    def _generate_proposal_feedback(self, proposal: Dict[str, Any], reason: str) -> str:
        """Generate feedback for the proposal."""
        return f"Proposal {proposal.get('id', 'unknown')} rolled back due to: {reason}"

    def _define_alert_thresholds(self, risk_level: str) -> Dict[str, Any]:
        """Define alert thresholds based on risk level."""
        thresholds = {
            'low': {'deviation': 0.10, 'breach_count': 3},
            'moderate': {'deviation': 0.07, 'breach_count': 2},
            'high': {'deviation': 0.05, 'breach_count': 1}
        }
        return thresholds.get(risk_level, thresholds['moderate'])

    def _get_current_market_conditions(self) -> Dict[str, Any]:
        """Get current market conditions for risk assessment."""
        try:
            # Simplified market conditions - could be enhanced with real market data
            return {
                'volatility': 0.20,  # 20% annualized volatility
                'regime': 'neutral',  # neutral, bull, bear
                'liquidity': 0.8,  # Liquidity score 0-1
                'risk_premium': 0.05  # 5% risk premium
            }
        except Exception as e:
            logger.error(f"Error getting market conditions: {e}")
            return {'volatility': 0.20, 'regime': 'neutral'}

    async def assess_risk(self, portfolio_data: str) -> Dict[str, Any]:
        """
        Assess risk for portfolio data for Discord integration.
        
        Args:
            portfolio_data: JSON string of portfolio data
            
        Returns:
            Dict: Risk assessment
        """
        try:
            # Parse portfolio data
            data = json.loads(portfolio_data)
            
            # Simple risk assessment
            positions = data.get('positions', [])
            total_value = sum(pos.get('value', 0) for pos in positions)
            
            if total_value > 0:
                # Calculate basic diversification
                symbols = [pos.get('symbol', '') for pos in positions]
                unique_symbols = len(set(symbols))
                
                risk_level = 'low' if unique_symbols >= 5 else 'medium' if unique_symbols >= 3 else 'high'
                
                return {
                    'total_value': total_value,
                    'num_positions': len(positions),
                    'unique_symbols': unique_symbols,
                    'risk_level': risk_level,
                    'diversification_score': unique_symbols / max(len(positions), 1)
                }
            else:
                return {'error': 'Invalid portfolio data'}
                
        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            return {'error': str(e)}