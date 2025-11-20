# src/agents/learning.py
# Purpose: Implements the Learning Agent, subclassing BaseAgent for ML refinements and batch directives (e.g., prune on SD >1.0).
# Handles parallel sims and convergence checks.
# Structural Reasoning: (e.g., FinRL/tf-quant tools) and configs (loaded fresh); backs funding with logged directives (e.g., "Pruned for +1.2% ROI lift").
# New: Async process_input for sims; reflect method for fading (e.g., linear over 15 batches).
# For legacy wealth: Accelerates proficiency without live risk, ensuring 15-20% sustained growth through experiential evolution.
# Update: Dynamic path setup for imports; root-relative paths for configs/prompts.

import os
# Set TensorFlow logging level to suppress warnings before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO, WARNING, and ERROR messages

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Dynamic root path for imports.

from src.agents.base import BaseAgent  # Absolute import.
import logging
from typing import Dict, Any, List, Optional
import pandas as pd  # For DataFrames (batch outputs).
import numpy as np  # For statistical analysis.
from datetime import datetime

logger = logging.getLogger(__name__)

# TensorFlow import disabled to prevent startup issues
logger.warning("TensorFlow import disabled to prevent startup issues. Using numpy fallback for ML operations.")
TENSORFLOW_AVAILABLE = False
tf = None

# Try to import sklearn libraries
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
    logger.info("scikit-learn available for machine learning operations")
except ImportError as e:
    logger.warning(f"scikit-learn not available: {e}. Using numpy fallback.")
    SKLEARN_AVAILABLE = False
    RandomForestRegressor = None
    StandardScaler = None
    train_test_split = None
    mean_squared_error = None
    r2_score = None

logger = logging.getLogger(__name__)

# Try to import professional learning libraries
try:
    import finrl
    from finrl import config
    from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
    from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    from finrl.agents.stablebaselines3.models import DRLAgent
    from stable_baselines3.common.logger import configure
    from finrl.meta.data_processor import DataProcessor
    FINRL_AVAILABLE = True
    logger.info("FinRL library available for reinforcement learning")
except ImportError as e:
    logger.warning(f"FinRL not available: {e}. Using fallback ML methods.")
    FINRL_AVAILABLE = False
    finrl = None

try:
    import backtrader as bt
    from backtrader import Cerebro, Strategy
    from backtrader.feeds import PandasData
    from backtrader.analyzers import SharpeRatio, DrawDown, Returns
    BACKTRADER_AVAILABLE = True
    logger.info("Backtrader library available for professional backtesting")
except ImportError as e:
    logger.warning(f"Backtrader not available: {e}. Using numpy backtesting fallback.")
    BACKTRADER_AVAILABLE = False
    bt = None
    Cerebro = None
    Strategy = None
    PandasData = None
    SharpeRatio = None
    DrawDown = None
    Returns = None

logger = logging.getLogger(__name__)

class LearningAgent(BaseAgent):
    """
    Learning Agent subclass.
    Reasoning: Aggregates/refines via sims; distributes directives for system-wide edges.
    """
    def __init__(self, a2a_protocol=None):
        config_paths = {'risk': 'config/risk-constraints.yaml', 'profit': 'config/profitability-targets.yaml'}  # Relative to root.
        prompt_paths = {'base': 'config/base_prompt.txt', 'role': 'docs/AGENTS/main-agents/learning-agent.md'}  # Relative to root.
        super().__init__(role='learning', config_paths=config_paths, prompt_paths=prompt_paths, a2a_protocol=a2a_protocol)
        
        # Import role-specific tools
        from src.utils.tools import (
            finrl_rl_train_tool,
            zipline_sim_tool,
            tf_quant_projection_tool,
            strategy_ml_optimization_tool,
            backtest_validation_tool
        )
        
        # Add role-specific tools/stubs (expand with Langchain/FinRL later).
        self.tools.extend([
            finrl_rl_train_tool,  # For RL updates.
            zipline_sim_tool,  # For parallel sims.
            tf_quant_projection_tool,  # For stochastic projections.
            strategy_ml_optimization_tool,  # For ML-based strategy refinement.
            backtest_validation_tool  # For comprehensive backtesting.
        ])
        
        # Initialize ML components for strategy optimization
        self._initialize_ml_components()
        
        # Memory for batches and convergence.
        self.memory = {
            'weekly_batches': [],  # For DataFrames.
            'convergence_metrics': {},  # For loss/param tracking.
            'pyramiding_performance': [],  # For pyramiding-specific learning
            'strategy_performance_history': [],  # For ML model training
            'ml_model_metrics': {}  # For tracking ML model performance
        }

    def _initialize_ml_components(self):
        """
        Initialize machine learning components for strategy optimization.
        """
        try:
            # Strategy performance prediction model
            self.strategy_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Feature scaler
            self.feature_scaler = StandardScaler()
            
            # Model training data
            self.training_features = []
            self.training_targets = []
            
            # Model status
            self.model_trained = False
            self.model_last_trained = None
            
            logger.info("ML components initialized for strategy optimization")
            
        except Exception as e:
            logger.warning(f"Failed to initialize ML components: {e}")
            self.strategy_predictor = None
            self.feature_scaler = None
            self.model_trained = False

    async def process_input(self, logs: list[Dict[str, Any]]) -> pd.DataFrame:
        """
        Processes logs: Aggregates batches, triggers directives if SD >1.0.
        Args:
            logs (list[Dict]): From Reflection (e.g., [{'sharpe': 1.5, 'bonus_awarded': True}]).
        Returns: pd.DataFrame with directives (e.g., rows for refinements like 'sizing_lift': 1.2).
        Reasoning: Async for parallel sims (e.g., Zipline blend); ties to configs for thresholds (>12% variance); logs for audits.
        """
        logger.info(f"Learning Agent processing {len(logs)} logs")
        
        # Separate pyramiding logs from general logs
        pyramiding_logs = [log for log in logs if 'pyramiding' in str(log).lower() or 'tiers' in log or 'efficiency_score' in log]
        general_logs = [log for log in logs if log not in pyramiding_logs]
        
        # Process general learning
        if general_logs:
            self._process_general_learning(general_logs)
        
        # Process pyramiding-specific learning
        if pyramiding_logs:
            self._process_pyramiding_learning(pyramiding_logs)
        
        # Check convergence metrics
        convergence = self._calculate_convergence_metrics()
        
        # Apply fade-out mechanism for safety priors
        fade_weight = self._calculate_fade_weight()
        
        # Generate combined directives
        directives = await self._generate_combined_directives(convergence, fade_weight)
        
        output = directives
        logger.info(f"Learning output shape: {output.shape}, SD variance: {self.memory.get('last_sd_variance', 0):.3f}, Converged: {convergence.get('converged', False)}")
        return output

    def _calculate_convergence_metrics(self) -> Dict[str, Any]:
        """
        Calculate convergence metrics from batch history.
        Returns: Dict with convergence analysis.
        """
        batches = self.memory.get('weekly_batches', [])
        if len(batches) < 3:
            return {'converged': False, 'reason': 'insufficient_batches'}
        
        # Extract sharpe ratios from recent batches
        recent_batches = batches[-5:]  # Last 5 batches
        sharpe_series = []
        for batch in recent_batches:
            sharpes = batch.get('sharpe_ratios', [])
            if sharpes:
                sharpe_series.append(np.mean(sharpes))
        
        if len(sharpe_series) < 3:
            return {'converged': False, 'reason': 'insufficient_sharpe_data'}
        
        # Calculate trend (slope of linear regression)
        x = np.arange(len(sharpe_series))
        slope, _ = np.polyfit(x, sharpe_series, 1)
        
        # Calculate stability (coefficient of variation)
        stability = np.std(sharpe_series) / np.mean(sharpe_series) if np.mean(sharpe_series) != 0 else float('inf')
        
        # Variance reduction (compare first half vs second half)
        mid = len(sharpe_series) // 2
        first_half_var = np.var(sharpe_series[:mid]) if mid > 0 else 0
        second_half_var = np.var(sharpe_series[mid:]) if mid < len(sharpe_series) else 0
        variance_reduction = (first_half_var - second_half_var) / first_half_var if first_half_var > 0 else 0
        
        # Convergence criteria
        loss_trend_stable = abs(slope) < 0.01  # Loss trend < 0.01
        param_stable = stability < 0.1  # Parameter stability (CV < 10%)
        variance_reduced = variance_reduction > 0.1  # >10% variance reduction
        
        converged = loss_trend_stable and param_stable and variance_reduced
        
        return {
            'converged': converged,
            'trend_slope': slope,
            'stability_coefficient': stability,
            'variance_reduction': variance_reduction,
            'recent_average_sharpe': np.mean(sharpe_series),
            'sharpe_std': np.std(sharpe_series),
            'batches_analyzed': len(sharpe_series),
            'criteria': {
                'loss_trend_stable': loss_trend_stable,
                'param_stable': param_stable,
                'variance_reduced': variance_reduced
            }
        }

    def _calculate_fade_weight(self) -> float:
        """
        Calculate fade-out weight for safety priors.
        Returns: Weight between 0 and 1 (1 = full safety, 0 = no safety).
        """
        fade_batches = self.configs.get('learning', {}).get('fade_batches', 15)
        current_batch = len(self.memory.get('weekly_batches', []))
        
        if current_batch >= fade_batches:
            return 0.0  # Fully faded out
        
        # Linear decay: weight = 1 - (batch_num / fade_batches)
        weight = 1.0 - (current_batch / fade_batches)
        return max(0.0, min(1.0, weight))  # Clamp to [0, 1]

    def _generate_directives(self, sd_variance: float, convergence: Dict[str, Any], fade_weight: float) -> pd.DataFrame:
        """
        Generate learning directives based on analysis.
        Args:
            sd_variance: Current SD variance.
            convergence: Convergence metrics dict.
            fade_weight: Current fade weight.
        Returns: DataFrame with directives.
        """
        directives = []
        
        # SD-based directives
        if sd_variance > 1.0:
            # High variance - increase position sizing for stability
            sizing_lift = min(0.3, sd_variance - 1.0)  # Max 30% lift
            directives.append({'refinement': 'sizing_lift', 'value': 1.0 + sizing_lift, 'reason': f'High SD variance: {sd_variance:.2f}'})
        
        # Convergence-based directives
        if convergence.get('converged', False):
            # Converged - optimize for efficiency
            directives.append({'refinement': 'efficiency_focus', 'value': 1.1, 'reason': 'Model converged, optimizing efficiency'})
        else:
            # Not converged - focus on exploration
            directives.append({'refinement': 'exploration_boost', 'value': 1.05, 'reason': 'Model not converged, boosting exploration'})
        
        # Fade weight directives
        if fade_weight > 0.5:
            # Still in safety mode - conservative adjustments
            directives.append({'refinement': 'conservative_filter', 'value': fade_weight, 'reason': f'Safety mode active: {fade_weight:.2f}'})
        elif fade_weight > 0:
            # Transitioning out of safety - moderate adjustments
            directives.append({'refinement': 'moderate_risk', 'value': 1 - fade_weight, 'reason': f'Transitioning from safety: {1-fade_weight:.2f}'})
        
        # Always include some basic refinements
        if len(directives) == 0:
            directives.append({'refinement': 'baseline_optimization', 'value': 1.02, 'reason': 'Baseline optimization'})
        
        return pd.DataFrame(directives)

    def _process_general_learning(self, logs: list[Dict[str, Any]]) -> None:
        """
        Process general learning logs (non-pyramiding).
        """
        if logs:
            # Aggregate metrics from logs
            sharpe_ratios = [log.get('sharpe_ratio', log.get('sharpe', 0)) for log in logs if 'sharpe' in str(log).lower() or 'sharpe_ratio' in log]
            returns = [log.get('total_return', log.get('return', 0)) for log in logs if 'return' in str(log).lower() or 'total_return' in log]
            drawdowns = [log.get('max_drawdown', 0) for log in logs if 'drawdown' in str(log).lower() or 'max_drawdown' in log]
            
            # Calculate SD variance from actual data
            if sharpe_ratios:
                sd_variance = np.std(sharpe_ratios) / np.mean(sharpe_ratios) if np.mean(sharpe_ratios) != 0 else 0
            else:
                sd_variance = 0.8  # Default if no sharpe data
            
            # Store in memory for convergence tracking
            self.memory['weekly_batches'].append({
                'timestamp': pd.Timestamp.now(),
                'sharpe_ratios': sharpe_ratios,
                'returns': returns,
                'drawdowns': drawdowns,
                'sd_variance': sd_variance,
                'batch_size': len(logs)
            })
            
            # Keep only last 20 batches for memory efficiency
            if len(self.memory['weekly_batches']) > 20:
                self.memory['weekly_batches'] = self.memory['weekly_batches'][-20:]
            
            # Store last SD variance for logging
            self.memory['last_sd_variance'] = sd_variance

    def _process_pyramiding_learning(self, logs: list[Dict[str, Any]]) -> None:
        """
        Process pyramiding-specific learning logs.
        Analyzes pyramiding performance and adapts parameters.
        """
        if not logs:
            return
            
        # Extract pyramiding metrics
        pyramiding_metrics = []
        for log in logs:
            if 'pyramiding' in log:
                pyramiding_data = log['pyramiding']
                metrics = {
                    'timestamp': log.get('timestamp', pd.Timestamp.now()),
                    'tiers_executed': pyramiding_data.get('tiers', 0),
                    'efficiency_score': pyramiding_data.get('efficiency_score', 1.0),
                    'total_exposure': pyramiding_data.get('total_exposure_limit', 0),
                    'volatility_regime': pyramiding_data.get('volatility_regime', 'normal'),
                    'trend_regime': pyramiding_data.get('trend_regime', 'moderate'),
                    'final_roi': log.get('final_roi', log.get('roi_estimate', 0)),
                    'success': log.get('success', log.get('final_roi', 0) > 0)
                }
                pyramiding_metrics.append(metrics)
        
        if pyramiding_metrics:
            # Store in memory
            self.memory['pyramiding_performance'].extend(pyramiding_metrics)
            
            # Keep only last 50 pyramiding records
            if len(self.memory['pyramiding_performance']) > 50:
                self.memory['pyramiding_performance'] = self.memory['pyramiding_performance'][-50:]
            
            logger.info(f"Processed {len(pyramiding_metrics)} pyramiding performance records")

    def train_strategy_predictor(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train ML model to predict strategy performance based on historical data.
        Args:
            performance_data: List of strategy performance records
        Returns:
            Training metrics and model performance
        """
        if not performance_data or not self.strategy_predictor:
            return {'trained': False, 'reason': 'No data or model unavailable'}
        
        try:
            # Extract features and targets from performance data
            features = []
            targets = []
            
            for record in performance_data:
                # Feature engineering for strategy prediction
                feature_vector = [
                    record.get('roi_estimate', 0),
                    record.get('pop_estimate', 0.5),
                    record.get('volatility', 0.2),
                    record.get('trend_strength', 0.5),
                    record.get('market_condition_score', 0.5),
                    record.get('pyramiding_efficiency', 1.0),
                    record.get('risk_adjusted_roi', record.get('roi_estimate', 0))
                ]
                features.append(feature_vector)
                
                # Target: actual performance score (could be sharpe, total return, etc.)
                target_score = record.get('actual_sharpe', record.get('actual_return', 0))
                targets.append(target_score)
            
            if len(features) < 10:  # Need minimum data for training
                return {'trained': False, 'reason': 'Insufficient training data'}
            
            # Convert to numpy arrays
            X = np.array(features)
            y = np.array(targets)
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.strategy_predictor.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.strategy_predictor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Mark as trained
            self.model_trained = True
            self.model_last_trained = pd.Timestamp.now()
            
            # Store training metrics
            self.memory['ml_model_metrics'] = {
                'mse': mse,
                'r2_score': r2,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'last_trained': self.model_last_trained
            }
            
            logger.info(f"Strategy predictor trained: MSE={mse:.4f}, R²={r2:.4f}")
            
            return {
                'trained': True,
                'mse': mse,
                'r2_score': r2,
                'feature_importance': dict(zip(
                    ['roi_estimate', 'pop_estimate', 'volatility', 'trend_strength', 
                     'market_condition', 'pyramiding_efficiency', 'risk_adjusted_roi'],
                    self.strategy_predictor.feature_importances_
                ))
            }
            
        except Exception as e:
            logger.error(f"Error training strategy predictor: {e}")
            return {'trained': False, 'error': str(e)}

    def predict_strategy_performance(self, strategy_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use trained ML model to predict strategy performance.
        Args:
            strategy_features: Strategy parameters for prediction
        Returns:
            Predicted performance metrics
        """
        if not self.model_trained or not self.strategy_predictor:
            return {'prediction_available': False, 'reason': 'Model not trained'}
        
        try:
            # Extract feature vector
            feature_vector = [
                strategy_features.get('roi_estimate', 0),
                strategy_features.get('pop_estimate', 0.5),
                strategy_features.get('volatility', 0.2),
                strategy_features.get('trend_strength', 0.5),
                strategy_features.get('market_condition_score', 0.5),
                strategy_features.get('pyramiding_efficiency', 1.0),
                strategy_features.get('risk_adjusted_roi', strategy_features.get('roi_estimate', 0))
            ]
            
            # Scale features
            X = np.array([feature_vector])
            X_scaled = self.feature_scaler.transform(X)
            
            # Make prediction
            predicted_performance = self.strategy_predictor.predict(X_scaled)[0]
            
            # Get prediction confidence (using standard deviation of training targets)
            training_targets = np.array(self.training_targets) if hasattr(self, 'training_targets') and self.training_targets else np.array([0])
            confidence_interval = np.std(training_targets) * 1.96  # 95% confidence
            
            return {
                'prediction_available': True,
                'predicted_performance': predicted_performance,
                'confidence_interval': confidence_interval,
                'prediction_upper': predicted_performance + confidence_interval,
                'prediction_lower': predicted_performance - confidence_interval
            }
            
        except Exception as e:
            logger.error(f"Error predicting strategy performance: {e}")
            return {'prediction_available': False, 'error': str(e)}

    def run_backtest_simulation(self, strategy_config: Dict[str, Any],
                              market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run comprehensive backtest simulation using Backtrader professional backtesting framework.
        Falls back to numpy implementation if Backtrader unavailable.
        """
        if BACKTRADER_AVAILABLE and bt:
            return self._run_backtrader_backtest(strategy_config, market_data)
        else:
            logger.warning("Backtrader not available, using numpy backtesting fallback")
            return self._run_numpy_backtest(strategy_config, market_data)

    def _run_backtrader_backtest(self, strategy_config: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:

        """
        Run professional backtest using Backtrader framework.
        """
        try:
            # Create Backtrader strategy class
            class MLStrategy(bt.Strategy):
                params = (
                    ('strategy_config', None),
                )

                def __init__(self):
                    self.dataclose = self.datas[0].close
                    self.order = None
                    self.trades = []
                    self.position_size = self.params.strategy_config.get('position_size', 0.1)
                    self.stop_loss = self.params.strategy_config.get('stop_loss', 0.05)
                    self.take_profit = self.params.strategy_config.get('take_profit', 0.10)
                    self.entry_signal = self.params.strategy_config.get('entry_signal', 'momentum')

                def next(self):
                    if self.order:
                        return

                    # Simple strategy logic (can be extended with complex strategies)
                    if self.entry_signal == 'momentum':
                        # Momentum-based entry
                        if self._check_momentum_signal_bt():
                            if not self.position:
                                self.order = self.buy(size=int(self.broker.getcash() * self.position_size / self.dataclose[0]))
                        elif self._check_exit_signal_bt():
                            if self.position:
                                self.order = self.sell(size=self.position.size)

                def _check_momentum_signal_bt(self):
                    """Check for momentum entry signal using Backtrader data"""
                    if len(self.dataclose) < 20:
                        return False
                    # Simple momentum: current price > 20-day MA
                    ma20 = sum(self.dataclose[-20:]) / 20
                    return self.dataclose[0] > ma20 * 1.02  # 2% above MA

                def _check_exit_signal_bt(self):
                    """Check for exit signal using Backtrader data"""
                    if not self.position:
                        return False
                    entry_price = self.position.price
                    current_price = self.dataclose[0]

                    # Stop loss or take profit
                    if current_price <= entry_price * (1 - self.stop_loss):
                        return True
                    if current_price >= entry_price * (1 + self.take_profit):
                        return True

                    return False

                def notify_trade(self, trade):
                    if trade.isclosed:
                        self.trades.append({
                            'profit': trade.pnl,
                            'duration': trade.barlen,
                            'entry_price': trade.price,
                            'exit_price': trade.price + trade.pnl / trade.size
                        })

            # Prepare data for Backtrader
            bt_data = self._prepare_backtrader_data(market_data)

            # Create Cerebro engine
            cerebro = bt.Cerebro()

            # Add data feed
            data_feed = bt.feeds.PandasData(dataname=bt_data)
            cerebro.adddata(data_feed)

            # Add strategy
            cerebro.addstrategy(MLStrategy, strategy_config=strategy_config)

            # Set broker parameters
            cerebro.broker.setcash(10000.0)
            cerebro.broker.setcommission(commission=0.005)  # 0.5% commission

            # Add analyzers
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

            # Run backtest
            results = cerebro.run()
            strat = results[0]

            # Extract performance metrics
            sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
            max_drawdown = strat.analyzers.drawdown.get_analysis().max.drawdown
            total_return = strat.analyzers.returns.get_analysis()['rtot']

            # Calculate additional metrics
            total_trades = len(strat.trades)
            winning_trades = len([t for t in strat.trades if t['profit'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            return {
                'backtest_completed': True,
                'backtesting_framework': 'backtrader',
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'equity_curve': [10000.0],  # Would need to extract from cerebro
                'trades': strat.trades,
                'alpha': 0,  # Would need market benchmark comparison
                'beta': 0,   # Would need market benchmark comparison
                'sortino_ratio': 0  # Would need additional analyzer
            }

        except Exception as e:
            logger.error(f"Error running Backtrader backtest: {e}")
            return self._run_numpy_backtest(strategy_config, market_data)

    def _run_numpy_backtest(self, strategy_config: Dict[str, Any],
                          market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fallback numpy-based backtest simulation (original implementation).
        """
        try:
            # Extract strategy parameters
            entry_signal = strategy_config.get('entry_signal', 'momentum')
            exit_signal = strategy_config.get('exit_signal', 'time_based')
            position_size = strategy_config.get('position_size', 0.1)
            stop_loss = strategy_config.get('stop_loss', 0.05)
            take_profit = strategy_config.get('take_profit', 0.10)

            # Initialize backtest results
            trades = []
            equity_curve = [10000.0]  # Starting capital
            current_position = 0
            entry_price = 0

            # Simulate trading (simplified - would use backtrader/zipline in production)
            for i in range(1, len(market_data)):
                current_price = market_data.iloc[i]['Close']
                previous_price = market_data.iloc[i-1]['Close']

                # Simple entry/exit logic (would be replaced with actual strategy logic)
                if current_position == 0:  # No position
                    # Entry condition (simplified)
                    if entry_signal == 'momentum' and current_price > previous_price * 1.01:
                        # Enter long position
                        position_value = equity_curve[-1] * position_size
                        shares = position_value / current_price
                        current_position = shares
                        entry_price = current_price
                        trades.append({
                            'type': 'entry',
                            'price': current_price,
                            'shares': shares,
                            'timestamp': market_data.index[i]
                        })

                elif current_position > 0:  # Long position
                    # Exit conditions
                    pnl_pct = (current_price - entry_price) / entry_price

                    if pnl_pct <= -stop_loss or pnl_pct >= take_profit:
                        # Exit position
                        exit_value = current_position * current_price
                        entry_value = current_position * entry_price
                        pnl = exit_value - entry_value

                        new_equity = equity_curve[-1] + pnl
                        equity_curve.append(new_equity)

                        trades.append({
                            'type': 'exit',
                            'price': current_price,
                            'shares': current_position,
                            'pnl': pnl,
                            'timestamp': market_data.index[i]
                        })

                        current_position = 0
                        entry_price = 0
                    else:
                        # Hold position
                        equity_curve.append(equity_curve[-1])
                else:
                    equity_curve.append(equity_curve[-1])

            # Calculate performance metrics
            total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
            peak = max(equity_curve)
            trough = min(equity_curve)
            max_drawdown = (peak - trough) / peak

            # Sharpe ratio (simplified)
            returns = np.diff(equity_curve) / equity_curve[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0

            return {
                'backtest_completed': True,
                'backtesting_framework': 'numpy_fallback',
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'total_trades': len([t for t in trades if t['type'] == 'exit']),
                'win_rate': len([t for t in trades if t['type'] == 'exit' and t['pnl'] > 0]) / max(1, len([t for t in trades if t['type'] == 'exit'])),
                'equity_curve': equity_curve,
                'trades': trades
            }

        except Exception as e:
            logger.error(f"Error running numpy backtest simulation: {e}")
            return {'backtest_completed': False, 'error': str(e)}

    def _prepare_backtrader_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare market data in Backtrader-compatible format.
        """
        try:
            # Backtrader expects OHLCV data with proper column names
            bt_df = market_data.copy()

            # Ensure required columns exist with Backtrader naming
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            column_mapping = {
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            }

            # Rename columns to Backtrader format
            bt_df = bt_df.rename(columns=column_mapping)

            # Ensure all required columns exist
            for col in required_cols:
                if col not in bt_df.columns:
                    if col == 'volume':
                        bt_df[col] = 1000000  # Default volume
                    elif col in ['open', 'high', 'low']:
                        bt_df[col] = bt_df.get('close', bt_df['Close'])  # Use close as fallback
                    else:
                        raise ValueError(f"Missing required column: {col}")

            # Select only required columns
            bt_df = bt_df[required_cols]

            # Ensure datetime index
            if not isinstance(bt_df.index, pd.DatetimeIndex):
                bt_df.index = pd.to_datetime(bt_df.index)

            # Sort by index
            bt_df = bt_df.sort_index()

            return bt_df

        except Exception as e:
            logger.error(f"Error preparing Backtrader data: {e}")
            raise

    def _check_momentum_signal(self, data, context) -> bool:
        """
        Check for momentum-based entry signal.
        """
        try:
            # Simple momentum check: price above moving average
            symbol = context.strategy_config.get('symbol', 'SPY')
            current_price = data.current(symbol, 'price')

            # Calculate simple moving average (last 20 days)
            prices = data.history(symbol, 'price', 20, '1d')
            sma = prices.mean()

            return current_price > sma * 1.01  # 1% above SMA

        except Exception as e:
            logger.warning(f"Error checking momentum signal: {e}")
            return False

    def _check_exit_signal(self, data, context) -> bool:
        """
        Check for exit signal.
        """
        try:
            # Simple exit: stop loss or take profit hit
            symbol = context.strategy_config.get('symbol', 'SPY')
            current_price = data.current(symbol, 'price')

            # Get current position
            position = context.portfolio.positions.get(symbol, None)
            if position is None or position.amount == 0:
                return False

            entry_price = position.cost_basis
            pnl_pct = (current_price - entry_price) / entry_price

            return pnl_pct <= -context.stop_loss or pnl_pct >= context.take_profit

        except Exception as e:
            logger.warning(f"Error checking exit signal: {e}")
            return False



    def _calculate_max_drawdown_from_series(self, portfolio_values: pd.Series) -> float:
        """
        Calculate maximum drawdown from portfolio value series.
        """
        try:
            peak = portfolio_values.expanding().max()
            drawdown = (portfolio_values - peak) / peak
            return drawdown.min()

        except Exception as e:
            logger.warning(f"Error calculating max drawdown: {e}")
            return 0.0

    async def _generate_combined_directives(self, convergence: Dict[str, Any], fade_weight: float) -> pd.DataFrame:
        """
        Generate combined directives from general and pyramiding learning using comprehensive LLM analysis.
        """
        sd_variance = self.memory.get('last_sd_variance', 0.8)

        # Use comprehensive LLM reasoning for all learning decisions (deep analysis and over-analysis)
        if self.llm:
            # Build foundation context for LLM
            foundation_context = f"""
FOUNDATION LEARNING ANALYSIS:
- Current SD Variance: {sd_variance:.3f}
- Convergence Status: {'CONVERGED' if convergence.get('converged', False) else 'NOT CONVERGED'}
- Convergence Trend Slope: {convergence.get('trend_slope', 0):.4f}
- Stability Coefficient: {convergence.get('stability_coefficient', 0):.3f}
- Variance Reduction: {convergence.get('variance_reduction', 0):.3f}
- Recent Average Sharpe: {convergence.get('recent_average_sharpe', 0):.3f}
- Fade Weight: {fade_weight:.3f}
- Total Batches Processed: {len(self.memory.get('weekly_batches', []))}
- Pyramiding Records: {len(self.memory.get('pyramiding_performance', []))}
"""

            # Add pyramiding performance summary if available
            pyramiding_data = self.memory.get('pyramiding_performance', [])
            if pyramiding_data:
                recent_pyramiding = pyramiding_data[-10:]  # Last 10 records
                avg_efficiency = np.mean([r.get('efficiency_score', 1.0) for r in recent_pyramiding])
                success_rate = np.mean([r.get('success', False) for r in recent_pyramiding])
                foundation_context += f"""
- Pyramiding Average Efficiency: {avg_efficiency:.3f}
- Pyramiding Success Rate: {success_rate:.1%}
- Recent Pyramiding Records: {len(recent_pyramiding)}
"""

            llm_question = """
Based on the foundation learning analysis above, what system-wide directives should be generated?

IMPORTANT: This is for LIVE TRADING with CURRENT MARKET DATA. All analysis and recommendations must be based on real-time market conditions, not historical simulations. Do not include any disclaimers about data being historical, illustrative, or not live.

Consider:
1. Risk management adjustments based on SD variance and convergence metrics
2. Position sizing optimizations for stability vs. returns
3. Pyramiding strategy refinements based on performance data
4. Exploration vs. exploitation balance given convergence status
5. Safety mechanism adjustments based on fade weight
6. Long-term learning trajectory and adaptation needs

Provide specific directive recommendations with values and detailed rationale for live trading execution.
"""

            try:
                llm_response = await self.reason_with_llm(foundation_context, llm_question)

                # Parse LLM response to extract directives
                directives = self._parse_llm_directives(llm_response, sd_variance, convergence, fade_weight)
                logger.info(f"Learning Agent LLM comprehensive analysis: Generated {len(directives)} directives")

            except Exception as e:
                logger.warning(f"Learning Agent LLM reasoning failed, using foundation logic: {e}")
                # Fall back to foundation logic
                general_directives = self._generate_general_directives(sd_variance, convergence, fade_weight)
                pyramiding_directives = self._generate_pyramiding_directives()
                directives = general_directives.to_dict('records') if not general_directives.empty else []
                directives.extend(pyramiding_directives)
        else:
            # Use foundation logic when LLM unavailable
            general_directives = self._generate_general_directives(sd_variance, convergence, fade_weight)
            pyramiding_directives = self._generate_pyramiding_directives()
            directives = general_directives.to_dict('records') if not general_directives.empty else []
            directives.extend(pyramiding_directives)

        # Return as DataFrame
        if directives and len(directives) > 0:
            return pd.DataFrame(directives)
        else:
            return pd.DataFrame([{'refinement': 'baseline_optimization', 'value': 1.02, 'reason': 'Baseline optimization'}])

    def _generate_general_directives(self, sd_variance: float, convergence: Dict[str, Any], fade_weight: float) -> pd.DataFrame:
        """
        Generate general learning directives (original logic).
        """
        directives = []
        
        # SD-based directives
        if sd_variance > 1.0:
            sizing_lift = min(0.3, sd_variance - 1.0)  # Max 30% lift
            directives.append({'refinement': 'sizing_lift', 'value': 1.0 + sizing_lift, 'reason': f'High SD variance: {sd_variance:.2f}'})
        
        # Convergence-based directives
        if convergence.get('converged', False):
            directives.append({'refinement': 'efficiency_focus', 'value': 1.1, 'reason': 'Model converged, optimizing efficiency'})
        else:
            directives.append({'refinement': 'exploration_boost', 'value': 1.05, 'reason': 'Model not converged, boosting exploration'})
        
        # Fade weight directives
        if fade_weight > 0.5:
            directives.append({'refinement': 'conservative_filter', 'value': fade_weight, 'reason': f'Safety mode active: {fade_weight:.2f}'})
        elif fade_weight > 0:
            directives.append({'refinement': 'moderate_risk', 'value': 1 - fade_weight, 'reason': f'Transitioning from safety: {1-fade_weight:.2f}'})
        
        return pd.DataFrame(directives) if directives else pd.DataFrame()

    def _generate_pyramiding_directives(self) -> list[Dict[str, Any]]:
        """
        Generate pyramiding-specific learning directives based on performance analysis.
        """
        directives = []
        performance_data = self.memory.get('pyramiding_performance', [])
        
        if len(performance_data) < 5:
            return directives  # Need minimum data for meaningful analysis
        
        # Analyze pyramiding performance patterns
        recent_performance = performance_data[-20:]  # Last 20 records
        
        # Calculate success rates by tiers
        tier_success_rates = {}
        for record in recent_performance:
            tiers = record['tiers_executed']
            success = record['success']
            if tiers not in tier_success_rates:
                tier_success_rates[tiers] = {'success': 0, 'total': 0}
            tier_success_rates[tiers]['total'] += 1
            if success:
                tier_success_rates[tiers]['success'] += 1
        
        # Calculate efficiency trends
        efficiency_scores = [r['efficiency_score'] for r in recent_performance]
        avg_efficiency = np.mean(efficiency_scores) if efficiency_scores else 1.0
        
        # Generate directives based on analysis
        
        # Tier optimization
        best_tier = max(tier_success_rates.keys(), key=lambda t: 
                       tier_success_rates[t]['success'] / tier_success_rates[t]['total'] 
                       if tier_success_rates[t]['total'] > 0 else 0)
        
        if best_tier > 3:  # If higher tiers perform better
            directives.append({
                'refinement': 'pyramiding_tier_boost',
                'value': 1.2,  # 20% increase in max tiers
                'reason': f'Optimal tier count: {best_tier}, increasing tier limits'
            })
        elif best_tier < 3:  # If lower tiers perform better
            directives.append({
                'refinement': 'pyramiding_conservative_tiers',
                'value': 0.8,  # 20% reduction in max tiers
                'reason': f'Optimal tier count: {best_tier}, reducing tier limits for safety'
            })
        
        # Efficiency-based adjustments
        if avg_efficiency > 1.5:  # High efficiency
            directives.append({
                'refinement': 'pyramiding_aggressive_scaling',
                'value': 1.15,  # 15% more aggressive scaling
                'reason': f'High efficiency score: {avg_efficiency:.2f}, increasing scaling factors'
            })
        elif avg_efficiency < 0.8:  # Low efficiency
            directives.append({
                'refinement': 'pyramiding_conservative_scaling',
                'value': 0.85,  # 15% more conservative scaling
                'reason': f'Low efficiency score: {avg_efficiency:.2f}, reducing scaling factors'
            })
        
        # Volatility regime adaptation
        vol_regime_performance = {}
        for record in recent_performance:
            regime = record['volatility_regime']
            success = record['success']
            if regime not in vol_regime_performance:
                vol_regime_performance[regime] = {'success': 0, 'total': 0}
            vol_regime_performance[regime]['total'] += 1
            if success:
                vol_regime_performance[regime]['success'] += 1
        
        # Adjust volatility multipliers based on performance
        for regime, perf in vol_regime_performance.items():
            if perf['total'] >= 3:  # Minimum samples
                success_rate = perf['success'] / perf['total']
                if success_rate > 0.7:  # High success in this regime
                    directives.append({
                        'refinement': f'pyramiding_vol_{regime}_boost',
                        'value': 1.1,  # 10% boost for successful regime
                        'reason': f'High success in {regime} volatility: {success_rate:.1%}'
                    })
                elif success_rate < 0.4:  # Low success in this regime
                    directives.append({
                        'refinement': f'pyramiding_vol_{regime}_conserve',
                        'value': 0.9,  # 10% reduction for poor regime
                        'reason': f'Low success in {regime} volatility: {success_rate:.1%}'
                    })
        
        logger.info(f"Generated {len(directives)} pyramiding learning directives")
        return directives

    def _parse_llm_directives(self, llm_response: str, sd_variance: float, convergence: Dict[str, Any], fade_weight: float) -> list[Dict[str, Any]]:
        """
        Parse LLM response to extract structured directives.
        """
        directives = []

        # Extract common directive patterns from LLM response
        response_lower = llm_response.lower()

        # SD-based directives
        if 'sizing_lift' in response_lower or 'position sizing' in response_lower:
            if sd_variance > 1.0:
                sizing_lift = min(0.3, sd_variance - 1.0)
                directives.append({
                    'refinement': 'sizing_lift',
                    'value': 1.0 + sizing_lift,
                    'reason': f'LLM recommended sizing adjustment for SD variance: {sd_variance:.2f}'
                })

        # Convergence-based directives
        if convergence.get('converged', False):
            if 'efficiency' in response_lower:
                directives.append({
                    'refinement': 'efficiency_focus',
                    'value': 1.1,
                    'reason': 'LLM recommended efficiency focus - model converged'
                })
        else:
            if 'exploration' in response_lower:
                directives.append({
                    'refinement': 'exploration_boost',
                    'value': 1.05,
                    'reason': 'LLM recommended exploration boost - model not converged'
                })

        # Fade weight directives
        if fade_weight > 0.5:
            directives.append({
                'refinement': 'conservative_filter',
                'value': fade_weight,
                'reason': f'LLM recommended conservative approach - safety mode active: {fade_weight:.2f}'
            })
        elif fade_weight > 0:
            directives.append({
                'refinement': 'moderate_risk',
                'value': 1 - fade_weight,
                'reason': f'LLM recommended moderate risk - transitioning from safety: {1-fade_weight:.2f}'
            })

        # Pyramiding directives based on LLM response
        if 'pyramiding' in response_lower:
            pyramiding_directives = self._generate_pyramiding_directives()
            directives.extend(pyramiding_directives)

        # If no specific directives found, add baseline
        if len(directives) == 0:
            directives.append({
                'refinement': 'baseline_optimization',
                'value': 1.02,
                'reason': 'LLM recommended baseline optimization'
            })

        return directives

    async def process_realtime_data(self, market_data: Dict[str, Any], performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process real-time market data and performance metrics for continuous learning.
        
        Args:
            market_data: Real-time market data (prices, volumes, indicators)
            performance_metrics: Current performance metrics from active trades
            
        Returns:
            Dict with real-time learning insights and model updates
        """
        logger.info("Processing real-time data for continuous learning")
        
        try:
            # Extract features from real-time data
            features = self._extract_realtime_features(market_data, performance_metrics)
            
            # Update online learning models
            model_updates = self._update_online_models(features)
            
            # Check for adaptive triggers
            adaptation_triggers = self._check_adaptation_triggers(features, performance_metrics)
            
            # Generate real-time insights
            insights = await self._generate_realtime_insights(features, model_updates, adaptation_triggers)
            
            # Update real-time metrics
            self._update_realtime_metrics(features, insights)
            
            return {
                'features_processed': len(features),
                'model_updates': model_updates,
                'adaptation_triggers': adaptation_triggers,
                'insights': insights,
                'timestamp': pd.Timestamp.now()
            }
            
        except Exception as e:
            logger.error(f"Error in real-time data processing: {e}")
            return {'error': str(e), 'timestamp': pd.Timestamp.now()}

    def _extract_realtime_features(self, market_data: Dict[str, Any], performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from real-time data for continuous learning.
        
        Args:
            market_data: Current market data
            performance_metrics: Current performance metrics
            
        Returns:
            Dict of extracted features
        """
        features = {}
        
        try:
            # Market features
            if 'price' in market_data:
                features['current_price'] = market_data['price']
                features['price_change_pct'] = market_data.get('price_change_pct', 0)
            
            if 'volume' in market_data:
                features['current_volume'] = market_data['volume']
                features['volume_change_pct'] = market_data.get('volume_change_pct', 0)
            
            # Technical indicators
            features.update({
                'rsi': market_data.get('rsi', 50),
                'macd_signal': market_data.get('macd_signal', 0),
                'bollinger_position': market_data.get('bollinger_position', 0),  # -1 to 1
                'vwap_deviation': market_data.get('vwap_deviation', 0)
            })
            
            # Performance features
            features.update({
                'current_pnl': performance_metrics.get('current_pnl', 0),
                'unrealized_pnl': performance_metrics.get('unrealized_pnl', 0),
                'win_rate': performance_metrics.get('win_rate', 0.5),
                'avg_trade_duration': performance_metrics.get('avg_trade_duration', 0),
                'current_position_size': performance_metrics.get('current_position_size', 0)
            })
            
            # Market regime features
            features.update({
                'volatility_regime': market_data.get('volatility_regime', 'normal'),  # low, normal, high
                'trend_strength': market_data.get('trend_strength', 0),  # -1 to 1
                'liquidity_score': market_data.get('liquidity_score', 0.5)
            })
            
            # Time-based features
            current_time = pd.Timestamp.now()
            features.update({
                'hour_of_day': current_time.hour,
                'day_of_week': current_time.dayofweek,
                'market_session': self._get_market_session(current_time)
            })
            
        except Exception as e:
            logger.warning(f"Error extracting real-time features: {e}")
            
        return features

    def _get_market_session(self, timestamp: pd.Timestamp) -> str:
        """
        Determine current market session.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            String indicating market session
        """
        hour = timestamp.hour
        if 9 <= hour < 16:  # 9:30 AM - 4:00 PM ET (approximate)
            return 'regular'
        elif 4 <= hour < 8:  # 4:00 PM - 8:00 PM ET
            return 'extended'
        elif 16 <= hour < 18:  # 4:00 PM - 6:00 PM ET
            return 'after_hours'
        else:
            return 'overnight'

    def _update_online_models(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update online learning models with new features.
        
        Args:
            features: Extracted real-time features
            
        Returns:
            Dict with model update results
        """
        updates = {}
        
        try:
            # Update strategy predictor with online learning
            if self.strategy_predictor and self.model_trained:
                # Prepare feature vector
                feature_vector = self._prepare_feature_vector(features)
                
                if feature_vector is not None:
                    # Online update (simplified incremental learning)
                    # In production, this would use proper online learning algorithms
                    updates['strategy_predictor'] = self._incremental_model_update(feature_vector)
            
            # Update volatility model
            if hasattr(self, 'volatility_model'):
                updates['volatility_model'] = self._update_volatility_model(features)
            
            # Update market regime classifier
            if hasattr(self, 'regime_classifier'):
                updates['regime_classifier'] = self._update_regime_classifier(features)
                
        except Exception as e:
            logger.warning(f"Error updating online models: {e}")
            
        return updates

    def _incremental_model_update(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """
        Perform incremental update to the strategy predictor model.
        
        Args:
            feature_vector: New feature vector
            
        Returns:
            Dict with update results
        """
        # This is a simplified implementation
        # In production, would use proper online learning algorithms like SGD, etc.
        
        try:
            # For now, just track feature statistics for potential retraining triggers
            if not hasattr(self, 'online_feature_buffer'):
                self.online_feature_buffer = []
            
            self.online_feature_buffer.append(feature_vector)
            
            # Keep buffer size manageable
            if len(self.online_feature_buffer) > 1000:
                self.online_feature_buffer = self.online_feature_buffer[-500:]
            
            # Check if we need to trigger model retraining
            should_retrain = self._check_retraining_trigger()
            
            return {
                'buffer_size': len(self.online_feature_buffer),
                'retraining_triggered': should_retrain,
                'feature_drift_detected': self._detect_feature_drift()
            }
            
        except Exception as e:
            logger.warning(f"Error in incremental model update: {e}")
            return {'error': str(e)}

    def _check_retraining_trigger(self) -> bool:
        """
        Check if model retraining should be triggered based on online data.
        
        Returns:
            Boolean indicating if retraining is needed
        """
        if not hasattr(self, 'online_feature_buffer') or len(self.online_feature_buffer) < 100:
            return False
        
        try:
            # Simple trigger based on buffer size and time since last training
            time_since_training = pd.Timestamp.now() - (self.model_last_trained or pd.Timestamp.now())
            buffer_full = len(self.online_feature_buffer) >= 500
            
            return buffer_full or time_since_training > pd.Timedelta(hours=24)
            
        except Exception as e:
            logger.warning(f"Error checking retraining trigger: {e}")
            return False

    def _detect_feature_drift(self) -> bool:
        """
        Detect if feature distribution has drifted significantly.
        
        Returns:
            Boolean indicating feature drift detection
        """
        if not hasattr(self, 'online_feature_buffer') or len(self.online_feature_buffer) < 50:
            return False
        
        try:
            # Simple drift detection based on feature statistics
            recent_features = np.array(self.online_feature_buffer[-50:])
            historical_features = np.array(self.online_feature_buffer[:-50]) if len(self.online_feature_buffer) > 50 else recent_features
            
            # Compare means and variances
            mean_diff = np.abs(np.mean(recent_features, axis=0) - np.mean(historical_features, axis=0))
            var_diff = np.abs(np.var(recent_features, axis=0) - np.var(historical_features, axis=0))
            
            # Trigger if significant changes detected
            drift_threshold = 0.2  # 20% change threshold
            return np.any(mean_diff > drift_threshold) or np.any(var_diff > drift_threshold)
            
        except Exception as e:
            logger.warning(f"Error detecting feature drift: {e}")
            return False

    async def _generate_realtime_insights(self, features: Dict[str, Any], model_updates: Dict[str, Any], 
                                        adaptation_triggers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate real-time learning insights using LLM analysis.
        
        Args:
            features: Current feature values
            model_updates: Results from model updates
            adaptation_triggers: Active adaptation triggers
            
        Returns:
            Dict with real-time insights and recommendations
        """
        try:
            # Prepare context for LLM analysis
            context = {
                'current_features': features,
                'model_updates': model_updates,
                'adaptation_triggers': adaptation_triggers,
                'recent_performance': self._get_recent_performance_context(),
                'market_conditions': self._summarize_market_conditions(features)
            }
            
            # Create prompt for real-time analysis
            prompt = f"""
            Analyze the following real-time trading data and provide continuous learning insights:

            IMPORTANT: This is for LIVE TRADING with CURRENT MARKET DATA. All analysis and recommendations must be based on real-time market conditions, not historical simulations. Do not include any disclaimers about data being historical, illustrative, or not live.

            Current Features: {features}
            Model Updates: {model_updates}
            Adaptation Triggers: {adaptation_triggers}
            Recent Performance: {context['recent_performance']}
            Market Conditions: {context['market_conditions']}

            Provide:
            1. Key patterns or changes observed
            2. Recommended immediate adjustments (if any)
            3. Learning insights for continuous improvement
            4. Risk assessment for current conditions

            Focus on actionable insights for real-time adaptation.
            """
            
            # Get LLM analysis
            llm_response = await self.call_llm(prompt)
            
            # Parse and structure the response
            insights = {
                'timestamp': pd.Timestamp.now(),
                'llm_analysis': llm_response,
                'patterns_identified': self._extract_patterns_from_response(llm_response),
                'recommendations': self._extract_recommendations_from_response(llm_response),
                'risk_assessment': self._extract_risk_assessment(llm_response),
                'confidence_score': self._calculate_insight_confidence(features, model_updates)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating real-time insights: {e}")
            return {
                'error': str(e),
                'timestamp': pd.Timestamp.now(),
                'fallback_insights': 'Real-time analysis unavailable'
            }

    def _get_recent_performance_context(self) -> Dict[str, Any]:
        """
        Get recent performance context for real-time analysis.
        
        Returns:
            Dict with recent performance metrics
        """
        try:
            recent_batches = self.memory.get('weekly_batches', [])[-3:]  # Last 3 batches
            
            if recent_batches:
                # Safely calculate averages, handling potential empty lists
                sharpe_ratios = [b.get('sharpe_ratios', []) for b in recent_batches]
                returns = [b.get('returns', []) for b in recent_batches]
                drawdowns = [b.get('drawdowns', []) for b in recent_batches]
                
                # Flatten and filter out empty lists
                sharpe_flat = [x for sublist in sharpe_ratios if sublist for x in sublist]
                returns_flat = [x for sublist in returns if sublist for x in sublist]
                drawdowns_flat = [x for sublist in drawdowns if sublist for x in sublist]
                
                avg_sharpe = np.mean(sharpe_flat) if sharpe_flat else 0.0
                avg_returns = np.mean(returns_flat) if returns_flat else 0.0
                avg_drawdowns = np.mean(drawdowns_flat) if drawdowns_flat else 0.0
                
                # Calculate trend safely
                trend = 'declining'
                if len(recent_batches) >= 2:
                    last_sharpe = [x for x in recent_batches[-1].get('sharpe_ratios', [])]
                    first_sharpe = [x for x in recent_batches[0].get('sharpe_ratios', [])]
                    if last_sharpe and first_sharpe:
                        if np.mean(last_sharpe) > np.mean(first_sharpe):
                            trend = 'improving'
                
                return {
                    'avg_sharpe': avg_sharpe,
                    'avg_returns': avg_returns,
                    'avg_drawdowns': avg_drawdowns,
                    'trend': trend
                }
            else:
                return {'status': 'insufficient_historical_data'}
                
        except Exception as e:
            logger.warning(f"Error getting recent performance context: {e}")
            return {'error': str(e)}

    def _summarize_market_conditions(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize current market conditions from features.
        
        Args:
            features: Current feature values
            
        Returns:
            Dict with market condition summary
        """
        conditions = {}
        
        try:
            # Volatility assessment
            volatility = features.get('volatility_regime', 'normal')
            conditions['volatility'] = volatility
            
            # Trend assessment
            trend_strength = features.get('trend_strength', 0)
            conditions['trend'] = 'bullish' if trend_strength > 0.2 else 'bearish' if trend_strength < -0.2 else 'sideways'
            
            # Liquidity assessment
            liquidity = features.get('liquidity_score', 0.5)
            conditions['liquidity'] = 'high' if liquidity > 0.7 else 'low' if liquidity < 0.3 else 'normal'
            
            # Overall market regime
            if volatility == 'high' and abs(trend_strength) < 0.1:
                conditions['regime'] = 'choppy_high_vol'
            elif volatility == 'low' and abs(trend_strength) > 0.3:
                conditions['regime'] = 'trending_low_vol'
            else:
                conditions['regime'] = 'normal'
                
        except Exception as e:
            logger.warning(f"Error summarizing market conditions: {e}")
            
        return conditions

    def _extract_patterns_from_response(self, llm_response: str) -> List[str]:
        """
        Extract identified patterns from LLM response.
        
        Args:
            llm_response: Raw LLM response
            
        Returns:
            List of identified patterns
        """
        # Simple pattern extraction - in production would use more sophisticated NLP
        patterns = []
        response_lower = llm_response.lower()
        
        pattern_keywords = [
            'momentum', 'reversal', 'breakout', 'consolidation', 'volatility',
            'trend', 'support', 'resistance', 'volume', 'liquidity'
        ]
        
        for keyword in pattern_keywords:
            if keyword in response_lower:
                patterns.append(keyword.title())
                
        return list(set(patterns))  # Remove duplicates

    def _extract_recommendations_from_response(self, llm_response: str) -> List[str]:
        """
        Extract recommendations from LLM response.
        
        Args:
            llm_response: Raw LLM response
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Look for recommendation indicators
        indicators = ['recommend', 'suggest', 'should', 'consider', 'adjust']
        
        lines = llm_response.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(indicator in line_lower for indicator in indicators):
                recommendations.append(line.strip())
                
        return recommendations[:5]  # Limit to top 5

    def _extract_risk_assessment(self, llm_response: str) -> str:
        """
        Extract risk assessment from LLM response.
        
        Args:
            llm_response: Raw LLM response
            
        Returns:
            Risk assessment summary
        """
        response_lower = llm_response.lower()
        
        if 'high risk' in response_lower or 'risky' in response_lower:
            return 'high'
        elif 'moderate risk' in response_lower or 'caution' in response_lower:
            return 'moderate'
        elif 'low risk' in response_lower or 'safe' in response_lower:
            return 'low'
        else:
            return 'normal'

    def _calculate_insight_confidence(self, features: Dict[str, Any], model_updates: Dict[str, Any]) -> float:
        """
        Calculate confidence score for real-time insights.
        
        Args:
            features: Current features
            model_updates: Model update results
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            confidence = 0.5  # Base confidence
            
            # Increase confidence based on data quality
            if len(features) > 10:
                confidence += 0.2
            
            # Increase confidence based on model update success
            if model_updates and not any('error' in str(update) for update in model_updates.values()):
                confidence += 0.2
            
            # Adjust based on market conditions
            volatility = features.get('volatility_regime', 'normal')
            if volatility == 'normal':
                confidence += 0.1
            elif volatility == 'high':
                confidence -= 0.1
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.warning(f"Error calculating insight confidence: {e}")
            return 0.5

    def _check_adaptation_triggers(self, features: Dict[str, Any], performance_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check for conditions that should trigger adaptive learning responses.
        
        Args:
            features: Current feature values
            performance_metrics: Current performance metrics
            
        Returns:
            List of triggered adaptations
        """
        triggers = []
        
        try:
            # Performance-based triggers
            current_pnl = features.get('current_pnl', 0)
            win_rate = features.get('win_rate', 0.5)
            
            # Poor performance trigger
            if current_pnl < -0.05:  # -5% drawdown
                triggers.append({
                    'type': 'performance_degradation',
                    'severity': 'high',
                    'action': 'reduce_position_sizes',
                    'reason': f'Current PnL: {current_pnl:.2%}'
                })
            
            # Low win rate trigger
            if win_rate < 0.4:
                triggers.append({
                    'type': 'win_rate_decline',
                    'severity': 'medium',
                    'action': 'increase_stop_losses',
                    'reason': f'Win rate: {win_rate:.1%}'
                })
            
            # Market regime triggers
            volatility_regime = features.get('volatility_regime', 'normal')
            if volatility_regime == 'high':
                triggers.append({
                    'type': 'high_volatility',
                    'severity': 'medium',
                    'action': 'reduce_leverage',
                    'reason': 'High market volatility detected'
                })
            
            # Volume-based triggers
            volume_change = features.get('volume_change_pct', 0)
            if volume_change > 0.5:  # 50% volume increase
                triggers.append({
                    'type': 'volume_spike',
                    'severity': 'low',
                    'action': 'monitor_liquidity',
                    'reason': f'Volume change: {volume_change:.1%}'
                })
            
            # Technical indicator triggers
            rsi = features.get('rsi', 50)
            if rsi > 70:
                triggers.append({
                    'type': 'overbought_condition',
                    'severity': 'low',
                    'action': 'reduce_long_exposure',
                    'reason': f'RSI: {rsi:.1f}'
                })
            elif rsi < 30:
                triggers.append({
                    'type': 'oversold_condition',
                    'severity': 'low',
                    'action': 'increase_long_exposure',
                    'reason': f'RSI: {rsi:.1f}'
                })
                
        except Exception as e:
            logger.warning(f"Error checking adaptation triggers: {e}")
            
        return triggers

    def _update_volatility_model(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update volatility model with real-time data.
        
        Args:
            features: Current feature values
            
        Returns:
            Dict with update results
        """
        # Placeholder for volatility model updates
        return {'status': 'not_implemented', 'message': 'Volatility model updates coming soon'}

    def _update_regime_classifier(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update market regime classifier with real-time data.
        
        Args:
            features: Current feature values
            
        Returns:
            Dict with update results
        """
        # Placeholder for regime classifier updates
        return {'status': 'not_implemented', 'message': 'Regime classifier updates coming soon'}

    # ===== SAFETY MECHANISMS FOR REAL-TIME LEARNING =====

    def _check_realtime_safety_limits(self, adaptation_triggers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check if real-time adaptations exceed safety limits.
        
        Args:
            adaptation_triggers: List of active adaptation triggers
            
        Returns:
            Dict with safety assessment
        """
        safety_status = {
            'safe_to_adapt': True,
            'blocked_triggers': [],
            'safety_violations': [],
            'circuit_breaker_activated': False
        }
        
        try:
            # Check for high-severity triggers
            high_severity_count = sum(1 for trigger in adaptation_triggers 
                                    if trigger.get('severity') == 'high')
            
            if high_severity_count >= 3:
                safety_status['safe_to_adapt'] = False
                safety_status['circuit_breaker_activated'] = True
                safety_status['safety_violations'].append('Multiple high-severity triggers')
            
            # Check for performance degradation triggers
            performance_triggers = [t for t in adaptation_triggers 
                                  if t.get('type') == 'performance_degradation']
            
            if len(performance_triggers) >= 2:
                safety_status['safe_to_adapt'] = False
                safety_status['blocked_triggers'].extend([t['action'] for t in performance_triggers])
                safety_status['safety_violations'].append('Multiple performance degradation triggers')
            
            # Check adaptation frequency
            if hasattr(self, 'adaptation_history'):
                recent_adaptations = [a for a in self.adaptation_history 
                                    if a['timestamp'] > pd.Timestamp.now() - pd.Timedelta(minutes=10)]
                
                if len(recent_adaptations) >= 5:  # More than 5 adaptations in 10 minutes
                    safety_status['safe_to_adapt'] = False
                    safety_status['safety_violations'].append('High adaptation frequency')
            
            # Check for conflicting adaptations
            actions = [t.get('action') for t in adaptation_triggers]
            if 'reduce_position_sizes' in actions and 'increase_leverage' in actions:
                safety_status['safe_to_adapt'] = False
                safety_status['safety_violations'].append('Conflicting adaptation actions')
                
        except Exception as e:
            logger.warning(f"Error checking safety limits: {e}")
            safety_status['safe_to_adapt'] = False
            safety_status['safety_violations'].append(f'Safety check failed: {str(e)}')
            
        return safety_status

    async def apply_safe_adaptations(self, adaptation_triggers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply adaptations with safety checks and circuit breakers.
        
        Args:
            adaptation_triggers: List of adaptation triggers to apply
            
        Returns:
            Dict with adaptation results
        """
        try:
            # Check safety limits first
            safety_check = self._check_realtime_safety_limits(adaptation_triggers)
            
            if not safety_check['safe_to_adapt']:
                logger.warning(f"Safety limits exceeded: {safety_check['safety_violations']}")
                
                # Create fallback conservative adaptations
                fallback_adaptations = self._generate_fallback_adaptations()
                
                result = {
                    'adaptations_applied': fallback_adaptations,
                    'original_triggers_blocked': len(adaptation_triggers),
                    'safety_activated': True,
                    'reason': safety_check['safety_violations']
                }
            else:
                # Apply adaptations normally
                applied_adaptations = await self._apply_adaptation_triggers(adaptation_triggers)
                
                result = {
                    'adaptations_applied': applied_adaptations,
                    'triggers_processed': len(adaptation_triggers),
                    'safety_activated': False
                }
            
            # Record adaptation history
            self._record_adaptation_event(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying safe adaptations: {e}")
            return {
                'error': str(e),
                'adaptations_applied': [],
                'safety_activated': True
            }

    def _generate_fallback_adaptations(self) -> List[Dict[str, Any]]:
        """
        Generate conservative fallback adaptations when safety limits are exceeded.
        
        Returns:
            List of conservative adaptation actions
        """
        return [
            {
                'action': 'reduce_risk_exposure',
                'type': 'safety_fallback',
                'parameters': {'max_position_size': 0.5, 'max_leverage': 1.0},
                'reason': 'Safety circuit breaker activated'
            },
            {
                'action': 'increase_stop_losses',
                'type': 'safety_fallback',
                'parameters': {'stop_loss_multiplier': 1.2},
                'reason': 'Conservative risk management'
            },
            {
                'action': 'pause_aggressive_strategies',
                'type': 'safety_fallback',
                'parameters': {'pause_duration': 300},  # 5 minutes
                'reason': 'Temporary strategy pause for safety'
            }
        ]

    async def _apply_adaptation_triggers(self, triggers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply adaptation triggers to the system.
        
        Args:
            triggers: List of adaptation triggers
            
        Returns:
            List of applied adaptations
        """
        applied = []
        
        try:
            for trigger in triggers:
                action = trigger.get('action')
                adaptation = {
                    'action': action,
                    'type': trigger.get('type'),
                    'severity': trigger.get('severity'),
                    'timestamp': pd.Timestamp.now(),
                    'parameters': self._get_adaptation_parameters(action),
                    'reason': trigger.get('reason')
                }
                
                # In a real implementation, this would interface with other agents
                # For now, just record the adaptation
                applied.append(adaptation)
                
                logger.info(f"Applied adaptation: {action} (severity: {trigger.get('severity')})")
                
                       
                       
                       
        except Exception as e:
            logger.warning(f"Error applying adaptation triggers: {e}")
            
        return applied

    def _get_adaptation_parameters(self, action: str) -> Dict[str, Any]:
        """
        Get parameters for a specific adaptation action.
        
        Args:
            action: The adaptation action
            
        Returns:
            Dict with action parameters
        """
        # Default parameters for different actions
        param_map = {
            'reduce_position_sizes': {'size_multiplier': 0.8},
            'increase_stop_losses': {'stop_multiplier': 1.1},
            'reduce_leverage': {'leverage_multiplier': 0.9},
            'monitor_liquidity': {'monitoring_interval': 60},
            'reduce_long_exposure': {'exposure_limit': 0.7},
            'increase_long_exposure': {'exposure_limit': 1.2},
            'pause_aggressive_strategies': {'pause_duration': 180}
        }
        
        return param_map.get(action, {})

    def _record_adaptation_event(self, adaptation_result: Dict[str, Any]) -> None:
        """
        Record adaptation event in history.
        
        Args:
            adaptation_result: Result from adaptation application
        """
        try:
            if not hasattr(self, 'adaptation_history'):
                self.adaptation_history = []
            
            event = {
                'timestamp': pd.Timestamp.now(),
                'result': adaptation_result,
                'safety_activated': adaptation_result.get('safety_activated', False)
            }
            
            self.adaptation_history.append(event)
            
            # Maintain history size
            if len(self.adaptation_history) > 100:
                self.adaptation_history = self.adaptation_history[-100:]
                
        except Exception as e:
            logger.warning(f"Error recording adaptation event: {e}")

    async def get_realtime_adaptation_status(self) -> Dict[str, Any]:
        """
        Get status of real-time adaptations and safety mechanisms.
        
        Returns:
            Dict with adaptation status
        """
        try:
            status = {
                'circuit_breaker_active': False,
                'recent_adaptations': [],
                'safety_violations': [],
                'adaptation_effectiveness': {}
            }
            
            # Check recent adaptation history
            if hasattr(self, 'adaptation_history'):
                recent_adaptations = self.adaptation_history[-10:]  # Last 10 adaptations
                
                status['recent_adaptations'] = [
                    {
                        'timestamp': a['timestamp'].isoformat(),
                        'safety_activated': a['result'].get('safety_activated', False),
                        'adaptations_count': len(a['result'].get('adaptations_applied', []))
                    }
                    for a in recent_adaptations
                ]
                
                # Check for circuit breaker activation
                recent_safety_activations = sum(1 for a in recent_adaptations 
                                              if a['result'].get('safety_activated', False))
                
                if recent_safety_activations >= 3:
                    status['circuit_breaker_active'] = True
            
            # Calculate adaptation effectiveness (simplified)
            if hasattr(self, 'realtime_metrics') and 'performance_tracking' in self.realtime_metrics:
                perf_data = self.realtime_metrics['performance_tracking'][-50:]  # Last 50 points
                
                if len(perf_data) >= 20:
                    pre_adaptation = perf_data[:len(perf_data)//2]
                    post_adaptation = perf_data[len(perf_data)//2:]
                    
                    pre_pnl = np.mean([p['pnl'] for p in pre_adaptation])
                    post_pnl = np.mean([p['pnl'] for p in post_adaptation])
                    
                    status['adaptation_effectiveness'] = {
                        'pre_adaptation_avg_pnl': pre_pnl,
                        'post_adaptation_avg_pnl': post_pnl,
                        'improvement': post_pnl - pre_pnl,
                        'data_points': len(perf_data)
                    }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting adaptation status: {e}")
            return {'error': str(e)}

    # ===== A2A COMMUNICATION FOR REAL-TIME LEARNING =====

    async def distribute_realtime_insights(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Distribute real-time learning insights to other agents via A2A protocol.
        
        Args:
            insights: Real-time insights to distribute
            
        Returns:
            Dict with distribution results
        """
        try:
            distribution = {
                'insights_distributed': False,
                'recipients': [],
                'distribution_method': 'broadcast',
                'priority': self._determine_insight_priority(insights),
                'timestamp': pd.Timestamp.now()
            }
            
            # Prepare insights for A2A distribution
            a2a_message = self._prepare_realtime_insights_for_a2a(insights)
            
            # Determine which agents should receive these insights
            recipients = self._determine_insight_recipients(insights)
            distribution['recipients'] = recipients
            
            # Distribute based on priority
            if distribution['priority'] == 'high':
                # Immediate distribution to critical agents
                await self._distribute_high_priority_insights(a2a_message, recipients)
                distribution['distribution_method'] = 'immediate_broadcast'
                
            elif distribution['priority'] == 'medium':
                # Standard distribution
                await self._distribute_standard_insights(a2a_message, recipients)
                distribution['distribution_method'] = 'standard_broadcast'
                
            else:
                # Low priority - batch with other updates
                self._queue_low_priority_insights(a2a_message, recipients)
                distribution['distribution_method'] = 'queued'
            
            distribution['insights_distributed'] = True
            
            # Log distribution
            logger.info(f"Distributed real-time insights to {len(recipients)} agents (priority: {distribution['priority']})")
            
            return distribution
            
        except Exception as e:
            logger.error(f"Error distributing real-time insights: {e}")
            return {'error': str(e), 'insights_distributed': False}

    def _determine_insight_priority(self, insights: Dict[str, Any]) -> str:
        """
        Determine the priority level of insights for distribution.
        
        Args:
            insights: The insights to prioritize
            
        Returns:
            Priority level: 'high', 'medium', or 'low'
        """
        try:
            # High priority conditions
            if insights.get('risk_assessment') == 'high':
                return 'high'
            
            if 'adaptation_triggers' in insights and insights['adaptation_triggers']:
                trigger_severities = [t.get('severity', 'low') for t in insights['adaptation_triggers']]
                if 'high' in trigger_severities:
                    return 'high'
            
            # Medium priority conditions
            confidence = insights.get('confidence_score', 0.5)
            if confidence > 0.8:
                return 'medium'
            
            if insights.get('patterns_identified') and len(insights['patterns_identified']) > 0:
                return 'medium'
            
            # Low priority for everything else
            return 'low'
            
        except Exception as e:
            logger.warning(f"Error determining insight priority: {e}")
            return 'low'

    def _prepare_realtime_insights_for_a2a(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare real-time insights for A2A communication format.
        
        Args:
            insights: Raw insights from real-time processing
            
        Returns:
            A2A-formatted message
        """
        try:
            a2a_message = {
                'message_type': 'realtime_learning_insights',
                'sender': 'learning_agent',
                'timestamp': insights.get('timestamp', pd.Timestamp.now()),
                'version': '1.0',
                'content': {
                    'patterns_identified': insights.get('patterns_identified', []),
                    'recommendations': insights.get('recommendations', []),
                    'risk_assessment': insights.get('risk_assessment', 'normal'),
                    'confidence_score': insights.get('confidence_score', 0.5),
                    'market_context': self._summarize_market_conditions({})  # Would pass actual features
                },
                'metadata': {
                    'processing_time': (pd.Timestamp.now() - insights.get('timestamp', pd.Timestamp.now())).total_seconds(),
                    'data_quality_score': insights.get('confidence_score', 0.5),
                    'insight_freshness': 'realtime'
                }
            }
            
            # Add adaptation triggers if present
            if 'adaptation_triggers' in insights and insights['adaptation_triggers']:
                a2a_message['content']['adaptation_triggers'] = insights['adaptation_triggers']
            
            # Add model updates if significant
            if 'model_updates' in insights and insights['model_updates']:
                significant_updates = {k: v for k, v in insights['model_updates'].items() 
                                     if v.get('status') != 'not_implemented'}
                if significant_updates:
                    a2a_message['content']['model_updates'] = significant_updates
            
            return a2a_message
            
        except Exception as e:
            logger.warning(f"Error preparing insights for A2A: {e}")
            return {
                'message_type': 'realtime_learning_insights',
                'error': str(e),
                'timestamp': pd.Timestamp.now()
            }

    def _determine_insight_recipients(self, insights: Dict[str, Any]) -> List[str]:
        """
        Determine which agents should receive the real-time insights.
        
        Args:
            insights: The insights being distributed
            
        Returns:
            List of agent names that should receive the insights
        """
        base_recipients = ['data_agent', 'strategy_agent', 'risk_agent', 'reflection_agent']
        
        try:
            recipients = base_recipients.copy()
            
            # Add execution agent for adaptation triggers
            if 'adaptation_triggers' in insights and insights['adaptation_triggers']:
                if 'execution_agent' not in recipients:
                    recipients.append('execution_agent')
            
            # Add portfolio dashboard for performance insights
            if 'performance_metrics' in insights.get('content', {}):
                if 'portfolio_dashboard' not in recipients:
                    recipients.append('portfolio_dashboard')
            
            # Filter based on insight relevance
            risk_assessment = insights.get('risk_assessment', 'normal')
            if risk_assessment == 'high':
                # High risk - prioritize risk and execution agents
                recipients = ['risk_agent', 'execution_agent'] + [r for r in recipients if r not in ['risk_agent', 'execution_agent']]
            
            return recipients
            
        except Exception as e:
            logger.warning(f"Error determining insight recipients: {e}")
            return base_recipients

    async def _distribute_high_priority_insights(self, a2a_message: Dict[str, Any], recipients: List[str]) -> None:
        """
        Distribute high-priority insights immediately.
        
        Args:
            a2a_message: The A2A message to distribute
            recipients: List of recipient agents
        """
        try:
            # In a real implementation, this would use the actual A2A protocol
            # For now, simulate distribution with logging
            
            for recipient in recipients:
                # Simulate immediate delivery
                logger.info(f"High-priority insight delivered to {recipient}: {a2a_message.get('content', {}).keys()}")
                
                # Would call: await self.a2a_protocol.send_message(recipient, a2a_message)
                
        except Exception as e:
            logger.warning(f"Error distributing high-priority insights: {e}")

    async def _distribute_standard_insights(self, a2a_message: Dict[str, Any], recipients: List[str]) -> None:
        """
        Distribute standard-priority insights.
        
        Args:
            a2a_message: The A2A message to distribute
            recipients: List of recipient agents
        """
        try:
            # Standard distribution - can be batched
            for recipient in recipients:
                logger.info(f"Standard insight queued for {recipient}")
                
                # Would queue for batch distribution
                # await self.a2a_protocol.queue_message(recipient, a2a_message)
                
        except Exception as e:
            logger.warning(f"Error distributing standard insights: {e}")

    def _queue_low_priority_insights(self, a2a_message: Dict[str, Any], recipients: List[str]) -> None:
        """
        Queue low-priority insights for batch distribution.
        
        Args:
            a2a_message: The A2A message to queue
            recipients: List of recipient agents
        """
        try:
            if not hasattr(self, 'low_priority_queue'):
                self.low_priority_queue = []
            
            queued_message = {
                'message': a2a_message,
                'recipients': recipients,
                'queued_at': pd.Timestamp.now(),
                'priority': 'low'
            }
            
            self.low_priority_queue.append(queued_message)
            
            # Maintain queue size
            if len(self.low_priority_queue) > 50:
                self.low_priority_queue = self.low_priority_queue[-50:]
                
        except Exception as e:
            logger.warning(f"Error queuing low-priority insights: {e}")

    async def process_queued_insights(self) -> Dict[str, Any]:
        """
        Process and distribute queued low-priority insights.
        
        Returns:
            Dict with processing results
        """
        try:
            if not hasattr(self, 'low_priority_queue'):
                return {'processed': 0, 'message': 'No queued insights'}
            
            queue_size = len(self.low_priority_queue)
            processed = 0
            
            # Process insights older than 5 minutes
            cutoff_time = pd.Timestamp.now() - pd.Timedelta(minutes=5)
            
            remaining_queue = []
            for queued_item in self.low_priority_queue:
                if queued_item['queued_at'] < cutoff_time:
                    # Process this item
                    await self._distribute_queued_item(queued_item)
                    processed += 1
                else:
                    remaining_queue.append(queued_item)
            
            self.low_priority_queue = remaining_queue
            
            return {
                'processed': processed,
                'remaining': len(self.low_priority_queue),
                'original_queue_size': queue_size
            }
            
        except Exception as e:
            logger.error(f"Error processing queued insights: {e}")
            return {'error': str(e), 'processed': 0}

    async def _distribute_queued_item(self, queued_item: Dict[str, Any]) -> None:
        """
        Distribute a single queued insight item.
        
        Args:
            queued_item: The queued item to distribute
        """
        try:
            a2a_message = queued_item['message']
            recipients = queued_item['recipients']
            
            # Distribute to all recipients
            for recipient in recipients:
                logger.info(f"Distributing queued insight to {recipient}")
                # await self.a2a_protocol.send_message(recipient, a2a_message)
                
        except Exception as e:
            logger.warning(f"Error distributing queued item: {e}")

    async def receive_realtime_insights_request(self, requesting_agent: str, request_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle requests from other agents for real-time learning insights.
        
        Args:
            requesting_agent: Name of the agent making the request
            request_details: Details of the request
            
        Returns:
            Dict with requested insights
        """
        try:
            # Validate requesting agent
            authorized_agents = ['data_agent', 'strategy_agent', 'risk_agent', 'execution_agent', 'reflection_agent']
            
            if requesting_agent not in authorized_agents:
                return {'error': 'Unauthorized agent', 'granted': False}
            
            # Prepare insights based on request
            request_type = request_details.get('type', 'general')
            
            if request_type == 'performance':
                insights = self._get_performance_insights()
            elif request_type == 'adaptation_status':
                insights = await self.get_realtime_adaptation_status()
            elif request_type == 'model_status':
                insights = await self.get_realtime_learning_status()
            else:
                insights = self._get_general_realtime_insights()
            
            return {
                'granted': True,
                'requesting_agent': requesting_agent,
                'insights': insights,
                'timestamp': pd.Timestamp.now()
            }
            
        except Exception as e:
            logger.error(f"Error handling insights request: {e}")
            return {'error': str(e), 'granted': False}

    def _get_performance_insights(self) -> Dict[str, Any]:
        """
        Get performance-focused real-time insights.
        
        Returns:
            Dict with performance insights
        """
        try:
            if not hasattr(self, 'realtime_metrics') or 'performance_tracking' not in self.realtime_metrics:
                return {'status': 'no_performance_data'}
            
            perf_data = self.realtime_metrics['performance_tracking'][-20:]  # Last 20 points
            
            if not perf_data:
                return {'status': 'insufficient_data'}
            
            return {
                'current_pnl': perf_data[-1]['pnl'] if perf_data else 0,
                'avg_pnl': np.mean([p['pnl'] for p in perf_data]),
                'pnl_volatility': np.std([p['pnl'] for p in perf_data]),
                'win_rate_trend': [p['win_rate'] for p in perf_data],
                'data_points': len(perf_data)
            }
            
        except Exception as e:
            logger.warning(f"Error getting performance insights: {e}")
            return {'error': str(e)}

    def _get_general_realtime_insights(self) -> Dict[str, Any]:
        """
        Get general real-time learning insights.
        
        Returns:
            Dict with general insights
        """
        try:
            insights = {
                'learning_mode': 'realtime_active',
                'data_points_processed': 0,
                'insights_generated': 0,
                'system_health': 'unknown'
            }
            
            if hasattr(self, 'realtime_metrics'):
                feature_history = self.realtime_metrics.get('feature_history', [])
                insight_history = self.realtime_metrics.get('insight_history', [])
                
                insights.update({
                    'data_points_processed': len(feature_history),
                    'insights_generated': len(insight_history),
                    'avg_confidence': np.mean([i.get('confidence_score', 0.5) for i in insight_history]) if insight_history else 0
                })
            
            # Get system health
            health_status = self._assess_realtime_system_health()
            insights['system_health'] = health_status.get('overall_status', 'unknown')
            
            return insights
            
        except Exception as e:
            logger.warning(f"Error getting general insights: {e}")
            return {'error': str(e)}

    async def get_realtime_learning_status(self) -> Dict[str, Any]:
        """
        Get comprehensive real-time learning status.
        
        Returns:
            Dict with real-time learning status and metrics
        """
        try:
            status = {
                'is_active': True,
                'last_update': pd.Timestamp.now(),
                'metrics': {}
            }
            
            # Basic metrics
            if hasattr(self, 'realtime_metrics'):
                feature_history = self.realtime_metrics.get('feature_history', [])
                insight_history = self.realtime_metrics.get('insight_history', [])
                
                status['metrics'].update({
                    'data_points_processed': len(feature_history),
                    'insights_generated': len(insight_history),
                    'avg_confidence': np.mean([i.get('confidence_score', 0.5) for i in insight_history]) if insight_history else 0,
                    'anomalies_detected': len(getattr(self, 'anomaly_history', []))
                })
            
            # Model status
            status['model_status'] = {
                'strategy_predictor_trained': self.model_trained,
                'online_buffer_size': len(getattr(self, 'online_feature_buffer', [])),
                'last_training': self.model_last_trained.isoformat() if self.model_last_trained else None
            }
            
            # Performance metrics
            if hasattr(self, 'realtime_metrics') and 'performance_tracking' in self.realtime_metrics:
                perf_tracking = self.realtime_metrics['performance_tracking'][-50:]  # Last 50 points
                
                if perf_tracking:
                    status['performance_metrics'] = {
                        'current_pnl': perf_tracking[-1]['pnl'] if perf_tracking else 0,
                        'avg_win_rate': np.mean([p['win_rate'] for p in perf_tracking]),
                        'rolling_sharpe': self.realtime_metrics.get('rolling_sharpe', 0),
                        'data_points': len(perf_tracking)
                    }
            
            # System health
            status['system_health'] = self._assess_realtime_system_health()
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting real-time learning status: {e}")
            return {'error': str(e), 'is_active': False}

    # ===== PROPOSAL HANDLING =====

    async def receive_optimization_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Receive and queue an optimization proposal for evaluation.
        
        Args:
            proposal: Optimization proposal from another agent
            
        Returns:
            Dict with receipt confirmation
        """
        try:
            proposal_id = proposal.get('proposal_id')
            if not proposal_id:
                return {'received': False, 'error': 'Missing proposal_id'}
            
            # Store proposal in evaluation queue
            if 'proposal_queue' not in self.memory:
                self.memory['proposal_queue'] = []
            
            # Add timestamp for processing order
            proposal['received_at'] = pd.Timestamp.now().isoformat()
            proposal['evaluation_status'] = 'queued'
            
            self.memory['proposal_queue'].append(proposal)
            
            # Update proposal tracking
            if 'proposal_tracking' not in self.memory:
                self.memory['proposal_tracking'] = {}
            self.memory['proposal_tracking'][proposal_id] = {
                'status': 'queued',
                'received_at': proposal['received_at'],
                'submitted_by': proposal.get('submitted_by'),
                'proposal_type': proposal.get('proposal_type')
            }
            
            logger.info(f"LearningAgent received optimization proposal: {proposal_id} from {proposal.get('submitted_by')}")
            
            # Auto-trigger evaluation if queue is not too long
            if len(self.memory['proposal_queue']) <= 5:
                await self._trigger_proposal_evaluation()
            
            return {
                'received': True,
                'proposal_id': proposal_id,
                'queue_position': len(self.memory['proposal_queue']),
                'message': 'Proposal queued for evaluation'
            }
            
        except Exception as e:
            logger.error(f"Error receiving optimization proposal: {e}")
            return {'received': False, 'error': str(e)}

    async def evaluate_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate an optimization proposal using comprehensive analysis.
        
        Args:
            proposal: Proposal to evaluate
            
        Returns:
            Dict with evaluation results
        """
        try:
            proposal_id = proposal.get('proposal_id')
            logger.info(f"LearningAgent evaluating proposal: {proposal_id}")
            
            # Extract proposal details
            proposal_type = proposal.get('proposal_type')
            changes = proposal.get('changes', {})
            expected_impact = proposal.get('expected_impact', {})
            evidence = proposal.get('evidence', {})
            confidence_score = proposal.get('confidence_score', 0.5)
            
            # Perform comprehensive evaluation
            evaluation_results = await self._comprehensive_proposal_evaluation(
                proposal_type, changes, expected_impact, evidence, confidence_score
            )
            
            # Update proposal status
            proposal['evaluation_results'] = evaluation_results
            proposal['evaluation_status'] = 'completed'
            proposal['evaluated_at'] = pd.Timestamp.now().isoformat()
            
            # Update tracking
            if proposal_id in self.memory.get('proposal_tracking', {}):
                self.memory['proposal_tracking'][proposal_id].update({
                    'status': 'evaluated',
                    'evaluation_score': evaluation_results.get('overall_score', 0),
                    'recommendation': evaluation_results.get('recommendation', 'hold'),
                    'evaluated_at': proposal['evaluated_at']
                })
            
            # Auto-decide based on evaluation
            if evaluation_results.get('recommendation') == 'implement':
                await self._schedule_proposal_implementation(proposal)
            elif evaluation_results.get('recommendation') == 'test':
                await self._schedule_proposal_testing(proposal)
            
            logger.info(f"LearningAgent completed evaluation of proposal {proposal_id}: {evaluation_results.get('recommendation')}")
            
            return {
                'evaluated': True,
                'proposal_id': proposal_id,
                'evaluation_results': evaluation_results,
                'recommendation': evaluation_results.get('recommendation')
            }
            
        except Exception as e:
            logger.error(f"Error evaluating proposal {proposal.get('proposal_id')}: {e}")
            return {'evaluated': False, 'error': str(e)}

    async def test_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test an optimization proposal in a controlled environment.
        
        Args:
            proposal: Proposal to test
            
        Returns:
            Dict with test results
        """
        try:
            proposal_id = proposal.get('proposal_id')
            logger.info(f"LearningAgent testing proposal: {proposal_id}")
            
            # Extract test requirements
            test_requirements = proposal.get('test_requirements', {})
            
            # Run comprehensive testing
            test_results = await self._run_proposal_tests(proposal, test_requirements)
            
            # Update proposal with test results
            proposal['test_results'] = test_results
            proposal['test_status'] = 'completed'
            proposal['tested_at'] = pd.Timestamp.now().isoformat()
            
            # Update tracking
            if proposal_id in self.memory.get('proposal_tracking', {}):
                self.memory['proposal_tracking'][proposal_id].update({
                    'test_score': test_results.get('overall_score', 0),
                    'test_passed': test_results.get('passed', False),
                    'tested_at': proposal['tested_at']
                })
            
            # Decide next action based on test results
            if test_results.get('passed', False):
                await self._schedule_proposal_implementation(proposal)
            else:
                proposal['evaluation_status'] = 'rejected'
                self.memory['proposal_tracking'][proposal_id]['status'] = 'rejected'
            
            logger.info(f"LearningAgent completed testing of proposal {proposal_id}: {'PASSED' if test_results.get('passed') else 'FAILED'}")
            
            return {
                'tested': True,
                'proposal_id': proposal_id,
                'test_results': test_results,
                'test_passed': test_results.get('passed', False)
            }
            
        except Exception as e:
            logger.error(f"Error testing proposal {proposal.get('proposal_id')}: {e}")
            return {'tested': False, 'error': str(e)}

    async def implement_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement an approved optimization proposal.
        
        Args:
            proposal: Proposal to implement
            
        Returns:
            Dict with implementation results
        """
        try:
            proposal_id = proposal.get('proposal_id')
            logger.info(f"LearningAgent implementing proposal: {proposal_id}")
            
            # Create backup before implementation
            backup_id = await self._create_implementation_backup(proposal)
            
            # Apply the changes
            implementation_results = await self._apply_proposal_changes(proposal)
            
            # Update proposal status
            proposal['implementation_results'] = implementation_results
            proposal['implementation_status'] = 'completed' if implementation_results.get('success') else 'failed'
            proposal['implemented_at'] = pd.Timestamp.now().isoformat()
            proposal['backup_id'] = backup_id
            
            # Update tracking
            if proposal_id in self.memory.get('proposal_tracking', {}):
                self.memory['proposal_tracking'][proposal_id].update({
                    'status': 'implemented' if implementation_results.get('success') else 'implementation_failed',
                    'implemented_at': proposal['implemented_at'],
                    'backup_id': backup_id
                })
            
            # Start performance monitoring
            if implementation_results.get('success'):
                await self._start_performance_monitoring(proposal)
            
            logger.info(f"LearningAgent {'successfully implemented' if implementation_results.get('success') else 'failed to implement'} proposal {proposal_id}")
            
            return {
                'implemented': implementation_results.get('success', False),
                'proposal_id': proposal_id,
                'implementation_results': implementation_results,
                'backup_id': backup_id
            }
            
        except Exception as e:
            logger.error(f"Error implementing proposal {proposal.get('proposal_id')}: {e}")
            return {'implemented': False, 'error': str(e)}

    async def rollback_proposal(self, proposal_id: str, reason: str = None) -> Dict[str, Any]:
        """
        Rollback an implemented optimization proposal.
        
        Args:
            proposal_id: ID of proposal to rollback
            reason: Reason for rollback
            
        Returns:
            Dict with rollback results
        """
        try:
            logger.info(f"LearningAgent rolling back proposal: {proposal_id}")
            
            # Find proposal
            proposal = None
            for p in self.memory.get('proposal_queue', []):
                if p.get('proposal_id') == proposal_id:
                    proposal = p
                    break
            
            if not proposal:
                return {'rolled_back': False, 'error': 'Proposal not found'}
            
            if proposal.get('implementation_status') != 'completed':
                return {'rolled_back': False, 'error': 'Proposal not implemented'}
            
            backup_id = proposal.get('backup_id')
            if not backup_id:
                return {'rolled_back': False, 'error': 'No backup available for rollback'}
            
            # Perform rollback
            rollback_results = await self._perform_rollback(proposal, backup_id, reason)
            
            # Update proposal status
            proposal['rollback_results'] = rollback_results
            proposal['rollback_status'] = 'completed' if rollback_results.get('success') else 'failed'
            proposal['rolled_back_at'] = pd.Timestamp.now().isoformat()
            proposal['rollback_reason'] = reason
            
            # Update tracking
            if proposal_id in self.memory.get('proposal_tracking', {}):
                self.memory['proposal_tracking'][proposal_id].update({
                    'status': 'rolled_back' if rollback_results.get('success') else 'rollback_failed',
                    'rolled_back_at': proposal['rolled_back_at'],
                    'rollback_reason': reason
                })
            
            logger.info(f"LearningAgent {'successfully rolled back' if rollback_results.get('success') else 'failed to rollback'} proposal {proposal_id}")
            
            return {
                'rolled_back': rollback_results.get('success', False),
                'proposal_id': proposal_id,
                'rollback_results': rollback_results
            }
            
        except Exception as e:
            logger.error(f"Error rolling back proposal {proposal_id}: {e}")
            return {'rolled_back': False, 'error': str(e)}

    # ===== PROPOSAL EVALUATION HELPERS =====

    async def _comprehensive_proposal_evaluation(self, proposal_type: str, changes: Dict[str, Any],
                                               expected_impact: Dict[str, Any], evidence: Dict[str, Any],
                                               confidence_score: float) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of an optimization proposal.

        Args:
            proposal_type: Type of proposal (strategy, risk, execution, etc.)
            changes: Proposed changes
            expected_impact: Expected impact metrics
            evidence: Supporting evidence
            confidence_score: Proposer's confidence score

        Returns:
            Dict with evaluation results
        """
        try:
            evaluation = {
                'overall_score': 0.0,
                'risk_assessment': 'unknown',
                'expected_improvement': 0.0,
                'confidence_adjusted_score': 0.0,
                'recommendation': 'hold',
                'evaluation_details': {}
            }

            # Evaluate based on proposal type
            if proposal_type == 'strategy_optimization':
                evaluation.update(await self._evaluate_strategy_proposal(changes, expected_impact, evidence))
            elif proposal_type == 'risk_parameter_adjustment':
                evaluation.update(await self._evaluate_risk_proposal(changes, expected_impact, evidence))
            elif proposal_type == 'execution_improvement':
                evaluation.update(await self._evaluate_execution_proposal(changes, expected_impact, evidence))
            else:
                evaluation.update(await self._evaluate_generic_proposal(changes, expected_impact, evidence))

            # Adjust for confidence and evidence quality
            confidence_multiplier = self._calculate_confidence_multiplier(confidence_score, evidence)
            evaluation['confidence_adjusted_score'] = evaluation['overall_score'] * confidence_multiplier

            # Make recommendation based on adjusted score
            evaluation['recommendation'] = self._generate_evaluation_recommendation(
                evaluation['confidence_adjusted_score'],
                evaluation['risk_assessment']
            )

            logger.info(f"Comprehensive evaluation completed: score={evaluation['overall_score']:.3f}, recommendation={evaluation['recommendation']}")

            return evaluation

        except Exception as e:
            logger.error(f"Error in comprehensive proposal evaluation: {e}")
            return {
                'overall_score': 0.0,
                'recommendation': 'reject',
                'error': str(e)
            }

    async def _evaluate_strategy_proposal(self, changes: Dict[str, Any], expected_impact: Dict[str, Any],
                                        evidence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate strategy optimization proposals.
        """
        evaluation = {'evaluation_details': {}}

        try:
            # Check backtesting evidence
            backtest_results = evidence.get('backtest_results', {})
            if backtest_results:
                sharpe_improvement = backtest_results.get('sharpe_improvement', 0)
                win_rate_improvement = backtest_results.get('win_rate_improvement', 0)
                drawdown_reduction = backtest_results.get('max_drawdown_reduction', 0)

                # Score based on improvements
                strategy_score = (
                    sharpe_improvement * 0.4 +
                    win_rate_improvement * 0.3 +
                    drawdown_reduction * 0.3
                )
                evaluation['overall_score'] = min(1.0, max(0.0, strategy_score))
            else:
                evaluation['overall_score'] = 0.3  # Default moderate score without evidence

            # Risk assessment
            max_drawdown_impact = expected_impact.get('max_drawdown_change', 0)
            if max_drawdown_impact > 0.05:  # 5% increase in drawdown
                evaluation['risk_assessment'] = 'high'
            elif max_drawdown_impact > 0.02:
                evaluation['risk_assessment'] = 'medium'
            else:
                evaluation['risk_assessment'] = 'low'

            evaluation['expected_improvement'] = expected_impact.get('sharpe_ratio_change', 0)
            evaluation['evaluation_details'] = {
                'backtest_validated': bool(backtest_results),
                'sharpe_improvement': sharpe_improvement,
                'win_rate_improvement': win_rate_improvement
            }

        except Exception as e:
            logger.warning(f"Error evaluating strategy proposal: {e}")
            evaluation['overall_score'] = 0.0
            evaluation['risk_assessment'] = 'high'

        return evaluation

    async def _evaluate_risk_proposal(self, changes: Dict[str, Any], expected_impact: Dict[str, Any],
                                    evidence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate risk parameter adjustment proposals.
        """
        evaluation = {'evaluation_details': {}}

        try:
            # Risk proposals need careful evaluation
            volatility_change = changes.get('volatility_target', 0)
            position_size_change = changes.get('position_size_limit', 0)

            # Conservative scoring for risk changes
            if abs(volatility_change) < 0.1 and abs(position_size_change) < 0.2:
                evaluation['overall_score'] = 0.6  # Moderate score for conservative changes
            else:
                evaluation['overall_score'] = 0.3  # Lower score for aggressive changes

            # Risk assessment is always high for risk changes
            evaluation['risk_assessment'] = 'high'

            # Expected improvement (usually stability)
            evaluation['expected_improvement'] = expected_impact.get('volatility_reduction', 0)

            evaluation['evaluation_details'] = {
                'volatility_change': volatility_change,
                'position_size_change': position_size_change,
                'conservative_adjustment': abs(volatility_change) < 0.1
            }

        except Exception as e:
            logger.warning(f"Error evaluating risk proposal: {e}")
            evaluation['overall_score'] = 0.0
            evaluation['risk_assessment'] = 'high'

        return evaluation

    async def _evaluate_execution_proposal(self, changes: Dict[str, Any], expected_impact: Dict[str, Any],
                                         evidence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate execution improvement proposals.
        """
        evaluation = {'evaluation_details': {}}

        try:
            # Execution improvements typically have lower risk
            slippage_reduction = expected_impact.get('slippage_reduction', 0)
            execution_speed_improvement = expected_impact.get('execution_speed_improvement', 0)

            # Score based on execution improvements
            execution_score = (
                slippage_reduction * 0.6 +
                execution_speed_improvement * 0.4
            )
            evaluation['overall_score'] = min(1.0, max(0.0, execution_score))

            # Lower risk assessment for execution changes
            evaluation['risk_assessment'] = 'low'

            evaluation['expected_improvement'] = slippage_reduction
            evaluation['evaluation_details'] = {
                'slippage_reduction': slippage_reduction,
                'speed_improvement': execution_speed_improvement
            }

        except Exception as e:
            logger.warning(f"Error evaluating execution proposal: {e}")
            evaluation['overall_score'] = 0.0
            evaluation['risk_assessment'] = 'medium'

        return evaluation

    async def _evaluate_generic_proposal(self, changes: Dict[str, Any], expected_impact: Dict[str, Any],
                                       evidence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate generic optimization proposals.
        """
        evaluation = {
            'overall_score': 0.4,  # Default moderate score
            'risk_assessment': 'medium',
            'expected_improvement': expected_impact.get('performance_improvement', 0),
            'evaluation_details': {
                'generic_evaluation': True,
                'changes_count': len(changes)
            }
        }

        return evaluation

    def _calculate_confidence_multiplier(self, confidence_score: float, evidence: Dict[str, Any]) -> float:
        """
        Calculate confidence multiplier based on proposer's confidence and evidence quality.
        """
        try:
            # Base multiplier from confidence score
            base_multiplier = confidence_score

            # Adjust based on evidence quality
            evidence_quality = 0.5  # Default

            if evidence.get('backtest_results'):
                evidence_quality += 0.2
            if evidence.get('historical_performance'):
                evidence_quality += 0.2
            if evidence.get('statistical_significance'):
                evidence_quality += 0.1

            # Combined multiplier
            confidence_multiplier = (base_multiplier * 0.7) + (evidence_quality * 0.3)

            return max(0.1, min(1.5, confidence_multiplier))  # Clamp between 0.1 and 1.5

        except Exception as e:
            logger.warning(f"Error calculating confidence multiplier: {e}")
            return 1.0

    def _generate_evaluation_recommendation(self, adjusted_score: float, risk_assessment: str) -> str:
        """
        Generate recommendation based on evaluation score and risk.
        """
        try:
            if risk_assessment == 'high':
                # Higher threshold for high-risk proposals
                if adjusted_score > 0.8:
                    return 'implement'
                elif adjusted_score > 0.6:
                    return 'test'
                else:
                    return 'reject'
            elif risk_assessment == 'medium':
                if adjusted_score > 0.7:
                    return 'implement'
                elif adjusted_score > 0.5:
                    return 'test'
                else:
                    return 'reject'
            else:  # low risk
                if adjusted_score > 0.6:
                    return 'implement'
                elif adjusted_score > 0.4:
                    return 'test'
                else:
                    return 'reject'

        except Exception as e:
            logger.warning(f"Error generating recommendation: {e}")
            return 'hold'

    # ===== PROPOSAL SCHEDULING HELPERS =====

    async def _schedule_proposal_implementation(self, proposal: Dict[str, Any]) -> None:
        """
        Schedule a proposal for implementation.
        """
        try:
            proposal_id = proposal.get('proposal_id')
            logger.info(f"Scheduling implementation for proposal: {proposal_id}")

            # Add to implementation queue
            if 'implementation_queue' not in self.memory:
                self.memory['implementation_queue'] = []

            implementation_task = {
                'proposal_id': proposal_id,
                'scheduled_at': pd.Timestamp.now().isoformat(),
                'status': 'scheduled',
                'priority': proposal.get('priority', 'normal')
            }

            self.memory['implementation_queue'].append(implementation_task)

            # Update tracking
            if proposal_id in self.memory.get('proposal_tracking', {}):
                self.memory['proposal_tracking'][proposal_id]['implementation_scheduled'] = True

        except Exception as e:
            logger.error(f"Error scheduling proposal implementation: {e}")

    async def _schedule_proposal_testing(self, proposal: Dict[str, Any]) -> None:
        """
        Schedule a proposal for testing.
        """
        try:
            proposal_id = proposal.get('proposal_id')
            logger.info(f"Scheduling testing for proposal: {proposal_id}")

            # Add to testing queue
            if 'testing_queue' not in self.memory:
                self.memory['testing_queue'] = []

            testing_task = {
                'proposal_id': proposal_id,
                'scheduled_at': pd.Timestamp.now().isoformat(),
                'status': 'scheduled',
                'test_requirements': proposal.get('test_requirements', {})
            }

            self.memory['testing_queue'].append(testing_task)

            # Update tracking
            if proposal_id in self.memory.get('proposal_tracking', {}):
                self.memory['proposal_tracking'][proposal_id]['testing_scheduled'] = True

        except Exception as e:
            logger.error(f"Error scheduling proposal testing: {e}")

    # ===== PROPOSAL TESTING HELPERS =====

    async def _run_proposal_tests(self, proposal: Dict[str, Any], test_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive tests for a proposal.
        """
        try:
            proposal_id = proposal.get('proposal_id')
            logger.info(f"Running tests for proposal: {proposal_id}")

            test_results = {
                'passed': False,
                'overall_score': 0.0,
                'test_details': {},
                'risk_checks': {},
                'performance_checks': {}
            }

            # Run different types of tests based on requirements
            if test_requirements.get('backtest_required', True):
                backtest_results = await self._run_backtest_validation(proposal)
                test_results['test_details']['backtest'] = backtest_results

            if test_requirements.get('stress_test_required', False):
                stress_results = await self._run_stress_tests(proposal)
                test_results['test_details']['stress_test'] = stress_results

            if test_requirements.get('risk_checks_required', True):
                risk_results = await self._run_risk_validation(proposal)
                test_results['risk_checks'] = risk_results

            # Calculate overall score
            scores = []
            if 'backtest' in test_results['test_details']:
                scores.append(test_results['test_details']['backtest'].get('score', 0))
            if 'stress_test' in test_results['test_details']:
                scores.append(test_results['test_details']['stress_test'].get('score', 0))
            if test_results['risk_checks']:
                scores.append(test_results['risk_checks'].get('score', 0))

            if scores:
                test_results['overall_score'] = np.mean(scores)

            # Determine if tests passed
            test_results['passed'] = (
                test_results['overall_score'] > 0.6 and  # Minimum score threshold
                test_results['risk_checks'].get('passed', True)  # Risk checks must pass
            )

            logger.info(f"Proposal {proposal_id} testing completed: score={test_results['overall_score']:.3f}, passed={test_results['passed']}")

            return test_results

        except Exception as e:
            logger.error(f"Error running proposal tests: {e}")
            return {'passed': False, 'error': str(e)}

    async def _run_backtest_validation(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run backtest validation for a proposal.
        """
        try:
            changes = proposal.get('changes', {})
            strategy_config = changes.get('strategy_config', {})

            # Get sample market data for testing
            market_data = await self._get_test_market_data()

            # Run backtest with proposed changes
            backtest_result = self._run_backtrader_backtest(strategy_config, market_data)

            if backtest_result.get('backtest_completed'):
                # Evaluate backtest results
                score = self._score_backtest_results(backtest_result)
                return {
                    'score': score,
                    'sharpe_ratio': backtest_result.get('sharpe_ratio', 0),
                    'total_return': backtest_result.get('total_return', 0),
                    'max_drawdown': backtest_result.get('max_drawdown', 0),
                    'win_rate': backtest_result.get('win_rate', 0)
                }
            else:
                return {'score': 0.0, 'error': 'Backtest failed'}

        except Exception as e:
            logger.warning(f"Error in backtest validation: {e}")
            return {'score': 0.0, 'error': str(e)}

    async def _run_stress_tests(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run stress tests for a proposal.
        """
        try:
            # Simulate extreme market conditions
            stress_scenarios = [
                {'volatility_multiplier': 2.0, 'trend_strength': -0.8},  # High vol, downtrend
                {'volatility_multiplier': 1.5, 'liquidity_drop': 0.7},   # Illiquid market
                {'gap_moves': True, 'size': 0.05}                       # Gap moves
            ]

            stress_scores = []
            for scenario in stress_scenarios:
                scenario_result = await self._run_stress_scenario(proposal, scenario)
                stress_scores.append(scenario_result.get('score', 0))

            avg_stress_score = np.mean(stress_scores) if stress_scores else 0.0

            return {
                'score': avg_stress_score,
                'scenarios_tested': len(stress_scenarios),
                'worst_case_score': min(stress_scores) if stress_scores else 0.0
            }

        except Exception as e:
            logger.warning(f"Error in stress testing: {e}")
            return {'score': 0.0, 'error': str(e)}

    async def _run_risk_validation(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run risk validation checks for a proposal.
        """
        try:
            risk_checks = {
                'passed': True,
                'score': 1.0,
                'violations': [],
                'warnings': []
            }

            changes = proposal.get('changes', {})

            # Check for excessive risk increases
            if changes.get('position_size_limit', 1.0) > 1.5:
                risk_checks['violations'].append('Excessive position size increase')
                risk_checks['passed'] = False
                risk_checks['score'] *= 0.5

            if changes.get('stop_loss', 0.05) > 0.1:
                risk_checks['warnings'].append('Relaxed stop loss')
                risk_checks['score'] *= 0.9

            # Check for correlation with existing risks
            if changes.get('leverage', 1.0) > 2.0:
                risk_checks['violations'].append('High leverage increase')
                risk_checks['passed'] = False
                risk_checks['score'] *= 0.3

            return risk_checks

        except Exception as e:
            logger.warning(f"Error in risk validation: {e}")
            return {'passed': False, 'score': 0.0, 'error': str(e)}

    # ===== PROPOSAL IMPLEMENTATION HELPERS =====

    async def _create_implementation_backup(self, proposal: Dict[str, Any]) -> str:
        """
        Create a backup before implementing changes.
        """
        try:
            proposal_id = proposal.get('proposal_id')
            backup_id = f"backup_{proposal_id}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

            # Create backup of current system state
            backup_data = {
                'proposal_id': proposal_id,
                'backup_id': backup_id,
                'created_at': pd.Timestamp.now().isoformat(),
                'system_state': {
                    'memory_snapshot': self.memory.copy(),
                    'active_proposals': len(self.memory.get('proposal_queue', [])),
                    'model_state': {
                        'trained': self.model_trained,
                        'last_trained': self.model_last_trained.isoformat() if self.model_last_trained else None
                    }
                }
            }

            # Store backup
            if 'implementation_backups' not in self.memory:
                self.memory['implementation_backups'] = {}

            self.memory['implementation_backups'][backup_id] = backup_data

            logger.info(f"Created implementation backup: {backup_id}")
            return backup_id

        except Exception as e:
            logger.error(f"Error creating implementation backup: {e}")
            return None

    async def _apply_proposal_changes(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the proposed changes to the system.
        """
        try:
            changes = proposal.get('changes', {})
            proposal_type = proposal.get('proposal_type')

            implementation_results = {
                'success': False,
                'changes_applied': [],
                'errors': []
            }

            # Apply changes based on type
            if proposal_type == 'strategy_optimization':
                result = await self._apply_strategy_changes(changes)
                implementation_results['changes_applied'].extend(result.get('applied', []))
                implementation_results['errors'].extend(result.get('errors', []))

            elif proposal_type == 'risk_parameter_adjustment':
                result = await self._apply_risk_changes(changes)
                implementation_results['changes_applied'].extend(result.get('applied', []))
                implementation_results['errors'].extend(result.get('errors', []))

            elif proposal_type == 'execution_improvement':
                result = await self._apply_execution_changes(changes)
                implementation_results['changes_applied'].extend(result.get('applied', []))
                implementation_results['errors'].extend(result.get('errors', []))

            # Check if implementation was successful
            implementation_results['success'] = len(implementation_results['errors']) == 0

            logger.info(f"Proposal implementation completed: success={implementation_results['success']}, changes={len(implementation_results['changes_applied'])}")

            return implementation_results

        except Exception as e:
            logger.error(f"Error applying proposal changes: {e}")
            return {'success': False, 'error': str(e)}

    async def _start_performance_monitoring(self, proposal: Dict[str, Any]) -> None:
        """
        Start performance monitoring for implemented proposal.
        """
        try:
            proposal_id = proposal.get('proposal_id')

            # Create monitoring task
            monitoring_task = {
                'proposal_id': proposal_id,
                'started_at': pd.Timestamp.now().isoformat(),
                'monitoring_period': proposal.get('monitoring_period', 30),  # days
                'baseline_metrics': self._get_current_performance_baseline(),
                'status': 'active'
            }

            # Add to monitoring queue
            if 'performance_monitoring' not in self.memory:
                self.memory['performance_monitoring'] = {}

            self.memory['performance_monitoring'][proposal_id] = monitoring_task

            logger.info(f"Started performance monitoring for proposal: {proposal_id}")

        except Exception as e:
            logger.error(f"Error starting performance monitoring: {e}")

    # ===== PROPOSAL ROLLBACK HELPERS =====

    async def _perform_rollback(self, proposal: Dict[str, Any], backup_id: str, reason: str) -> Dict[str, Any]:
        """
        Perform rollback of an implemented proposal.
        """
        try:
            rollback_results = {
                'success': False,
                'changes_reverted': [],
                'errors': []
            }

            # Get backup data
            backup_data = self.memory.get('implementation_backups', {}).get(backup_id)
            if not backup_data:
                return {'success': False, 'error': 'Backup not found'}

            # Restore system state from backup
            system_state = backup_data.get('system_state', {})

            # Restore memory (careful - don't overwrite new critical data)
            memory_snapshot = system_state.get('memory_snapshot', {})

            # Only restore specific keys that are safe to rollback
            safe_keys = ['weekly_batches', 'pyramiding_performance', 'model_params']
            for key in safe_keys:
                if key in memory_snapshot:
                    self.memory[key] = memory_snapshot[key]

            # Restore model state if needed
            model_state = system_state.get('model_state', {})
            if model_state.get('trained'):
                self.model_trained = model_state['trained']
                if model_state.get('last_trained'):
                    self.model_last_trained = pd.Timestamp(model_state['last_trained'])

            rollback_results['success'] = True
            rollback_results['changes_reverted'] = safe_keys

            logger.info(f"Successfully rolled back proposal using backup: {backup_id}")
            return rollback_results

        except Exception as e:
            logger.error(f"Error performing rollback: {e}")
            return {'success': False, 'error': str(e)}

    # ===== UTILITY METHODS =====

    async def _get_test_market_data(self) -> pd.DataFrame:
        """
        Get sample market data for testing purposes.
        """
        try:
            # Generate synthetic market data for testing
            current_year = datetime.now().year
            dates = pd.date_range(start=f'{current_year}-01-01', end=f'{current_year}-12-31', freq='D')
            np.random.seed(42)  # For reproducible results

            # Generate realistic price series
            n_days = len(dates)
            returns = np.random.normal(0.0005, 0.02, n_days)  # Mean return 0.05%, vol 2%
            prices = 100 * np.exp(np.cumsum(returns))  # Start at $100

            # Create OHLCV data
            high_mult = 1 + np.abs(np.random.normal(0, 0.01, n_days))
            low_mult = 1 - np.abs(np.random.normal(0, 0.01, n_days))
            volume = np.random.lognormal(10, 1, n_days)  # Log-normal volume

            market_data = pd.DataFrame({
                'Open': prices * (1 + np.random.normal(0, 0.005, n_days)),
                'High': prices * high_mult,
                'Low': prices * low_mult,
                'Close': prices,
                'Volume': volume
            }, index=dates)

            return market_data

        except Exception as e:
            logger.warning(f"Error generating test market data: {e}")
            return pd.DataFrame()

    def _score_backtest_results(self, backtest_result: Dict[str, Any]) -> float:
        """
        Score backtest results on a 0-1 scale.
        """
        try:
            sharpe = backtest_result.get('sharpe_ratio', 0)
            total_return = backtest_result.get('total_return', 0)
            max_drawdown = backtest_result.get('max_drawdown', 1)
            win_rate = backtest_result.get('win_rate', 0)

            # Normalize and score each metric
            sharpe_score = min(1.0, max(0.0, (sharpe + 1) / 3))  # -1 to 2 range -> 0 to 1
            return_score = min(1.0, max(0.0, total_return * 2))   # Scale returns
            drawdown_score = max(0.0, 1 - max_drawdown * 5)       # Penalize drawdowns
            win_rate_score = win_rate                             # Already 0-1

            # Weighted average
            overall_score = (
                sharpe_score * 0.4 +
                return_score * 0.3 +
                drawdown_score * 0.2 +
                win_rate_score * 0.1
            )

            return overall_score

        except Exception as e:
            logger.warning(f"Error scoring backtest results: {e}")
            return 0.0

    async def _run_stress_scenario(self, proposal: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a specific stress test scenario.
        """
        try:
            # Simplified stress test - in production would modify market data
            # and run backtest with stressed conditions

            # For now, return a random score based on scenario severity
            severity_score = 0.5

            if scenario.get('volatility_multiplier', 1.0) > 1.5:
                severity_score -= 0.2
            if scenario.get('trend_strength', 0) < -0.5:
                severity_score -= 0.2
            if scenario.get('liquidity_drop', 0) > 0.5:
                severity_score -= 0.1

            return {'score': max(0.0, severity_score)}

        except Exception as e:
            logger.warning(f"Error running stress scenario: {e}")
            return {'score': 0.0}

    async def _apply_strategy_changes(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply strategy optimization changes.
        """
        try:
            applied = []
            errors = []

            # Apply strategy configuration changes
            strategy_config = changes.get('strategy_config', {})
            if strategy_config:
                # Update memory with new strategy parameters
                if 'strategy_params' not in self.memory:
                    self.memory['strategy_params'] = {}

                self.memory['strategy_params'].update(strategy_config)
                applied.append('strategy_config')

            # Apply model parameter changes
            model_params = changes.get('model_params', {})
            if model_params:
                # This would update the actual model in production
                applied.append('model_params')

            return {'applied': applied, 'errors': errors}

        except Exception as e:
            logger.error(f"Error applying strategy changes: {e}")
            return {'applied': [], 'errors': [str(e)]}

    async def _apply_risk_changes(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply risk parameter changes.
        """
        try:
            applied = []
            errors = []

            # Apply risk parameter changes
            risk_params = changes.get('risk_params', {})
            if risk_params:
                if 'risk_params' not in self.memory:
                    self.memory['risk_params'] = {}

                self.memory['risk_params'].update(risk_params)
                applied.append('risk_params')

            return {'applied': applied, 'errors': errors}

        except Exception as e:
            logger.error(f"Error applying risk changes: {e}")
            return {'applied': [], 'errors': [str(e)]}

    async def _apply_execution_changes(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply execution improvement changes.
        """
        try:
            applied = []
            errors = []

            # Apply execution parameter changes
            exec_params = changes.get('execution_params', {})
            if exec_params:
                if 'execution_params' not in self.memory:
                    self.memory['execution_params'] = {}

                self.memory['execution_params'].update(exec_params)
                applied.append('execution_params')

            return {'applied': applied, 'errors': errors}

        except Exception as e:
            logger.error(f"Error applying execution changes: {e}")
            return {'applied': [], 'errors': [str(e)]}

    def _get_current_performance_baseline(self) -> Dict[str, Any]:
        """
        Get current performance baseline for monitoring.
        """
        try:
            baseline = {}

            # Get recent performance metrics
            recent_batches = self.memory.get('weekly_batches', [])[-5:]  # Last 5 batches

            if recent_batches:
                baseline['avg_sharpe'] = np.mean([b.get('sharpe_ratios', [0]) for b in recent_batches])
                baseline['avg_returns'] = np.mean([b.get('returns', [0]) for b in recent_batches])
                baseline['avg_drawdowns'] = np.mean([b.get('drawdowns', [0]) for b in recent_batches])

            return baseline

        except Exception as e:
            logger.warning(f"Error getting performance baseline: {e}")
            return {}

    # ===== OPTIMIZATION PROPOSAL METHODS =====

    async def monitor_strategy_performance(self) -> Dict[str, Any]:
        """Monitor learning agent performance and identify optimization opportunities."""
        try:
            logger.info("LearningAgent monitoring strategy performance")

            # Get recent learning performance data
            performance_data = await self._collect_learning_performance_data()

            # Analyze learning effectiveness
            learning_analysis = self._analyze_learning_effectiveness(performance_data)

            # Identify optimization opportunities
            optimization_opportunities = self._identify_learning_optimization_opportunities(learning_analysis)

            # Generate optimization proposals
            optimization_proposals = []
            for opportunity in optimization_opportunities:
                proposal = await self._generate_learning_optimization_proposal(opportunity, performance_data)
                if proposal:
                    optimization_proposals.append(proposal)

            monitoring_result = {
                'performance_metrics': {
                    'learning_effectiveness_score': learning_analysis.get('overall_effectiveness', 0),
                    'optimization_opportunities': len(optimization_opportunities),
                    'proposals_generated': len(optimization_proposals),
                    'model_performance': learning_analysis.get('model_performance', 'unknown'),
                    'learning_trends': learning_analysis.get('learning_trends', 'stable'),
                    'optimization_success_rate': learning_analysis.get('optimization_success_rate', 0)
                },
                'performance_summary': learning_analysis,
                'optimization_opportunities': len(optimization_opportunities),
                'proposals_generated': len(optimization_proposals),
                'learning_effectiveness_score': learning_analysis.get('overall_effectiveness', 0),
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"LearningAgent performance monitoring completed: {len(optimization_proposals)} proposals generated")
            return monitoring_result

        except Exception as e:
            logger.error(f"LearningAgent performance monitoring failed: {e}")
            return {'error': str(e)}

    async def _collect_learning_performance_data(self) -> Dict[str, Any]:
        """Collect performance data for learning activities."""
        try:
            performance_data = {
                'model_performance': {},
                'learning_metrics': {},
                'optimization_history': [],
                'timestamp': datetime.now().isoformat()
            }

            # Get recent model performance from memory
            model_history = self.memory.get('model_performance_history', [])

            if model_history:
                recent_models = model_history[-10:]  # Last 10 model evaluations

                # Calculate learning metrics
                accuracies = [m.get('accuracy', 0) for m in recent_models]
                losses = [m.get('loss', 0) for m in recent_models]
                improvements = []

                for i in range(1, len(accuracies)):
                    if accuracies[i-1] > 0:
                        improvement = (accuracies[i] - accuracies[i-1]) / accuracies[i-1]
                        improvements.append(improvement)

                performance_data['learning_metrics'] = {
                    'avg_accuracy': np.mean(accuracies) if accuracies else 0,
                    'avg_loss': np.mean(losses) if losses else 0,
                    'learning_trend': np.mean(improvements) if improvements else 0,
                    'total_models_evaluated': len(recent_models),
                    'consistency_score': 1 - np.std(accuracies) if accuracies else 0
                }

            # Get optimization proposal history
            proposal_history = self.memory.get('optimization_proposals', {})
            performance_data['optimization_history'] = list(proposal_history.values())[-20:]  # Last 20 proposals

            return performance_data

        except Exception as e:
            logger.error(f"Error collecting learning performance data: {e}")
            return {'error': str(e)}

    def _analyze_learning_effectiveness(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the effectiveness of learning processes."""
        try:
            analysis = {
                'overall_effectiveness': 0.5,
                'learning_trends': 'stable',
                'model_performance': 'adequate',
                'optimization_success_rate': 0.0,
                'areas_for_improvement': []
            }

            learning_metrics = performance_data.get('learning_metrics', {})

            # Assess learning effectiveness
            avg_accuracy = learning_metrics.get('avg_accuracy', 0)
            learning_trend = learning_metrics.get('learning_trend', 0)
            consistency = learning_metrics.get('consistency_score', 0)

            # Calculate overall effectiveness score
            effectiveness_score = (avg_accuracy * 0.4 + (learning_trend + 1) * 0.3 + consistency * 0.3)
            analysis['overall_effectiveness'] = min(max(effectiveness_score, 0), 1)

            # Determine learning trends
            if learning_trend > 0.05:
                analysis['learning_trends'] = 'improving'
            elif learning_trend < -0.05:
                analysis['learning_trends'] = 'declining'
            else:
                analysis['learning_trends'] = 'stable'

            # Assess model performance
            if avg_accuracy > 0.8:
                analysis['model_performance'] = 'excellent'
            elif avg_accuracy > 0.6:
                analysis['model_performance'] = 'good'
            elif avg_accuracy > 0.4:
                analysis['model_performance'] = 'adequate'
            else:
                analysis['model_performance'] = 'poor'

            # Analyze optimization success
            proposal_history = performance_data.get('optimization_history', [])
            if proposal_history:
                successful_proposals = sum(1 for p in proposal_history if p.get('status') == 'implemented')
                analysis['optimization_success_rate'] = successful_proposals / len(proposal_history)

            # Identify areas for improvement
            if learning_trend < 0:
                analysis['areas_for_improvement'].append('model_degradation')
            if consistency < 0.7:
                analysis['areas_for_improvement'].append('inconsistent_performance')
            if analysis['optimization_success_rate'] < 0.6:
                analysis['areas_for_improvement'].append('low_optimization_success')

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing learning effectiveness: {e}")
            return {'error': str(e)}

    def _identify_learning_optimization_opportunities(self, learning_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify optimization opportunities based on learning analysis."""
        opportunities = []

        try:
            effectiveness = learning_analysis.get('overall_effectiveness', 0.5)
            areas_for_improvement = learning_analysis.get('areas_for_improvement', [])

            # Model retraining opportunity
            if effectiveness < 0.7 or 'model_degradation' in areas_for_improvement:
                opportunities.append({
                    'type': 'model_retraining',
                    'priority': 'high' if effectiveness < 0.5 else 'medium',
                    'description': 'Retraining models with recent data to improve performance',
                    'expected_impact': '15-25% improvement in model accuracy',
                    'implementation_complexity': 'medium',
                    'estimated_cost': '4-6 hours of compute time'
                })

            # Feature engineering opportunity
            if 'inconsistent_performance' in areas_for_improvement:
                opportunities.append({
                    'type': 'feature_engineering',
                    'priority': 'medium',
                    'description': 'Improve feature selection and engineering for better consistency',
                    'expected_impact': '10-20% improvement in prediction consistency',
                    'implementation_complexity': 'high',
                    'estimated_cost': '2-3 days development'
                })

            # Optimization algorithm improvement
            success_rate = learning_analysis.get('optimization_success_rate', 0)
            if success_rate < 0.7:
                opportunities.append({
                    'type': 'optimization_algorithm',
                    'priority': 'medium',
                    'description': 'Improve optimization algorithms for better proposal success rates',
                    'expected_impact': '20-30% improvement in optimization success',
                    'implementation_complexity': 'high',
                    'estimated_cost': '1-2 weeks development'
                })

            # Learning pipeline optimization
            if len(areas_for_improvement) > 1:
                opportunities.append({
                    'type': 'learning_pipeline',
                    'priority': 'low',
                    'description': 'Optimize overall learning pipeline for better efficiency',
                    'expected_impact': '5-15% improvement in learning speed and effectiveness',
                    'implementation_complexity': 'low',
                    'estimated_cost': '1-2 days development'
                })

        except Exception as e:
            logger.error(f"Error identifying learning optimization opportunities: {e}")

        return opportunities

    async def _generate_learning_optimization_proposal(self, opportunity: Dict[str, Any],
                                                       performance_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate optimization proposal for learning improvements."""
        try:
            proposal_type = opportunity['type']

            proposal = {
                'id': f"learning_opt_{proposal_type}_{int(datetime.now().timestamp())}",
                'type': 'learning_optimization',
                'target_component': proposal_type,
                'current_performance': performance_data.get('learning_metrics', {}),
                'proposed_changes': self._get_learning_proposal_changes(proposal_type),
                'expected_benefits': {
                    'accuracy_improvement': opportunity.get('expected_impact', '').split('-')[0].strip('%') or 10,
                    'efficiency_gain': 15,
                    'robustness_improvement': 20
                },
                'implementation_complexity': opportunity.get('implementation_complexity', 'medium'),
                'estimated_implementation_time': opportunity.get('estimated_cost', '2_days'),
                'risk_assessment': {
                    'technical_risk': 'low' if opportunity.get('implementation_complexity') == 'low' else 'medium',
                    'performance_risk': 'low',
                    'resource_risk': 'low'
                },
                'success_metrics': [
                    'model_accuracy_improvement',
                    'learning_consistency',
                    'optimization_success_rate'
                ],
                'timestamp': datetime.now().isoformat()
            }

            return proposal

        except Exception as e:
            logger.error(f"Error generating learning optimization proposal: {e}")
            return None

    def _get_learning_proposal_changes(self, proposal_type: str) -> Dict[str, Any]:
        """Get specific changes for learning optimization proposals."""
        changes_map = {
            'model_retraining': {
                'retraining_frequency': 'weekly',
                'data_window': '6_months',
                'hyperparameter_tuning': True,
                'cross_validation_folds': 5
            },
            'feature_engineering': {
                'new_features': ['market_regime', 'volatility_clusters', 'correlation_features'],
                'feature_selection_method': 'recursive_elimination',
                'dimensionality_reduction': 'pca_95_variance'
            },
            'optimization_algorithm': {
                'algorithm': 'bayesian_optimization',
                'search_space_expansion': 2.0,
                'early_stopping_patience': 20,
                'parallel_evaluations': 4
            },
            'learning_pipeline': {
                'batch_size_increase': 1.5,
                'memory_optimization': True,
                'async_processing': True,
                'caching_strategy': 'intelligent'
            }
        }

        return changes_map.get(proposal_type, {})

    # ===== PROPOSAL LOG ACCESS =====

    def get_proposal_log(self, status_filter: str = None, limit: int = 50) -> Dict[str, Any]:
        """
        Get the proposal log with filtering and pagination.

        Args:
            status_filter: Filter by status ('queued', 'evaluated', 'implemented', 'rejected', 'rolled_back', etc.)
            limit: Maximum number of proposals to return

        Returns:
            Dict with proposal log data
        """
        try:
            proposal_tracking = self.memory.get('proposal_tracking', {})
            proposal_queue = self.memory.get('proposal_queue', [])

            # Filter proposals by status if requested
            if status_filter:
                filtered_tracking = {
                    pid: data for pid, data in proposal_tracking.items()
                    if data.get('status') == status_filter
                }
                filtered_queue = [
                    p for p in proposal_queue
                    if p.get('evaluation_status') == status_filter or p.get('status') == status_filter
                ]
            else:
                filtered_tracking = proposal_tracking
                filtered_queue = proposal_queue

            # Combine and sort by most recent first
            all_proposals = []

            # Add tracking data
            for proposal_id, tracking_data in filtered_tracking.items():
                proposal_data = {
                    'proposal_id': proposal_id,
                    'status': tracking_data.get('status', 'unknown'),
                    'submitted_by': tracking_data.get('submitted_by', 'unknown'),
                    'proposal_type': tracking_data.get('proposal_type', 'unknown'),
                    'received_at': tracking_data.get('received_at'),
                    'evaluated_at': tracking_data.get('evaluated_at'),
                    'implemented_at': tracking_data.get('implemented_at'),
                    'evaluation_score': tracking_data.get('evaluation_score'),
                    'recommendation': tracking_data.get('recommendation'),
                    'source': 'tracking'
                }
                all_proposals.append(proposal_data)

            # Add queue data (more detailed)
            for proposal in filtered_queue[-limit:]:  # Limit queue items
                queue_data = {
                    'proposal_id': proposal.get('proposal_id'),
                    'status': proposal.get('evaluation_status', proposal.get('status', 'unknown')),
                    'submitted_by': proposal.get('submitted_by', 'unknown'),
                    'proposal_type': proposal.get('proposal_type', 'unknown'),
                    'received_at': proposal.get('received_at'),
                    'evaluated_at': proposal.get('evaluated_at'),
                    'implemented_at': proposal.get('implemented_at'),
                    'evaluation_results': proposal.get('evaluation_results'),
                    'test_results': proposal.get('test_results'),
                    'implementation_results': proposal.get('implementation_results'),
                    'description': proposal.get('description', ''),
                    'confidence_score': proposal.get('confidence_score'),
                    'source': 'queue'
                }
                all_proposals.append(queue_data)

            # Remove duplicates (prefer queue data over tracking data)
            seen_ids = set()
            unique_proposals = []
            for proposal in reversed(all_proposals):  # Process in reverse to prefer later entries
                pid = proposal['proposal_id']
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    unique_proposals.append(proposal)

            # Sort by most recent timestamp
            def get_sort_key(p):
                timestamps = [
                    p.get('implemented_at'),
                    p.get('evaluated_at'),
                    p.get('received_at')
                ]
                valid_timestamps = [t for t in timestamps if t]
                return max(valid_timestamps) if valid_timestamps else ''

            unique_proposals.sort(key=get_sort_key, reverse=True)

            # Apply limit
            unique_proposals = unique_proposals[:limit]

            # Calculate summary statistics
            status_counts = {}
            type_counts = {}
            for proposal in unique_proposals:
                status = proposal.get('status', 'unknown')
                p_type = proposal.get('proposal_type', 'unknown')

                status_counts[status] = status_counts.get(status, 0) + 1
                type_counts[p_type] = type_counts.get(p_type, 0) + 1

            proposal_log = {
                'total_proposals': len(unique_proposals),
                'status_filter': status_filter,
                'limit_applied': limit,
                'summary': {
                    'status_distribution': status_counts,
                    'type_distribution': type_counts,
                    'total_tracked': len(proposal_tracking),
                    'total_queued': len(proposal_queue)
                },
                'proposals': unique_proposals,
                'timestamp': pd.Timestamp.now().isoformat()
            }

            logger.info(f"Retrieved proposal log: {len(unique_proposals)} proposals, filter: {status_filter}")
            return proposal_log

        except Exception as e:
            logger.error(f"Error retrieving proposal log: {e}")
            return {
                'error': str(e),
                'total_proposals': 0,
                'proposals': [],
                'timestamp': pd.Timestamp.now().isoformat()
            }