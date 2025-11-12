# src/utils/risk_analytics_framework.py
# Purpose: Comprehensive risk analytics framework for historical simulations
# Provides advanced risk metrics, performance attribution, and benchmarking

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)

class RiskAnalyticsFramework:
    """
    Comprehensive framework for analyzing portfolio risk in historical simulations.
    Includes advanced risk metrics, stress testing, and performance attribution.
    """

    def __init__(self, simulation_results: Dict[str, Any]):
        self.simulation_results = simulation_results
        self.portfolio_history = pd.DataFrame(simulation_results.get('portfolio_history', []))
        self.trades = simulation_results.get('trades', [])
        self.benchmark_data = simulation_results.get('benchmark_comparison', {})

        logger.info(f"RiskAnalyticsFramework: portfolio_history columns: {list(self.portfolio_history.columns) if not self.portfolio_history.empty else 'empty'}")
        logger.info(f"RiskAnalyticsFramework: portfolio_history shape: {self.portfolio_history.shape}")

        if not self.portfolio_history.empty:
            # Check if 'date' column exists
            if 'date' in self.portfolio_history.columns:
                self.portfolio_history['date'] = pd.to_datetime(self.portfolio_history['date'])
                self.portfolio_history = self.portfolio_history.set_index('date')
                self.portfolio_history['returns'] = self.portfolio_history['portfolio_value'].pct_change()
                logger.info("RiskAnalyticsFramework: successfully processed portfolio_history")
            else:
                logger.error(f"RiskAnalyticsFramework: 'date' column not found in portfolio_history. Columns: {list(self.portfolio_history.columns)}")
                self.portfolio_history = pd.DataFrame()  # Set to empty to avoid further errors

    def calculate_advanced_risk_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics beyond basic statistics.

        Returns:
            Dict with advanced risk metrics
        """
        if self.portfolio_history.empty:
            return {'error': 'No portfolio history available'}

        returns = self.portfolio_history['returns'].dropna()

        metrics = {}

        # Basic risk metrics
        metrics['volatility'] = returns.std() * np.sqrt(252)
        metrics['downside_volatility'] = returns[returns < 0].std() * np.sqrt(252)
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()

        # Value at Risk (VaR) calculations
        confidence_levels = [0.95, 0.99]
        for conf in confidence_levels:
            # Historical VaR
            metrics[f'var_historical_{int(conf*100)}'] = np.percentile(returns, (1-conf) * 100)

            # Parametric VaR (assuming normal distribution)
            metrics[f'var_parametric_{int(conf*100)}'] = stats.norm.ppf(1-conf, returns.mean(), returns.std())

            # Cornish-Fisher VaR (adjusts for skewness and kurtosis)
            z_score = stats.norm.ppf(1-conf)
            cf_adjustment = (z_score**2 - 1) * metrics['skewness']/6 + (z_score**3 - 3*z_score) * metrics['kurtosis']/24 - z_score**3 * metrics['kurtosis']/36
            metrics[f'var_cornish_fisher_{int(conf*100)}'] = returns.mean() + (z_score + cf_adjustment) * returns.std()

        # Conditional VaR (CVaR/Expected Shortfall)
        for conf in confidence_levels:
            var_threshold = np.percentile(returns, (1-conf) * 100)
            cvar_returns = returns[returns <= var_threshold]
            metrics[f'cvar_{int(conf*100)}'] = cvar_returns.mean() if len(cvar_returns) > 0 else var_threshold

        # Maximum Drawdown Analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        metrics['max_drawdown'] = drawdown.min()
        metrics['avg_drawdown'] = drawdown[drawdown < 0].mean()
        metrics['drawdown_std'] = drawdown[drawdown < 0].std()

        # Recovery analysis
        drawdown_periods = self._identify_drawdown_periods(drawdown)
        if drawdown_periods:
            recovery_times = [period['recovery_days'] for period in drawdown_periods if period['recovery_days'] is not None]
            if recovery_times:
                metrics['avg_recovery_time'] = np.mean(recovery_times)
                metrics['max_recovery_time'] = max(recovery_times)
                metrics['recovery_success_rate'] = len(recovery_times) / len(drawdown_periods)

        # Stress testing metrics
        stress_metrics = self._calculate_stress_metrics(returns)
        metrics.update(stress_metrics)

        # Liquidity risk metrics
        liquidity_metrics = self._calculate_liquidity_risk()
        metrics.update(liquidity_metrics)

        return metrics

    def _identify_drawdown_periods(self, drawdown_series: pd.Series) -> List[Dict[str, Any]]:
        """
        Identify individual drawdown periods with start, peak, and recovery dates.

        Args:
            drawdown_series: Series of drawdown values

        Returns:
            List of drawdown period dictionaries
        """
        drawdown_periods = []
        in_drawdown = False
        drawdown_start = None
        max_drawdown = 0
        max_drawdown_date = None

        for date, dd_value in drawdown_series.items():
            if dd_value < 0 and not in_drawdown:
                # Start of new drawdown
                in_drawdown = True
                drawdown_start = date
                max_drawdown = dd_value
                max_drawdown_date = date
            elif dd_value < 0 and in_drawdown:
                # Continuing drawdown
                if dd_value < max_drawdown:
                    max_drawdown = dd_value
                    max_drawdown_date = date
            elif dd_value >= 0 and in_drawdown:
                # End of drawdown - recovery
                in_drawdown = False
                recovery_days = (date - max_drawdown_date).days if max_drawdown_date else None

                drawdown_periods.append({
                    'start_date': drawdown_start,
                    'peak_date': max_drawdown_date,
                    'recovery_date': date,
                    'max_drawdown': max_drawdown,
                    'duration_days': (date - drawdown_start).days,
                    'recovery_days': recovery_days
                })

        return drawdown_periods

    def _calculate_stress_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Calculate stress testing metrics for extreme market conditions.

        Args:
            returns: Portfolio returns series

        Returns:
            Dict with stress testing metrics
        """
        stress_metrics = {}

        # Define stress scenarios
        scenarios = {
            'black_monday_1987': -0.22,  # -22% in one day
            'dot_com_crash': -0.10,      # -10% in one day
            'financial_crisis_2008': -0.09,  # -9% in one day
            'covid_2020': -0.12,         # -12% in one day
            'weekly_stress': -0.05,      # -5% in one week
            'monthly_stress': -0.15      # -15% in one month
        }

        for scenario_name, shock_return in scenarios.items():
            # Calculate portfolio impact
            portfolio_impact = shock_return * (1 + shock_return)  # Compounded effect

            # Recovery time estimation (simplified model)
            volatility = returns.std()
            expected_recovery_days = abs(shock_return) / (volatility / np.sqrt(252)) if volatility > 0 else 30

            stress_metrics[f'{scenario_name}_impact'] = portfolio_impact
            stress_metrics[f'{scenario_name}_recovery_days'] = expected_recovery_days

        # Historical stress periods analysis
        stress_periods = self._identify_stress_periods(returns)
        stress_metrics['historical_stress_periods'] = len(stress_periods)

        if stress_periods:
            stress_metrics['avg_stress_duration'] = np.mean([p['duration'] for p in stress_periods])
            stress_metrics['max_stress_impact'] = min([p['impact'] for p in stress_periods])

        return stress_metrics

    def _identify_stress_periods(self, returns: pd.Series) -> List[Dict[str, Any]]:
        """
        Identify periods of market stress based on extreme negative returns.

        Args:
            returns: Portfolio returns series

        Returns:
            List of stress period dictionaries
        """
        stress_periods = []
        stress_threshold = returns.quantile(0.05)  # Bottom 5% of returns

        in_stress = False
        stress_start = None
        stress_returns = []

        for date, ret in returns.items():
            if ret <= stress_threshold and not in_stress:
                # Start of stress period
                in_stress = True
                stress_start = date
                stress_returns = [ret]
            elif ret <= stress_threshold and in_stress:
                # Continuing stress period
                stress_returns.append(ret)
            elif ret > stress_threshold and in_stress:
                # End of stress period
                in_stress = False
                duration = len(stress_returns)
                total_impact = np.prod(1 + np.array(stress_returns)) - 1

                stress_periods.append({
                    'start_date': stress_start,
                    'end_date': date,
                    'duration': duration,
                    'impact': total_impact,
                    'avg_daily_return': np.mean(stress_returns),
                    'cumulative_impact': total_impact
                })

        return stress_periods

    def _calculate_liquidity_risk(self) -> Dict[str, Any]:
        """
        Calculate liquidity risk metrics based on trading activity.

        Returns:
            Dict with liquidity risk metrics
        """
        if not self.trades:
            return {'liquidity_note': 'No trading data available for liquidity analysis'}

        liquidity_metrics = {}

        # Trading frequency analysis
        trade_dates = [pd.to_datetime(trade['date']) for trade in self.trades]
        if trade_dates:
            trading_days = len(set(trade_dates))
            total_days = (max(trade_dates) - min(trade_dates)).days if len(trade_dates) > 1 else 1
            liquidity_metrics['trading_frequency'] = trading_days / total_days if total_days > 0 else 0

        # Trade size analysis
        trade_values = [trade['value'] for trade in self.trades if trade['value'] > 0]
        if trade_values:
            liquidity_metrics['avg_trade_size'] = np.mean(trade_values)
            liquidity_metrics['median_trade_size'] = np.median(trade_values)
            liquidity_metrics['trade_size_volatility'] = np.std(trade_values) / np.mean(trade_values) if np.mean(trade_values) > 0 else 0

            # Concentration analysis
            sorted_trades = sorted(trade_values, reverse=True)
            top_5_pct = sum(sorted_trades[:max(1, int(len(sorted_trades) * 0.05))]) / sum(trade_values)
            liquidity_metrics['trade_concentration_top_5pct'] = top_5_pct

        # Transaction cost analysis
        total_commissions = sum(trade['commission'] for trade in self.trades)
        total_value = sum(trade['value'] for trade in self.trades)
        liquidity_metrics['total_transaction_costs'] = total_commissions
        liquidity_metrics['transaction_cost_ratio'] = total_commissions / total_value if total_value > 0 else 0

        return liquidity_metrics

    def perform_performance_attribution(self) -> Dict[str, Any]:
        """
        Perform detailed performance attribution analysis.

        Returns:
            Dict with performance attribution results
        """
        if self.portfolio_history.empty:
            return {'error': 'No portfolio history available'}

        attribution_results = {}

        # Benchmark comparison
        if self.benchmark_data and 'benchmark_cumulative_returns' in self.benchmark_data:
            benchmark_returns = pd.Series(self.benchmark_data['benchmark_cumulative_returns'])

            if not benchmark_returns.empty and len(benchmark_returns) == len(self.portfolio_history):
                try:
                    portfolio_cumulative = (1 + self.portfolio_history['returns']).cumprod() - 1
                    benchmark_cumulative = benchmark_returns

                    # Ensure indices align for comparison
                    if not portfolio_cumulative.index.equals(benchmark_cumulative.index):
                        # Reindex benchmark to match portfolio dates
                        benchmark_cumulative = benchmark_cumulative.reindex(portfolio_cumulative.index, method='ffill')

                    # Calculate excess returns
                    excess_returns = portfolio_cumulative - benchmark_cumulative

                    attribution_results['benchmark_comparison'] = {
                        'portfolio_total_return': portfolio_cumulative.iloc[-1],
                        'benchmark_total_return': benchmark_cumulative.iloc[-1],
                        'excess_return': excess_returns.iloc[-1],
                        'tracking_error': excess_returns.std(),
                        'information_ratio': excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
                    }
                except Exception as e:
                    logger.warning(f"Benchmark comparison failed: {e}")
                    attribution_results['benchmark_comparison'] = {'error': str(e)}
            else:
                attribution_results['benchmark_comparison'] = {'note': 'Benchmark data length mismatch or empty'}
        else:
            attribution_results['benchmark_comparison'] = {'note': 'No benchmark data available'}

        # Sector/asset contribution analysis
        if self.trades:
            attribution_results['asset_contribution'] = self._calculate_asset_contribution()

        # Timing vs Security Selection
        timing_attribution = self._calculate_timing_attribution()
        if timing_attribution:
            attribution_results['timing_attribution'] = timing_attribution

        return attribution_results

    def _calculate_asset_contribution(self) -> Dict[str, Any]:
        """
        Calculate the contribution of each asset to portfolio performance.

        Returns:
            Dict with asset contribution analysis
        """
        asset_contributions = {}

        # Group trades by symbol
        symbol_trades = {}
        for trade in self.trades:
            symbol = trade['symbol']
            if symbol not in symbol_trades:
                symbol_trades[symbol] = []
            symbol_trades[symbol].append(trade)

        for symbol, trades in symbol_trades.items():
            # Calculate total invested and current value
            total_invested = sum(trade['value'] for trade in trades if trade['action'] == 'BUY')
            total_returned = sum(trade['value'] for trade in trades if trade['action'] == 'SELL')

            # Calculate holding period returns
            buy_trades = [t for t in trades if t['action'] == 'BUY']
            sell_trades = [t for t in trades if t['action'] == 'SELL']

            if buy_trades and sell_trades:
                # Simple average holding period
                total_pnl = sum(trade['value'] for trade in sell_trades) - sum(trade['value'] for trade in buy_trades)
                total_cost = sum(trade['value'] for trade in buy_trades)

                asset_contributions[symbol] = {
                    'total_invested': total_invested,
                    'total_returned': total_returned,
                    'total_pnl': total_pnl,
                    'return_on_investment': total_pnl / total_cost if total_cost > 0 else 0,
                    'trade_count': len(trades)
                }

        return asset_contributions

    def _calculate_timing_attribution(self) -> Optional[Dict[str, Any]]:
        """
        Calculate market timing vs security selection attribution.

        Returns:
            Dict with timing attribution results or None if insufficient data
        """
        if self.portfolio_history.empty or not self.benchmark_data or 'benchmark_cumulative_returns' not in self.benchmark_data:
            return None

        # This is a simplified timing attribution model
        # In practice, this would use more sophisticated factor models

        portfolio_returns = self.portfolio_history['returns'].dropna()
        benchmark_returns = pd.Series(self.benchmark_data.get('benchmark_cumulative_returns', []))

        if len(benchmark_returns) != len(portfolio_returns):
            return None

        try:
            # Ensure both series have the same index for calculations
            if not portfolio_returns.index.equals(benchmark_returns.index):
                benchmark_returns = benchmark_returns.reindex(portfolio_returns.index, method='ffill')

            # Calculate beta (market sensitivity)
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0

            # Calculate timing skill (simplified)
            # Positive timing skill means portfolio outperforms when market is up
            benchmark_median = benchmark_returns.median()
            up_market_returns = portfolio_returns[benchmark_returns > benchmark_median]
            down_market_returns = portfolio_returns[benchmark_returns < benchmark_median]

            if len(up_market_returns) > 0 and len(down_market_returns) > 0:
                up_market_alpha = up_market_returns.mean() - beta * benchmark_returns[benchmark_returns > benchmark_median].mean()
                down_market_alpha = down_market_returns.mean() - beta * benchmark_returns[benchmark_returns < benchmark_median].mean()

                timing_skill = up_market_alpha - down_market_alpha

                return {
                    'beta': beta,
                    'up_market_alpha': up_market_alpha,
                    'down_market_alpha': down_market_alpha,
                    'timing_skill': timing_skill,
                    'market_timing_quality': 'good' if timing_skill > 0.02 else 'poor' if timing_skill < -0.02 else 'neutral'
                }
        except Exception as e:
            logger.warning(f"Timing attribution calculation failed: {e}")

        return None

    def generate_risk_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive risk report with all analytics.

        Returns:
            Complete risk analysis report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'simulation_summary': {
                'start_date': self.simulation_results.get('simulation_config', {}).get('start_date'),
                'end_date': self.simulation_results.get('simulation_config', {}).get('end_date'),
                'initial_capital': self.simulation_results.get('simulation_config', {}).get('initial_capital'),
                'final_value': self.simulation_results.get('trading_statistics', {}).get('final_portfolio_value'),
                'total_return': self.simulation_results.get('performance_metrics', {}).get('total_return')
            }
        }

        # Add advanced risk metrics
        report['risk_metrics'] = self.calculate_advanced_risk_metrics()

        # Add performance attribution
        report['performance_attribution'] = self.perform_performance_attribution()

        # Add risk assessment
        risk_assessment = self._generate_risk_assessment(report['risk_metrics'])
        report['risk_assessment'] = risk_assessment

        # Add recommendations
        report['recommendations'] = self._generate_risk_recommendations(report)

        return report

    def _generate_risk_assessment(self, risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate overall risk assessment based on calculated metrics.

        Args:
            risk_metrics: Dict of calculated risk metrics

        Returns:
            Dict with risk assessment summary
        """
        assessment = {
            'overall_risk_level': 'unknown',
            'risk_factors': [],
            'confidence_level': 'medium'
        }

        # Assess volatility risk
        volatility = risk_metrics.get('volatility', 0)
        if volatility > 0.4:  # >40% annualized volatility
            assessment['risk_factors'].append('extreme_volatility')
        elif volatility > 0.3:  # >30% annualized volatility
            assessment['risk_factors'].append('high_volatility')
        elif volatility > 0.2:  # >20% annualized volatility
            assessment['risk_factors'].append('moderate_volatility')

        # Assess drawdown risk
        max_drawdown = risk_metrics.get('max_drawdown', 0)
        if max_drawdown < -0.5:  # >50% drawdown
            assessment['risk_factors'].append('severe_drawdown_risk')
        elif max_drawdown < -0.3:  # >30% drawdown
            assessment['risk_factors'].append('significant_drawdown_risk')
        elif max_drawdown < -0.2:  # >20% drawdown
            assessment['risk_factors'].append('moderate_drawdown_risk')

        # Assess tail risk (CVaR vs VaR)
        cvar_95 = risk_metrics.get('cvar_95', 0)
        var_95 = risk_metrics.get('var_historical_95', 0)
        tail_risk_ratio = cvar_95 / var_95 if var_95 < 0 else 1
        if tail_risk_ratio > 1.5:
            assessment['risk_factors'].append('high_tail_risk')

        # Determine overall risk level
        risk_factor_count = len(assessment['risk_factors'])
        if risk_factor_count >= 3:
            assessment['overall_risk_level'] = 'very_high'
        elif risk_factor_count >= 2:
            assessment['overall_risk_level'] = 'high'
        elif risk_factor_count >= 1:
            assessment['overall_risk_level'] = 'moderate'
        else:
            assessment['overall_risk_level'] = 'low'

        # Assess confidence in risk metrics
        data_points = len(self.portfolio_history) if not self.portfolio_history.empty else 0
        if data_points > 500:  # More than ~2 years of daily data
            assessment['confidence_level'] = 'high'
        elif data_points > 100:  # More than ~6 months
            assessment['confidence_level'] = 'medium'
        else:
            assessment['confidence_level'] = 'low'

        return assessment

    def _generate_risk_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """
        Generate risk management recommendations based on analysis.

        Args:
            report: Complete risk analysis report

        Returns:
            List of risk management recommendations
        """
        recommendations = []

        risk_metrics = report.get('risk_metrics', {})
        risk_assessment = report.get('risk_assessment', {})

        # Volatility-based recommendations
        volatility = risk_metrics.get('volatility', 0)
        if volatility > 0.3:
            recommendations.append("Consider implementing volatility-based position sizing to reduce portfolio swings")
            recommendations.append("Evaluate adding diversification across uncorrelated assets")

        # Drawdown-based recommendations
        max_drawdown = risk_metrics.get('max_drawdown', 0)
        if max_drawdown < -0.2:
            recommendations.append("Implement stop-loss mechanisms to limit downside risk")
            recommendations.append("Consider dynamic asset allocation based on market conditions")

        # Tail risk recommendations
        cvar_95 = risk_metrics.get('cvar_95', 0)
        if cvar_95 < -0.1:  # More than 10% expected loss in worst 5% of cases
            recommendations.append("Portfolio exhibits significant tail risk - consider tail risk hedging strategies")
            recommendations.append("Evaluate put options or other downside protection mechanisms")

        # Liquidity recommendations
        liquidity_metrics = risk_metrics.get('liquidity_risk', {})
        if liquidity_metrics.get('trade_concentration_top_5pct', 0) > 0.5:
            recommendations.append("High concentration in large trades - consider breaking up positions for better liquidity")

        # Overall risk level recommendations
        risk_level = risk_assessment.get('overall_risk_level', 'unknown')
        if risk_level in ['very_high', 'high']:
            recommendations.append("Portfolio risk level is elevated - consider risk reduction strategies")
            recommendations.append("Regular stress testing and scenario analysis recommended")
        elif risk_level == 'low':
            recommendations.append("Portfolio maintains low risk profile - monitor for changes in market conditions")

        # Data quality recommendations
        confidence = risk_assessment.get('confidence_level', 'medium')
        if confidence == 'low':
            recommendations.append("Limited historical data - risk metrics have lower confidence")
            recommendations.append("Consider extending backtest period for more robust analysis")

        return recommendations

def analyze_portfolio_risk(simulation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to perform complete risk analysis on simulation results.

    Args:
        simulation_results: Results from historical portfolio simulation

    Returns:
        Comprehensive risk analysis report
    """
    try:
        framework = RiskAnalyticsFramework(simulation_results)
        return framework.generate_risk_report()
    except Exception as e:
        logger.error(f"Risk analysis failed: {e}")
        return {'error': str(e)}