# [LABEL:AGENT:reflection] [LABEL:COMPONENT:audit] [LABEL:FRAMEWORK:pyfolio] [LABEL:FRAMEWORK:zipline] [LABEL:FRAMEWORK:asyncio]
# [LABEL:AUTHOR:GitHub Copilot] [LABEL:UPDATED:2024-11-20] [LABEL:REVIEWED:yes]
#
# Purpose: Implements the Reflection Agent, subclassing BaseAgent for outcome reviews and audits (e.g., polls for bonuses). Handles mini-loops and escalations for sanity.
# Dependencies: sys, pathlib, src.agents.base, logging, typing, numpy, pandas, datetime, asyncio, src.utils.tools
# Related: docs/AGENTS/reflection-agent.md, config/base_prompt.txt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Dynamic root path for imports.

from src.agents.base import BaseAgent  # Absolute import.
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from src.utils.tools import (
    audit_poll_tool, pyfolio_metrics_tool, zipline_backtest_tool,
    sanity_check_tool, convergence_check_tool
)

logger = logging.getLogger(__name__)

class ReflectionAgent(BaseAgent):
    """
    Reflection Agent subclass.
    Reasoning: Reviews/logs outcomes with polls; refines via reflections for experiential audits.
    """
    def __init__(self, a2a_protocol=None):
        config_paths = {'risk': 'config/risk-constraints.yaml', 'profit': 'config/profitability-targets.yaml'}  # Relative to root.
        prompt_paths = {'base': 'config/base_prompt.txt', 'role': 'docs/AGENTS/main-agents/reflection-agent.md'}  # Relative to root.
        super().__init__(role='reflection', config_paths=config_paths, prompt_paths=prompt_paths, a2a_protocol=a2a_protocol)

        # Initialize tools
        self.tools = [
            audit_poll_tool,
            pyfolio_metrics_tool,
            zipline_backtest_tool,
            sanity_check_tool,
            convergence_check_tool
        ]

        # Memory is now loaded automatically by BaseAgent
        # Initialize memory structure if empty (first run)
        if not self.memory:
            self.memory = {
                'audit_history': [],  # For quarterly audits
                'bonus_stack': {},  # For POP credits with fade tracking
                'performance_history': [],  # For convergence analysis
                'mini_loop_history': [],  # For sanity check tracking
                'quarterly_audits': [],  # For long-horizon reviews
                'last_quarterly_audit': None
            }
            # Save initial memory structure
            self.save_memory()

        # Quarterly audit thresholds
        self.quarterly_thresholds = {
            'min_cumulative_return': 0.30,  # 30% minimum for no penalties
            'convergence_threshold': 0.01,  # 1% slope threshold
            'stability_threshold': 0.20,  # 20% CV threshold
            'bonus_fade_rate': 0.25  # 25% quarterly fade
        }

        logger.info(f"Reflection Agent initialized with {len(self.tools)} tools and quarterly audit system")

        # Initialize real-time auditing
        self._initialize_real_time_auditing()

        # Initialize audit metrics tracking
        self.audit_metrics = {
            'total_audits': 0,
            'last_audit_time': None,
            'anomalies_detected': 0,
            'validation_checks': 0,
            'current_risk_level': 'low',
            'performance_history': []
        }

    async def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes execution outcome or proposal for review.
        Args:
            input_data: Either execution outcome or trading proposal.
        Returns: Dict with insights, bonuses, or validation results.
        """
        logger.info(f"Reflection Agent processing: {input_data.keys()}")

        # Determine input type and route accordingly
        if 'p&l' in input_data or 'outcome' in input_data:
            # Post-execution review
            return await self._process_execution_outcome(input_data)
        elif 'executed' in input_data:
            # Execution result from execution agent
            return await self._process_execution_result(input_data)
        elif 'symbol' in input_data and 'quantity' in input_data:
            # Pre-execution sanity check
            return await self._process_pre_execution_check(input_data)
        elif 'audit_type' in input_data:
            # Quarterly audit request
            return await self._process_quarterly_audit(input_data)
        else:
            return {"error": "Unknown input type for reflection processing"}

    async def _process_execution_outcome(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process post-execution outcomes with metrics and bonus polling.
        """
        try:
            # Extract outcome data
            pnl_pct = outcome.get('p&l', 0)
            metrics = outcome.get('metrics', {})
            symbol = outcome.get('symbol', 'UNKNOWN')

            # Calculate comprehensive metrics using pyfolio tool
            if 'returns_data' in outcome:
                data_str = outcome['returns_data']
                metrics_result = pyfolio_metrics_tool.invoke({"data": data_str})
                if isinstance(metrics_result, str) and metrics_result.startswith("Error"):
                    # Fallback to basic metrics based on actual P&L
                    # Calculate Sharpe-like ratio from P&L volatility estimate
                    pnl_volatility = abs(pnl_pct) * 0.5  # Estimate volatility from P&L magnitude
                    sharpe = pnl_pct / pnl_volatility if pnl_volatility > 0 else 1.5
                    max_drawdown = abs(pnl_pct) * 0.3  # Conservative drawdown estimate
                elif isinstance(metrics_result, dict):
                    sharpe = metrics_result.get('sharpe_ratio', 1.5)
                    max_drawdown = metrics_result.get('max_drawdown', 0.15)
                else:
                    # Fallback for unexpected result type
                    pnl_volatility = abs(pnl_pct) * 0.5
                    sharpe = pnl_pct / pnl_volatility if pnl_volatility > 0 else 1.5
                    max_drawdown = abs(pnl_pct) * 0.3
            else:
                # Calculate basic metrics from available P&L data
                pnl_volatility = abs(pnl_pct) * 0.5  # Estimate volatility from P&L magnitude
                sharpe = pnl_pct / pnl_volatility if pnl_volatility > 0 else 1.5
                max_drawdown = abs(pnl_pct) * 0.3  # Conservative drawdown estimate

            # Store in performance history
            performance_entry = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'pnl_pct': pnl_pct,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'outcome': outcome,
                # Add pyramiding performance tracking
                'pyramiding': self._extract_pyramiding_metrics(outcome)
            }
            self.append_to_memory_list('performance_history', performance_entry, max_items=1000)

            # Use comprehensive LLM reasoning for all reflection decisions (deep analysis and over-analysis)
            estimate = outcome.get('estimate', pnl_pct * 100)  # Convert to percentage
            bonus_threshold = self.configs['profit']['bonuses']['threshold']

            bonus_awarded = False
            bonus_amount = 0
            bonus_rationale = ""

            if self.llm:
                # Build foundation context for LLM
                foundation_context = f"""
FOUNDATION REFLECTION ANALYSIS:
- Symbol: {symbol}
- P&L Percentage: {pnl_pct:.3f}
- Sharpe Ratio: {sharpe:.3f}
- Max Drawdown: {max_drawdown:.3f}
- Estimate Threshold: {bonus_threshold:.1f}%
- Current Performance History: {len(self.memory['performance_history'])} entries
- Recent Trend: {self._analyze_performance_trend().get('trend', 'unknown')}
- Pyramiding Metrics: {self._extract_pyramiding_metrics(outcome)}
"""

                llm_question = """
Based on the foundation reflection analysis above, should a bonus be awarded for this execution outcome?

Consider:
1. Performance quality relative to risk-adjusted metrics (Sharpe, drawdown)
2. Estimate accuracy and consistency with actual outcomes
3. Recent performance trends and system stability
4. Pyramiding execution effectiveness and risk management
5. Overall contribution to long-term portfolio objectives
6. Whether this outcome justifies POP credit adjustments

Provide a clear AWARD/NO-AWARD recommendation with detailed rationale.
"""

                try:
                    llm_response = await self.reason_with_llm(foundation_context, llm_question)

                    if "AWARD" in llm_response.upper() and not "NO-AWARD" in llm_response.upper():
                        bonus_awarded = True
                        bonus_amount = min(15, max(5, estimate / 10))
                        self._award_bonus(symbol, bonus_amount, estimate)
                        bonus_rationale = f"LLM Comprehensive Analysis Awarded: {llm_response[:200]}..."
                        logger.info(f"Reflection Agent LLM comprehensive analysis: Bonus awarded with deep reasoning")
                    elif "NO-AWARD" in llm_response.upper():
                        bonus_awarded = False
                        bonus_rationale = f"LLM Comprehensive Analysis No-Award: {llm_response[:200]}..."
                        logger.info(f"Reflection Agent LLM comprehensive analysis: No bonus with deep reasoning")
                    else:
                        # Use foundation logic if LLM is unclear
                        if estimate > bonus_threshold:
                            poll_question = f"Approve bonus for {estimate:.1f}% estimate on {symbol}? (Threshold: {bonus_threshold}%)"
                            poll_result = audit_poll_tool.invoke({"question": poll_question})
                            if isinstance(poll_result, dict) and poll_result.get('consensus', 'no') == 'yes' and poll_result.get('confidence', 0) >= 0.5:
                                bonus_awarded = True
                                bonus_amount = min(15, max(5, estimate / 10))
                                self._award_bonus(symbol, bonus_amount, estimate)
                                bonus_rationale = f"LLM Unclear, Foundation Poll Approved"
                            else:
                                bonus_awarded = False
                                bonus_rationale = f"LLM Unclear, Foundation Poll Rejected"
                        else:
                            bonus_awarded = False
                            bonus_rationale = f"LLM Unclear, Below Threshold"

                except Exception as e:
                    logger.warning(f"Reflection Agent LLM reasoning failed, using foundation logic: {e}")
                    # Foundation logic fallback
                    if estimate > bonus_threshold:
                        poll_question = f"Approve bonus for {estimate:.1f}% estimate on {symbol}? (Threshold: {bonus_threshold}%)"
                        poll_result = audit_poll_tool.invoke({"question": poll_question})
                        if isinstance(poll_result, dict) and poll_result.get('consensus', 'no') == 'yes' and poll_result.get('confidence', 0) >= 0.5:
                            bonus_awarded = True
                            bonus_amount = min(15, max(5, estimate / 10))
                            self._award_bonus(symbol, bonus_amount, estimate)
                            bonus_rationale = f"Foundation Poll Approved"
                        else:
                            bonus_awarded = False
                            bonus_rationale = f"Foundation Poll Rejected"
                    else:
                        bonus_awarded = False
                        bonus_rationale = f"Below Threshold"
            else:
                # Use foundation logic when LLM unavailable
                if estimate > bonus_threshold:
                    poll_question = f"Approve bonus for {estimate:.1f}% estimate on {symbol}? (Threshold: {bonus_threshold}%)"
                    poll_result = audit_poll_tool.invoke({"question": poll_question})
                    if isinstance(poll_result, dict) and poll_result.get('consensus', 'no') == 'yes' and poll_result.get('confidence', 0) >= 0.5:
                        bonus_awarded = True
                        bonus_amount = min(15, max(5, estimate / 10))
                        self._award_bonus(symbol, bonus_amount, estimate)
                        bonus_rationale = f"Foundation Poll Approved"
                    else:
                        bonus_awarded = False
                        bonus_rationale = f"Foundation Poll Rejected"
                else:
                    bonus_awarded = False
                    bonus_rationale = f"Below Threshold"            # Generate insights and recommendations
            insights = self._generate_insights(performance_entry, bonus_awarded)

            # Check for escalation triggers
            adjustments = self.reflect({
                'delta': pnl_pct,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown
            })

            result = {
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'bonus_awarded': bonus_awarded,
                'bonus_amount': bonus_amount if bonus_awarded else 0,
                'bonus_rationale': bonus_rationale,
                'insights': insights,
                'adjustments': adjustments,
                'recommendations': self._generate_recommendations(performance_entry),
                'audit_logged': True,
                'rationale': f"Reviewed {symbol} outcome: {pnl_pct:.2f}% P&L, Sharpe {sharpe:.2f}, "
                           f"bonus {'awarded' if bonus_awarded else 'not awarded'}"
            }

            # Log to audit history
            audit_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'execution_review',
                'outcome': outcome,
                'analysis': result
            }
            self.append_to_memory_list('audit_history', audit_entry, max_items=500)

            logger.info(f"Post-execution review complete: {result['rationale']}")
            return result

        except Exception as e:
            logger.error(f"Error processing execution outcome: {e}")
            return {"error": f"Outcome processing failed: {str(e)}"}

    async def _process_execution_result(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process execution result from execution agent with comprehensive analysis.
        """
        try:
            executed = execution_result.get('executed', False)
            symbol = execution_result.get('symbol', 'UNKNOWN')
            reason = execution_result.get('reason', 'unknown')
            roi_estimate = execution_result.get('roi_estimate', 0.0)

            # Extract execution details
            execution_details = {
                'executed': executed,
                'symbol': symbol,
                'reason': reason,
                'roi_estimate': roi_estimate,
                'price': execution_result.get('price'),
                'quantity': execution_result.get('quantity'),
                'total_value': execution_result.get('total_value'),
                'commission': execution_result.get('commission', 0),
                'simulated': execution_result.get('simulated', True),
                'source': execution_result.get('source', 'unknown'),
                'rationale': execution_result.get('rationale', ''),
                'timestamp': datetime.now().isoformat()
            }

            # Store execution result in performance history
            self.append_to_memory_list('performance_history', execution_details, max_items=1000)

            # Analyze execution patterns
            pattern_analysis = self._analyze_execution_patterns(execution_details)

            # Generate insights and recommendations
            insights = self._generate_execution_insights(execution_details, pattern_analysis)
            recommendations = self._generate_execution_recommendations(execution_details, pattern_analysis)

            # Determine if bonus should be awarded
            bonus_analysis = await self._analyze_execution_bonus(execution_details)

            result = {
                'execution_processed': True,
                'symbol': symbol,
                'executed': executed,
                'reason': reason,
                'roi_estimate': roi_estimate,
                'pattern_analysis': pattern_analysis,
                'insights': insights,
                'recommendations': recommendations,
                'bonus_analysis': bonus_analysis,
                'rationale': f"Processed execution result for {symbol}: {'Executed' if executed else 'Not executed'} ({reason})"
            }

            # Log to audit history
            audit_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'execution_result',
                'result': execution_result,
                'analysis': result
            }
            self.append_to_memory_list('audit_history', audit_entry, max_items=500)

            logger.info(f"Execution result processed: {result['rationale']}")
            return result

        except Exception as e:
            logger.error(f"Error processing execution result: {e}")
            return {"error": f"Execution result processing failed: {str(e)}"}

    def _analyze_execution_patterns(self, execution_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze execution patterns and trends.
        """
        try:
            symbol = execution_details.get('symbol', 'UNKNOWN')
            executed = execution_details.get('executed', False)
            reason = execution_details.get('reason', '')

            # Get recent executions for this symbol
            recent_executions = [
                entry for entry in self.memory.get('performance_history', [])
                if entry.get('symbol') == symbol and 'executed' in entry
            ][-10:]  # Last 10 executions

            if len(recent_executions) < 2:
                return {
                    'pattern': 'insufficient_history',
                    'execution_rate': 0.0 if len(recent_executions) == 0 else (sum(1 for e in recent_executions if e.get('executed')) / len(recent_executions)),
                    'common_reasons': [reason] if reason else []
                }

            # Calculate execution rate
            execution_rate = sum(1 for e in recent_executions if e.get('executed')) / len(recent_executions)

            # Analyze common rejection reasons
            rejection_reasons = [e.get('reason', '') for e in recent_executions if not e.get('executed', True)]
            common_reasons = list(set(rejection_reasons)) if rejection_reasons else []

            # Determine pattern
            if execution_rate > 0.8:
                pattern = 'high_execution_rate'
            elif execution_rate > 0.5:
                pattern = 'moderate_execution_rate'
            elif execution_rate > 0.2:
                pattern = 'low_execution_rate'
            else:
                pattern = 'very_low_execution_rate'

            # Analyze timing patterns (if timestamps available)
            timing_patterns = self._analyze_timing_patterns(recent_executions)

            return {
                'pattern': pattern,
                'execution_rate': execution_rate,
                'common_reasons': common_reasons,
                'timing_patterns': timing_patterns,
                'sample_size': len(recent_executions)
            }

        except Exception as e:
            logger.warning(f"Error analyzing execution patterns: {e}")
            return {'pattern': 'analysis_error', 'error': str(e)}

    def _analyze_timing_patterns(self, executions: list) -> Dict[str, Any]:
        """
        Analyze timing patterns in executions.
        """
        try:
            if len(executions) < 3:
                return {'pattern': 'insufficient_data'}

            # Extract timestamps and check for patterns
            timestamps = []
            for execution in executions:
                timestamp_str = execution.get('timestamp')
                if timestamp_str:
                    try:
                        timestamps.append(datetime.fromisoformat(timestamp_str))
                    except:
                        continue

            if len(timestamps) < 3:
                return {'pattern': 'insufficient_timestamps'}

            # Check for market hours clustering
            market_hours_executions = sum(1 for ts in timestamps if 9 <= ts.hour <= 16)
            market_hours_rate = market_hours_executions / len(timestamps)

            # Check for day-of-week patterns
            weekday_executions = sum(1 for ts in timestamps if ts.weekday() < 5)  # Monday-Friday
            weekday_rate = weekday_executions / len(timestamps)

            if market_hours_rate > 0.8:
                timing_pattern = 'market_hours_focused'
            elif weekday_rate > 0.9:
                timing_pattern = 'weekday_focused'
            else:
                timing_pattern = 'mixed_timing'

            return {
                'pattern': timing_pattern,
                'market_hours_rate': market_hours_rate,
                'weekday_rate': weekday_rate
            }

        except Exception as e:
            logger.warning(f"Error analyzing timing patterns: {e}")
            return {'pattern': 'analysis_error', 'error': str(e)}

    def _generate_execution_insights(self, execution_details: Dict[str, Any], pattern_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate insights from execution result and patterns.
        """
        insights = []
        executed = execution_details.get('executed', False)
        symbol = execution_details.get('symbol', 'UNKNOWN')
        reason = execution_details.get('reason', '')
        execution_rate = pattern_analysis.get('execution_rate', 0)

        if executed:
            insights.append(f"Successfully executed trade for {symbol}")
            if execution_details.get('simulated'):
                insights.append("Execution was simulated - monitor real execution performance")
        else:
            insights.append(f"Trade not executed for {symbol}: {reason}")
            if reason == 'market_closed':
                insights.append("Market timing issue - review execution scheduling")
            elif 'position_sizing' in reason:
                insights.append("Position sizing constraints preventing execution")
            elif 'poor_microstructure' in reason:
                insights.append("Market microstructure conditions not favorable")

        # Pattern insights
        if execution_rate < 0.3:
            insights.append(f"Low execution rate ({execution_rate:.1%}) for {symbol} - review strategy parameters")
        elif execution_rate > 0.8:
            insights.append(f"High execution rate ({execution_rate:.1%}) for {symbol} - strategy well-aligned with conditions")

        return insights

    def _generate_execution_recommendations(self, execution_details: Dict[str, Any], pattern_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on execution result and patterns.
        """
        recommendations = []
        executed = execution_details.get('executed', False)
        reason = execution_details.get('reason', '')
        execution_rate = pattern_analysis.get('execution_rate', 0)
        timing_pattern = pattern_analysis.get('timing_patterns', {}).get('pattern', '')

        if not executed:
            if reason == 'market_closed':
                recommendations.append("Adjust execution timing to align with market hours")
            elif 'position_sizing' in reason:
                recommendations.append("Review and adjust position sizing constraints")
            elif 'poor_microstructure' in reason:
                recommendations.append("Wait for better market microstructure conditions or adjust execution strategy")

        if execution_rate < 0.5:
            recommendations.append("Consider modifying strategy parameters to improve execution rate")
            if timing_pattern == 'mixed_timing':
                recommendations.append("Focus executions during optimal market hours")

        if execution_details.get('simulated') and executed:
            recommendations.append("Validate simulation parameters against real market conditions")

        return recommendations

    async def _analyze_execution_bonus(self, execution_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze if execution result warrants a bonus.
        """
        try:
            executed = execution_details.get('executed', False)
            roi_estimate = execution_details.get('roi_estimate', 0.0)

            # Bonuses are typically for successful executions with good ROI potential
            if not executed:
                return {
                    'bonus_awarded': False,
                    'reason': 'No execution occurred',
                    'amount': 0
                }

            # Check against bonus threshold
            bonus_threshold = self.configs['profit']['bonuses']['threshold']

            if roi_estimate >= bonus_threshold:
                # Award bonus for successful execution with good ROI potential
                bonus_amount = min(10, max(2, roi_estimate / 20))  # Scale with ROI
                self._award_bonus(execution_details.get('symbol', 'UNKNOWN'), bonus_amount, roi_estimate)

                return {
                    'bonus_awarded': True,
                    'amount': bonus_amount,
                    'reason': f'Successful execution with {roi_estimate:.1f}% ROI estimate above {bonus_threshold:.1f}% threshold',
                    'roi_estimate': roi_estimate,
                    'threshold': bonus_threshold
                }
            else:
                return {
                    'bonus_awarded': False,
                    'reason': f'ROI estimate {roi_estimate:.1f}% below {bonus_threshold:.1f}% threshold',
                    'roi_estimate': roi_estimate,
                    'threshold': bonus_threshold
                }

        except Exception as e:
            logger.warning(f"Error analyzing execution bonus: {e}")
            return {
                'bonus_awarded': False,
                'reason': f'Analysis error: {str(e)}',
                'amount': 0
            }

    async def _process_pre_execution_check(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform pre-execution sanity checks and mini-loops.
        """
        try:
            # Convert proposal dict to readable string for sanity check tool
            proposal_str = f"{proposal.get('direction', 'trade').title()} {proposal.get('quantity', 0)} shares of {proposal.get('symbol', 'UNKNOWN')} at ${proposal.get('price', 0):.2f}"

            # Use sanity check tool
            sanity_result = sanity_check_tool.invoke(proposal_str)

            if isinstance(sanity_result, dict) and sanity_result.get('proposal_valid', False):
                # Additional checks for mini-loop
                mini_loop_checks = await self._run_mini_loop(proposal)

                result = {
                    'sanity_check': sanity_result,
                    'mini_loop': mini_loop_checks,
                    'overall_approval': sanity_result['proposal_valid'] and mini_loop_checks['approved'],
                    'warnings': sanity_result.get('warnings', []) + mini_loop_checks.get('warnings', []),
                    'recommendation': 'proceed' if (sanity_result['proposal_valid'] and mini_loop_checks['approved']) else 'reject'
                }
            else:
                result = {
                    'sanity_check': sanity_result,
                    'mini_loop': {'approved': False, 'reason': 'Sanity check failed'},
                    'overall_approval': False,
                    'warnings': sanity_result.get('failed_checks', []) if isinstance(sanity_result, dict) else ['Sanity check error'],
                    'recommendation': 'reject'
                }

            # Log mini-loop result
            mini_loop_entry = {
                'timestamp': datetime.now().isoformat(),
                'proposal': proposal,
                'validation': result
            }
            self.append_to_memory_list('mini_loop_history', mini_loop_entry, max_items=200)

            logger.info(f"Pre-execution check complete: {result['recommendation']}")
            return result

        except Exception as e:
            logger.error(f"Error in pre-execution check: {e}")
            return {"error": f"Pre-execution check failed: {str(e)}"}

    async def _process_quarterly_audit(self, audit_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform quarterly audit with convergence analysis and long-horizon review.
        """
        try:
            audit_type = audit_request.get('audit_type', 'quarterly')

            # Check if enough time has passed since last audit
            now = datetime.now()
            if self.memory['last_quarterly_audit']:
                last_audit = datetime.fromisoformat(self.memory['last_quarterly_audit'])
                days_since = (now - last_audit).days
                if days_since < 90:  # Less than 3 months
                    return {"warning": f"Only {days_since} days since last audit. Quarterly audits should be 90+ days apart."}

            # Analyze performance history
            if len(self.memory['performance_history']) < 5:
                return {"warning": "Insufficient performance history for meaningful quarterly audit"}

            # Use convergence check tool
            history_data = self.memory['performance_history'][-20:]  # Last 20 entries
            convergence_result = convergence_check_tool.invoke({"performance_data": {"metrics": {}, "learning_history": history_data}})

            # Calculate cumulative metrics
            recent_performance = self.memory['performance_history'][-10:]  # Last 10 trades/outcomes
            cumulative_return = sum(p.get('pnl_pct', 0) for p in recent_performance)

            # Determine audit outcome
            min_threshold = self.quarterly_thresholds['min_cumulative_return']
            converged = convergence_result.get('converged', False) if isinstance(convergence_result, dict) else False

            if cumulative_return >= min_threshold:
                # Pure review - no penalties
                audit_outcome = "satisfactory"
                adjustments = []
                bonus_review = await self._review_bonus_stack()
            else:
                # Escalation needed
                audit_outcome = "requires_adjustment"
                adjustments = self._generate_escalation_adjustments(cumulative_return, convergence_result)
                bonus_review = {"action": "no_changes", "reason": "Performance below threshold"}

            # Update bonus stack (fade old bonuses)
            self._fade_bonus_stack()

            audit_result = {
                'audit_type': audit_type,
                'period_analyzed': f"{len(recent_performance)} recent outcomes",
                'cumulative_return': cumulative_return,
                'threshold': min_threshold,
                'outcome': audit_outcome,
                'convergence_analysis': convergence_result,
                'adjustments': adjustments,
                'bonus_review': bonus_review,
                'recommendations': self._generate_audit_recommendations(audit_outcome, convergence_result),
                'next_audit_due': (now + timedelta(days=90)).isoformat()
            }

            # Log audit
            audit_entry = {
                'timestamp': now.isoformat(),
                'audit': audit_result
            }
            self.append_to_memory_list('quarterly_audits', audit_entry, max_items=50)
            self.update_memory('last_quarterly_audit', now.isoformat())

            logger.info(f"Quarterly audit complete: {audit_outcome}")
            return audit_result

        except Exception as e:
            logger.error(f"Error in quarterly audit: {e}")
            return {"error": f"Quarterly audit failed: {str(e)}"}

    async def _run_mini_loop(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run mini-loop validation with 3-5 iterations max.
        """
        max_iterations = 5
        warnings = []

        for i in range(max_iterations):
            # Check market conditions
            market_open = await self._check_market_conditions(proposal)
            if not market_open:
                return {'approved': False, 'reason': 'Market not open', 'iterations': i+1}

            # Check position feasibility
            feasible = await self._check_position_feasibility(proposal)
            if not feasible['feasible']:
                warnings.append(feasible['reason'])
                if i == max_iterations - 1:  # Last attempt
                    return {'approved': False, 'reason': 'Position not feasible', 'warnings': warnings, 'iterations': i+1}

            # Check for delusions (overconfidence)
            confidence_check = self._check_overconfidence(proposal)
            if confidence_check['overconfident']:
                warnings.append(confidence_check['warning'])
                # Reduce position size as correction
                proposal['quantity'] = int(proposal['quantity'] * 0.8)

            # If we get here, proposal looks good
            if i >= 2:  # Minimum 3 iterations for thorough check
                return {
                    'approved': True,
                    'iterations': i+1,
                    'warnings': warnings,
                    'corrections_applied': len(warnings) > 0
                }

        return {'approved': False, 'reason': 'Mini-loop max iterations reached', 'iterations': max_iterations}

    async def _check_market_conditions(self, proposal: Dict[str, Any]) -> bool:
        """
        Check if market conditions are suitable for execution.
        """
        # Stub: In real implementation, check exchange calendars, volatility, etc.
        symbol = proposal.get('symbol', '')
        current_hour = datetime.now().hour

        # Basic market hours check (9:30 AM - 4:00 PM ET)
        if 9 <= current_hour <= 16:
            return True
        else:
            return False

    async def _check_position_feasibility(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if proposed position is feasible.
        """
        quantity = proposal.get('quantity', 0)
        price = proposal.get('price', 0)
        symbol = proposal.get('symbol', '')

        # Basic feasibility checks
        if quantity <= 0:
            return {'feasible': False, 'reason': 'Invalid quantity'}

        if price <= 0 or price > 10000:  # Basic price sanity
            return {'feasible': False, 'reason': 'Invalid price'}

        if not symbol or len(symbol) > 10:
            return {'feasible': False, 'reason': 'Invalid symbol'}

        return {'feasible': True}

    def _check_overconfidence(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for overconfidence/delusions in proposal.
        """
        confidence = proposal.get('confidence', 0.5)

        if confidence > 0.95:
            return {
                'overconfident': True,
                'warning': f'High confidence ({confidence:.1%}) may indicate overconfidence'
            }

        return {'overconfident': False}

    def _award_bonus(self, symbol: str, amount: float, estimate: float):
        """
        Award POP bonus credit with metadata.
        """
        bonus_entry = {
            'symbol': symbol,
            'amount': amount,
            'estimate': estimate,
            'timestamp': datetime.now().isoformat(),
            'quarter_awarded': self._get_current_quarter(),
            'fade_rate': self.quarterly_thresholds['bonus_fade_rate']
        }

        if symbol not in self.memory['bonus_stack']:
            self.memory['bonus_stack'][symbol] = []

        self.memory['bonus_stack'][symbol].append(bonus_entry)
        self.save_memory()  # Save after bonus award

    def _fade_bonus_stack(self):
        """
        Apply quarterly fade to bonus stack.
        """
        fade_rate = self.quarterly_thresholds['bonus_fade_rate']

        for symbol, bonuses in self.memory['bonus_stack'].items():
            active_bonuses = []
            for bonus in bonuses:
                quarters_old = self._get_current_quarter() - bonus['quarter_awarded']
                if quarters_old > 0:
                    # Apply fade: weight -= max(0, fade_rate * quarters_old)
                    faded_amount = bonus['amount'] * max(0, 1 - (fade_rate * quarters_old))
                    bonus['amount'] = faded_amount

                # Keep bonuses with remaining value
                if bonus['amount'] > 0.1:  # Minimum threshold
                    active_bonuses.append(bonus)

            self.memory['bonus_stack'][symbol] = active_bonuses

        self.save_memory()  # Save after bonus fade

    def _get_current_quarter(self) -> int:
        """
        Get current quarter number.
        """
        now = datetime.now()
        return ((now.month - 1) // 3) + 1

    async def _review_bonus_stack(self) -> Dict[str, Any]:
        """
        Review and potentially adjust bonus stack.
        """
        total_active_bonuses = sum(
            sum(b['amount'] for b in bonuses)
            for bonuses in self.memory['bonus_stack'].values()
        )

        return {
            'total_active_bonuses': total_active_bonuses,
            'symbols_with_bonuses': list(self.memory['bonus_stack'].keys()),
            'action': 'maintained',
            'reason': 'Satisfactory performance - no bonus adjustments needed'
        }

    def _generate_insights(self, performance_entry: Dict[str, Any], bonus_awarded: bool) -> List[str]:
        """
        Generate insights from performance data.
        """
        insights = []

        pnl_pct = performance_entry['pnl_pct']
        sharpe = performance_entry['sharpe_ratio']
        symbol = performance_entry['symbol']

        if pnl_pct > 0.05:
            insights.append(f"Strong performance on {symbol}: +{pnl_pct:.1f}%")
        elif pnl_pct < -0.05:
            insights.append(f"Loss on {symbol}: {pnl_pct:.1f}% - review risk management")

        if sharpe > 1.5:
            insights.append(f"Good risk-adjusted returns (Sharpe: {sharpe:.2f})")
        elif sharpe < 1.0:
            insights.append(f"Poor risk-adjusted returns (Sharpe: {sharpe:.2f}) - consider position sizing")

        if bonus_awarded:
            insights.append("Bonus awarded for strong estimate accuracy")

        return insights

    def _generate_recommendations(self, performance_entry: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on performance.
        """
        recommendations = []

        pnl_pct = performance_entry['pnl_pct']
        max_drawdown = performance_entry['max_drawdown']

        if pnl_pct > 0.10:
            recommendations.append("Consider scaling up position sizes for similar setups")
        elif pnl_pct < -0.10:
            recommendations.append("Review entry/exit timing and reduce position sizes")

        if max_drawdown > 0.20:
            recommendations.append("Implement stricter stop-loss rules to limit drawdowns")

        if len(self.memory['performance_history']) > 10:
            recent_trades = self.memory['performance_history'][-5:]
            win_rate = sum(1 for t in recent_trades if t['pnl_pct'] > 0) / len(recent_trades)
            if win_rate > 0.7:
                recommendations.append("Strong recent performance - maintain current strategy")
            elif win_rate < 0.3:
                recommendations.append("Poor recent performance - reassess strategy parameters")

        return recommendations

    def _generate_escalation_adjustments(self, cumulative_return: float, convergence_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate escalation adjustments for poor performance.
        """
        adjustments = []

        deficit = self.quarterly_thresholds['min_cumulative_return'] - cumulative_return

        # Risk reduction
        adjustments.append({
            'type': 'risk_reduction',
            'parameter': 'max_position_size',
            'adjustment': -0.2,  # Reduce by 20%
            'reason': f'Performance {deficit:.1%} below threshold'
        })

        # Volatility adjustment
        if convergence_result.get('stability_coefficient', 0) > 0.3:
            adjustments.append({
                'type': 'volatility_filter',
                'parameter': 'max_volatility_threshold',
                'adjustment': -0.1,  # Tighten by 10%
                'reason': 'High volatility contributing to losses'
            })

        return adjustments

    def _generate_audit_recommendations(self, audit_outcome: str, convergence_result: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations from quarterly audit.
        """
        recommendations = []

        if audit_outcome == "satisfactory":
            recommendations.append("Continue current strategy with periodic monitoring")
            if convergence_result.get('converged', False):
                recommendations.append("Strategy has converged - focus on execution consistency")
        else:
            recommendations.append("Implement risk controls and position size reductions")
            recommendations.append("Review strategy parameters and market conditions")
            if not convergence_result.get('converged', True):
                recommendations.append("Strategy still evolving - allow more time or adjust parameters")

        return recommendations

    def reflect(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced reflection with trend analysis and escalation triggers.
        """
        adjustments = super().reflect(metrics)

        # Additional reflection logic for reflection agent
        delta = metrics.get('delta', 0)

        # Check for significant changes (>5% delta)
        if abs(delta) > 0.05:
            trend_analysis = self._analyze_performance_trend()

            if trend_analysis['trend'] == 'deteriorating':
                adjustments['escalation_triggered'] = True
                adjustments['escalation_reason'] = 'Performance deterioration detected'
                adjustments['recommended_action'] = 'Reduce risk exposure'

            elif trend_analysis['trend'] == 'improving':
                adjustments['positive_trend'] = True
                adjustments['recommended_action'] = 'Monitor for scaling opportunities'

        # Quarterly audit trigger check
        if self._should_trigger_quarterly_audit():
            adjustments['quarterly_audit_due'] = True
            adjustments['audit_reason'] = 'Performance milestone or time-based trigger'

        return adjustments

    def _analyze_performance_trend(self) -> Dict[str, Any]:
        """
        Analyze recent performance trends.
        """
        if len(self.memory['performance_history']) < 5:
            return {'trend': 'insufficient_data'}

        recent = self.memory['performance_history'][-10:]
        pnl_values = [p['pnl_pct'] for p in recent]

        # Simple trend analysis
        slope = np.polyfit(range(len(pnl_values)), pnl_values, 1)[0]

        if slope > 0.002:
            trend = 'improving'
        elif slope < -0.002:
            trend = 'deteriorating'
        else:
            trend = 'stable'

        return {
            'trend': trend,
            'slope': slope,
            'periods': len(recent),
            'avg_performance': np.mean(pnl_values)
        }

    def _extract_pyramiding_metrics(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract pyramiding-specific performance metrics from execution outcome.
        """
        pyramiding_data = outcome.get('pyramiding', {})

        if not pyramiding_data:
            return {}

        metrics = {
            'tiers_executed': pyramiding_data.get('tiers_executed', 0),
            'efficiency_score': pyramiding_data.get('efficiency_score', 0.0),
            'volatility_regime': pyramiding_data.get('volatility_regime', 'unknown'),
            'trend_regime': pyramiding_data.get('trend_regime', 'unknown'),
            'final_roi': pyramiding_data.get('final_roi', 0.0),
            'success': pyramiding_data.get('success', False),
            'exposure_utilization': pyramiding_data.get('exposure_utilization', 0.0)
        }

        return metrics

    def _should_trigger_quarterly_audit(self) -> bool:
        """
        Check if quarterly audit should be triggered.
        """
        now = datetime.now()

        # Time-based trigger (every 3 months)
        if self.memory['last_quarterly_audit']:
            last_audit = datetime.fromisoformat(self.memory['last_quarterly_audit'])
            if (now - last_audit).days >= 90:
                return True

        # Performance-based trigger (significant cumulative change)
        if len(self.memory['performance_history']) >= 20:
            recent = self.memory['performance_history'][-20:]
            cumulative = sum(p['pnl_pct'] for p in recent)

            if abs(cumulative) > 0.50:  # 50% cumulative change
                return True

        return False

    # ===== REAL-TIME AUDITING CAPABILITIES =====

    async def process_realtime_audit(self, system_metrics: Dict[str, Any], performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process real-time audit data for continuous system monitoring and validation.

        Args:
            system_metrics: Current system health metrics
            performance_data: Real-time performance data

        Returns:
            Dict with real-time audit results and alerts
        """
        logger.info("Processing real-time audit data for continuous monitoring")

        try:
            # Extract audit features from real-time data
            audit_features = self._extract_audit_features(system_metrics, performance_data)

            # Perform continuous validation checks
            validation_results = await self._perform_continuous_validation(audit_features)

            # Check for audit anomalies
            anomalies = self._detect_audit_anomalies(audit_features)

            # Assess real-time risk levels
            risk_assessment = self._assess_realtime_risk(audit_features, validation_results)

            # Generate audit alerts if needed
            alerts = self._generate_audit_alerts(anomalies, risk_assessment, validation_results)

            # Update audit metrics
            self._update_audit_metrics(audit_features, validation_results, anomalies)

            audit_result = {
                'timestamp': datetime.now().isoformat(),
                'audit_features': audit_features,
                'validation_results': validation_results,
                'anomalies_detected': len(anomalies),
                'anomalies': anomalies,
                'risk_assessment': risk_assessment,
                'alerts': alerts,
                'system_health_score': self._calculate_system_health_score(validation_results, anomalies),
                'audit_status': 'completed'
            }

            # Distribute critical alerts via A2A
            if alerts:
                await self._distribute_audit_alerts(alerts)

            # Log audit result
            self._log_audit_result(audit_result)

            return audit_result

        except Exception as e:
            logger.error(f"Error in real-time audit processing: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'audit_status': 'failed'
            }

    def _extract_audit_features(self, system_metrics: Dict[str, Any], performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract audit-relevant features from real-time data.

        Args:
            system_metrics: System health metrics
            performance_data: Performance data

        Returns:
            Dict of audit features
        """
        features = {}

        try:
            # System health features
            features.update({
                'cpu_usage': system_metrics.get('cpu_usage', 0),
                'memory_usage': system_metrics.get('memory_usage', 0),
                'api_response_time': system_metrics.get('api_response_time', 0),
                'error_rate': system_metrics.get('error_rate', 0),
                'uptime_hours': system_metrics.get('uptime_hours', 0)
            })

            # Performance features
            features.update({
                'current_pnl': performance_data.get('current_pnl', 0),
                'daily_pnl': performance_data.get('daily_pnl', 0),
                'win_rate': performance_data.get('win_rate', 0.5),
                'avg_trade_size': performance_data.get('avg_trade_size', 0),
                'active_positions': performance_data.get('active_positions', 0),
                'pending_orders': performance_data.get('pending_orders', 0)
            })

            # Risk metrics
            features.update({
                'portfolio_volatility': performance_data.get('portfolio_volatility', 0),
                'max_drawdown': performance_data.get('max_drawdown', 0),
                'var_95': performance_data.get('var_95', 0),
                'sharpe_ratio': performance_data.get('sharpe_ratio', 0),
                'concentration_risk': performance_data.get('concentration_risk', 0)
            })

            # Operational features
            features.update({
                'orders_per_minute': performance_data.get('orders_per_minute', 0),
                'execution_success_rate': performance_data.get('execution_success_rate', 1.0),
                'data_latency_ms': system_metrics.get('data_latency_ms', 0),
                'system_load': system_metrics.get('system_load', 0)
            })

            # Time-based features
            current_time = datetime.now()
            features.update({
                'hour_of_day': current_time.hour,
                'day_of_week': current_time.weekday(),
                'is_market_hours': 9 <= current_time.hour <= 16 and current_time.weekday() < 5
            })

        except Exception as e:
            logger.warning(f"Error extracting audit features: {e}")

        return features

    async def _perform_continuous_validation(self, audit_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform continuous validation checks on system health and performance.

        Args:
            audit_features: Extracted audit features

        Returns:
            Dict with validation results
        """
        validation_results = {
            'system_health': 'healthy',
            'performance_status': 'normal',
            'risk_level': 'acceptable',
            'operational_status': 'normal',
            'validation_checks': [],
            'warnings': [],
            'critical_issues': []
        }

        try:
            # System health validation
            system_checks = self._validate_system_health(audit_features)
            validation_results['validation_checks'].extend(system_checks['checks'])
            validation_results['warnings'].extend(system_checks['warnings'])
            validation_results['critical_issues'].extend(system_checks['critical'])

            if system_checks['critical']:
                validation_results['system_health'] = 'critical'
            elif system_checks['warnings']:
                validation_results['system_health'] = 'warning'

            # Performance validation
            performance_checks = self._validate_performance_metrics(audit_features)
            validation_results['validation_checks'].extend(performance_checks['checks'])

            if performance_checks.get('degraded_performance', False):
                validation_results['performance_status'] = 'degraded'

            # Risk validation
            risk_checks = self._validate_risk_metrics(audit_features)
            validation_results['validation_checks'].extend(risk_checks['checks'])

            if risk_checks.get('high_risk', False):
                validation_results['risk_level'] = 'high'
            elif risk_checks.get('elevated_risk', False):
                validation_results['risk_level'] = 'elevated'

            # Operational validation
            operational_checks = self._validate_operational_metrics(audit_features)
            validation_results['validation_checks'].extend(operational_checks['checks'])

            if operational_checks.get('operational_issues', False):
                validation_results['operational_status'] = 'issues'

        except Exception as e:
            logger.warning(f"Error in continuous validation: {e}")
            validation_results['validation_checks'].append({
                'check': 'validation_system',
                'status': 'error',
                'message': f'Validation failed: {str(e)}'
            })

        return validation_results

    def _validate_system_health(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate system health metrics including tool availability.

        Args:
            features: Audit features

        Returns:
            Dict with system health validation results
        """
        checks = []
        warnings = []
        critical = []

        # CPU usage check
        cpu_usage = features.get('cpu_usage', 0)
        if cpu_usage > 90:
            critical.append(f'Critical CPU usage: {cpu_usage}%')
            checks.append({'check': 'cpu_usage', 'status': 'critical', 'value': cpu_usage})
        elif cpu_usage > 75:
            warnings.append(f'High CPU usage: {cpu_usage}%')
            checks.append({'check': 'cpu_usage', 'status': 'warning', 'value': cpu_usage})
        else:
            checks.append({'check': 'cpu_usage', 'status': 'normal', 'value': cpu_usage})

        # Memory usage check
        memory_usage = features.get('memory_usage', 0)
        if memory_usage > 95:
            critical.append(f'Critical memory usage: {memory_usage}%')
            checks.append({'check': 'memory_usage', 'status': 'critical', 'value': memory_usage})
        elif memory_usage > 85:
            warnings.append(f'High memory usage: {memory_usage}%')
            checks.append({'check': 'memory_usage', 'status': 'warning', 'value': memory_usage})
        else:
            checks.append({'check': 'memory_usage', 'status': 'normal', 'value': memory_usage})

        # Error rate check
        error_rate = features.get('error_rate', 0)
        if error_rate > 0.1:  # 10% error rate
            critical.append(f'High error rate: {error_rate:.1%}')
            checks.append({'check': 'error_rate', 'status': 'critical', 'value': error_rate})
        elif error_rate > 0.05:  # 5% error rate
            warnings.append(f'Elevated error rate: {error_rate:.1%}')
            checks.append({'check': 'error_rate', 'status': 'warning', 'value': error_rate})
        else:
            checks.append({'check': 'error_rate', 'status': 'normal', 'value': error_rate})

        # API response time check
        response_time = features.get('api_response_time', 0)
        if response_time > 5000:  # 5 seconds
            critical.append(f'Very slow API response: {response_time}ms')
            checks.append({'check': 'api_response_time', 'status': 'critical', 'value': response_time})
        elif response_time > 2000:  # 2 seconds
            warnings.append(f'Slow API response: {response_time}ms')
            checks.append({'check': 'api_response_time', 'status': 'warning', 'value': response_time})
        else:
            checks.append({'check': 'api_response_time', 'status': 'normal', 'value': response_time})

        # Tool health check - NEW: Monitor critical tool availability
        tool_health = self._check_tool_health()
        if tool_health['critical_failures']:
            critical.extend([f"Critical tool failure: {tool}" for tool in tool_health['critical_failures']])
            checks.append({'check': 'tool_health', 'status': 'critical', 'value': tool_health})
        elif tool_health['warning_failures']:
            warnings.extend([f"Tool warning: {tool}" for tool in tool_health['warning_failures']])
            checks.append({'check': 'tool_health', 'status': 'warning', 'value': tool_health})
        else:
            checks.append({'check': 'tool_health', 'status': 'normal', 'value': tool_health})

        return {'checks': checks, 'warnings': warnings, 'critical': critical}

    def _check_tool_health(self) -> Dict[str, Any]:
        """
        Check health status of critical tools across agents.

        Returns:
            Dict with tool health status
        """
        tool_health = {
            'critical_failures': [],
            'warning_failures': [],
            'healthy_tools': [],
            'total_tools_checked': 0,
            'health_score': 100
        }

        try:
            # Define critical tools that must be available
            critical_tools = [
                'yfinance_data_tool',
                'sentiment_analysis_tool',
                'pyfolio_metrics_tool',
                'zipline_backtest_tool',
                'options_greeks_calc_tool',
                'flow_alpha_calc_tool',
                'qlib_ml_refine_tool'
            ]

            # Define important tools (warnings if unavailable)
            important_tools = [
                'news_data_tool',
                'economic_data_tool',
                'marketdataapp_api_tool',
                'twitter_sentiment_tool',
                'fundamental_data_tool',
                'microstructure_analysis_tool'
            ]

            # Check critical tools
            for tool_name in critical_tools:
                tool_health['total_tools_checked'] += 1
                if not self._is_tool_available(tool_name):
                    tool_health['critical_failures'].append(tool_name)
                    tool_health['health_score'] -= 15  # Reduce score for each critical failure

            # Check important tools
            for tool_name in important_tools:
                tool_health['total_tools_checked'] += 1
                if not self._is_tool_available(tool_name):
                    tool_health['warning_failures'].append(tool_name)
                    tool_health['health_score'] -= 5  # Reduce score for each warning

            # Ensure score doesn't go below 0
            tool_health['health_score'] = max(0, tool_health['health_score'])

            # Add healthy tools count
            total_problematic = len(tool_health['critical_failures']) + len(tool_health['warning_failures'])
            tool_health['healthy_tools_count'] = tool_health['total_tools_checked'] - total_problematic

        except Exception as e:
            logger.warning(f"Error checking tool health: {e}")
            tool_health['critical_failures'].append('tool_health_check_error')
            tool_health['health_score'] = 0

        return tool_health

    def _is_tool_available(self, tool_name: str) -> bool:
        """
        Check if a specific tool is available and functional.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if tool is available, False otherwise
        """
        try:
            # Import tools module to check availability
            from src.utils.tools import get_available_tools

            # Get list of available tools
            available_tools = get_available_tools()

            # Check if tool name is in available tools
            if tool_name in available_tools:
                # Additional check: try to instantiate the tool to ensure it's functional
                tool_class = getattr(available_tools[tool_name], '__class__', None)
                if tool_class and hasattr(tool_class, 'run'):
                    return True

            return False

        except Exception as e:
            logger.warning(f"Error checking availability of tool {tool_name}: {e}")
            return False

    def _initialize_real_time_auditing(self):
        """
        Initialize real-time auditing and performance monitoring system.
        """
        logger.info("Initializing real-time auditing system...")

        # Real-time auditing state
        self.real_time_auditing = {
            'active': True,
            'audit_interval_seconds': 60,  # Audit every minute
            'last_audit': None,
            'performance_thresholds': {
                'min_win_rate': 0.55,  # 55% minimum win rate
                'max_avg_loss': 0.03,  # 3% maximum average loss
                'min_sharpe_ratio': 1.5,  # Minimum Sharpe ratio
                'max_drawdown': 0.08,  # 8% maximum drawdown
                'min_profit_factor': 1.2  # Minimum profit factor
            },
            'strategy_adjustment_triggers': {
                'consecutive_losses': 5,  # Adjust after 5 consecutive losses
                'win_rate_drop': 0.1,  # Adjust if win rate drops by 10%
                'sharpe_decline': 0.5,  # Adjust if Sharpe drops by 0.5
                'drawdown_threshold': 0.05  # Adjust if drawdown exceeds 5%
            },
            'active_adjustments': [],
            'audit_history': [],
            'performance_metrics': {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'consecutive_wins': 0,
                'consecutive_losses': 0,
                'current_drawdown': 0.0,
                'peak_portfolio_value': 0.0
            }
        }

        # Automated strategy adjustment parameters
        self.strategy_adjustments = {
            'position_sizing': {
                'reduction_factor': 0.8,  # Reduce position size by 20%
                'min_size': 0.01,  # Minimum 1% position size
                'max_size': 0.10  # Maximum 10% position size
            },
            'risk_management': {
                'stop_loss_tightening': 0.9,  # Tighten stops by 10%
                'take_profit_adjustment': 1.1,  # Increase take profit targets by 10%
                'max_holding_period': 5  # Maximum 5 days holding
            },
            'entry_filters': {
                'volatility_filter': True,
                'momentum_filter': True,
                'correlation_filter': True,
                'liquidity_filter': True
            }
        }

        logger.info("Real-time auditing system initialized successfully")

    async def start_real_time_auditing(self):
        """
        Start the real-time auditing and performance monitoring loop.
        """
        if not hasattr(self, 'real_time_auditing'):
            self._initialize_real_time_auditing()

        logger.info("Starting real-time auditing loop...")

        while self.real_time_auditing['active']:
            try:
                await self._perform_performance_audit()
                await asyncio.sleep(self.real_time_auditing['audit_interval_seconds'])
            except Exception as e:
                logger.error(f"Error in real-time auditing: {e}")
                await asyncio.sleep(120)  # Wait longer on error

    async def _perform_performance_audit(self):
        """
        Perform comprehensive real-time performance audit and strategy assessment.
        """
        try:
            current_time = datetime.now()

            # Get current performance data
            performance_data = await self._get_current_performance_data()

            # Calculate real-time performance metrics
            audit_metrics = await self._calculate_audit_metrics(performance_data)

            # Check for strategy adjustment triggers
            adjustment_triggers = await self._check_adjustment_triggers(audit_metrics)

            # Update audit history
            self._update_audit_history(audit_metrics, current_time)

            # Execute automated adjustments if needed
            if adjustment_triggers:
                await self._execute_strategy_adjustments(adjustment_triggers, audit_metrics)

            # Log significant findings
            await self._log_audit_findings(audit_metrics, adjustment_triggers)

            # Update auditing timestamp
            self.real_time_auditing['last_audit'] = current_time

        except Exception as e:
            logger.error(f"Error performing performance audit: {e}")

    async def _get_current_performance_data(self) -> Dict[str, Any]:
        """
        Get current performance data from execution history, portfolio state, and shared memory.
        """
        try:
            # Try to get portfolio data from shared memory first
            portfolio_data = await self._get_portfolio_data_from_shared_memory()
            
            # Get recent trades from memory
            recent_trades = []
            if 'outcome_logs' in self.memory and self.memory['outcome_logs']:
                # Get last 100 trades for analysis
                recent_trades = self.memory['outcome_logs'][-100:]
            
            # Calculate actual performance metrics from trade history
            performance_metrics = self._calculate_actual_performance_metrics(recent_trades)
            
            # Get current positions from shared memory or trade history
            current_positions = await self._get_current_positions()
            
            # Combine all data
            performance_data = {
                'recent_trades': recent_trades,
                'total_trades': len(recent_trades),
                'time_period': 'recent_100_trades',
                'portfolio_value': portfolio_data.get('portfolio_value', performance_metrics.get('calculated_portfolio_value', 100000.0)),
                'cash_position': portfolio_data.get('cash_position', performance_metrics.get('calculated_cash_position', 20000.0)),
                'current_positions': current_positions,
                'performance_metrics': performance_metrics,
                'data_source': 'real_calculated' if recent_trades else 'estimated_defaults'
            }

            return performance_data

        except Exception as e:
            logger.error(f"Error getting performance data: {e}")
            # Return basic structure with defaults
            return {
                'recent_trades': [],
                'total_trades': 0,
                'time_period': 'no_data_available',
                'portfolio_value': 100000.0,  # Conservative default
                'cash_position': 20000.0,     # Conservative default
                'current_positions': {},
                'performance_metrics': {},
                'data_source': 'error_fallback'
            }

    async def _get_portfolio_data_from_shared_memory(self) -> Dict[str, Any]:
        """
        Get portfolio data from shared memory system.
        """
        try:
            if not self.shared_memory_coordinator:
                return {}
                
            # Try to get portfolio data from execution agent namespace
            portfolio_ns = self.shared_memory_coordinator.a2a_protocol.get_namespace("portfolio")
            if portfolio_ns:
                portfolio_value = await portfolio_ns.retrieve_shared_memory("current_value", "reflection")
                cash_position = await portfolio_ns.retrieve_shared_memory("cash_position", "reflection")
                
                return {
                    'portfolio_value': portfolio_value if portfolio_value is not None else None,
                    'cash_position': cash_position if cash_position is not None else None
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting portfolio data from shared memory: {e}")
            return {}

    def _calculate_actual_performance_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate actual performance metrics from trade history.
        """
        try:
            if not trades:
                return {
                    'calculated_portfolio_value': 100000.0,
                    'calculated_cash_position': 20000.0,
                    'total_pnl': 0.0,
                    'win_rate': 0.0,
                    'avg_trade_pnl': 0.0
                }
            
            # Calculate basic metrics from trades
            total_pnl = sum(trade.get('pnl', 0) for trade in trades if 'pnl' in trade)
            winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
            total_trades = len(trades)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Estimate current portfolio value based on trade P&L
            # This is a simplified calculation - in reality would need proper position tracking
            initial_capital = 100000.0  # Assumed starting capital
            calculated_portfolio_value = initial_capital + total_pnl
            
            # Estimate cash position (simplified)
            # In reality, this would track actual cash flows
            calculated_cash_position = max(20000.0, initial_capital * 0.2 - total_pnl * 0.1)
            
            return {
                'calculated_portfolio_value': max(0, calculated_portfolio_value),
                'calculated_cash_position': max(0, calculated_cash_position),
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'avg_trade_pnl': total_pnl / total_trades if total_trades > 0 else 0,
                'total_trades_analyzed': total_trades
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {
                'calculated_portfolio_value': 100000.0,
                'calculated_cash_position': 20000.0,
                'error': str(e)
            }

    async def _get_current_positions(self) -> Dict[str, Any]:
        """
        Get current positions from shared memory or trade history.
        """
        try:
            # Try to get positions from shared memory
            if self.shared_memory_coordinator:
                portfolio_ns = self.shared_memory_coordinator.a2a_protocol.get_namespace("portfolio")
                if portfolio_ns:
                    positions = await portfolio_ns.retrieve_shared_memory("current_positions", "reflection")
                    if positions:
                        return positions
            
            # Fallback: derive positions from recent trades
            # This is a simplified approach - real position tracking would be more complex
            positions = {}
            
            # Get recent trades to infer current positions
            if 'outcome_logs' in self.memory and self.memory['outcome_logs']:
                recent_trades = self.memory['outcome_logs'][-50:]  # Last 50 trades
                
                for trade in recent_trades:
                    symbol = trade.get('symbol')
                    if symbol:
                        if symbol not in positions:
                            positions[symbol] = {
                                'quantity': 0,
                                'avg_price': 0,
                                'current_value': 0,
                                'unrealized_pnl': 0
                            }
                        
                        # This is a very simplified position calculation
                        # In reality, would need proper position tracking with buys/sells
                        quantity = trade.get('quantity', 0)
                        price = trade.get('price', 0)
                        
                        if trade.get('side') == 'buy':
                            positions[symbol]['quantity'] += quantity
                            # Update average price (simplified)
                            positions[symbol]['avg_price'] = (
                                (positions[symbol]['avg_price'] * (positions[symbol]['quantity'] - quantity)) +
                                (price * quantity)
                            ) / positions[symbol]['quantity'] if positions[symbol]['quantity'] > 0 else price
                        elif trade.get('side') == 'sell':
                            positions[symbol]['quantity'] -= quantity
            
            # Clean up zero positions
            positions = {k: v for k, v in positions.items() if v['quantity'] != 0}
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting current positions: {e}")
            return {}

    async def _calculate_audit_metrics(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive audit metrics from performance data.
        """
        try:
            trades = performance_data.get('recent_trades', [])
            metrics = {}

            if not trades:
                return self._get_default_audit_metrics()

            # Basic trade metrics
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) < 0]

            metrics['total_trades'] = len(trades)
            metrics['winning_trades'] = len(winning_trades)
            metrics['losing_trades'] = len(losing_trades)
            metrics['win_rate'] = len(winning_trades) / len(trades) if trades else 0

            # P&L metrics
            pnl_values = [t.get('pnl', 0) for t in trades]
            metrics['total_pnl'] = sum(pnl_values)
            metrics['avg_win'] = sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
            metrics['avg_loss'] = sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
            metrics['largest_win'] = max(pnl_values) if pnl_values else 0
            metrics['largest_loss'] = min(pnl_values) if pnl_values else 0
            metrics['profit_factor'] = abs(sum(t.get('pnl', 0) for t in winning_trades) / sum(t.get('pnl', 0) for t in losing_trades)) if losing_trades and sum(t.get('pnl', 0) for t in losing_trades) != 0 else float('inf')

            # Risk metrics
            metrics['avg_trade_pnl'] = sum(pnl_values) / len(trades) if trades else 0
            metrics['pnl_std'] = np.std(pnl_values) if len(pnl_values) > 1 else 0
            metrics['sharpe_ratio'] = (metrics['avg_trade_pnl'] / metrics['pnl_std']) * np.sqrt(252) if metrics['pnl_std'] > 0 else 0

            # Consecutive trade analysis
            consecutive_wins, consecutive_losses = self._calculate_consecutive_trades(trades)
            metrics['max_consecutive_wins'] = consecutive_wins
            metrics['max_consecutive_losses'] = consecutive_losses

            # Current streak
            current_streak = self._calculate_current_streak(trades)
            metrics['current_streak_type'] = current_streak['type']
            metrics['current_streak_length'] = current_streak['length']

            # Drawdown analysis
            metrics['current_drawdown'] = self._calculate_current_drawdown(trades)
            metrics['max_drawdown'] = self._calculate_max_drawdown(trades)

            # Time-based metrics
            metrics['avg_holding_period'] = self._calculate_avg_holding_period(trades)
            metrics['best_day_pnl'] = self._calculate_best_day_pnl(trades)
            metrics['worst_day_pnl'] = self._calculate_worst_day_pnl(trades)

            return metrics

        except Exception as e:
            logger.error(f"Error calculating audit metrics: {e}")
            return self._get_default_audit_metrics()

    def _get_default_audit_metrics(self) -> Dict[str, Any]:
        """
        Get default audit metrics when no data is available.
        """
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'profit_factor': 1.0,
            'avg_trade_pnl': 0.0,
            'pnl_std': 0.0,
            'sharpe_ratio': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'current_streak_type': 'none',
            'current_streak_length': 0,
            'current_drawdown': 0.0,
            'max_drawdown': 0.0,
            'avg_holding_period': 0.0,
            'best_day_pnl': 0.0,
            'worst_day_pnl': 0.0
        }

    def _calculate_consecutive_trades(self, trades: List[Dict[str, Any]]) -> tuple:
        """
        Calculate maximum consecutive wins and losses.
        """
        if not trades:
            return 0, 0

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in trades:
            pnl = trade.get('pnl', 0)
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0

        return max_wins, max_losses

    def _calculate_current_streak(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate current winning/losing streak.
        """
        if not trades:
            return {'type': 'none', 'length': 0}

        streak_type = 'none'
        streak_length = 0

        # Check from most recent trade backwards
        for trade in reversed(trades):
            pnl = trade.get('pnl', 0)
            if pnl > 0:
                if streak_type == 'win' or streak_type == 'none':
                    streak_type = 'win'
                    streak_length += 1
                else:
                    break
            elif pnl < 0:
                if streak_type == 'loss' or streak_type == 'none':
                    streak_type = 'loss'
                    streak_length += 1
                else:
                    break
            else:
                break

        return {'type': streak_type, 'length': streak_length}

    def _calculate_current_drawdown(self, trades: List[Dict[str, Any]]) -> float:
        """
        Calculate current drawdown from peak.
        """
        if not trades:
            return 0.0

        # Calculate cumulative P&L
        cumulative_pnl = []
        running_total = 0
        for trade in trades:
            running_total += trade.get('pnl', 0)
            cumulative_pnl.append(running_total)

        if not cumulative_pnl:
            return 0.0

        # Find peak
        peak = max(cumulative_pnl)
        current = cumulative_pnl[-1]

        # Current drawdown
        drawdown = (peak - current) / (1 + peak) if peak > 0 else 0
        return max(0, drawdown)

    def _calculate_max_drawdown(self, trades: List[Dict[str, Any]]) -> float:
        """
        Calculate maximum drawdown over the period.
        """
        if not trades:
            return 0.0

        # Calculate cumulative P&L
        cumulative_pnl = []
        running_total = 0
        for trade in trades:
            running_total += trade.get('pnl', 0)
            cumulative_pnl.append(running_total)

        if not cumulative_pnl:
            return 0.0

        # Calculate drawdowns
        peak = cumulative_pnl[0]
        max_drawdown = 0

        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl
            drawdown = (peak - pnl) / (1 + peak) if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def _calculate_avg_holding_period(self, trades: List[Dict[str, Any]]) -> float:
        """
        Calculate average holding period in days.
        """
        if not trades:
            return 0.0

        holding_periods = []
        for trade in trades:
            entry_time = trade.get('entry_time')
            exit_time = trade.get('exit_time')

            if entry_time and exit_time:
                try:
                    if isinstance(entry_time, str):
                        entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                    if isinstance(exit_time, str):
                        exit_time = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))

                    holding_period = (exit_time - entry_time).total_seconds() / (24 * 3600)  # days
                    holding_periods.append(holding_period)
                except:
                    continue

        return sum(holding_periods) / len(holding_periods) if holding_periods else 0.0

    def _calculate_best_day_pnl(self, trades: List[Dict[str, Any]]) -> float:
        """
        Calculate best daily P&L.
        """
        if not trades:
            return 0.0

        # Group trades by date
        daily_pnl = {}
        for trade in trades:
            trade_date = trade.get('exit_time', '').split('T')[0] if trade.get('exit_time') else 'unknown'
            pnl = trade.get('pnl', 0)
            daily_pnl[trade_date] = daily_pnl.get(trade_date, 0) + pnl

        return max(daily_pnl.values()) if daily_pnl else 0.0

    def _calculate_worst_day_pnl(self, trades: List[Dict[str, Any]]) -> float:
        """
        Calculate worst daily P&L.
        """
        if not trades:
            return 0.0

        # Group trades by date
        daily_pnl = {}
        for trade in trades:
            trade_date = trade.get('exit_time', '').split('T')[0] if trade.get('exit_time') else 'unknown'
            pnl = trade.get('pnl', 0)
            daily_pnl[trade_date] = daily_pnl.get(trade_date, 0) + pnl

        return min(daily_pnl.values()) if daily_pnl else 0.0

    async def _check_adjustment_triggers(self, audit_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check for strategy adjustment triggers based on audit metrics.
        """
        try:
            triggers = []
            thresholds = self.real_time_auditing['strategy_adjustment_triggers']

            # Check consecutive losses trigger
            consecutive_losses = audit_metrics.get('max_consecutive_losses', 0)
            if consecutive_losses >= thresholds['consecutive_losses']:
                triggers.append({
                    'type': 'consecutive_losses',
                    'severity': 'high',
                    'reason': f"{consecutive_losses} consecutive losses (threshold: {thresholds['consecutive_losses']})",
                    'suggested_action': 'reduce_position_sizes'
                })

            # Check win rate drop
            win_rate = audit_metrics.get('win_rate', 0)
            min_win_rate = self.real_time_auditing['performance_thresholds']['min_win_rate']
            if win_rate < min_win_rate:
                triggers.append({
                    'type': 'win_rate_below_threshold',
                    'severity': 'medium',
                    'reason': f"Win rate {win_rate:.1%} below threshold {min_win_rate:.1%}",
                    'suggested_action': 'add_entry_filters'
                })

            # Check Sharpe ratio decline
            sharpe_ratio = audit_metrics.get('sharpe_ratio', 0)
            min_sharpe = self.real_time_auditing['performance_thresholds']['min_sharpe_ratio']
            if sharpe_ratio < min_sharpe:
                triggers.append({
                    'type': 'sharpe_ratio_below_threshold',
                    'severity': 'medium',
                    'reason': f"Sharpe ratio {sharpe_ratio:.2f} below threshold {min_sharpe:.2f}",
                    'suggested_action': 'tighten_stops'
                })

            # Check drawdown threshold
            current_drawdown = audit_metrics.get('current_drawdown', 0)
            drawdown_threshold = thresholds['drawdown_threshold']
            if current_drawdown > drawdown_threshold:
                triggers.append({
                    'type': 'drawdown_threshold_exceeded',
                    'severity': 'high',
                    'reason': f"Current drawdown {current_drawdown:.1%} exceeds threshold {drawdown_threshold:.1%}",
                    'suggested_action': 'reduce_risk_exposure'
                })

            # Check profit factor
            profit_factor = audit_metrics.get('profit_factor', 1.0)
            min_profit_factor = self.real_time_auditing['performance_thresholds']['min_profit_factor']
            if profit_factor < min_profit_factor:
                triggers.append({
                    'type': 'profit_factor_below_threshold',
                    'severity': 'medium',
                    'reason': f"Profit factor {profit_factor:.2f} below threshold {min_profit_factor:.2f}",
                    'suggested_action': 'adjust_entry_exit_logic'
                })

            return triggers

        except Exception as e:
            logger.error(f"Error checking adjustment triggers: {e}")
            return []

    async def _execute_strategy_adjustments(self, triggers: List[Dict[str, Any]], audit_metrics: Dict[str, Any]):
        """
        Execute automated strategy adjustments based on triggers.
        """
        try:
            for trigger in triggers:
                action = trigger.get('suggested_action')
                severity = trigger.get('severity')

                # Skip if adjustment already active
                active_adjustments = [adj['action'] for adj in self.real_time_auditing['active_adjustments']]
                if action in active_adjustments:
                    continue

                # Execute adjustment based on action type
                if action == 'reduce_position_sizes':
                    await self._adjust_position_sizes(severity or 'medium')
                elif action == 'add_entry_filters':
                    await self._add_entry_filters(severity or 'medium')
                elif action == 'tighten_stops':
                    await self._tighten_stop_losses(severity or 'medium')
                elif action == 'reduce_risk_exposure':
                    await self._reduce_risk_exposure(severity or 'medium')
                elif action == 'adjust_entry_exit_logic':
                    await self._adjust_entry_exit_logic(severity or 'medium')

                # Record the adjustment
                adjustment_record = {
                    'timestamp': datetime.now(),
                    'trigger': trigger,
                    'action': action,
                    'severity': severity,
                    'audit_metrics': audit_metrics.copy()
                }

                self.real_time_auditing['active_adjustments'].append(adjustment_record)

                logger.info(f"Executed strategy adjustment: {action} (trigger: {trigger['type']})")

        except Exception as e:
            logger.error(f"Error executing strategy adjustments: {e}")

    async def _adjust_position_sizes(self, severity: str):
        """
        Adjust position sizes based on performance.
        """
        try:
            adjustment_factor = self.strategy_adjustments['position_sizing']['reduction_factor']
            if severity == 'high':
                adjustment_factor *= 0.8  # Additional 20% reduction for high severity

            # In production, this would communicate with strategy agent to adjust sizing
            logger.info(f"Adjusting position sizes by factor: {adjustment_factor}")

        except Exception as e:
            logger.error(f"Error adjusting position sizes: {e}")

    async def _add_entry_filters(self, severity: str):
        """
        Add additional entry filters to improve trade quality.
        """
        try:
            # Enable additional filters based on severity
            if severity == 'high':
                self.strategy_adjustments['entry_filters']['volatility_filter'] = True
                self.strategy_adjustments['entry_filters']['correlation_filter'] = True

            logger.info(f"Enhanced entry filters for {severity} severity trigger")

        except Exception as e:
            logger.error(f"Error adding entry filters: {e}")

    async def _tighten_stop_losses(self, severity: str):
        """
        Tighten stop loss levels.
        """
        try:
            tightening_factor = self.strategy_adjustments['risk_management']['stop_loss_tightening']
            if severity == 'high':
                tightening_factor *= 0.9  # Additional tightening for high severity

            # In production, this would adjust stop loss parameters
            logger.info(f"Tightening stop losses by factor: {tightening_factor}")

        except Exception as e:
            logger.error(f"Error tightening stop losses: {e}")

    async def _reduce_risk_exposure(self, severity: str):
        """
        Reduce overall risk exposure.
        """
        try:
            # Multiple risk reduction measures
            await self._adjust_position_sizes(severity)
            await self._tighten_stop_losses(severity)

            # Reduce maximum holding period
            if severity == 'high':
                self.strategy_adjustments['risk_management']['max_holding_period'] = 3  # Reduce to 3 days

            logger.info(f"Reduced risk exposure for {severity} severity trigger")

        except Exception as e:
            logger.error(f"Error reducing risk exposure: {e}")

    async def _adjust_entry_exit_logic(self, severity: str):
        """
        Adjust entry and exit logic parameters.
        """
        try:
            # Adjust take profit targets
            tp_adjustment = self.strategy_adjustments['risk_management']['take_profit_adjustment']
            if severity == 'high':
                tp_adjustment = 0.9  # Reduce take profit targets for high severity

            logger.info(f"Adjusted entry/exit logic for {severity} severity trigger")

        except Exception as e:
            logger.error(f"Error adjusting entry/exit logic: {e}")

    def _update_audit_history(self, audit_metrics: Dict[str, Any], timestamp):
        """
        Update audit history for trend analysis.
        """
        try:
            history_entry = {
                'timestamp': timestamp,
                'metrics': audit_metrics.copy()
            }

            self.real_time_auditing['audit_history'].append(history_entry)

            # Keep only last 500 entries to prevent memory issues
            if len(self.real_time_auditing['audit_history']) > 500:
                self.real_time_auditing['audit_history'] = self.real_time_auditing['audit_history'][-500:]

        except Exception as e:
            logger.error(f"Error updating audit history: {e}")

    async def _log_audit_findings(self, audit_metrics: Dict[str, Any], triggers: List[Dict[str, Any]]):
        """
        Log significant audit findings and triggers.
        """
        try:
            # Log key metrics
            win_rate = audit_metrics.get('win_rate', 0)
            sharpe_ratio = audit_metrics.get('sharpe_ratio', 0)
            current_drawdown = audit_metrics.get('current_drawdown', 0)
            total_trades = audit_metrics.get('total_trades', 0)

            logger.info(f"Audit Summary - Trades: {total_trades}, Win Rate: {win_rate:.1%}, Sharpe: {sharpe_ratio:.2f}, Drawdown: {current_drawdown:.1%}")

            # Log triggers
            if triggers:
                for trigger in triggers:
                    logger.warning(f"Audit Trigger: {trigger['type']} - {trigger['reason']}")
            else:
                logger.info("No adjustment triggers detected")

        except Exception as e:
            logger.error(f"Error logging audit findings: {e}")

    async def get_audit_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive audit dashboard data for monitoring and reporting.
        """
        try:
            # Get latest audit metrics
            performance_data = await self._get_current_performance_data()
            audit_metrics = await self._calculate_audit_metrics(performance_data)

            dashboard_data = {
                'timestamp': datetime.now(),
                'audit_metrics': audit_metrics,
                'active_adjustments': self.real_time_auditing.get('active_adjustments', []),
                'recent_triggers': [],  # Would populate from recent triggers
                'performance_history': self.real_time_auditing.get('audit_history', [])[-20:],  # Last 20 audits
                'strategy_settings': self.strategy_adjustments.copy(),
                'thresholds': self.real_time_auditing.get('performance_thresholds', {})
            }

            return dashboard_data

        except Exception as e:
            logger.error(f"Error getting audit dashboard data: {e}")
            return {}

    def _detect_audit_anomalies(self, audit_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in audit features and performance metrics.

        Args:
            audit_features: Audit features and metrics

        Returns:
            List of detected anomalies
        """
        anomalies = []

        try:
            # Performance anomalies
            win_rate = audit_features.get('win_rate', 0)
            if win_rate < 0.3:  # Below 30% win rate
                anomalies.append({
                    'type': 'performance',
                    'severity': 'high',
                    'metric': 'win_rate',
                    'value': win_rate,
                    'threshold': 0.3,
                    'description': f'Win rate {win_rate:.1%} below acceptable threshold'
                })

            # Risk anomalies
            current_drawdown = audit_features.get('current_drawdown', 0)
            if current_drawdown > 0.15:  # Above 15% drawdown
                anomalies.append({
                    'type': 'risk',
                    'severity': 'high',
                    'metric': 'drawdown',
                    'value': current_drawdown,
                    'threshold': 0.15,
                    'description': f'Current drawdown {current_drawdown:.1%} exceeds threshold'
                })

            # Operational anomalies
            error_rate = audit_features.get('error_rate', 0)
            if error_rate > 0.05:  # Above 5% error rate
                anomalies.append({
                    'type': 'operational',
                    'severity': 'medium',
                    'metric': 'error_rate',
                    'value': error_rate,
                    'threshold': 0.05,
                    'description': f'Error rate {error_rate:.1%} above acceptable level'
                })

        except Exception as e:
            logger.warning(f"Error detecting audit anomalies: {e}")

        return anomalies

    def _assess_realtime_risk(self, audit_features: Dict[str, Any], validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess real-time risk based on audit features and validation results.

        Args:
            audit_features: Audit features and metrics
            validation_results: Validation results

        Returns:
            Risk assessment dictionary
        """
        risk_assessment = {
            'overall_risk_level': 'low',
            'risk_factors': [],
            'recommendations': []
        }

        try:
            # Assess drawdown risk
            current_drawdown = audit_features.get('current_drawdown', 0)
            if current_drawdown > 0.20:
                risk_assessment['risk_factors'].append('extreme_drawdown')
                risk_assessment['overall_risk_level'] = 'critical'
                risk_assessment['recommendations'].append('Immediate position reduction required')
            elif current_drawdown > 0.10:
                risk_assessment['risk_factors'].append('high_drawdown')
                risk_assessment['overall_risk_level'] = 'high'
                risk_assessment['recommendations'].append('Consider reducing position sizes')

            # Assess volatility risk
            volatility = audit_features.get('volatility', 0)
            if volatility > 0.30:  # 30% volatility
                risk_assessment['risk_factors'].append('high_volatility')
                if risk_assessment['overall_risk_level'] == 'low':
                    risk_assessment['overall_risk_level'] = 'medium'
                risk_assessment['recommendations'].append('Implement tighter stop losses')

            # Assess performance risk
            win_rate = audit_features.get('win_rate', 0)
            if win_rate < 0.25:
                risk_assessment['risk_factors'].append('poor_performance')
                if risk_assessment['overall_risk_level'] == 'low':
                    risk_assessment['overall_risk_level'] = 'medium'
                risk_assessment['recommendations'].append('Review and adjust trading strategy')

        except Exception as e:
            logger.warning(f"Error assessing real-time risk: {e}")
            risk_assessment['overall_risk_level'] = 'unknown'

        return risk_assessment

    def _generate_audit_alerts(self, anomalies: List[Dict[str, Any]], risk_assessment: Dict[str, Any], validation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate audit alerts based on anomalies, risk assessment, and validation results.

        Args:
            anomalies: Detected anomalies
            risk_assessment: Risk assessment results
            validation_results: Validation results

        Returns:
            List of audit alerts
        """
        alerts = []

        try:
            # Generate alerts from anomalies
            for anomaly in anomalies:
                alerts.append({
                    'type': 'anomaly',
                    'severity': anomaly['severity'],
                    'category': anomaly['type'],
                    'message': anomaly['description'],
                    'timestamp': datetime.now(),
                    'data': anomaly
                })

            # Generate alerts from risk assessment
            if risk_assessment['overall_risk_level'] in ['high', 'critical']:
                alerts.append({
                    'type': 'risk',
                    'severity': risk_assessment['overall_risk_level'],
                    'category': 'risk_management',
                    'message': f"Risk level elevated to {risk_assessment['overall_risk_level']}",
                    'timestamp': datetime.now(),
                    'data': risk_assessment
                })

            # Generate alerts from validation failures
            validation_checks = validation_results.get('validation_checks', [])
            for check in validation_checks:
                if check.get('status') in ['error', 'critical']:
                    alerts.append({
                        'type': 'validation',
                        'severity': 'medium',
                        'category': 'system_health',
                        'message': f"Validation check failed: {check.get('check', 'unknown')}",
                        'timestamp': datetime.now(),
                        'data': check
                    })

        except Exception as e:
            logger.warning(f"Error generating audit alerts: {e}")

        return alerts

    def _update_audit_metrics(self, audit_features: Dict[str, Any], validation_results: Dict[str, Any], anomalies: List[Dict[str, Any]]):
        """
        Update audit metrics with new data and findings.

        Args:
            audit_features: Audit features and metrics
            validation_results: Validation results
            anomalies: Detected anomalies
        """
        try:
            # Update performance metrics
            self.audit_metrics.update({
                'last_audit_time': datetime.now(),
                'total_audits': self.audit_metrics.get('total_audits', 0) + 1,
                'anomalies_detected': len(anomalies),
                'validation_checks': len(validation_results.get('validation_checks', [])),
                'current_risk_level': validation_results.get('risk_level', 'unknown')
            })

            # Update rolling averages
            if 'performance_history' not in self.audit_metrics:
                self.audit_metrics['performance_history'] = []

            self.audit_metrics['performance_history'].append({
                'timestamp': datetime.now(),
                'win_rate': audit_features.get('win_rate', 0),
                'drawdown': audit_features.get('current_drawdown', 0),
                'sharpe_ratio': audit_features.get('sharpe_ratio', 0)
            })

            # Keep only last 100 entries
            if len(self.audit_metrics['performance_history']) > 100:
                self.audit_metrics['performance_history'] = self.audit_metrics['performance_history'][-100:]

        except Exception as e:
            logger.error(f"Error updating audit metrics: {e}")

    def _calculate_system_health_score(self, validation_results: Dict[str, Any], anomalies: List[Dict[str, Any]]) -> float:
        """
        Calculate overall system health score based on validation and anomalies.

        Args:
            validation_results: Validation results
            anomalies: Detected anomalies

        Returns:
            Health score between 0-100
        """
        base_score = 100.0

        try:
            # Deduct points for validation failures
            validation_checks = validation_results.get('validation_checks', [])
            error_checks = [c for c in validation_checks if c.get('status') in ['error', 'critical']]
            base_score -= len(error_checks) * 5  # 5 points per error

            warning_checks = [c for c in validation_checks if c.get('status') == 'warning']
            base_score -= len(warning_checks) * 2  # 2 points per warning

            # Deduct points for anomalies
            high_severity = [a for a in anomalies if a.get('severity') == 'high']
            base_score -= len(high_severity) * 10  # 10 points per high severity anomaly

            medium_severity = [a for a in anomalies if a.get('severity') == 'medium']
            base_score -= len(medium_severity) * 5  # 5 points per medium severity anomaly

            # Ensure score doesn't go below 0
            base_score = max(0.0, base_score)

        except Exception as e:
            logger.warning(f"Error calculating system health score: {e}")
            base_score = 50.0  # Default to 50 if calculation fails

        return base_score

    async def _distribute_audit_alerts(self, alerts: List[Dict[str, Any]]):
        """
        Distribute audit alerts to relevant agents and systems.

        Args:
            alerts: List of audit alerts to distribute
        """
        try:
            # Filter critical alerts
            critical_alerts = [a for a in alerts if a.get('severity') in ['high', 'critical']]

            if critical_alerts:
                # Send to risk management agent
                await self._send_alert_to_agent('risk_management_agent', critical_alerts)

                # Send to portfolio management agent
                await self._send_alert_to_agent('portfolio_management_agent', critical_alerts)

            # Send all alerts to monitoring system
            await self._send_alert_to_monitoring(alerts)

        except Exception as e:
            logger.error(f"Error distributing audit alerts: {e}")

    async def _send_alert_to_agent(self, agent_name: str, alerts: List[Dict[str, Any]]):
        """
        Send alerts to a specific agent.

        Args:
            agent_name: Name of the target agent
            alerts: Alerts to send
        """
        try:
            # This would integrate with the A2A communication system
            alert_message = {
                'type': 'audit_alert',
                'alerts': alerts,
                'source': 'reflection_agent',
                'timestamp': datetime.now()
            }

            # Placeholder for actual agent communication
            logger.info(f"Sending {len(alerts)} alerts to {agent_name}")

        except Exception as e:
            logger.warning(f"Error sending alerts to {agent_name}: {e}")

    async def _send_alert_to_monitoring(self, alerts: List[Dict[str, Any]]):
        """
        Send alerts to the monitoring system.

        Args:
            alerts: Alerts to send
        """
        try:
            # Log alerts for monitoring
            for alert in alerts:
                logger.warning(f"AUDIT ALERT: {alert['type']} - {alert['message']}")

        except Exception as e:
            logger.warning(f"Error sending alerts to monitoring: {e}")

    def _log_audit_result(self, audit_result: Dict[str, Any]):
        """
        Log audit results for monitoring and analysis.

        Args:
            audit_result: Complete audit result to log
        """
        try:
            logger.info(f"Audit completed - Health Score: {audit_result.get('system_health_score', 'N/A')}, "
                       f"Risk Level: {audit_result.get('risk_level', 'N/A')}, "
                       f"Anomalies: {len(audit_result.get('anomalies', []))}")

            # Log detailed results if there are issues
            if audit_result.get('system_health_score', 100) < 80:
                logger.warning(f"Low health score detected: {audit_result['system_health_score']}")

        except Exception as e:
            logger.error(f"Error logging audit result: {e}")

    def _validate_performance_metrics(self, audit_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate performance metrics against thresholds.

        Args:
            audit_features: Audit features containing performance metrics

        Returns:
            Performance validation results
        """
        checks = []
        high_risk = False

        try:
            # Win rate validation
            win_rate = audit_features.get('win_rate', 0)
            if win_rate < 0.25:
                checks.append({
                    'check': 'win_rate',
                    'status': 'critical',
                    'message': f'Win rate {win_rate:.1%} below 25% threshold'
                })
                high_risk = True
            elif win_rate < 0.35:
                checks.append({
                    'check': 'win_rate',
                    'status': 'warning',
                    'message': f'Win rate {win_rate:.1%} below 35% threshold'
                })

            # Sharpe ratio validation
            sharpe_ratio = audit_features.get('sharpe_ratio', 0)
            if sharpe_ratio < 0.5:
                checks.append({
                    'check': 'sharpe_ratio',
                    'status': 'warning',
                    'message': f'Sharpe ratio {sharpe_ratio:.2f} below 0.5 threshold'
                })

            # Total return validation
            total_return = audit_features.get('total_return', 0)
            if total_return < -0.10:  # More than 10% loss
                checks.append({
                    'check': 'total_return',
                    'status': 'critical',
                    'message': f'Total return {total_return:.1%} shows significant losses'
                })
                high_risk = True

        except Exception as e:
            logger.warning(f"Error validating performance metrics: {e}")
            checks.append({
                'check': 'performance_validation',
                'status': 'error',
                'message': f'Performance validation failed: {str(e)}'
            })

        return {'checks': checks, 'high_risk': high_risk}

    def _validate_risk_metrics(self, audit_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate risk metrics against thresholds.

        Args:
            audit_features: Audit features containing risk metrics

        Returns:
            Risk validation results
        """
        checks = []
        high_risk = False
        elevated_risk = False

        try:
            # Drawdown validation
            current_drawdown = audit_features.get('current_drawdown', 0)
            if current_drawdown > 0.25:
                checks.append({
                    'check': 'current_drawdown',
                    'status': 'critical',
                    'message': f'Current drawdown {current_drawdown:.1%} exceeds 25% threshold'
                })
                high_risk = True
            elif current_drawdown > 0.15:
                checks.append({
                    'check': 'current_drawdown',
                    'status': 'warning',
                    'message': f'Current drawdown {current_drawdown:.1%} exceeds 15% threshold'
                })
                elevated_risk = True

            # Volatility validation
            volatility = audit_features.get('volatility', 0)
            if volatility > 0.40:
                checks.append({
                    'check': 'volatility',
                    'status': 'critical',
                    'message': f'Volatility {volatility:.1%} exceeds 40% threshold'
                })
                high_risk = True
            elif volatility > 0.25:
                checks.append({
                    'check': 'volatility',
                    'status': 'warning',
                    'message': f'Volatility {volatility:.1%} exceeds 25% threshold'
                })
                elevated_risk = True

            # Value at Risk validation
            var_95 = audit_features.get('var_95', 0)
            if var_95 > 0.15:  # 15% VaR at 95% confidence
                checks.append({
                    'check': 'value_at_risk',
                    'status': 'warning',
                    'message': f'VaR (95%) {var_95:.1%} exceeds 15% threshold'
                })
                elevated_risk = True

        except Exception as e:
            logger.warning(f"Error validating risk metrics: {e}")
            checks.append({
                'check': 'risk_validation',
                'status': 'error',
                'message': f'Risk validation failed: {str(e)}'
            })

        return {'checks': checks, 'high_risk': high_risk, 'elevated_risk': elevated_risk}

    def _validate_operational_metrics(self, audit_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate operational metrics against thresholds.

        Args:
            audit_features: Audit features containing operational metrics

        Returns:
            Operational validation results
        """
        checks = []
        operational_issues = False

        try:
            # Error rate validation
            error_rate = audit_features.get('error_rate', 0)
            if error_rate > 0.10:
                checks.append({
                    'check': 'error_rate',
                    'status': 'critical',
                    'message': f'Error rate {error_rate:.1%} exceeds 10% threshold'
                })
                operational_issues = True
            elif error_rate > 0.05:
                checks.append({
                    'check': 'error_rate',
                    'status': 'warning',
                    'message': f'Error rate {error_rate:.1%} exceeds 5% threshold'
                })
                operational_issues = True

            # Response time validation
            response_time = audit_features.get('api_response_time', 0)
            if response_time > 10000:  # 10 seconds
                checks.append({
                    'check': 'response_time',
                    'status': 'critical',
                    'message': f'API response time {response_time}ms exceeds 10s threshold'
                })
                operational_issues = True
            elif response_time > 5000:  # 5 seconds
                checks.append({
                    'check': 'response_time',
                    'status': 'warning',
                    'message': f'API response time {response_time}ms exceeds 5s threshold'
                })
                operational_issues = True

            # CPU usage validation
            cpu_usage = audit_features.get('cpu_usage', 0)
            if cpu_usage > 95:
                checks.append({
                    'check': 'cpu_usage',
                    'status': 'critical',
                    'message': f'CPU usage {cpu_usage}% exceeds 95% threshold'
                })
                operational_issues = True
            elif cpu_usage > 85:
                checks.append({
                    'check': 'cpu_usage',
                    'status': 'warning',
                    'message': f'CPU usage {cpu_usage}% exceeds 85% threshold'
                })
                operational_issues = True

            # Memory usage validation
            memory_usage = audit_features.get('memory_usage', 0)
            if memory_usage > 95:
                checks.append({
                    'check': 'memory_usage',
                    'status': 'critical',
                    'message': f'Memory usage {memory_usage}% exceeds 95% threshold'
                })
                operational_issues = True
            elif memory_usage > 85:
                checks.append({
                    'check': 'memory_usage',
                    'status': 'warning',
                    'message': f'Memory usage {memory_usage}% exceeds 85% threshold'
                })
                operational_issues = True

        except Exception as e:
            logger.warning(f"Error validating operational metrics: {e}")
            checks.append({
                'check': 'operational_validation',
                'status': 'error',
                'message': f'Operational validation failed: {str(e)}'
            })

        return {'checks': checks, 'operational_issues': operational_issues}