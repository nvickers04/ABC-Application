#!/usr/bin/env python3
"""
Unified Workflow Orchestrator - Paper Trading Ready
Consolidates Live Workflow, Continuous Trading, and 24/6 operations into
one bullet-proof system with multiple operating modes.

Supports paper trading without Discord integration.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

# Import core components
from src.agents.base import BaseAgent
from src.agents.data import DataAgent
from src.agents.strategy import StrategyAgent
from src.agents.risk import RiskAgent
from src.agents.execution import ExecutionAgent
from src.agents.reflection import ReflectionAgent
from src.agents.learning import LearningAgent
from src.agents.macro import MacroAgent
from src.agents.memory import MemoryAgent

# Import utilities
from src.utils.a2a_protocol import A2AProtocol
from src.utils.logging_config import get_logger
from src.utils.redis_cache import RedisCacheManager
from src.utils.alert_manager import get_alert_manager
from src.utils.shared_memory import get_multi_agent_coordinator

# Setup logging
logger = get_logger(__name__)

class WorkflowMode(Enum):
    """Operating modes for the unified workflow orchestrator."""
    ANALYSIS = "analysis"      # Full collaborative analysis with human intervention
    EXECUTION = "execution"    # Automated trading execution only
    HYBRID = "hybrid"         # Analysis + automated execution with oversight
    BACKTEST = "backtest"      # Historical simulation and validation

class UnifiedWorkflowOrchestrator:
    """
    Unified Workflow Orchestrator - The single source of truth for all trading operations.

    This consolidates Live Workflow, Continuous Trading, and 24/6 operations into
    one bullet-proof system with multiple operating modes.
    """

    def __init__(self,
                 mode: WorkflowMode = WorkflowMode.HYBRID,
                 enable_discord: bool = True,
                 enable_health_monitoring: bool = True,
                 symbols: Optional[List[str]] = None):
        """
        Initialize the unified workflow orchestrator.

        Args:
            mode: Operating mode (ANALYSIS, EXECUTION, HYBRID, BACKTEST)
            enable_discord: Whether to enable Discord integration
            enable_health_monitoring: Whether to enable health monitoring
            symbols: List of symbols to trade (default: ['SPY'])
        """
        self.mode = mode
        self.enable_discord = enable_discord
        self.enable_health_monitoring = enable_health_monitoring
        self.symbols = symbols or ['SPY']

        # Core components
        self.a2a_protocol = None
        self.agents = {}
        self.redis_cache = None
        self.alert_manager = None
        self.collaborative_coordinator = None

        # State tracking
        self.initialized = False
        self.running = False
        self.active_sessions = {}  # Track active collaborative sessions

        logger.info(f"Initialized UnifiedWorkflowOrchestrator in {mode.value} mode")

    async def initialize(self) -> bool:
        """
        Initialize all components and agents.

        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("🚀 Initializing Unified Workflow Orchestrator...")

            # Initialize Redis cache (no fallbacks)
            self.redis_cache = RedisCacheManager()
            # RedisCacheManager initializes automatically in __init__
            logger.info("✅ Redis cache initialized")

            # Initialize alert manager
            self.alert_manager = get_alert_manager()
            logger.info("✅ Alert manager initialized")

            # Initialize A2A protocol
            self.a2a_protocol = A2AProtocol()
            logger.info("✅ A2A protocol initialized")

            # Initialize collaborative coordinator
            self.collaborative_coordinator = get_multi_agent_coordinator()
            logger.info("✅ Collaborative coordinator initialized")

            # Initialize agents
            await self._initialize_agents()
            logger.info("✅ All agents initialized")

            # Health monitoring (if enabled)
            if self.enable_health_monitoring:
                await self._start_health_monitoring()
                logger.info("✅ Health monitoring started")

            self.initialized = True
            logger.info("🎯 Unified Workflow Orchestrator initialization complete!")
            return True

        except Exception as e:
            logger.error(f"❌ Orchestrator initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _initialize_agents(self):
        """Initialize all trading agents."""
        # Create agents
        self.agents['macro'] = MacroAgent(a2a_protocol=self.a2a_protocol)
        self.agents['data'] = DataAgent(a2a_protocol=self.a2a_protocol)
        self.agents['strategy'] = StrategyAgent(a2a_protocol=self.a2a_protocol)
        self.agents['risk'] = RiskAgent(a2a_protocol=self.a2a_protocol)
        self.agents['execution'] = ExecutionAgent(a2a_protocol=self.a2a_protocol)
        self.agents['reflection'] = ReflectionAgent(a2a_protocol=self.a2a_protocol)
        self.agents['learning'] = LearningAgent(a2a_protocol=self.a2a_protocol)
        self.agents['memory'] = MemoryAgent(a2a_protocol=self.a2a_protocol)

        # Register agents with A2A protocol
        for name, agent in self.agents.items():
            self.a2a_protocol.register_agent(name, agent)
            logger.info(f"📋 Registered agent: {name}")

        # Initialize each agent
        for name, agent in self.agents.items():
            try:
                if hasattr(agent, 'initialize'):
                    await agent.initialize()
                logger.info(f"✅ Agent {name} initialized")
            except Exception as e:
                logger.warning(f"⚠️ Agent {name} initialization failed: {e}")

    async def _start_health_monitoring(self):
        """Start health monitoring systems."""
        try:
            from src.utils.api_health_monitor import start_health_monitoring
            start_health_monitoring(check_interval=300)  # Check every 5 minutes
            logger.info("🏥 API health monitoring started")
        except Exception as e:
            logger.warning(f"⚠️ Health monitoring failed to start: {e}")

    async def start(self):
        """Start the orchestrator workflow."""
        if not self.initialized:
            logger.error("❌ Cannot start: orchestrator not initialized")
            return

        logger.info(f"🎯 Starting {self.mode.value} mode workflow...")
        self.running = True

        try:
            if self.mode == WorkflowMode.HYBRID:
                await self._run_hybrid_workflow()
            elif self.mode == WorkflowMode.ANALYSIS:
                await self._run_analysis_workflow()
            elif self.mode == WorkflowMode.EXECUTION:
                await self._run_execution_workflow()
            elif self.mode == WorkflowMode.BACKTEST:
                await self._run_backtest_workflow()

        except Exception as e:
            logger.error(f"❌ Workflow execution failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False

    async def _run_hybrid_workflow(self):
        """Run hybrid analysis + execution workflow using A2A orchestration."""
        logger.info("🔄 Starting hybrid workflow (analysis + execution)")

        assert self.a2a_protocol is not None, "A2A protocol not initialized"

        while self.running:
            try:
                # Check if collaborative analysis is needed for complex scenarios
                collaborative_needed = await self._should_run_collaborative_analysis()

                if collaborative_needed:
                    logger.info("🎯 Complex analysis scenario detected - running collaborative session")
                    collab_result = await self.run_collaborative_analysis(
                        topic="Complex Market Analysis",
                        context={'symbols': self.symbols, 'market_conditions': collaborative_needed}
                    )

                    # Use collaborative insights in the main orchestration
                    enhanced_data = {
                        'symbols': self.symbols,
                        'mode': 'hybrid',
                        'collaborative_insights': collab_result
                    }
                    logger.info("🚀 Running enhanced A2A orchestration with collaborative insights")
                    result = await self.a2a_protocol.run_orchestration(enhanced_data)
                else:
                    # Use standard A2A protocol orchestration
                    logger.info("🚀 Running standard A2A orchestration workflow")
                    initial_data = {'symbols': self.symbols, 'mode': 'hybrid'}
                    result = await self.a2a_protocol.run_orchestration(initial_data)

                logger.info(f"✅ Orchestration completed: {result}")

                # Wait before next cycle
                logger.info("⏱️ Waiting 15 minutes before next cycle...")
                await asyncio.sleep(900)  # 15 minutes

            except Exception as e:
                logger.error(f"❌ Hybrid workflow cycle failed: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error



    async def _run_analysis_workflow(self):
        """Run analysis-only workflow."""
        logger.info("🔍 Running analysis-only workflow")
        assert self.a2a_protocol is not None, "A2A protocol not initialized"
        while self.running:
            try:
                initial_data = {'symbols': self.symbols, 'mode': 'analysis'}
                result = await self.a2a_protocol.run_orchestration(initial_data)
                logger.info(f"✅ Analysis orchestration completed: {result}")
                await asyncio.sleep(900)  # 15 minutes
            except Exception as e:
                logger.error(f"❌ Analysis workflow failed: {e}")
                await asyncio.sleep(300)

    async def _run_execution_workflow(self):
        """Run execution-only workflow."""
        logger.info("💰 Running execution-only workflow")
        assert self.a2a_protocol is not None, "A2A protocol not initialized"
        while self.running:
            try:
                initial_data = {'symbols': self.symbols, 'mode': 'execution'}
                result = await self.a2a_protocol.run_orchestration(initial_data)
                logger.info(f"✅ Execution orchestration completed: {result}")
                await asyncio.sleep(300)  # 5 minutes
            except Exception as e:
                logger.error(f"❌ Execution workflow failed: {e}")
                await asyncio.sleep(300)

    async def _run_backtest_workflow(self):
        """Run backtesting workflow."""
        logger.info("📈 Running backtesting workflow")
        logger.info("Backtesting mode not yet implemented")

    async def create_collaborative_session(self, topic: str, participants: List[str],
                                         context: Dict[str, Any] = None) -> Optional[str]:
        """
        Create a collaborative session for complex analysis tasks.

        Args:
            topic: Session topic (e.g., "High Volatility SPY Analysis")
            participants: List of agent roles to participate
            context: Initial context data for the session

        Returns:
            Session ID if created successfully, None otherwise
        """
        if not self.collaborative_coordinator:
            logger.warning("Collaborative coordinator not available")
            return None

        try:
            # Create the session
            session_id = await self.collaborative_coordinator.create_collaborative_session(
                creator_agent="orchestrator",
                topic=topic,
                max_participants=len(participants) + 1,  # +1 for orchestrator
                session_timeout=1800  # 30 minutes
            )

            if session_id:
                # Store session info
                self.active_sessions[session_id] = {
                    'topic': topic,
                    'participants': participants,
                    'context': context or {},
                    'created_at': datetime.now()
                }

                # Have agents join the session
                for agent_role in participants:
                    if agent_role in self.agents:
                        await self.collaborative_coordinator.join_collaborative_session(
                            session_id=session_id,
                            agent_role=agent_role,
                            agent_context={'orchestrator_initiated': True, 'topic': topic}
                        )
                        logger.info(f"Agent {agent_role} joined collaborative session {session_id}")
                    else:
                        logger.warning(f"Agent {agent_role} not found for session {session_id}")

                # Add initial context if provided
                if context:
                    await self.collaborative_coordinator.update_session_context(
                        session_id=session_id,
                        agent_role="orchestrator",
                        context_type="initial_analysis",
                        context_data=context
                    )

                logger.info(f"✅ Created collaborative session: {topic} ({session_id})")
                return session_id

        except Exception as e:
            logger.error(f"Failed to create collaborative session: {e}")
            return None

    async def contribute_to_session(self, session_id: str, agent_role: str,
                                  insight: Dict[str, Any]) -> bool:
        """
        Have an agent contribute an insight to a collaborative session.

        Args:
            session_id: Session ID
            agent_role: Contributing agent role
            insight: Insight data

        Returns:
            Success status
        """
        if not self.collaborative_coordinator:
            return False

        try:
            return await self.collaborative_coordinator.contribute_to_session(
                session_id=session_id,
                agent_role=agent_role,
                insight=insight
            )
        except Exception as e:
            logger.error(f"Failed to contribute to session {session_id}: {e}")
            return False

    async def get_session_insights(self, session_id: str, agent_role: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get insights from a collaborative session.

        Args:
            session_id: Session ID
            agent_role: Filter by specific agent (optional)

        Returns:
            List of insights
        """
        if not self.collaborative_coordinator:
            return []

        try:
            return await self.collaborative_coordinator.get_session_insights(session_id, agent_role)
        except Exception as e:
            logger.error(f"Failed to get session insights for {session_id}: {e}")
            return []

    async def archive_session(self, session_id: str) -> bool:
        """
        Archive a completed collaborative session.

        Args:
            session_id: Session ID

        Returns:
            Success status
        """
        if not self.collaborative_coordinator:
            return False

        try:
            success = await self.collaborative_coordinator.archive_session(
                session_id=session_id,
                agent_role="orchestrator"
            )

            if success and session_id in self.active_sessions:
                del self.active_sessions[session_id]
                logger.info(f"✅ Archived collaborative session {session_id}")

            return success
        except Exception as e:
            logger.error(f"Failed to archive session {session_id}: {e}")
            return False

    async def run_collaborative_analysis(self, topic: str, context: Dict[str, Any],
                                       participants: List[str] = None) -> Dict[str, Any]:
        """
        Run a collaborative analysis session for complex tasks.

        Args:
            topic: Analysis topic
            context: Analysis context data
            participants: Agent participants (default: strategy, risk, data)

        Returns:
            Analysis results with collaborative insights
        """
        if participants is None:
            participants = ['strategy', 'risk', 'data']

        logger.info(f"🤝 Starting collaborative analysis: {topic}")

        # Create collaborative session
        session_id = await self.create_collaborative_session(
            topic=f"Collaborative Analysis: {topic}",
            participants=participants,
            context=context
        )

        if not session_id:
            logger.error("Failed to create collaborative session")
            return {'error': 'Failed to create session'}

        try:
            # Trigger agent contributions to the session
            await self._trigger_agent_contributions(session_id, participants, context)

            # Allow agents to check for and respond to messages
            await self._process_agent_messages(participants)

            # Give agents time to contribute insights
            await asyncio.sleep(15)  # 15 seconds for contributions

            # Collect insights from all participants
            all_insights = await self.get_session_insights(session_id)

            # Structure the results
            result = {
                'session_id': session_id,
                'topic': topic,
                'participants': participants,
                'total_insights': len(all_insights),
                'insights': all_insights,
                'consensus_analysis': self._analyze_collaborative_insights(all_insights, context)
            }

            logger.info(f"✅ Collaborative analysis complete: {len(all_insights)} insights collected")
            return result

        finally:
            # Archive the session
            await self.archive_session(session_id)

    async def _trigger_agent_contributions(self, session_id: str, participants: List[str],
                                         context: Dict[str, Any]):
        """
        Trigger participating agents to contribute insights to the collaborative session.

        Args:
            session_id: Session ID
            participants: List of participating agent roles
            context: Analysis context
        """
        for agent_role in participants:
            if agent_role in self.agents:
                try:
                    agent = self.agents[agent_role]

                    # Create a contribution request
                    contribution_prompt = {
                        'type': 'collaborative_contribution',
                        'session_id': session_id,
                        'context': context,
                        'request': f'Please contribute your analysis insights for: {context.get("topic", "current analysis")}'
                    }

                    # Send contribution request via A2A protocol
                    await self.a2a_protocol.send_message(
                        sender='orchestrator',
                        receiver=agent_role,
                        message_type='collaboration_request',
                        content=contribution_prompt
                    )

                    logger.info(f"Sent contribution request to {agent_role} for session {session_id}")

                except Exception as e:
                    logger.warning(f"Failed to trigger contribution from {agent_role}: {e}")

    async def _process_agent_messages(self, agent_roles: List[str]):
        """
        Process A2A messages for participating agents to handle collaborative requests.

        Args:
            agent_roles: List of agent roles to check for messages
        """
        for agent_role in agent_roles:
            if agent_role in self.agents:
                try:
                    agent = self.agents[agent_role]
                    if hasattr(agent, 'check_a2a_messages'):
                        await agent.check_a2a_messages()
                        logger.debug(f"Processed messages for agent {agent_role}")
                except Exception as e:
                    logger.warning(f"Error processing messages for agent {agent_role}: {e}")

    async def _should_run_collaborative_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Determine if collaborative analysis is needed based on current conditions.

        Returns:
            Dict with conditions that triggered collaborative analysis, or None
        """
        try:
            # Check market volatility (placeholder - integrate with real market data)
            volatility_condition = await self._check_market_volatility()
            if volatility_condition:
                return {
                    'trigger': 'high_volatility',
                    'reason': f"Market volatility: {volatility_condition}",
                    'severity': 'high'
                }

            # Check for complex multi-asset scenarios
            multi_asset_condition = await self._check_multi_asset_complexity()
            if multi_asset_condition:
                return {
                    'trigger': 'multi_asset_complexity',
                    'reason': f"Complex multi-asset scenario: {multi_asset_condition}",
                    'severity': 'medium'
                }

            # Check for conflicting signals from agents
            signal_conflict = await self._check_agent_signal_conflicts()
            if signal_conflict:
                return {
                    'trigger': 'signal_conflicts',
                    'reason': f"Agent signal conflicts detected: {signal_conflict}",
                    'severity': 'medium'
                }

            # Periodic deep analysis (every few cycles)
            if hasattr(self, '_cycle_count'):
                self._cycle_count += 1
            else:
                self._cycle_count = 1

            if self._cycle_count % 5 == 0:  # Every 5th cycle
                return {
                    'trigger': 'periodic_deep_analysis',
                    'reason': 'Scheduled comprehensive analysis',
                    'severity': 'low'
                }

            return None

        except Exception as e:
            logger.warning(f"Error checking collaborative analysis conditions: {e}")
            return None

    async def _check_market_volatility(self) -> Optional[str]:
        """Check if market conditions warrant collaborative analysis."""
        try:
            # Get VIX level using yfinance
            import yfinance as yf

            # Fetch VIX data for the last 5 trading days
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="5d")

            if vix_data.empty:
                logger.warning("Could not fetch VIX data")
                return None

            # Get the latest VIX close
            latest_vix = vix_data['Close'].iloc[-1]
            logger.info(f"Current VIX level: {latest_vix:.2f}")

            # Check for high volatility conditions
            if latest_vix > 30:
                return f"VIX at {latest_vix:.1f} - extreme volatility detected"
            elif latest_vix > 25:
                return f"VIX at {latest_vix:.1f} - high volatility environment"
            elif latest_vix > 20:
                return f"VIX at {latest_vix:.1f} - elevated volatility"

            # Also check for recent volatility spikes (VIX change > 20% in last 2 days)
            if len(vix_data) >= 2:
                recent_change = (latest_vix - vix_data['Close'].iloc[-2]) / vix_data['Close'].iloc[-2]
                if abs(recent_change) > 0.20:  # 20% change
                    direction = "spike up" if recent_change > 0 else "drop"
                    return f"VIX {direction} {recent_change:.1%} in 1 day - volatility event"

        except Exception as e:
            logger.warning(f"Error checking market volatility: {e}")
            return None

    async def _check_multi_asset_complexity(self) -> Optional[str]:
        """Check for complex multi-asset scenarios."""
        try:
            if len(self.symbols) <= 1:
                return None

            # For multi-asset scenarios, check correlation complexity
            if len(self.symbols) > 3:
                return f"Complex multi-asset portfolio: {len(self.symbols)} symbols requiring correlation analysis"

            # Check for sector diversity (simplified)
            sectors = set()
            import yfinance as yf

            for symbol in self.symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    sector = info.get('sector', 'Unknown')
                    if sector != 'Unknown':
                        sectors.add(sector)
                except Exception:
                    continue

            # If we have assets from different sectors, it's more complex
            if len(sectors) > 1:
                return f"Multi-sector analysis: {len(sectors)} sectors ({', '.join(sectors)}) requiring cross-sector correlation analysis"

            # Check for high correlation between assets
            if len(self.symbols) == 2:
                correlation = await self._calculate_asset_correlation(self.symbols[0], self.symbols[1])
                if correlation and abs(correlation) > 0.7:  # Highly correlated
                    return f"Highly correlated pair: {self.symbols[0]}-{self.symbols[1]} ({correlation:.2f}) requiring coordinated analysis"

        except Exception as e:
            logger.warning(f"Error checking multi-asset complexity: {e}")
            return None

    async def _calculate_asset_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """Calculate correlation between two assets over the last 30 trading days."""
        try:
            import yfinance as yf
            import numpy as np

            # Fetch data for both symbols
            data1 = yf.Ticker(symbol1).history(period="2mo")
            data2 = yf.Ticker(symbol2).history(period="2mo")

            if data1.empty or data2.empty:
                return None

            # Calculate daily returns
            returns1 = data1['Close'].pct_change().dropna()
            returns2 = data2['Close'].pct_change().dropna()

            # Align the data by date
            common_dates = returns1.index.intersection(returns2.index)
            if len(common_dates) < 20:  # Need at least 20 data points
                return None

            returns1_aligned = returns1.loc[common_dates]
            returns2_aligned = returns2.loc[common_dates]

            # Calculate correlation
            correlation = returns1_aligned.corr(returns2_aligned)
            return correlation

        except Exception as e:
            logger.warning(f"Error calculating correlation between {symbol1} and {symbol2}: {e}")
            return None

    async def _check_agent_signal_conflicts(self) -> Optional[str]:
        """Check for conflicting signals between agents."""
        try:
            conflicts = []

            # Check strategy vs risk agent alignment
            strategy_signals = await self._get_recent_agent_signals('strategy')
            risk_signals = await self._get_recent_agent_signals('risk')

            if strategy_signals and risk_signals:
                # Check for risk-adjusted return conflicts
                strategy_returns = [s.get('expected_return', 0) for s in strategy_signals]
                risk_limits = [r.get('risk_limit', 0.1) for r in risk_signals]

                if strategy_returns and risk_limits:
                    avg_strategy_return = sum(strategy_returns) / len(strategy_returns)
                    avg_risk_limit = sum(risk_limits) / len(risk_limits)

                    # If expected returns significantly exceed risk limits, flag conflict
                    if avg_strategy_return > avg_risk_limit * 2:  # 2x risk limit
                        conflicts.append(f"Strategy expected returns ({avg_strategy_return:.1%}) exceed risk limits ({avg_risk_limit:.1%})")

            # Check for directional conflicts between agents
            directional_conflicts = await self._check_directional_conflicts()
            if directional_conflicts:
                conflicts.extend(directional_conflicts)

            if conflicts:
                return f"Agent signal conflicts detected: {'; '.join(conflicts[:2])}"  # Limit to 2 conflicts

        except Exception as e:
            logger.warning(f"Error checking agent signal conflicts: {e}")
            return None

    async def _get_recent_agent_signals(self, agent_role: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent signals from a specific agent."""
        try:
            if agent_role not in self.agents:
                return []

            agent = self.agents[agent_role]

            # Try to get recent analysis or recommendations from agent memory
            if hasattr(agent, 'memory') and agent.memory:
                recent_items = agent.memory.get('recent_analysis', [])
                return recent_items[-limit:] if recent_items else []

            # Fallback: check if agent has recent outputs
            if hasattr(agent, 'recent_outputs'):
                return agent.recent_outputs[-limit:] if agent.recent_outputs else []

        except Exception as e:
            logger.debug(f"Error getting signals from {agent_role}: {e}")

        return []

    async def _check_directional_conflicts(self) -> List[str]:
        """Check for directional conflicts between agents (bullish vs bearish signals)."""
        conflicts = []

        try:
            # Get directional signals from different agents
            strategy_direction = await self._get_agent_direction('strategy')
            macro_direction = await self._get_agent_direction('macro')

            if strategy_direction and macro_direction:
                if strategy_direction != macro_direction:
                    conflicts.append(f"Strategy ({strategy_direction}) vs Macro ({macro_direction}) directional conflict")

        except Exception as e:
            logger.debug(f"Error checking directional conflicts: {e}")

        return conflicts

    async def _get_agent_direction(self, agent_role: str) -> Optional[str]:
        """Get directional bias from an agent (bullish/bearish/neutral)."""
        try:
            signals = await self._get_recent_agent_signals(agent_role, limit=3)

            if not signals:
                return None

            # Analyze signals for directional bias
            bullish_signals = 0
            bearish_signals = 0

            for signal in signals:
                direction = str(signal).lower()
                if 'bull' in direction or 'long' in direction or 'positive' in direction:
                    bullish_signals += 1
                elif 'bear' in direction or 'short' in direction or 'negative' in direction:
                    bearish_signals += 1

            if bullish_signals > bearish_signals:
                return 'bullish'
            elif bearish_signals > bullish_signals:
                return 'bearish'
            else:
                return 'neutral'

        except Exception as e:
            logger.debug(f"Error getting direction from {agent_role}: {e}")
            return None

    def _analyze_collaborative_insights(self, insights: List[Dict[str, Any]],
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze collaborative insights to find consensus and key findings.

        Args:
            insights: List of insights from the session
            context: Original analysis context

        Returns:
            Consensus analysis
        """
        if not insights:
            return {'consensus': 'No insights provided', 'confidence': 0}

        # Simple consensus analysis (can be enhanced with ML)
        agent_insights = {}
        for insight in insights:
            agent = insight.get('agent', 'unknown')
            if agent not in agent_insights:
                agent_insights[agent] = []
            agent_insights[agent].append(insight)

        # Calculate basic consensus metrics
        total_insights = len(insights)
        unique_agents = len(agent_insights)

        return {
            'total_insights': total_insights,
            'participating_agents': unique_agents,
            'insights_per_agent': {agent: len(ins) for agent, ins in agent_insights.items()},
            'consensus_level': 'High' if unique_agents >= 2 else 'Limited',
            'key_findings': [i.get('insight', {}).get('summary', 'No summary') for i in insights[:3]]
        }

    async def stop(self):
        """Stop the orchestrator."""
        logger.info("🛑 Stopping Unified Workflow Orchestrator...")

        # Archive any remaining active sessions
        for session_id in list(self.active_sessions.keys()):
            await self.archive_session(session_id)

        self.running = False
        logger.info("✅ Orchestrator stopped")

def main():
    """Main entry point with command-line arguments."""
    parser = argparse.ArgumentParser(description="Unified Workflow Orchestrator")
    parser.add_argument(
        '--mode',
        choices=['analysis', 'execution', 'hybrid', 'backtest'],
        default='hybrid',
        help='Operating mode'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['SPY'],
        help='Symbols to trade/analyze'
    )
    parser.add_argument(
        '--no-discord',
        action='store_true',
        help='Disable Discord integration'
    )

    args = parser.parse_args()

    mode_map = {
        'analysis': WorkflowMode.ANALYSIS,
        'execution': WorkflowMode.EXECUTION,
        'hybrid': WorkflowMode.HYBRID,
        'backtest': WorkflowMode.BACKTEST
    }

    orchestrator = UnifiedWorkflowOrchestrator(
        mode=mode_map[args.mode],
        enable_discord=not args.no_discord,
        symbols=args.symbols
    )

    # Run initialization and start
    asyncio.run(orchestrator.initialize())
    asyncio.run(orchestrator.start())

if __name__ == "__main__":
    main()