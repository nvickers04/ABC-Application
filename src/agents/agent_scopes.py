# Agent Scope Definitions and Command Routing
# This file defines clear scopes for each agent while enabling collaborative "hive mind" behavior

from typing import Dict, List, Set, Any, Optional
from enum import Enum
import re

class AgentScope(Enum):
    """Defines the primary scope/responsibility of each agent"""
    MACRO = "macro_analysis"
    DATA = "data_collection"
    STRATEGY = "strategy_generation"
    RISK = "risk_assessment"
    EXECUTION = "trade_execution"
    REFLECTION = "performance_review"
    LEARNING = "model_optimization"

class CollaborationMode(Enum):
    """Defines how agents can collaborate"""
    DIRECT_REQUEST = "direct_request"  # Agent asks another agent directly
    BROADCAST_LISTEN = "broadcast_listen"  # Agent listens to all broadcasts but filters relevant
    VETO_CONSENSUS = "veto_consensus"  # Agent participates in approval/consensus processes
    SHARED_CONTEXT = "shared_context"  # Agent contributes to shared knowledge base

class AgentScopeDefinition:
    """Defines the scope, capabilities, and collaboration rules for each agent"""

    def __init__(self):
        self.agent_scopes = self._define_agent_scopes()
        self.collaboration_rules = self._define_collaboration_rules()
        self.command_filters = self._define_command_filters()

    def _define_agent_scopes(self) -> Dict[str, Dict[str, Any]]:
        """Define the primary scope and capabilities of each agent"""
        return {
            'macro': {
                'scope': AgentScope.MACRO,
                'primary_responsibilities': [
                    'sector_analysis',
                    'asset_universe_scanning',
                    'market_regime_assessment',
                    'macro_economic_trends',
                    'sector_rotation_signals'
                ],
                'can_request_from': ['data', 'strategy', 'risk'],
                'provides_to': ['strategy', 'risk', 'execution'],
                'decision_authority': 'recommendatory',  # Makes recommendations, not final decisions
                'data_dependencies': ['market_data', 'economic_indicators', 'sector_performance']
            },

            'data': {
                'scope': AgentScope.DATA,
                'primary_responsibilities': [
                    'market_data_collection',
                    'economic_data_aggregation',
                    'news_sentiment_analysis',
                    'institutional_activity_tracking',
                    'real_time_data_processing'
                ],
                'can_request_from': ['macro', 'strategy'],  # Limited requests
                'provides_to': ['macro', 'strategy', 'risk', 'execution'],
                'decision_authority': 'informational',  # Provides data, no decisions
                'data_dependencies': []  # Data source, no dependencies
            },

            'strategy': {
                'scope': AgentScope.STRATEGY,
                'primary_responsibilities': [
                    'trade_strategy_generation',
                    'options_strategy_design',
                    'pyramiding_logic',
                    'entry_exit_criteria',
                    'position_sizing'
                ],
                'can_request_from': ['data', 'macro', 'risk', 'execution'],
                'provides_to': ['risk', 'execution', 'reflection'],
                'decision_authority': 'propositional',  # Proposes strategies
                'data_dependencies': ['market_data', 'technical_indicators', 'volatility_data']
            },

            'risk': {
                'scope': AgentScope.RISK,
                'primary_responsibilities': [
                    'drawdown_assessment',
                    'volatility_analysis',
                    'correlation_analysis',
                    'stress_testing',
                    'risk_limit_enforcement'
                ],
                'can_request_from': ['data', 'macro', 'strategy'],
                'provides_to': ['strategy', 'execution', 'reflection'],
                'decision_authority': 'veto',  # Can veto trades
                'data_dependencies': ['position_data', 'market_data', 'volatility_data']
            },

            'execution': {
                'scope': AgentScope.EXECUTION,
                'primary_responsibilities': [
                    'order_execution',
                    'slippage_management',
                    'timing_optimization',
                    'position_monitoring',
                    'execution_cost_analysis'
                ],
                'can_request_from': ['strategy', 'risk', 'data'],
                'provides_to': ['reflection', 'learning'],
                'decision_authority': 'implementational',  # Executes approved trades
                'data_dependencies': ['broker_data', 'market_data', 'position_data']
            },

            'reflection': {
                'scope': AgentScope.REFLECTION,
                'primary_responsibilities': [
                    'performance_analysis',
                    'outcome_assessment',
                    'process_improvement',
                    'audit_conduct',
                    'lesson_extraction'
                ],
                'can_request_from': ['execution', 'risk', 'strategy', 'data'],
                'provides_to': ['learning', 'strategy', 'risk'],
                'decision_authority': 'analytical',  # Analyzes past performance
                'data_dependencies': ['trade_history', 'performance_metrics']
            },

            'learning': {
                'scope': AgentScope.LEARNING,
                'primary_responsibilities': [
                    'model_training',
                    'strategy_optimization',
                    'backtesting_improvement',
                    'predictive_model_updates',
                    'system_adaptation'
                ],
                'can_request_from': ['reflection', 'strategy', 'data'],
                'provides_to': ['strategy', 'risk', 'execution'],
                'decision_authority': 'optimizational',  # Improves system performance
                'data_dependencies': ['historical_data', 'performance_data', 'model_metrics']
            }
        }

    def _define_collaboration_rules(self) -> Dict[str, Dict[str, Any]]:
        """Define how agents can collaborate with each other"""
        return {
            'direct_request': {
                'description': 'Agent can directly request information from other agents',
                'allowed_pairs': [
                    ('macro', 'data'),      # Macro asks data for initial data
                    ('strategy', 'data'),   # Strategy asks data for market info
                    ('strategy', 'risk'),   # Strategy asks risk for vetting
                    ('strategy', 'execution'), # Strategy asks execution for timing
                    ('execution', 'data'),  # Execution asks data for real-time info
                    ('reflection', 'execution'), # Reflection asks execution for results
                    ('learning', 'reflection'), # Learning asks reflection for insights
                ],
                'protocol': 'a2a_message',
                'response_required': True
            },

            'broadcast_listen': {
                'description': 'All agents receive broadcasts but filter for relevance',
                'universal_access': True,
                'filtering_required': True,
                'protocol': 'broadcast_with_filter'
            },

            'veto_consensus': {
                'description': 'Critical decisions require consensus from multiple agents',
                'required_participants': {
                    'trade_approval': ['strategy', 'risk', 'execution'],
                    'system_changes': ['reflection', 'learning', 'risk'],
                    'risk_limit_changes': ['risk', 'reflection', 'execution']
                },
                'veto_power': ['risk', 'execution'],  # These agents can veto
                'protocol': 'consensus_voting'
            },

            'shared_context': {
                'description': 'Agents contribute to and access shared knowledge base',
                'shared_namespaces': [
                    'market_context',
                    'strategy_templates',
                    'risk_parameters',
                    'performance_history',
                    'system_configuration'
                ],
                'access_level': 'read_write',  # All agents can read and write
                'protocol': 'shared_memory'
            }
        }

    def _define_command_filters(self) -> Dict[str, Dict[str, Any]]:
        """Define which commands each agent should process vs ignore"""
        return {
            'macro': {
                'process_commands': [
                    r'.*macro.*analysis.*',
                    r'.*sector.*selection.*',
                    r'.*market.*regime.*',
                    r'.*economic.*trend.*',
                    r'.*asset.*universe.*',
                    r'.*market.*condition.*'
                ],
                'ignore_commands': [
                    r'.*execute.*trade.*',
                    r'.*place.*order.*',
                    r'.*backtest.*strategy.*',
                    r'.*model.*training.*'
                ],
                'delegate_to': {
                    r'.*data.*collection.*': 'data',
                    r'.*strategy.*generation.*': 'strategy',
                    r'.*risk.*assessment.*': 'risk'
                }
            },

            'data': {
                'process_commands': [
                    r'.*fetch.*data.*',
                    r'.*collect.*market.*',
                    r'.*economic.*indicator.*',
                    r'.*news.*sentiment.*',
                    r'.*real.*time.*data.*',
                    r'.*institutional.*activity.*'
                ],
                'ignore_commands': [
                    r'.*execute.*trade.*',
                    r'.*strategy.*design.*',
                    r'.*risk.*limit.*',
                    r'.*performance.*review.*'
                ],
                'delegate_to': {}
            },

            'strategy': {
                'process_commands': [
                    r'.*strategy.*generation.*',
                    r'.*trade.*setup.*',
                    r'.*options.*strategy.*',
                    r'.*pyramiding.*logic.*',
                    r'.*entry.*exit.*criteria.*',
                    r'.*position.*sizing.*'
                ],
                'ignore_commands': [
                    r'.*execute.*order.*',
                    r'.*monitor.*position.*',
                    r'.*model.*training.*',
                    r'.*audit.*performance.*'
                ],
                'delegate_to': {
                    r'.*risk.*vetting.*': 'risk',
                    r'.*execution.*timing.*': 'execution',
                    r'.*market.*data.*': 'data'
                }
            },

            'risk': {
                'process_commands': [
                    r'.*risk.*assessment.*',
                    r'.*drawdown.*analysis.*',
                    r'.*volatility.*analysis.*',
                    r'.*correlation.*analysis.*',
                    r'.*stress.*test.*',
                    r'.*risk.*limit.*'
                ],
                'ignore_commands': [
                    r'.*execute.*trade.*',
                    r'.*strategy.*design.*',
                    r'.*data.*collection.*',
                    r'.*model.*optimization.*'
                ],
                'delegate_to': {
                    r'.*market.*data.*': 'data',
                    r'.*strategy.*validation.*': 'strategy'
                }
            },

            'execution': {
                'process_commands': [
                    r'.*execute.*trade.*',
                    r'.*place.*order.*',
                    r'.*slippage.*management.*',
                    r'.*timing.*optimization.*',
                    r'.*position.*monitoring.*',
                    r'.*execution.*cost.*'
                ],
                'ignore_commands': [
                    r'.*strategy.*generation.*',
                    r'.*data.*collection.*',
                    r'.*model.*training.*',
                    r'.*macro.*analysis.*'
                ],
                'delegate_to': {
                    r'.*strategy.*validation.*': 'strategy',
                    r'.*risk.*check.*': 'risk'
                }
            },

            'reflection': {
                'process_commands': [
                    r'.*performance.*analysis.*',
                    r'.*outcome.*assessment.*',
                    r'.*process.*improvement.*',
                    r'.*audit.*conduct.*',
                    r'.*lesson.*extraction.*'
                ],
                'ignore_commands': [
                    r'.*execute.*trade.*',
                    r'.*strategy.*design.*',
                    r'.*data.*collection.*',
                    r'.*real.*time.*monitoring.*'
                ],
                'delegate_to': {
                    r'.*trade.*result.*': 'execution',
                    r'.*strategy.*performance.*': 'strategy'
                }
            },

            'learning': {
                'process_commands': [
                    r'.*model.*training.*',
                    r'.*strategy.*optimization.*',
                    r'.*backtesting.*improvement.*',
                    r'.*predictive.*model.*',
                    r'.*system.*adaptation.*'
                ],
                'ignore_commands': [
                    r'.*execute.*trade.*',
                    r'.*real.*time.*data.*',
                    r'.*position.*monitoring.*',
                    r'.*macro.*analysis.*'
                ],
                'delegate_to': {
                    r'.*performance.*data.*': 'reflection',
                    r'.*strategy.*template.*': 'strategy'
                }
            }
        }

    def get_agent_scope(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get the scope definition for a specific agent"""
        return self.agent_scopes.get(agent_name)

    def can_agent_request_from(self, requesting_agent: str, target_agent: str) -> bool:
        """Check if one agent can request information from another"""
        scope = self.get_agent_scope(requesting_agent)
        if not scope:
            return False
        return target_agent in scope.get('can_request_from', [])

    def should_agent_process_command(self, agent_name: str, command: str) -> Dict[str, Any]:
        """
        Determine if an agent should process a command and how

        Returns:
            {
                'should_process': bool,
                'action': 'process'|'delegate'|'ignore',
                'delegate_to': str or None,
                'confidence': float
            }
        """
        filters = self.command_filters.get(agent_name, {})
        if not filters:
            return {'should_process': True, 'action': 'process', 'delegate_to': None, 'confidence': 0.5}

        command_lower = command.lower()

        # Check if command should be ignored
        for ignore_pattern in filters.get('ignore_commands', []):
            if re.search(ignore_pattern, command_lower, re.IGNORECASE):
                return {'should_process': False, 'action': 'ignore', 'delegate_to': None, 'confidence': 0.9}

        # Check if command should be delegated
        for delegate_pattern, delegate_agent in filters.get('delegate_to', {}).items():
            if re.search(delegate_pattern, command_lower, re.IGNORECASE):
                return {'should_process': False, 'action': 'delegate', 'delegate_to': delegate_agent, 'confidence': 0.8}

        # Check if command should be processed
        for process_pattern in filters.get('process_commands', []):
            if re.search(process_pattern, command_lower, re.IGNORECASE):
                return {'should_process': True, 'action': 'process', 'delegate_to': None, 'confidence': 0.8}

        # Default: process with low confidence (allows for learning)
        return {'should_process': True, 'action': 'process', 'delegate_to': None, 'confidence': 0.3}

    def get_collaboration_path(self, from_agent: str, to_agent: str, task_type: str) -> Optional[str]:
        """Get the appropriate collaboration method for a task between agents"""
        rules = self.collaboration_rules

        # Check direct request
        if task_type == 'information_request':
            allowed_pairs = rules['direct_request']['allowed_pairs']
            if (from_agent, to_agent) in allowed_pairs:
                return 'direct_request'

        # Check veto consensus
        if task_type == 'approval':
            required_participants = rules['veto_consensus']['required_participants']
            for process, participants in required_participants.items():
                if from_agent in participants and to_agent in participants:
                    return 'veto_consensus'

        # Default to shared context
        return 'shared_context'

# Global instance for easy access
agent_scope_definitions = AgentScopeDefinition()

def get_agent_scope_definitions():
    """Get the global agent scope definitions instance"""
    return agent_scope_definitions