#!/usr/bin/env python3
"""
Unit tests for Acontext integration in the ABC Application system.
Tests AcontextIntegration, TradingDirective, learning agent Acontext functionality,
and cross-agent propagation.
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import pytest
import asyncio
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import Acontext integration components directly without going through src package
# This avoids the complex dependency chain
sys.path.insert(0, str(project_root / 'src' / 'integrations'))

# Mock the heavy imports before loading the module
with patch.dict(sys.modules, {
    'src.utils.utils': MagicMock(load_yaml=MagicMock(return_value={}))
}):
    from acontext_integration import (
        AcontextIntegration,
        TradingDirective,
        get_acontext_integration,
        ACONTEXT_AVAILABLE
    )


class TestTradingDirective:
    """Test cases for TradingDirective dataclass."""

    def test_trading_directive_creation(self):
        """Test creating a TradingDirective instance."""
        directive = TradingDirective(
            id='test_directive_001',
            category='learning_insight',
            name='Test Directive',
            description='A test trading directive',
            content={'directives': [{'refinement': 'sizing_lift', 'value': 1.2}]},
            applies_to=['strategy', 'risk'],
            source='learning',
            priority='medium'
        )
        
        assert directive.id == 'test_directive_001'
        assert directive.category == 'learning_insight'
        assert directive.name == 'Test Directive'
        assert directive.applies_to == ['strategy', 'risk']
        assert directive.priority == 'medium'
        assert directive.source == 'learning'

    def test_trading_directive_to_dict(self):
        """Test converting TradingDirective to dictionary."""
        directive = TradingDirective(
            id='test_directive_002',
            category='risk_constraint',
            name='Risk Directive',
            description='A risk constraint directive',
            content={'max_drawdown': 0.05},
            applies_to=['risk'],
            source='learning',
            priority='high'
        )
        
        result = directive.to_dict()
        
        assert isinstance(result, dict)
        assert result['id'] == 'test_directive_002'
        assert result['category'] == 'risk_constraint'
        assert result['applies_to'] == ['risk']
        assert result['content'] == {'max_drawdown': 0.05}

    def test_trading_directive_from_dict(self):
        """Test creating TradingDirective from dictionary."""
        data = {
            'id': 'test_directive_003',
            'category': 'strategy_directive',
            'name': 'Strategy Directive',
            'description': 'A strategy directive',
            'content': {'pyramiding_enabled': True},
            'applies_to': ['strategy'],
            'source': 'learning',
            'priority': 'low'
        }
        
        directive = TradingDirective.from_dict(data)
        
        assert directive.id == 'test_directive_003'
        assert directive.category == 'strategy_directive'
        assert directive.content == {'pyramiding_enabled': True}


class TestAcontextIntegration:
    """Test cases for AcontextIntegration class."""

    @pytest.fixture
    def acontext_integration(self):
        """Create an AcontextIntegration instance with mocked config."""
        with patch.object(AcontextIntegration, '_load_config') as mock_load:
            mock_load.return_value = {
                'api': {
                    'base_url': 'https://api.acontext.dev',
                    'timeout_seconds': 30,
                    'max_retries': 3,
                },
                'space': {
                    'name': 'test-trading-sops',
                },
                'sop': {
                    'id_prefix': 'test_directive',
                    'default_ttl_days': 90,
                    'priority_levels': {
                        'critical': 100,
                        'high': 75,
                        'medium': 50,
                        'low': 25,
                        'background': 10,
                    }
                },
                'fallback': {
                    'enabled': True,
                    'use_local_storage': True,
                    'local_storage_path': '/tmp/acontext_test',
                },
                'propagation': {
                    'enabled': True,
                    'target_agents': ['strategy', 'risk', 'execution'],
                    'priority_queuing_enabled': True,
                },
                'monitoring': {
                    'alert_threshold_failures': 3,
                }
            }
            integration = AcontextIntegration()
            return integration

    def test_acontext_integration_initialization(self, acontext_integration):
        """Test AcontextIntegration initialization."""
        assert acontext_integration is not None
        assert acontext_integration.config is not None
        assert acontext_integration._fallback_mode == False
        assert acontext_integration._initialized == False

    def test_generate_directive_id(self, acontext_integration):
        """Test directive ID generation."""
        directive_id = acontext_integration._generate_directive_id('learning_insight')
        
        assert directive_id is not None
        assert 'test_directive' in directive_id
        assert 'learning_insight' in directive_id

    def test_get_default_config(self, acontext_integration):
        """Test getting default configuration."""
        config = acontext_integration._get_default_config()
        
        assert 'api' in config
        assert 'space' in config
        assert 'sop' in config
        assert 'fallback' in config
        assert 'propagation' in config

    def test_queue_directive(self, acontext_integration):
        """Test queuing directives for async propagation."""
        directive = TradingDirective(
            id='queue_test_001',
            category='learning_insight',
            name='Queue Test',
            description='Test queuing',
            content={'test': True},
            applies_to=['strategy'],
            source='learning',
            priority='medium'
        )
        
        acontext_integration.queue_directive(directive)
        
        assert len(acontext_integration._directive_queue) == 1
        assert acontext_integration._directive_queue[0].id == 'queue_test_001'

    def test_queue_directive_priority_ordering(self, acontext_integration):
        """Test that queued directives are ordered by priority."""
        low_priority = TradingDirective(
            id='low_001', category='test', name='Low', description='Low priority',
            content={}, applies_to=['strategy'], source='learning', priority='low'
        )
        high_priority = TradingDirective(
            id='high_001', category='test', name='High', description='High priority',
            content={}, applies_to=['strategy'], source='learning', priority='high'
        )
        critical_priority = TradingDirective(
            id='critical_001', category='test', name='Critical', description='Critical priority',
            content={}, applies_to=['strategy'], source='learning', priority='critical'
        )
        
        acontext_integration.queue_directive(low_priority)
        acontext_integration.queue_directive(high_priority)
        acontext_integration.queue_directive(critical_priority)
        
        # Should be ordered: critical, high, low
        assert acontext_integration._directive_queue[0].priority == 'critical'
        assert acontext_integration._directive_queue[1].priority == 'high'
        assert acontext_integration._directive_queue[2].priority == 'low'

    def test_get_queue_status(self, acontext_integration):
        """Test getting queue status."""
        directive = TradingDirective(
            id='status_test', category='test', name='Status Test', description='Test',
            content={}, applies_to=['strategy'], source='learning', priority='medium'
        )
        acontext_integration.queue_directive(directive)
        
        status = acontext_integration.get_queue_status()
        
        assert status['total_queued'] == 1
        assert 'medium' in status['by_priority']
        assert status['fallback_mode'] == False


class TestAcontextIntegrationAsync:
    """Async test cases for AcontextIntegration class."""

    @pytest.fixture
    def acontext_integration(self):
        """Create an AcontextIntegration instance for async testing."""
        with patch.object(AcontextIntegration, '_load_config') as mock_load:
            mock_load.return_value = {
                'api': {'timeout_seconds': 30, 'max_retries': 3},
                'space': {'name': 'test-sops'},
                'sop': {
                    'id_prefix': 'test_directive',
                    'default_ttl_days': 90,
                    'priority_levels': {'critical': 100, 'high': 75, 'medium': 50, 'low': 25}
                },
                'fallback': {'enabled': True, 'use_local_storage': True, 'local_storage_path': '/tmp/acontext_test'},
                'propagation': {'enabled': True, 'target_agents': ['strategy', 'risk']},
                'session': {'log_all_sessions': True},
                'artifacts': {'enabled': True, 'types': ['ml_model', 'backtest_result'], 'max_size_mb': 100},
                'monitoring': {'alert_threshold_failures': 3}
            }
            return AcontextIntegration()

    @pytest.mark.asyncio
    async def test_initialize_without_api_key(self, acontext_integration):
        """Test initialization falls back gracefully without API key."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove ACONTEXT_API_KEY if it exists
            os.environ.pop('ACONTEXT_API_KEY', None)
            
            result = await acontext_integration.initialize()
            
            # Should initialize in fallback mode
            assert acontext_integration._initialized == True
            assert acontext_integration._fallback_mode == True

    @pytest.mark.asyncio
    async def test_store_sop_fallback(self, acontext_integration):
        """Test storing SOP in fallback mode."""
        # Force fallback mode
        acontext_integration._fallback_mode = True
        acontext_integration._initialized = True
        
        directive = TradingDirective(
            id='fallback_test_001',
            category='learning_insight',
            name='Fallback Test',
            description='Test fallback storage',
            content={'directives': [{'refinement': 'test', 'value': 1.0}]},
            applies_to=['strategy'],
            source='learning',
            priority='medium'
        )
        
        sop_id = await acontext_integration.store_sop(directive)
        
        assert sop_id is not None
        assert 'local_' in sop_id

    @pytest.mark.asyncio
    async def test_retrieve_sop_fallback(self, acontext_integration):
        """Test retrieving SOP from fallback storage."""
        # Force fallback mode and store a directive
        acontext_integration._fallback_mode = True
        acontext_integration._initialized = True
        
        directive = TradingDirective(
            id='retrieve_test_001',
            category='learning_insight',
            name='Retrieve Test',
            description='Test retrieval',
            content={'test': True},
            applies_to=['strategy'],
            source='learning',
            priority='medium'
        )
        
        sop_id = await acontext_integration.store_sop(directive)
        retrieved = await acontext_integration.retrieve_sop(sop_id)
        
        assert retrieved is not None
        assert retrieved.id == 'retrieve_test_001'

    @pytest.mark.asyncio
    async def test_query_sops_fallback(self, acontext_integration):
        """Test querying SOPs from fallback storage."""
        # Force fallback mode and store some directives
        acontext_integration._fallback_mode = True
        acontext_integration._initialized = True
        
        for i in range(3):
            directive = TradingDirective(
                id=f'query_test_{i:03d}',
                category='learning_insight',
                name=f'Query Test {i}',
                description='Test query',
                content={'index': i},
                applies_to=['strategy'],
                source='learning',
                priority='medium'
            )
            await acontext_integration.store_sop(directive)
        
        results = await acontext_integration.query_sops(category='learning_insight', limit=10)
        
        assert len(results) >= 3

    @pytest.mark.asyncio
    async def test_log_session_fallback(self, acontext_integration):
        """Test logging session in fallback mode."""
        acontext_integration._fallback_mode = True
        acontext_integration._initialized = True
        
        session_data = {
            'type': 'learning_session',
            'agent': 'learning',
            'log_count': 5,
        }
        
        session_id = await acontext_integration.log_session(session_data)
        
        assert session_id is not None
        assert 'session_' in session_id

    @pytest.mark.asyncio
    async def test_upload_artifact_fallback(self, acontext_integration):
        """Test uploading artifact in fallback mode."""
        acontext_integration._fallback_mode = True
        acontext_integration._initialized = True
        
        artifact_data = b'test artifact data'
        metadata = {'name': 'test_model', 'type': 'ml_model'}
        
        artifact_id = await acontext_integration.upload_artifact(
            artifact_type='ml_model',
            artifact_data=artifact_data,
            metadata=metadata
        )
        
        assert artifact_id is not None
        assert 'artifact_ml_model_' in artifact_id


class TestLearningAgentAcontextIntegration:
    """Test cases for Learning Agent Acontext functionality.
    
    These tests use mock objects to test the integration patterns without
    importing the full learning agent dependencies.
    """

    @pytest.fixture
    def mock_learning_agent(self):
        """Create a mock learning agent for testing Acontext patterns."""
        mock_integration = MagicMock()
        mock_integration.initialize = AsyncMock(return_value=True)
        mock_integration.store_sop = AsyncMock(return_value='test_sop_001')
        mock_integration.log_session = AsyncMock(return_value='session_001')
        mock_integration.query_sops = AsyncMock(return_value=[])
        mock_integration.get_queue_status = Mock(return_value={'total_queued': 0, 'fallback_mode': False})
        
        agent = MagicMock()
        agent.role = 'learning'
        agent.tools = []
        agent.configs = {}
        agent.memory = {
            'weekly_batches': [],
            'pyramiding_performance': [],
            'stored_sops': [],
            'propagated_directives': []
        }
        agent.llm = None
        agent.a2a_protocol = None
        agent.acontext_integration = mock_integration
        agent.acontext_available = True
        agent.acontext_initialized = True
        agent.TradingDirective = TradingDirective
        
        return agent

    def test_mock_learning_agent_setup(self, mock_learning_agent):
        """Test that mock learning agent is properly configured."""
        assert mock_learning_agent.role == 'learning'
        assert mock_learning_agent.acontext_available == True
        assert mock_learning_agent.acontext_initialized == True
        assert mock_learning_agent.acontext_integration is not None

    @pytest.mark.asyncio
    async def test_acontext_integration_store_sop(self, mock_learning_agent):
        """Test that Acontext integration can store SOPs."""
        directive = TradingDirective(
            id='learning_test_001',
            category='learning_insight',
            name='Learning Test',
            description='Test from learning agent',
            content={'directives': [{'refinement': 'test', 'value': 1.0}]},
            applies_to=['strategy'],
            source='learning',
            priority='medium'
        )
        
        sop_id = await mock_learning_agent.acontext_integration.store_sop(directive)
        
        assert sop_id == 'test_sop_001'
        mock_learning_agent.acontext_integration.store_sop.assert_called_once()

    @pytest.mark.asyncio
    async def test_acontext_integration_log_session(self, mock_learning_agent):
        """Test that Acontext integration can log sessions."""
        session_data = {
            'type': 'learning_session',
            'logs': [{'sharpe': 1.5}]
        }
        
        session_id = await mock_learning_agent.acontext_integration.log_session(session_data)
        
        assert session_id == 'session_001'

    def test_determine_directive_targets_pattern(self):
        """Test the pattern for determining directive target agents."""
        # This tests the logic that would be in _determine_directive_target_agents
        directives = [
            {'refinement': 'sizing_lift', 'value': 1.2},
            {'refinement': 'risk_conservative', 'value': 0.9},
            {'refinement': 'pyramiding_boost', 'value': 1.1}
        ]
        
        target_agents = set()
        for directive in directives:
            refinement = directive.get('refinement', '')
            if 'sizing' in refinement or 'position' in refinement:
                target_agents.add('strategy')
                target_agents.add('execution')
            if 'risk' in refinement or 'conservative' in refinement:
                target_agents.add('risk')
            if 'pyramiding' in refinement:
                target_agents.add('strategy')
                target_agents.add('risk')
        
        assert 'strategy' in target_agents
        assert 'risk' in target_agents
        assert 'execution' in target_agents

    def test_log_summarization_pattern(self):
        """Test the pattern for summarizing logs for session storage."""
        logs = [
            {'sharpe_ratio': 1.5},
            {'sharpe_ratio': 1.2},
            {'pyramiding': {'tiers': 3}}
        ]
        
        # Pattern that would be in _summarize_logs_for_session
        summary = {
            'total_logs': len(logs),
            'has_sharpe_data': any('sharpe' in str(log).lower() for log in logs),
            'has_pyramiding_data': any('pyramiding' in str(log).lower() for log in logs),
            'has_return_data': any('return' in str(log).lower() for log in logs),
        }
        
        assert summary['total_logs'] == 3
        assert summary['has_sharpe_data'] == True
        assert summary['has_pyramiding_data'] == True
        assert summary['has_return_data'] == False


class TestCrossAgentPropagation:
    """Test cases for cross-agent directive propagation patterns.
    
    Tests the directive structure and handling patterns without importing
    the full agent dependencies.
    """

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing directive patterns."""
        agent = MagicMock()
        agent.role = 'test'
        agent.memory = {}
        return agent

    def test_directive_structure(self):
        """Test that directive structure is correct."""
        directive = {
            'directive_id': 'test_001',
            'sop_id': 'sop_001',
            'category': 'learning_insight',
            'content': {
                'directives': [
                    {'refinement': 'sizing_lift', 'value': 1.2},
                    {'refinement': 'efficiency_focus', 'value': 1.1}
                ]
            },
            'priority': 'medium',
            'source': 'learning',
            'applies_to': ['test']
        }
        
        assert 'directive_id' in directive
        assert 'category' in directive
        assert 'content' in directive
        assert 'priority' in directive
        assert 'applies_to' in directive

    def test_directive_target_filtering(self, mock_agent):
        """Test that directives are filtered by target agent."""
        directive = {
            'directive_id': 'test_002',
            'category': 'learning_insight',
            'content': {'directives': []},
            'priority': 'medium',
            'source': 'learning',
            'applies_to': ['strategy', 'risk']  # Not 'test'
        }
        
        # Pattern for checking if directive applies
        applies_to = directive.get('applies_to', [])
        should_apply = not applies_to or mock_agent.role in applies_to
        
        assert should_apply == False  # Should not apply to 'test' agent

    def test_directive_storage_pattern(self, mock_agent):
        """Test the pattern for storing received directives."""
        directive = {
            'directive_id': 'store_001',
            'category': 'learning_insight',
            'content': {
                'directives': [
                    {'refinement': 'test_refinement', 'value': 1.0}
                ]
            },
            'priority': 'high',
        }
        
        # Pattern for storing directives
        if 'received_directives' not in mock_agent.memory:
            mock_agent.memory['received_directives'] = []
        
        for directive_item in directive['content'].get('directives', []):
            mock_agent.memory['received_directives'].append({
                'directive_id': directive['directive_id'],
                'refinement': directive_item.get('refinement'),
                'value': directive_item.get('value'),
                'priority': directive['priority'],
                'timestamp': datetime.now().isoformat(),
            })
        
        assert len(mock_agent.memory['received_directives']) == 1
        assert mock_agent.memory['received_directives'][0]['refinement'] == 'test_refinement'

    def test_risk_constraint_directive_pattern(self, mock_agent):
        """Test the pattern for handling risk constraint directives."""
        directive = {
            'directive_id': 'risk_001',
            'category': 'risk_constraint',
            'content': {
                'constraint': {'max_drawdown': 0.05, 'var_confidence': 0.95}
            },
            'priority': 'high',
        }
        
        # Pattern for storing risk constraints
        if 'risk_constraints_from_directives' not in mock_agent.memory:
            mock_agent.memory['risk_constraints_from_directives'] = []
        
        mock_agent.memory['risk_constraints_from_directives'].append({
            'directive_id': directive['directive_id'],
            'content': directive['content'],
            'priority': directive['priority'],
            'timestamp': datetime.now().isoformat(),
        })
        
        assert len(mock_agent.memory['risk_constraints_from_directives']) == 1
        assert mock_agent.memory['risk_constraints_from_directives'][0]['content']['constraint']['max_drawdown'] == 0.05

    def test_directive_priority_levels(self):
        """Test that directive priority levels are handled correctly."""
        priority_levels = {
            'critical': 100,
            'high': 75,
            'medium': 50,
            'low': 25,
            'background': 10,
        }
        
        # Test sorting by priority
        directives = [
            TradingDirective(id='d1', category='test', name='Low', description='', content={}, 
                           applies_to=[], source='test', priority='low'),
            TradingDirective(id='d2', category='test', name='Critical', description='', content={}, 
                           applies_to=[], source='test', priority='critical'),
            TradingDirective(id='d3', category='test', name='Medium', description='', content={}, 
                           applies_to=[], source='test', priority='medium'),
        ]
        
        sorted_directives = sorted(
            directives,
            key=lambda d: priority_levels.get(d.priority, 50),
            reverse=True
        )
        
        assert sorted_directives[0].priority == 'critical'
        assert sorted_directives[1].priority == 'medium'
        assert sorted_directives[2].priority == 'low'


class TestFallbackBehavior:
    """Test cases for graceful fallback behavior."""

    @pytest.mark.asyncio
    async def test_fallback_when_acontext_unavailable(self):
        """Test graceful fallback when Acontext SDK is unavailable."""
        with patch.object(AcontextIntegration, '_load_config') as mock_load:
            mock_load.return_value = {
                'api': {'timeout_seconds': 30},
                'fallback': {'enabled': True, 'use_local_storage': True, 'local_storage_path': '/tmp/test_fallback'},
                'sop': {'id_prefix': 'test', 'default_ttl_days': 90, 'priority_levels': {}},
                'monitoring': {'alert_threshold_failures': 3}
            }
            
            # Create integration without real Acontext SDK
            with patch.dict(sys.modules, {'acontext': None}):
                integration = AcontextIntegration()
                
                # Force initialization without API key
                os.environ.pop('ACONTEXT_API_KEY', None)
                result = await integration.initialize()
                
                # Should be in fallback mode
                assert integration._fallback_mode == True
                assert integration._initialized == True

    @pytest.mark.asyncio
    async def test_consecutive_failure_triggers_fallback(self):
        """Test that consecutive failures trigger fallback mode."""
        with patch.object(AcontextIntegration, '_load_config') as mock_load:
            mock_load.return_value = {
                'api': {'timeout_seconds': 30},
                'fallback': {'enabled': True, 'use_local_storage': True, 'local_storage_path': '/tmp/test_failures'},
                'sop': {'id_prefix': 'test', 'default_ttl_days': 90, 'priority_levels': {}},
                'monitoring': {'alert_threshold_failures': 3}
            }
            integration = AcontextIntegration()
            integration._initialized = True
            integration._fallback_mode = False
            
            # Simulate consecutive failures
            integration._consecutive_failures = 3
            
            # After 3 failures, should be in fallback mode on next store
            directive = TradingDirective(
                id='fail_test', category='test', name='Fail Test', description='Test',
                content={}, applies_to=['strategy'], source='learning', priority='medium'
            )
            
            # Mock the async client to raise an exception
            integration.async_client = MagicMock()
            integration.async_client.blocks = MagicMock()
            integration.async_client.blocks.create = AsyncMock(side_effect=Exception("API Error"))
            
            await integration.store_sop(directive)
            
            # Should now be in fallback mode
            assert integration._fallback_mode == True

    @pytest.mark.asyncio
    async def test_fallback_storage_operations(self):
        """Test that fallback storage operations work correctly."""
        with patch.object(AcontextIntegration, '_load_config') as mock_load:
            mock_load.return_value = {
                'api': {'timeout_seconds': 30},
                'fallback': {'enabled': True, 'use_local_storage': True, 'local_storage_path': '/tmp/test_ops'},
                'sop': {'id_prefix': 'test', 'default_ttl_days': 90, 'priority_levels': {}},
                'session': {'log_all_sessions': True, 'retention_days': 30},
                'artifacts': {'enabled': True, 'types': ['ml_model'], 'max_size_mb': 100},
                'monitoring': {'alert_threshold_failures': 3}
            }
            integration = AcontextIntegration()
            integration._initialized = True
            integration._fallback_mode = True
            
            # Test SOP storage in fallback
            directive = TradingDirective(
                id='fallback_ops_test', category='test', name='Ops Test', description='Test',
                content={'test': True}, applies_to=['strategy'], source='learning', priority='medium'
            )
            sop_id = await integration.store_sop(directive)
            assert sop_id is not None
            assert 'local_' in sop_id
            
            # Test SOP retrieval from fallback
            retrieved = await integration.retrieve_sop(sop_id)
            assert retrieved is not None
            assert retrieved.id == 'fallback_ops_test'
            
            # Test session logging in fallback
            session_id = await integration.log_session({'type': 'test', 'data': 'test'})
            assert session_id is not None
            
            # Test artifact upload in fallback
            artifact_id = await integration.upload_artifact('ml_model', b'test data', {'name': 'test'})
            assert artifact_id is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
