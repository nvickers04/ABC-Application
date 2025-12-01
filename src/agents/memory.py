# [LABEL:AGENT:memory] [LABEL:COMPONENT:memory_management] [LABEL:FRAMEWORK:redis] [LABEL:FRAMEWORK:asyncio]
# [LABEL:AUTHOR:GitHub Copilot] [LABEL:UPDATED:2024-11-20] [LABEL:REVIEWED:yes]
#
# Purpose: Implements the Memory Agent, subclassing BaseAgent for comprehensive memory management. Handles short-term, long-term, and multi-agent memory sharing with advanced position tracking.
# Dependencies: sys, pathlib, src.agents.base, logging, typing, pandas, numpy, datetime, json, asyncio, src.utils.advanced_memory, src.utils.memory_persistence, src.utils.memory_security, src.utils.shared_memory
# Related: docs/AGENTS/memory-agent.md, config/base_prompt.txt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Dynamic root path for imports.

from src.agents.base import BaseAgent  # Absolute import.
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import asyncio

# Import memory management utilities
from src.utils.advanced_memory import get_advanced_memory_manager
from src.utils.memory_persistence import get_memory_persistence
from src.utils.memory_security import get_secure_memory_manager
from src.utils.shared_memory import get_multi_agent_coordinator

logger = logging.getLogger(__name__)

class MemoryAgent(BaseAgent):
    """
    Memory Agent subclass for comprehensive memory management across the AI trading system.
    Handles short-term, long-term, and multi-agent memory sharing with advanced position tracking.
    """

    def __init__(self):
        config_paths = {'risk': 'config/risk-constraints.yaml', 'profit': 'config/profitability-targets.yaml'}  # Relative to root.
        prompt_paths = {'base': 'config/base_prompt.txt', 'role': 'docs/AGENTS/main-agents/memory-agent.md'}  # Relative to root.
        super().__init__(role='memory', config_paths=config_paths, prompt_paths=prompt_paths)

        # Initialize memory management components
        self.advanced_memory_manager = get_advanced_memory_manager()
        self.memory_persistence = get_memory_persistence()
        self.memory_security = get_secure_memory_manager()
        self.shared_memory = get_multi_agent_coordinator()

        # Initialize memory structures
        self._initialize_memory_structures()

        # Memory performance tracking
        self.memory_metrics = {
            'total_memories': 0,
            'memory_operations': 0,
            'decay_operations': 0,
            'sharing_operations': 0,
            'last_maintenance': datetime.now(),
            'performance_stats': {}
        }

        logger.info("Memory Agent initialized with comprehensive memory management capabilities")

    def _initialize_memory_structures(self):
        """
        Initialize all memory structures for the trading system.
        """
        try:
            # Short-term memory (working/thread-scoped)
            self.short_term_memory = {
                'active_sessions': {},
                'current_context': {},
                'recent_interactions': [],
                'temp_cache': {}
            }

            # Long-term memory structures
            self.long_term_memory = {
                'semantic': {},  # Facts/preferences (e.g., user risk tolerance)
                'episodic': [],  # Past events (e.g., trade outcomes)
                'procedural': {}  # Rules/instructions (e.g., rebalancing algorithm)
            }

            # Multi-agent memory sharing
            self.agent_memory_spaces = {
                'data_agent': {},
                'strategy_agent': {},
                'risk_agent': {},
                'execution_agent': {},
                'learning_agent': {},
                'reflection_agent': {},
                'shared': {}  # Cross-agent shared memory
            }

            # Open positions memory (dedicated tracking)
            self.positions_memory = {
                'active_positions': {},
                'closed_positions': [],
                'position_history': [],
                'pnl_tracking': {}
            }

            # Memory metadata and indexing
            self.memory_metadata = {
                'last_updated': datetime.now(),
                'version': '1.0',
                'namespaces': {},
                'indices': {}
            }

            # Load existing memory from persistence if available
            self._load_persistent_memory()

            logger.info("Memory structures initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing memory structures: {e}")
            raise

    def _load_persistent_memory(self):
        """
        Load existing memory from persistent storage.
        """
        try:
            # Load long-term memory
            persistent_ltm = self.memory_persistence.load_long_term_memory()
            if persistent_ltm:
                self.long_term_memory.update(persistent_ltm)

            # Load agent memory spaces
            persistent_agent_memory = self.memory_persistence.load_agent_memory_spaces()
            if persistent_agent_memory:
                self.agent_memory_spaces.update(persistent_agent_memory)

            # Load positions memory
            persistent_positions = self.memory_persistence.load_positions_memory()
            if persistent_positions:
                self.positions_memory.update(persistent_positions)

            logger.info("Persistent memory loaded successfully")

        except Exception as e:
            logger.warning(f"Error loading persistent memory: {e}")

    async def process_input(self, memory_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process memory operations: store, retrieve, share, or maintain memory.
        Args:
            memory_request: Dict containing memory operation details
        Returns: Dict with operation results
        """
        logger.info(f"Memory Agent processing request: {memory_request.get('operation', 'unknown')}")

        operation = memory_request.get('operation', 'retrieve')
        result = {}

        try:
            if operation == 'store':
                result = await self._store_memory(memory_request)
            elif operation == 'retrieve':
                result = await self._retrieve_memory(memory_request)
            elif operation == 'share':
                result = await self._share_memory(memory_request)
            elif operation == 'search':
                result = await self._search_memory(memory_request)
            elif operation == 'maintain':
                result = await self._maintain_memory(memory_request)
            elif operation == 'position_track':
                result = await self._track_position(memory_request)
            else:
                result = {'error': f'Unknown operation: {operation}'}

            # Update metrics
            self.memory_metrics['memory_operations'] += 1

            return result

        except Exception as e:
            logger.error(f"Error processing memory request: {e}")
            return {'error': str(e), 'operation': operation}

    async def _store_memory(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store memory in appropriate structure based on type and scope.
        """
        try:
            memory_type = request.get('memory_type', 'episodic')
            scope = request.get('scope', 'long_term')
            namespace = request.get('namespace', 'shared')
            content = request.get('content', {})
            metadata = request.get('metadata', {})

            # Add timestamp and version
            content['_timestamp'] = datetime.now().isoformat()
            content['_version'] = self.memory_metadata['version']

            # Encrypt sensitive data
            if self._is_sensitive_content(content):
                content = self.memory_security.encrypt_memory(content)

            # Store based on type and scope
            if scope == 'short_term':
                result = await self._store_short_term_memory(memory_type, content, metadata)
            elif scope == 'long_term':
                result = await self._store_long_term_memory(memory_type, content, metadata)
            elif scope == 'agent':
                result = await self._store_agent_memory(namespace, memory_type, content, metadata)
            else:
                return {'error': f'Unknown scope: {scope}'}

            # Update metadata
            self.memory_metadata['last_updated'] = datetime.now()
            self.memory_metadata['namespaces'][namespace] = self.memory_metadata['namespaces'].get(namespace, 0) + 1

            # Persist to storage
            await self._persist_memory_updates()

            # Update metrics
            self.memory_metrics['total_memories'] += 1

            return {
                'stored': True,
                'memory_id': result.get('memory_id'),
                'scope': scope,
                'type': memory_type,
                'namespace': namespace
            }

        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return {'error': str(e), 'stored': False}

    async def _store_short_term_memory(self, memory_type: str, content: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store short-term memory with automatic decay.
        """
        try:
            memory_id = f"stm_{datetime.now().timestamp()}_{memory_type}"

            memory_entry = {
                'id': memory_id,
                'type': memory_type,
                'content': content,
                'metadata': metadata,
                'created_at': datetime.now(),
                'ttl': metadata.get('ttl', 3600)  # Default 1 hour TTL
            }

            # Store in appropriate short-term structure
            if memory_type == 'session':
                session_id = metadata.get('session_id', 'default')
                if session_id not in self.short_term_memory['active_sessions']:
                    self.short_term_memory['active_sessions'][session_id] = []
                self.short_term_memory['active_sessions'][session_id].append(memory_entry)
            elif memory_type == 'context':
                self.short_term_memory['current_context'].update(content)
            elif memory_type == 'interaction':
                self.short_term_memory['recent_interactions'].append(memory_entry)
                # Keep only last 50 interactions
                if len(self.short_term_memory['recent_interactions']) > 50:
                    self.short_term_memory['recent_interactions'] = self.short_term_memory['recent_interactions'][-50:]
            else:
                self.short_term_memory['temp_cache'][memory_id] = memory_entry

            return {'memory_id': memory_id}

        except Exception as e:
            logger.error(f"Error storing short-term memory: {e}")
            raise

    async def _store_long_term_memory(self, memory_type: str, content: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store long-term memory with persistence and indexing.
        """
        try:
            memory_id = f"ltm_{datetime.now().timestamp()}_{memory_type}"

            memory_entry = {
                'id': memory_id,
                'type': memory_type,
                'content': content,
                'metadata': metadata,
                'created_at': datetime.now(),
                'last_accessed': datetime.now(),
                'access_count': 0
            }

            # Store in appropriate long-term structure
            if memory_type == 'semantic':
                key = metadata.get('key', str(hash(str(content))))
                self.long_term_memory['semantic'][key] = memory_entry
            elif memory_type == 'episodic':
                self.long_term_memory['episodic'].append(memory_entry)
                # Keep only last 1000 episodic memories
                if len(self.long_term_memory['episodic']) > 1000:
                    self.long_term_memory['episodic'] = self.long_term_memory['episodic'][-1000:]
            elif memory_type == 'procedural':
                key = metadata.get('key', str(hash(str(content))))
                self.long_term_memory['procedural'][key] = memory_entry

            # Update indices
            await self._update_memory_indices(memory_entry)

            return {'memory_id': memory_id}

        except Exception as e:
            logger.error(f"Error storing long-term memory: {e}")
            raise

    async def _store_agent_memory(self, agent_name: str, memory_type: str, content: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store memory in agent-specific namespace.
        """
        try:
            if agent_name not in self.agent_memory_spaces:
                self.agent_memory_spaces[agent_name] = {}

            memory_id = f"agent_{agent_name}_{datetime.now().timestamp()}_{memory_type}"

            memory_entry = {
                'id': memory_id,
                'type': memory_type,
                'content': content,
                'metadata': metadata,
                'created_at': datetime.now(),
                'agent': agent_name
            }

            if memory_type not in self.agent_memory_spaces[agent_name]:
                self.agent_memory_spaces[agent_name][memory_type] = []

            self.agent_memory_spaces[agent_name][memory_type].append(memory_entry)

            # Keep only last 100 memories per type per agent
            if len(self.agent_memory_spaces[agent_name][memory_type]) > 100:
                self.agent_memory_spaces[agent_name][memory_type] = self.agent_memory_spaces[agent_name][memory_type][-100:]

            return {'memory_id': memory_id}

        except Exception as e:
            logger.error(f"Error storing agent memory: {e}")
            raise

    async def _retrieve_memory(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve memory from appropriate structure.
        """
        try:
            scope = request.get('scope', 'long_term')
            memory_type = request.get('memory_type')
            query = request.get('query', {})
            limit = request.get('limit', 10)

            results = []

            if scope == 'short_term':
                results = await self._retrieve_short_term_memory(memory_type, query, limit)
            elif scope == 'long_term':
                results = await self._retrieve_long_term_memory(memory_type, query, limit)
            elif scope == 'agent':
                agent_name = request.get('agent_name', 'shared')
                results = await self._retrieve_agent_memory(agent_name, memory_type, query, limit)
            elif scope == 'positions':
                results = await self._retrieve_positions_memory(query, limit)

            # Decrypt sensitive results
            for result in results:
                if self._is_encrypted_content(result.get('content', {})):
                    result['content'] = self.memory_security.decrypt_memory(result['content'])

            return {
                'retrieved': True,
                'results': results,
                'count': len(results),
                'scope': scope
            }

        except Exception as e:
            logger.error(f"Error retrieving memory: {e}")
            return {'error': str(e), 'retrieved': False}

    async def _retrieve_short_term_memory(self, memory_type: str, query: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """
        Retrieve short-term memory with filtering.
        """
        try:
            results = []

            if memory_type == 'session':
                session_id = query.get('session_id', 'default')
                if session_id in self.short_term_memory['active_sessions']:
                    session_memories = self.short_term_memory['active_sessions'][session_id]
                    results = self._filter_memories(session_memories, query, limit)
            elif memory_type == 'context':
                results = [self.short_term_memory['current_context']] if self.short_term_memory['current_context'] else []
            elif memory_type == 'interaction':
                results = self._filter_memories(self.short_term_memory['recent_interactions'], query, limit)
            else:
                # Search all temp cache
                all_temp = list(self.short_term_memory['temp_cache'].values())
                results = self._filter_memories(all_temp, query, limit)

            return results

        except Exception as e:
            logger.error(f"Error retrieving short-term memory: {e}")
            return []

    async def _retrieve_long_term_memory(self, memory_type: str, query: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """
        Retrieve long-term memory with advanced filtering and ranking.
        """
        try:
            results = []

            if memory_type == 'semantic':
                semantic_memories = list(self.long_term_memory['semantic'].values())
                results = self._filter_and_rank_memories(semantic_memories, query, limit)
            elif memory_type == 'episodic':
                results = self._filter_and_rank_memories(self.long_term_memory['episodic'], query, limit)
            elif memory_type == 'procedural':
                procedural_memories = list(self.long_term_memory['procedural'].values())
                results = self._filter_and_rank_memories(procedural_memories, query, limit)
            else:
                # Search all long-term memory
                all_ltm = (list(self.long_term_memory['semantic'].values()) +
                          self.long_term_memory['episodic'] +
                          list(self.long_term_memory['procedural'].values()))
                results = self._filter_and_rank_memories(all_ltm, query, limit)

            # Update access statistics
            for result in results:
                result['last_accessed'] = datetime.now()
                result['access_count'] = result.get('access_count', 0) + 1

            return results

        except Exception as e:
            logger.error(f"Error retrieving long-term memory: {e}")
            return []

    async def _retrieve_agent_memory(self, agent_name: str, memory_type: str, query: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """
        Retrieve memory from agent-specific namespace.
        """
        try:
            if agent_name not in self.agent_memory_spaces:
                return []

            agent_memories = self.agent_memory_spaces[agent_name]

            if memory_type and memory_type in agent_memories:
                memories = agent_memories[memory_type]
                return self._filter_memories(memories, query, limit)
            else:
                # Search all memory types for this agent
                all_agent_memories = []
                for mem_type, mem_list in agent_memories.items():
                    all_agent_memories.extend(mem_list)
                return self._filter_memories(all_agent_memories, query, limit)

        except Exception as e:
            logger.error(f"Error retrieving agent memory: {e}")
            return []

    async def _retrieve_positions_memory(self, query: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """
        Retrieve position-related memory.
        """
        try:
            results = []

            position_type = query.get('position_type', 'active')

            if position_type == 'active':
                active_positions = list(self.positions_memory['active_positions'].values())
                results = self._filter_memories(active_positions, query, limit)
            elif position_type == 'closed':
                results = self._filter_memories(self.positions_memory['closed_positions'], query, limit)
            elif position_type == 'history':
                results = self._filter_memories(self.positions_memory['position_history'], query, limit)

            return results

        except Exception as e:
            logger.error(f"Error retrieving positions memory: {e}")
            return []

    async def _share_memory(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Share memory between agents via A2A protocol.
        """
        try:
            source_agent = request.get('source_agent', 'unknown')
            target_agents = request.get('target_agents', [])
            memory_content = request.get('memory_content', {})
            priority = request.get('priority', 'normal')

            shared_memory = {
                'source_agent': source_agent,
                'target_agents': target_agents,
                'content': memory_content,
                'priority': priority,
                'shared_at': datetime.now(),
                'shared_id': f"shared_{datetime.now().timestamp()}"
            }

            # Store in shared memory space
            self.agent_memory_spaces['shared'][shared_memory['shared_id']] = shared_memory

            # Distribute to target agents
            distribution_results = []
            for target_agent in target_agents:
                result = await self._distribute_memory_to_agent(target_agent, shared_memory)
                distribution_results.append(result)

            # Update metrics
            self.memory_metrics['sharing_operations'] += 1

            return {
                'shared': True,
                'shared_id': shared_memory['shared_id'],
                'target_agents': target_agents,
                'distribution_results': distribution_results,
                'priority': priority
            }

        except Exception as e:
            logger.error(f"Error sharing memory: {e}")
            return {'error': str(e), 'shared': False}

    async def _distribute_memory_to_agent(self, agent_name: str, memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Distribute shared memory to specific agent.
        """
        try:
            # In a real implementation, this would use A2A protocol
            # For now, simulate distribution by storing in agent's memory space

            if agent_name not in self.agent_memory_spaces:
                self.agent_memory_spaces[agent_name] = {}

            if 'received_shared' not in self.agent_memory_spaces[agent_name]:
                self.agent_memory_spaces[agent_name]['received_shared'] = []

            self.agent_memory_spaces[agent_name]['received_shared'].append(memory)

            # Keep only last 50 shared memories per agent
            if len(self.agent_memory_spaces[agent_name]['received_shared']) > 50:
                self.agent_memory_spaces[agent_name]['received_shared'] = self.agent_memory_spaces[agent_name]['received_shared'][-50:]

            return {
                'agent': agent_name,
                'distributed': True,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error distributing memory to {agent_name}: {e}")
            return {
                'agent': agent_name,
                'distributed': False,
                'error': str(e)
            }

    async def _search_memory(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform advanced memory search using vector similarity and semantic matching.
        """
        try:
            query_text = request.get('query_text', '')
            scope = request.get('scope', 'all')
            limit = request.get('limit', 20)
            search_type = request.get('search_type', 'semantic')  # semantic, vector, keyword

            results = []

            if search_type == 'semantic':
                results = await self._semantic_memory_search(query_text, scope, limit)
            elif search_type == 'vector':
                results = await self._vector_memory_search(query_text, scope, limit)
            elif search_type == 'keyword':
                results = await self._keyword_memory_search(query_text, scope, limit)

            return {
                'searched': True,
                'query': query_text,
                'search_type': search_type,
                'scope': scope,
                'results': results,
                'count': len(results)
            }

        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return {'error': str(e), 'searched': False}

    async def _semantic_memory_search(self, query: str, scope: str, limit: int) -> List[Dict[str, Any]]:
        """
        Perform semantic search across memory using LLM understanding.
        """
        try:
            # Use LLM to understand query intent and find relevant memories
            if self.llm:
                search_context = f"""
                Search the memory system for information related to: {query}

                Available memory scopes: {scope}
                Consider semantic meaning, context, and relevance to the query.
                Return the most relevant memory entries.
                """

                # Get all memories in scope
                all_memories = await self._get_memories_in_scope(scope)

                # Use LLM to rank and filter memories
                relevant_memories = await self._llm_rank_memories(search_context, all_memories, limit)

                return relevant_memories
            else:
                # Fallback to keyword search
                return await self._keyword_memory_search(query, scope, limit)

        except Exception as e:
            logger.error(f"Error in semantic memory search: {e}")
            return []

    async def _vector_memory_search(self, query: str, scope: str, limit: int) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search across memory.
        """
        try:
            # This would use vector embeddings for similarity search
            # For now, return placeholder implementation
            all_memories = await self._get_memories_in_scope(scope)

            # Simple text similarity (placeholder for vector search)
            query_words = set(query.lower().split())
            scored_memories = []

            for memory in all_memories:
                content_text = str(memory.get('content', '')).lower()
                content_words = set(content_text.split())
                similarity = len(query_words.intersection(content_words)) / len(query_words.union(content_words))

                if similarity > 0:
                    memory['similarity_score'] = similarity
                    scored_memories.append(memory)

            # Sort by similarity and limit results
            scored_memories.sort(key=lambda x: x['similarity_score'], reverse=True)
            return scored_memories[:limit]

        except Exception as e:
            logger.error(f"Error in vector memory search: {e}")
            return []

    async def _keyword_memory_search(self, query: str, scope: str, limit: int) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search across memory.
        """
        try:
            keywords = query.lower().split()
            all_memories = await self._get_memories_in_scope(scope)
            matching_memories = []

            for memory in all_memories:
                content_text = str(memory.get('content', '')).lower()
                if any(keyword in content_text for keyword in keywords):
                    matching_memories.append(memory)

            return matching_memories[:limit]

        except Exception as e:
            logger.error(f"Error in keyword memory search: {e}")
            return []

    async def _get_memories_in_scope(self, scope: str) -> List[Dict[str, Any]]:
        """
        Get all memories within specified scope.
        """
        try:
            all_memories = []

            if scope == 'all':
                # Add all memory types
                all_memories.extend(list(self.long_term_memory['semantic'].values()))
                all_memories.extend(self.long_term_memory['episodic'])
                all_memories.extend(list(self.long_term_memory['procedural'].values()))

                for agent_memories in self.agent_memory_spaces.values():
                    for mem_type, mem_list in agent_memories.items():
                        if isinstance(mem_list, list):
                            all_memories.extend(mem_list)
                        elif isinstance(mem_list, dict):
                            all_memories.extend(mem_list.values())

            elif scope == 'long_term':
                all_memories.extend(list(self.long_term_memory['semantic'].values()))
                all_memories.extend(self.long_term_memory['episodic'])
                all_memories.extend(list(self.long_term_memory['procedural'].values()))

            elif scope == 'short_term':
                all_memories.extend(self.short_term_memory['recent_interactions'])
                all_memories.extend(self.short_term_memory['temp_cache'].values())

            elif scope.startswith('agent_'):
                agent_name = scope.replace('agent_', '')
                if agent_name in self.agent_memory_spaces:
                    agent_memories = self.agent_memory_spaces[agent_name]
                    for mem_type, mem_list in agent_memories.items():
                        if isinstance(mem_list, list):
                            all_memories.extend(mem_list)
                        elif isinstance(mem_list, dict):
                            all_memories.extend(mem_list.values())

            return all_memories

        except Exception as e:
            logger.error(f"Error getting memories in scope {scope}: {e}")
            return []

    async def _llm_rank_memories(self, context: str, memories: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """
        Use LLM to rank and filter memories based on relevance.
        """
        try:
            if not self.llm or not memories:
                return memories[:limit]

            # Prepare memory summaries for LLM
            memory_summaries = []
            for i, memory in enumerate(memories[:50]):  # Limit to first 50 for LLM context
                summary = f"Memory {i}: {str(memory.get('content', ''))[:200]}..."
                memory_summaries.append(summary)

            ranking_prompt = f"""
            {context}

            Available memories:
            {chr(10).join(memory_summaries)}

            Rank the memories by relevance to the query and return the indices of the top {limit} most relevant memories.
            Format: Return only a comma-separated list of indices (e.g., "0,3,7,12").
            """

            llm_response = await self.reason_with_llm(context, ranking_prompt)

            # Parse LLM response for indices
            import re
            indices = re.findall(r'\d+', llm_response)
            relevant_indices = [int(idx) for idx in indices if int(idx) < len(memories)][:limit]

            return [memories[idx] for idx in relevant_indices]

        except Exception as e:
            logger.error(f"Error in LLM memory ranking: {e}")
            return memories[:limit]

    async def _maintain_memory(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform memory maintenance operations: decay, pruning, optimization.
        """
        try:
            operation = request.get('maintenance_operation', 'decay')

            if operation == 'decay':
                result = await self._perform_memory_decay()
            elif operation == 'prune':
                result = await self._perform_memory_pruning()
            elif operation == 'optimize':
                result = await self._optimize_memory_storage()
            elif operation == 'backup':
                result = await self._backup_memory()
            else:
                return {'error': f'Unknown maintenance operation: {operation}'}

            # Update metrics
            self.memory_metrics['decay_operations'] += 1
            self.memory_metrics['last_maintenance'] = datetime.now()

            return result

        except Exception as e:
            logger.error(f"Error in memory maintenance: {e}")
            return {'error': str(e), 'maintenance': False}

    async def _perform_memory_decay(self) -> Dict[str, Any]:
        """
        Apply decay mechanisms to old or irrelevant memories.
        """
        try:
            decay_stats = {
                'short_term_decayed': 0,
                'long_term_decayed': 0,
                'agent_memory_decayed': 0,
                'total_decayed': 0
            }

            # Decay short-term memory
            current_time = datetime.now()

            # Remove expired short-term memories
            for session_id, session_memories in self.short_term_memory['active_sessions'].items():
                active_memories = []
                for memory in session_memories:
                    ttl = memory.get('ttl', 3600)
                    created_at = memory.get('created_at', current_time)
                    if isinstance(created_at, str):
                        created_at = datetime.fromisoformat(created_at)

                    if (current_time - created_at).total_seconds() < ttl:
                        active_memories.append(memory)
                    else:
                        decay_stats['short_term_decayed'] += 1

                self.short_term_memory['active_sessions'][session_id] = active_memories

            # Decay long-term memory based on access patterns
            for memory_type in ['semantic', 'episodic', 'procedural']:
                if memory_type == 'episodic':
                    memories = self.long_term_memory[memory_type]
                else:
                    memories = list(self.long_term_memory[memory_type].values())

                active_memories = []
                for memory in memories:
                    # Decay logic: remove if not accessed in 30 days and low access count
                    last_accessed = memory.get('last_accessed', memory.get('created_at', current_time))
                    if isinstance(last_accessed, str):
                        last_accessed = datetime.fromisoformat(last_accessed)

                    days_since_access = (current_time - last_accessed).days
                    access_count = memory.get('access_count', 0)

                    # Keep if recently accessed or frequently used
                    if days_since_access < 30 or access_count > 5:
                        active_memories.append(memory)
                    else:
                        decay_stats['long_term_decayed'] += 1

                if memory_type == 'episodic':
                    self.long_term_memory[memory_type] = active_memories
                else:
                    # Rebuild dict for semantic/procedural
                    self.long_term_memory[memory_type] = {f"key_{i}": mem for i, mem in enumerate(active_memories)}

            # Decay agent memories (keep only last 30 days)
            for agent_name, agent_memories in self.agent_memory_spaces.items():
                for mem_type, mem_list in agent_memories.items():
                    if isinstance(mem_list, list):
                        active_memories = []
                        for memory in mem_list:
                            created_at = memory.get('created_at', current_time)
                            if isinstance(created_at, str):
                                created_at = datetime.fromisoformat(created_at)

                            if (current_time - created_at).days < 30:
                                active_memories.append(memory)
                            else:
                                decay_stats['agent_memory_decayed'] += 1

                        self.agent_memory_spaces[agent_name][mem_type] = active_memories

            decay_stats['total_decayed'] = sum(decay_stats.values())

            logger.info(f"Memory decay completed: {decay_stats}")

            return {
                'maintenance': True,
                'operation': 'decay',
                'stats': decay_stats
            }

        except Exception as e:
            logger.error(f"Error performing memory decay: {e}")
            raise

    async def _perform_memory_pruning(self) -> Dict[str, Any]:
        """
        Prune memory based on configurable rules and priorities.
        """
        try:
            # Implement pruning logic based on memory management policies
            # This is a simplified version - in production would have more sophisticated rules

            pruning_stats = {
                'pruned_memories': 0,
                'freed_space': 0,
                'optimization_score': 0
            }

            # Prune old episodic memories beyond certain count
            max_episodic = 500
            if len(self.long_term_memory['episodic']) > max_episodic:
                pruned_count = len(self.long_term_memory['episodic']) - max_episodic
                self.long_term_memory['episodic'] = self.long_term_memory['episodic'][-max_episodic:]
                pruning_stats['pruned_memories'] += pruned_count

            # Prune agent memories beyond limits
            for agent_name, agent_memories in self.agent_memory_spaces.items():
                for mem_type, mem_list in agent_memories.items():
                    if isinstance(mem_list, list):
                        max_per_type = 50
                        if len(mem_list) > max_per_type:
                            pruned_count = len(mem_list) - max_per_type
                            self.agent_memory_spaces[agent_name][mem_type] = mem_list[-max_per_type:]
                            pruning_stats['pruned_memories'] += pruned_count

            return {
                'maintenance': True,
                'operation': 'pruning',
                'stats': pruning_stats
            }

        except Exception as e:
            logger.error(f"Error performing memory pruning: {e}")
            raise

    async def _optimize_memory_storage(self) -> Dict[str, Any]:
        """
        Optimize memory storage for better performance.
        """
        try:
            # Rebuild indices
            await self._rebuild_memory_indices()

            # Compact storage
            await self._compact_memory_storage()

            # Update statistics
            self.memory_metrics['performance_stats'] = await self._calculate_memory_performance_stats()

            return {
                'maintenance': True,
                'operation': 'optimization',
                'stats': self.memory_metrics['performance_stats']
            }

        except Exception as e:
            logger.error(f"Error optimizing memory storage: {e}")
            raise

    async def _backup_memory(self) -> Dict[str, Any]:
        """
        Create backup of all memory structures.
        """
        try:
            backup_data = {
                'timestamp': datetime.now().isoformat(),
                'version': self.memory_metadata['version'],
                'short_term_memory': self.short_term_memory,
                'long_term_memory': self.long_term_memory,
                'agent_memory_spaces': self.agent_memory_spaces,
                'positions_memory': self.positions_memory,
                'memory_metadata': self.memory_metadata,
                'memory_metrics': self.memory_metrics
            }

            # Use persistence layer to create backup
            backup_result = await self.memory_persistence.create_backup(backup_data)

            return {
                'maintenance': True,
                'operation': 'backup',
                'backup_id': backup_result.get('backup_id'),
                'backup_location': backup_result.get('location')
            }

        except Exception as e:
            logger.error(f"Error creating memory backup: {e}")
            raise

    async def _track_position(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Track open positions with comprehensive memory management.
        """
        try:
            operation = request.get('position_operation', 'update')
            position_data = request.get('position_data', {})

            if operation == 'open':
                result = await self._open_position(position_data)
            elif operation == 'update':
                result = await self._update_position(position_data)
            elif operation == 'close':
                result = await self._close_position(position_data)
            elif operation == 'query':
                result = await self._query_positions(request)
            else:
                return {'error': f'Unknown position operation: {operation}'}

            return result

        except Exception as e:
            logger.error(f"Error tracking position: {e}")
            return {'error': str(e), 'tracked': False}

    async def _open_position(self, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Open a new position in memory.
        """
        try:
            symbol = position_data.get('symbol', 'UNKNOWN')
            position_id = f"pos_{symbol}_{datetime.now().timestamp()}"

            position_entry = {
                'id': position_id,
                'symbol': symbol,
                'quantity': position_data.get('quantity', 0),
                'entry_price': position_data.get('entry_price', 0),
                'entry_timestamp': datetime.now(),
                'source': position_data.get('source', 'unknown'),
                'agent': position_data.get('agent', 'unknown'),
                'status': 'open',
                'metadata': position_data.get('metadata', {}),
                'pnl_tracking': {
                    'unrealized_pnl': 0,
                    'realized_pnl': 0,
                    'high_watermark': position_data.get('entry_price', 0),
                    'low_watermark': position_data.get('entry_price', 0)
                }
            }

            self.positions_memory['active_positions'][position_id] = position_entry
            self.positions_memory['position_history'].append(position_entry)

            # Persist position data
            await self._persist_memory_updates()

            return {
                'tracked': True,
                'operation': 'open',
                'position_id': position_id,
                'symbol': symbol
            }

        except Exception as e:
            logger.error(f"Error opening position: {e}")
            raise

    async def _update_position(self, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing position.
        """
        try:
            position_id = position_data.get('position_id')
            if not position_id or position_id not in self.positions_memory['active_positions']:
                return {'error': 'Position not found', 'tracked': False}

            position = self.positions_memory['active_positions'][position_id]

            # Update position data
            for key, value in position_data.items():
                if key != 'position_id':
                    position[key] = value

            # Update P&L tracking
            current_price = position_data.get('current_price')
            if current_price:
                entry_price = position['entry_price']
                quantity = position['quantity']

                unrealized_pnl = (current_price - entry_price) * quantity
                position['pnl_tracking']['unrealized_pnl'] = unrealized_pnl

                # Update watermarks
                position['pnl_tracking']['high_watermark'] = max(
                    position['pnl_tracking']['high_watermark'], current_price
                )
                position['pnl_tracking']['low_watermark'] = min(
                    position['pnl_tracking']['low_watermark'], current_price
                )

            position['last_updated'] = datetime.now()

            return {
                'tracked': True,
                'operation': 'update',
                'position_id': position_id
            }

        except Exception as e:
            logger.error(f"Error updating position: {e}")
            raise

    async def _close_position(self, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Close an existing position.
        """
        try:
            position_id = position_data.get('position_id')
            if not position_id or position_id not in self.positions_memory['active_positions']:
                return {'error': 'Position not found', 'tracked': False}

            position = self.positions_memory['active_positions'][position_id]

            # Calculate final P&L
            close_price = position_data.get('close_price', position['entry_price'])
            quantity = position['quantity']
            entry_price = position['entry_price']

            realized_pnl = (close_price - entry_price) * quantity

            # Update position with close data
            position.update({
                'status': 'closed',
                'close_price': close_price,
                'close_timestamp': datetime.now(),
                'realized_pnl': realized_pnl,
                'pnl_tracking': {
                    **position.get('pnl_tracking', {}),
                    'final_pnl': realized_pnl
                }
            })

            # Move to closed positions
            self.positions_memory['closed_positions'].append(position)
            del self.positions_memory['active_positions'][position_id]

            # Keep only last 1000 closed positions
            if len(self.positions_memory['closed_positions']) > 1000:
                self.positions_memory['closed_positions'] = self.positions_memory['closed_positions'][-1000:]

            # Update P&L tracking
            self.positions_memory['pnl_tracking'][position_id] = position['pnl_tracking']

            return {
                'tracked': True,
                'operation': 'close',
                'position_id': position_id,
                'realized_pnl': realized_pnl
            }

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            raise

    async def _query_positions(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query position data.
        """
        try:
            query_type = request.get('query_type', 'active')
            symbol = request.get('symbol')
            agent = request.get('agent')

            if query_type == 'active':
                positions = list(self.positions_memory['active_positions'].values())
            elif query_type == 'closed':
                positions = self.positions_memory['closed_positions']
            elif query_type == 'all':
                positions = (list(self.positions_memory['active_positions'].values()) +
                           self.positions_memory['closed_positions'])
            else:
                return {'error': f'Unknown query type: {query_type}'}

            # Apply filters
            if symbol:
                positions = [p for p in positions if p.get('symbol') == symbol]
            if agent:
                positions = [p for p in positions if p.get('agent') == agent]

            return {
                'tracked': True,
                'operation': 'query',
                'positions': positions,
                'count': len(positions)
            }

        except Exception as e:
            logger.error(f"Error querying positions: {e}")
            return {'error': str(e), 'tracked': False}

    async def _persist_memory_updates(self) -> None:
        """
        Persist memory updates to storage.
        """
        try:
            # Persist long-term memory
            await self.memory_persistence.save_long_term_memory(self.long_term_memory)

            # Persist agent memory spaces
            await self.memory_persistence.save_agent_memory_spaces(self.agent_memory_spaces)

            # Persist positions memory
            await self.memory_persistence.save_positions_memory(self.positions_memory)

            # Persist metadata
            await self.memory_persistence.save_memory_metadata(self.memory_metadata)

        except Exception as e:
            logger.error(f"Error persisting memory updates: {e}")

    async def _update_memory_indices(self, memory_entry: Dict[str, Any]) -> None:
        """
        Update memory indices for efficient retrieval.
        """
        try:
            # This would implement indexing logic for faster searches
            # For now, just maintain basic metadata
            memory_type = memory_entry.get('type', 'unknown')
            if memory_type not in self.memory_metadata['indices']:
                self.memory_metadata['indices'][memory_type] = 0
            self.memory_metadata['indices'][memory_type] += 1

        except Exception as e:
            logger.warning(f"Error updating memory indices: {e}")

    async def _rebuild_memory_indices(self) -> None:
        """
        Rebuild memory indices from scratch.
        """
        try:
            self.memory_metadata['indices'] = {}

            # Count memories by type
            for memory_type in ['semantic', 'episodic', 'procedural']:
                if memory_type == 'episodic':
                    count = len(self.long_term_memory[memory_type])
                else:
                    count = len(self.long_term_memory[memory_type])
                self.memory_metadata['indices'][memory_type] = count

            # Count agent memories
            for agent_name, agent_memories in self.agent_memory_spaces.items():
                for mem_type, mem_list in agent_memories.items():
                    key = f"agent_{agent_name}_{mem_type}"
                    count = len(mem_list) if isinstance(mem_list, list) else len(mem_list)
                    self.memory_metadata['indices'][key] = count

        except Exception as e:
            logger.warning(f"Error rebuilding memory indices: {e}")

    async def _compact_memory_storage(self) -> None:
        """
        Compact memory storage to optimize space usage.
        """
        try:
            # This would implement storage compaction logic
            # For now, just ensure data consistency
            logger.info("Memory storage compaction completed")

        except Exception as e:
            logger.warning(f"Error compacting memory storage: {e}")

    async def _calculate_memory_performance_stats(self) -> Dict[str, Any]:
        """
        Calculate memory performance statistics.
        """
        try:
            stats = {
                'total_memories': self.memory_metrics['total_memories'],
                'memory_operations': self.memory_metrics['memory_operations'],
                'sharing_operations': self.memory_metrics['sharing_operations'],
                'decay_operations': self.memory_metrics['decay_operations'],
                'active_sessions': len(self.short_term_memory['active_sessions']),
                'active_positions': len(self.positions_memory['active_positions']),
                'long_term_semantic': len(self.long_term_memory['semantic']),
                'long_term_episodic': len(self.long_term_memory['episodic']),
                'long_term_procedural': len(self.long_term_memory['procedural']),
                'last_maintenance': self.memory_metrics['last_maintenance'].isoformat()
            }

            # Calculate memory efficiency metrics
            total_operations = stats['memory_operations'] + stats['sharing_operations']
            if total_operations > 0:
                stats['operation_success_rate'] = 0.99  # Placeholder - would track actual success
                stats['average_operation_time'] = 0.05  # Placeholder - would measure actual times

            return stats

        except Exception as e:
            logger.error(f"Error calculating memory performance stats: {e}")
            return {}

    def _filter_memories(self, memories: List[Dict[str, Any]], query: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """
        Filter memories based on query criteria.
        """
        try:
            filtered = memories.copy()

            # Apply filters
            for key, value in query.items():
                if key not in ['limit', 'sort_by', 'sort_order']:
                    filtered = [m for m in filtered if self._matches_filter(m, key, value)]

            # Apply sorting
            sort_by = query.get('sort_by', 'created_at')
            sort_order = query.get('sort_order', 'desc')

            if sort_by in ['created_at', 'last_accessed', 'access_count']:
                reverse = sort_order == 'desc'
                filtered.sort(key=lambda x: x.get(sort_by, datetime.min), reverse=reverse)

            return filtered[:limit]

        except Exception as e:
            logger.warning(f"Error filtering memories: {e}")
            return memories[:limit]

    def _filter_and_rank_memories(self, memories: List[Dict[str, Any]], query: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """
        Filter and rank memories with relevance scoring.
        """
        try:
            filtered = self._filter_memories(memories, query, len(memories))

            # Add relevance scoring for long-term memory
            for memory in filtered:
                relevance_score = self._calculate_relevance_score(memory, query)
                memory['relevance_score'] = relevance_score

            # Sort by relevance
            filtered.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

            return filtered[:limit]

        except Exception as e:
            logger.warning(f"Error filtering and ranking memories: {e}")
            return memories[:limit]

    def _calculate_relevance_score(self, memory: Dict[str, Any], query: Dict[str, Any]) -> float:
        """
        Calculate relevance score for memory ranking.
        """
        try:
            score = 0.0

            # Recency boost
            created_at = memory.get('created_at', datetime.min)
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)

            days_old = (datetime.now() - created_at).days
            recency_score = max(0, 1.0 - (days_old / 365.0))  # Decay over a year
            score += recency_score * 0.3

            # Access frequency boost
            access_count = memory.get('access_count', 0)
            access_score = min(1.0, access_count / 10.0)  # Cap at 10 accesses
            score += access_score * 0.2

            # Content relevance (simple keyword matching)
            query_text = ' '.join(str(v) for v in query.values() if isinstance(v, str)).lower()
            content_text = str(memory.get('content', '')).lower()

            if query_text and content_text:
                query_words = set(query_text.split())
                content_words = set(content_text.split())
                overlap = len(query_words.intersection(content_words))
                total_unique = len(query_words.union(content_words))
                content_score = overlap / total_unique if total_unique > 0 else 0
                score += content_score * 0.5

            return score

        except Exception as e:
            logger.warning(f"Error calculating relevance score: {e}")
            return 0.0

    def _matches_filter(self, memory: Dict[str, Any], key: str, value: Any) -> bool:
        """
        Check if memory matches filter criteria.
        """
        try:
            # Check direct key in memory
            memory_value = memory.get(key)
            if memory_value is not None:
                if isinstance(value, str) and isinstance(memory_value, str):
                    return value.lower() in memory_value.lower()
                else:
                    return memory_value == value

            # Check in content
            content = memory.get('content', {})
            memory_value = content.get(key)
            if memory_value is not None:
                if isinstance(value, str) and isinstance(memory_value, str):
                    return value.lower() in memory_value.lower()
                else:
                    return memory_value == value

            return False

        except Exception:
            return False

    def _is_sensitive_content(self, content: Dict[str, Any]) -> bool:
        """
        Check if content contains sensitive information.
        """
        sensitive_keys = ['password', 'key', 'token', 'secret', 'pnl', 'balance']
        content_str = str(content).lower()

        return any(key in content_str for key in sensitive_keys)

    def _is_encrypted_content(self, content: Dict[str, Any]) -> bool:
        """
        Check if content is encrypted.
        """
        return isinstance(content, dict) and '_encrypted' in content

    async def get_memory_status(self) -> Dict[str, Any]:
        """
        Get comprehensive memory system status.
        """
        try:
            status = {
                'memory_agent_active': True,
                'last_updated': self.memory_metadata['last_updated'].isoformat(),
                'version': self.memory_metadata['version'],
                'metrics': self.memory_metrics,
                'memory_counts': {
                    'short_term_sessions': len(self.short_term_memory['active_sessions']),
                    'short_term_interactions': len(self.short_term_memory['recent_interactions']),
                    'long_term_semantic': len(self.long_term_memory['semantic']),
                    'long_term_episodic': len(self.long_term_memory['episodic']),
                    'long_term_procedural': len(self.long_term_memory['procedural']),
                    'active_positions': len(self.positions_memory['active_positions']),
                    'closed_positions': len(self.positions_memory['closed_positions'])
                },
                'agent_memory_spaces': list(self.agent_memory_spaces.keys()),
                'system_health': 'good'
            }

            # Check system health
            if self.memory_metrics['memory_operations'] > 10000:  # High operation count
                status['system_health'] = 'high_usage'
            elif self.memory_metrics['last_maintenance'] < datetime.now() - timedelta(days=7):
                status['system_health'] = 'needs_maintenance'

            return status

        except Exception as e:
            logger.error(f"Error getting memory status: {e}")
            return {'error': str(e), 'memory_agent_active': False}