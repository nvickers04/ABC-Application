# src/utils/shared_memory.py
# Purpose: Multi-agent memory sharing system
# Enables agents to share memories and coordinate through shared namespaces
# Implements A2A (Agent-to-Agent) protocols for memory exchange

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import json
import numpy as np
import pandas as pd

from src.utils.advanced_memory import get_advanced_memory_manager
from src.utils.memory_persistence import get_memory_persistence

logger = logging.getLogger(__name__)

def sanitize_for_json(data: Any, max_depth: int = 3, current_depth: int = 0, seen_objects: Optional[set] = None) -> Any:
    """
    Recursively sanitize data for JSON serialization by converting complex objects
    to simple serializable types, handling circular references and deep nesting.

    Args:
        data: Data to sanitize
        max_depth: Maximum recursion depth to prevent infinite loops
        current_depth: Current recursion depth
        seen_objects: Set of object IDs already processed to detect circular references

    Returns:
        JSON-serializable version of the data
    """
    if seen_objects is None:
        seen_objects = set()

    # Check for circular references
    if id(data) in seen_objects:
        return "<circular_reference>"

    if current_depth >= max_depth:
        return str(data)

    # Handle None
    if data is None:
        return None

    # Handle basic types
    if isinstance(data, (int, float, str, bool)):
        return data

    # Handle datetime objects
    if isinstance(data, datetime):
        return data.isoformat()

    # Handle numpy types
    if isinstance(data, (np.integer, np.floating, np.bool_)):
        return data.item()
    if isinstance(data, np.ndarray):
        return {
            "type": "ndarray",
            "shape": data.shape,
            "dtype": str(data.dtype),
            "data": data.flatten()[:10].tolist() if data.size > 0 else [],
            "size": int(data.size)
        }

    # Handle pandas DataFrames and Series
    if isinstance(data, pd.DataFrame):
        return {
            "type": "DataFrame",
            "columns": list(data.columns),
            "shape": data.shape,
            "data": data.head(5).to_dict('records') if len(data) > 0 else [],
            "summary": str(data.describe()) if len(data) > 0 else "Empty DataFrame"
        }
    elif isinstance(data, pd.Series):
        return {
            "type": "Series",
            "name": str(data.name),
            "data": data.head(5).tolist() if len(data) > 0 else [],
            "length": len(data),
            "summary": str(data.describe()) if len(data) > 0 else "Empty Series"
        }

    # Handle lists and tuples
    if isinstance(data, (list, tuple)):
        seen_objects.add(id(data))
        try:
            result = []
            for item in data:
                if current_depth < max_depth - 1:
                    result.append(sanitize_for_json(item, max_depth, current_depth + 1, seen_objects))
                else:
                    result.append(str(item))
            return result
        except RecursionError:
            return [str(item) for item in data]
        finally:
            seen_objects.discard(id(data))

    # Handle dictionaries
    if isinstance(data, dict):
        seen_objects.add(id(data))
        try:
            result = {}
            for k, v in data.items():
                key_str = str(k)
                if current_depth < max_depth - 1:
                    result[key_str] = sanitize_for_json(v, max_depth, current_depth + 1, seen_objects)
                else:
                    result[key_str] = str(v)
            return result
        except RecursionError:
            return {str(k): str(v) for k, v in data.items()}
        finally:
            seen_objects.discard(id(data))

    # Handle any other object - convert to string representation
    try:
        # Check if it's a complex object that might cause issues
        if hasattr(data, '__dict__'):
            seen_objects.add(id(data))
            try:
                # For objects with __dict__, create a safe representation
                obj_dict = {}
                for k, v in data.__dict__.items():
                    if not k.startswith('_'):  # Skip private attributes
                        try:
                            if current_depth < max_depth - 1:
                                obj_dict[k] = sanitize_for_json(v, max_depth, current_depth + 1, seen_objects)
                            else:
                                obj_dict[k] = str(v)
                        except:
                            obj_dict[k] = str(v)
                return {
                    "type": data.__class__.__name__,
                    "attributes": obj_dict
                }
            finally:
                seen_objects.discard(id(data))
        else:
            return str(data)
    except:
        # Fallback to string representation
        return str(data)

class SharedMemoryNamespace:
    """
    Represents a shared memory namespace that multiple agents can access.
    """

    def __init__(self, namespace: str, access_control: Optional[Dict[str, List[str]]] = None):
        """
        Initialize shared memory namespace.

        Args:
            namespace: Namespace identifier
            access_control: Dict mapping agent roles to allowed operations
        """
        self.namespace = namespace
        self.access_control = access_control or {}
        self.subscribers = {}  # agent_role -> callback function
        self.memory_manager = get_advanced_memory_manager()

    async def store_shared_memory(self, key: str, data: Any, agent_role: str,
                                memory_type: str = "shared") -> bool:
        """
        Store data in shared namespace with access control.

        Args:
            key: Memory key
            data: Data to store
            agent_role: Role of agent making the request
            memory_type: Type of memory

        Returns:
            bool: Success status
        """
        # Check access control
        if not self._check_write_access(agent_role):
            logger.warning(f"Agent {agent_role} denied write access to namespace {self.namespace}")
            return False

        try:
            # Sanitize data for JSON serialization to prevent recursion errors
            sanitized_data = sanitize_for_json(data)

            # Create full key with namespace
            full_key = f"shared:{self.namespace}:{key}"

            # Add sharing metadata
            metadata = {
                "shared_by": agent_role,
                "namespace": self.namespace,
                "shared_at": datetime.now().isoformat(),
                "access_level": "shared",
                "original_data_type": type(data).__name__
            }

            # Store in advanced memory system
            success = await self.memory_manager.store_memory(
                full_key, sanitized_data, memory_type, metadata
            )

            if success:
                # Notify subscribers with sanitized data
                await self._notify_subscribers("store", {
                    "key": key,
                    "data": sanitized_data,
                    "agent_role": agent_role,
                    "namespace": self.namespace
                })

                logger.info(f"Agent {agent_role} stored shared memory in {self.namespace}:{key}")

            return success
        except Exception as e:
            logger.error(f"Failed to store shared memory {self.namespace}:{key}: {e}")
            return False

    async def retrieve_shared_memory(self, key: str, agent_role: str) -> Optional[Any]:
        """
        Retrieve data from shared namespace with access control.

        Args:
            key: Memory key
            agent_role: Role of agent making the request

        Returns:
            Retrieved data or None
        """
        # Check access control
        if not self._check_read_access(agent_role):
            logger.warning(f"Agent {agent_role} denied read access to namespace {self.namespace}")
            return None

        try:
            full_key = f"shared:{self.namespace}:{key}"
            data = await self.memory_manager.retrieve_memory(full_key)

            if data is not None:
                logger.debug(f"Agent {agent_role} retrieved shared memory from {self.namespace}:{key}")

            return data
        except Exception as e:
            logger.error(f"Failed to retrieve shared memory {self.namespace}:{key}: {e}")
            return None

    async def search_shared_memories(self, query: str, agent_role: str,
                                   limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search shared memories in this namespace.

        Args:
            query: Search query
            agent_role: Role of agent making the request
            limit: Maximum results

        Returns:
            List of search results
        """
        # Check access control
        if not self._check_read_access(agent_role):
            logger.warning(f"Agent {agent_role} denied search access to namespace {self.namespace}")
            return []

        try:
            # Search with namespace filter
            namespace_query = f"{self.namespace} {query}"
            results = await self.memory_manager.search_memories(namespace_query, limit=limit)

            # Filter results to this namespace
            namespace_results = []
            for result in results:
                if f"shared:{self.namespace}:" in result.get("key", ""):
                    namespace_results.append(result)

            return namespace_results
        except Exception as e:
            logger.error(f"Failed to search shared memories in {self.namespace}: {e}")
            return []

    def subscribe(self, agent_role: str, callback: Callable):
        """
        Subscribe to namespace updates.

        Args:
            agent_role: Agent role subscribing
            callback: Callback function for updates
        """
        self.subscribers[agent_role] = callback
        logger.info(f"Agent {agent_role} subscribed to namespace {self.namespace}")

    def unsubscribe(self, agent_role: str):
        """
        Unsubscribe from namespace updates.

        Args:
            agent_role: Agent role unsubscribing
        """
        self.subscribers.pop(agent_role, None)
        logger.info(f"Agent {agent_role} unsubscribed from namespace {self.namespace}")

    async def _notify_subscribers(self, event_type: str, event_data: Dict[str, Any]):
        """
        Notify all subscribers of namespace updates.

        Args:
            event_type: Type of event (store, update, delete)
            event_data: Event data
        """
        for agent_role, callback in self.subscribers.items():
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, event_data)
                else:
                    callback(event_type, event_data)
            except Exception as e:
                logger.error(f"Failed to notify subscriber {agent_role}: {e}")

    def _check_read_access(self, agent_role: str) -> bool:
        """
        Check if agent has read access to this namespace.

        Args:
            agent_role: Agent role

        Returns:
            bool: Access granted
        """
        if not self.access_control:
            return True  # No access control = open access

        allowed_operations = self.access_control.get(agent_role, [])
        return "read" in allowed_operations or "write" in allowed_operations

    def _check_write_access(self, agent_role: str) -> bool:
        """
        Check if agent has write access to this namespace.

        Args:
            agent_role: Agent role

        Returns:
            bool: Access granted
        """
        if not self.access_control:
            return True  # No access control = open access

        allowed_operations = self.access_control.get(agent_role, [])
        return "write" in allowed_operations

class AgentToAgentProtocol:
    """
    Implements A2A (Agent-to-Agent) protocols for memory sharing and coordination.
    """

    def __init__(self):
        self.namespaces = {}  # namespace -> SharedMemoryNamespace
        self.agent_registrations = {}  # agent_role -> agent_info
        self.message_queue = asyncio.Queue()
        self.running = False

    def register_agent(self, agent_role: str, agent_info: Optional[Dict[str, Any]] = None):
        """
        Register an agent with the A2A protocol.

        Args:
            agent_role: Agent role
            agent_info: Agent information and capabilities
        """
        self.agent_registrations[agent_role] = {
            **(agent_info or {}),
            "registered_at": datetime.now().isoformat(),
            "status": "active"
        }
        logger.info(f"Registered agent: {agent_role}")

    def unregister_agent(self, agent_role: str):
        """
        Unregister an agent from the A2A protocol.

        Args:
            agent_role: Agent role
        """
        self.agent_registrations.pop(agent_role, None)

        # Unsubscribe from all namespaces
        for namespace in self.namespaces.values():
            namespace.unsubscribe(agent_role)

        logger.info(f"Unregistered agent: {agent_role}")

    def create_namespace(self, namespace: str, access_control: Optional[Dict[str, List[str]]] = None) -> SharedMemoryNamespace:
        """
        Create a new shared memory namespace.

        Args:
            namespace: Namespace identifier
            access_control: Access control rules

        Returns:
            SharedMemoryNamespace: Created namespace
        """
        if namespace in self.namespaces:
            logger.warning(f"Namespace {namespace} already exists")
            return self.namespaces[namespace]

        shared_namespace = SharedMemoryNamespace(namespace, access_control)
        self.namespaces[namespace] = shared_namespace

        logger.info(f"Created shared namespace: {namespace}")
        return shared_namespace

    def get_namespace(self, namespace: str) -> Optional[SharedMemoryNamespace]:
        """
        Get a shared memory namespace.

        Args:
            namespace: Namespace identifier

        Returns:
            SharedMemoryNamespace or None
        """
        return self.namespaces.get(namespace)

    async def send_message(self, from_agent: str, to_agent: str, message_type: str,
                          payload: Optional[Dict[str, Any]] = None):
        """
        Send a message from one agent to another.

        Args:
            from_agent: Sending agent role
            to_agent: Receiving agent role
            message_type: Type of message
            payload: Message payload
        """
        message = {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "message_type": message_type,
            "payload": payload or {},
            "timestamp": datetime.now().isoformat(),
            "message_id": f"{from_agent}_{to_agent}_{datetime.now().timestamp()}"
        }

        await self.message_queue.put(message)
        logger.debug(f"Queued message from {from_agent} to {to_agent}: {message_type}")

    async def broadcast_message(self, from_agent: str, message_type: str,
                               payload: Optional[Dict[str, Any]] = None, target_roles: Optional[List[str]] = None):
        """
        Broadcast a message to multiple agents.

        Args:
            from_agent: Sending agent role
            message_type: Type of message
            payload: Message payload
            target_roles: List of target agent roles (None = all registered agents)
        """
        targets = target_roles or list(self.agent_registrations.keys())

        for target_role in targets:
            if target_role != from_agent:  # Don't send to self
                await self.send_message(from_agent, target_role, message_type, payload)

    async def process_messages(self):
        """
        Process incoming messages (should run in background task).
        """
        self.running = True
        logger.info("Started A2A message processing")

        try:
            while self.running:
                # Wait for message
                message = await self.message_queue.get()

                try:
                    await self._handle_message(message)
                except Exception as e:
                    logger.error(f"Failed to handle message {message.get('message_id')}: {e}")
                finally:
                    self.message_queue.task_done()

        except asyncio.CancelledError:
            logger.info("A2A message processing cancelled")
        except Exception as e:
            logger.error(f"A2A message processing error: {e}")

    async def _handle_message(self, message: Dict[str, Any]):
        """
        Handle an incoming message.

        Args:
            message: Message to handle
        """
        from_agent = message["from_agent"]
        to_agent = message["to_agent"]
        message_type = message["message_type"]
        payload = message["payload"]

        logger.debug(f"Handling message {message_type} from {from_agent} to {to_agent}")

        # Route message based on type
        if message_type == "memory_share":
            await self._handle_memory_share(from_agent, to_agent, payload)
        elif message_type == "memory_request":
            await self._handle_memory_request(from_agent, to_agent, payload)
        elif message_type == "coordination_signal":
            await self._handle_coordination_signal(from_agent, to_agent, payload)
        else:
            logger.warning(f"Unknown message type: {message_type}")

    async def _handle_memory_share(self, from_agent: str, to_agent: str, payload: Dict[str, Any]):
        """
        Handle memory sharing message.
        """
        namespace = payload.get("namespace")
        key = payload.get("key")
        data = payload.get("data")

        if namespace and key and data is not None:
            # Store in shared namespace
            shared_ns = self.get_namespace(namespace)
            if shared_ns:
                success = await shared_ns.store_shared_memory(key, data, from_agent)
                if success:
                    logger.info(f"Shared memory from {from_agent} to {namespace}:{key}")
                else:
                    logger.warning(f"Failed to share memory from {from_agent} to {namespace}:{key}")

    async def _handle_memory_request(self, from_agent: str, to_agent: str, payload: Dict[str, Any]):
        """
        Handle memory request message.
        """
        namespace = payload.get("namespace")
        key = payload.get("key")

        if namespace and key:
            # Retrieve from shared namespace
            shared_ns = self.get_namespace(namespace)
            if shared_ns:
                data = await shared_ns.retrieve_shared_memory(key, to_agent)
                if data is not None:
                    # Send response back
                    response_payload = {
                        "namespace": namespace,
                        "key": key,
                        "data": data,
                        "request_id": payload.get("request_id")
                    }
                    await self.send_message(to_agent, from_agent, "memory_response", response_payload)

    async def _handle_coordination_signal(self, from_agent: str, to_agent: str, payload: Dict[str, Any]):
        """
        Handle coordination signal message.
        """
        signal_type = payload.get("signal_type")
        signal_data = payload.get("signal_data", {})

        logger.info(f"Coordination signal {signal_type} from {from_agent} to {to_agent}: {signal_data}")

        # Could trigger specific coordination actions based on signal type
        # For now, just log the coordination event

    def get_registered_agents(self) -> List[str]:
        """
        Get list of registered agent roles.

        Returns:
            List of agent roles
        """
        return list(self.agent_registrations.keys())

    def get_namespace_info(self) -> Dict[str, Any]:
        """
        Get information about all namespaces.

        Returns:
            Dict with namespace information
        """
        return {
            namespace: {
                "subscribers": list(ns.subscribers.keys()),
                "access_control": ns.access_control
            }
            for namespace, ns in self.namespaces.items()
        }

class CollaborativeSession:
    """
    Represents a collaborative session between multiple agents.
    """

    def __init__(self, session_id: str, creator: str, topic: str, max_participants: int, timeout: int):
        self.session_id = session_id
        self.creator = creator
        self.topic = topic
        self.participants = {}
        self.max_participants = max_participants
        self.session_timeout = timeout
        self.created_at = datetime.now().isoformat()
        self.status = "active"
        self.insights = []
        self.decisions = []
        self.shared_context = {}
        self.last_activity = self.created_at

    @property
    def creator_agent(self):
        return self.creator

    @property
    def session_data(self):
        return {
            "insights": self.insights,
            "decisions": self.decisions
        }

    def is_expired(self) -> bool:
        """
        Check if the session has expired based on last activity and timeout.
        """
        try:
            activity_time = datetime.fromisoformat(self.last_activity)
        except ValueError:
            activity_time = datetime.fromisoformat(self.created_at)
        elapsed = (datetime.now() - activity_time).total_seconds()
        return elapsed > self.session_timeout

    def join(self, agent_role, context=None):
        if len(self.participants) > self.max_participants:
            return False
        if agent_role not in self.participants:
            self.participants[agent_role] = {"context": context or {}}
            self.insights.append({
                "agent": agent_role,
                "type": "joined",
                "timestamp": datetime.now().isoformat(),
                "context": context or {}
            })
            self.last_activity = datetime.now().isoformat()
        return True

    def leave(self, agent_role):
        if agent_role in self.participants:
            del self.participants[agent_role]
            self.last_activity = datetime.now().isoformat()
            return True
        return False

    def contribute_insight(self, agent_role, insight):
        if agent_role not in self.participants:
            return False
        contribution = {
            "agent": agent_role,
            "timestamp": datetime.now().isoformat(),
            **insight
        }
        if "validated_by" not in contribution:
            contribution["validated_by"] = []
        self.insights.append(contribution)
        self.last_activity = datetime.now().isoformat()
        return True

    def record_decision(self, agent_role, decision):
        if agent_role not in self.participants:
            return False
        decision_record = {
            "agent": agent_role,
            "participants": list(self.participants.keys()),
            "timestamp": datetime.now().isoformat(),
            **decision
        }
        self.decisions.append(decision_record)
        self.last_activity = datetime.now().isoformat()
        return True

    def get_session_summary(self):
        if self.status != "active":
            return None
        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "creator": self.creator,
            "participant_count": len(self.participants),
            "insights_count": len(self.insights),
            "decisions_count": len(self.decisions),
            "status": self.status
        }

    def archive(self, agent_role=None):
        if agent_role and agent_role != self.creator:
            return False
        self.status = "archived"
        return True

class MultiAgentMemoryCoordinator:
    """
    Coordinates memory sharing and collaboration between multiple agents.
    Extends AgentToAgentProtocol with collaborative session management.
    """

    def __init__(self):
        self.a2a_protocol = AgentToAgentProtocol()
        self.collaborative_sessions = {}  # session_id -> CollaborativeSession
        self.session_counter = 0

    def get_session(self, session_id):
        return self.collaborative_sessions.get(session_id)

    async def share_memory(self, from_agent: str, to_agent: str, namespace: str,
                          key: str, data: Any) -> bool:
        """
        Share memory between agents through A2A protocol.

        Args:
            from_agent: Sending agent
            to_agent: Receiving agent
            namespace: Memory namespace
            key: Memory key
            data: Data to share

        Returns:
            bool: Success status
        """
        try:
            # Store in shared namespace
            shared_ns = self.a2a_protocol.get_namespace(namespace)
            if shared_ns is None:
                shared_ns = self.a2a_protocol.create_namespace(namespace)

            success = await shared_ns.store_shared_memory(key, data, from_agent)
            if success:
                # Send notification via A2A protocol
                await self.a2a_protocol.send_message(
                    from_agent, to_agent, "memory_shared",
                    {"namespace": namespace, "key": key}
                )
            return success
        except Exception as e:
            logger.error(f"Failed to share memory: {e}")
            return False

    async def broadcast_coordination_signal(self, from_agent: str, signal_type: str,
                                          signal_data: Optional[Dict[str, Any]] = None):
        """
        Broadcast coordination signal to all registered agents.

        Args:
            from_agent: Sending agent
            signal_type: Type of coordination signal
            signal_data: Additional signal data
        """
        await self.a2a_protocol.broadcast_message(
            from_agent, "coordination_signal",
            {"signal_type": signal_type, "signal_data": signal_data or {}}
        )

    async def create_collaborative_session(self, creator_agent: str, topic: str,
                                         max_participants: int = 10,
                                         session_timeout: int = 3600) -> Optional[str]:
        """
        Create a new collaborative session.

        Args:
            creator_agent: Agent creating the session
            topic: Session topic
            max_participants: Maximum participants
            session_timeout: Session timeout in seconds

        Returns:
            str: Session ID or None if failed
        """
        self.session_counter += 1
        session_id = f"session_{creator_agent}_{self.session_counter}"

        session = CollaborativeSession(session_id, creator_agent, topic, max_participants, session_timeout)
        self.collaborative_sessions[session_id] = session
        logger.info(f"Created collaborative session: {session_id} by {creator_agent}")
        return session_id

    async def join_collaborative_session(self, session_id: str, agent_role: str,
                                       agent_context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Join an existing collaborative session.

        Args:
            session_id: Session ID
            agent_role: Joining agent role
            agent_context: Agent context information

        Returns:
            bool: Success status
        """
        session = self.collaborative_sessions.get(session_id)
        if not session or session.status != "active":
            return False

        if len(session.participants) >= session.max_participants:
            return False

        if agent_role not in session.participants:
            session.participants[agent_role] = {"context": agent_context or {}}
            session.last_activity = datetime.now().isoformat()

        logger.info(f"Agent {agent_role} joined session {session_id}")
        return True

    async def leave_collaborative_session(self, session_id: str, agent_role: str) -> bool:
        """
        Leave a collaborative session.

        Args:
            session_id: Session ID
            agent_role: Leaving agent role

        Returns:
            bool: Success status
        """
        session = self.collaborative_sessions.get(session_id)
        if not session or agent_role not in session.participants:
            return False

        del session.participants[agent_role]
        session.last_activity = datetime.now().isoformat()
        logger.info(f"Agent {agent_role} left session {session_id}")
        return True

    async def contribute_to_session(self, session_id: str, agent_role: str,
                                  insight: Dict[str, Any]) -> bool:
        """
        Contribute an insight to a collaborative session.

        Args:
            session_id: Session ID
            agent_role: Contributing agent
            insight: Insight data

        Returns:
            bool: Success status
        """
        session = self.collaborative_sessions.get(session_id)
        if not session or session.status != "active":
            return False

        if agent_role not in session.participants:
            return False

        contribution = {
            "agent": agent_role,
            "timestamp": datetime.now().isoformat(),
            "insight": insight
        }
        if "validated_by" not in contribution:
            contribution["validated_by"] = []

        session.insights.append(contribution)
        session.last_activity = datetime.now().isoformat()
        logger.info(f"Agent {agent_role} contributed to session {session_id}")
        return True

    async def validate_session_insight(self, session_id: str, agent_role: str, insight_index: int,
                                     validation: Dict[str, Any]) -> bool:
        """
        Validate an insight in a collaborative session.

        Args:
            session_id: Session ID
            agent_role: Validating agent
            insight_index: Index of insight to validate
            validation: Validation data

        Returns:
            bool: Success status
        """
        session = self.collaborative_sessions.get(session_id)
        if not session or session.status != "active":
            return False

        if agent_role not in session.participants:
            return False

        if insight_index >= len(session.insights):
            return False

        insight = session.insights[insight_index]
        if "validated_by" not in insight:
            insight["validated_by"] = []

        insight["validated_by"].append({
            "validator": agent_role,
            "validation": validation,
            "timestamp": datetime.now().isoformat()
        })

        session.last_activity = datetime.now().isoformat()
        logger.info(f"Agent {agent_role} validated insight {insight_index} in session {session_id}")
        return True

    async def update_session_context(self, session_id: str, agent_role: str,
                                   context_type: str, context_data: Dict[str, Any]) -> bool:
        """
        Update session context with shared information.

        Args:
            session_id: Session ID
            agent_role: Updating agent
            context_type: Type of context (e.g., 'position', 'workflow')
            context_data: Context data to share

        Returns:
            bool: Success status
        """
        session = self.collaborative_sessions.get(session_id)
        if not session or session.status != "active":
            return False

        if agent_role != session.creator and agent_role not in session.participants:
            return False

        session.shared_context[context_type] = {
            "agent": agent_role,
            "timestamp": datetime.now().isoformat(),
            "data": context_data
        }

        session.last_activity = datetime.now().isoformat()
        logger.info(f"Agent {agent_role} updated {context_type} context in session {session_id}")
        return True

    async def get_session_context(self, session_id: str, context_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get session context data.

        Args:
            session_id: Session ID
            context_type: Specific context type to retrieve (optional)

        Returns:
            Dict containing context data
        """
        session = self.collaborative_sessions.get(session_id)
        if not session:
            return {}

        context = session.shared_context

        if context_type:
            return context.get(context_type, {})

        return context

    async def record_session_decision(self, session_id: str, agent_role: str,
                                    decision: Dict[str, Any]) -> bool:
        """
        Record a collaborative decision in the session.

        Args:
            session_id: Session ID
            agent_role: Agent recording the decision
            decision: Decision data

        Returns:
            bool: Success status
        """
        session = self.collaborative_sessions.get(session_id)
        if not session or session.status != "active":
            return False

        if agent_role != session.creator and agent_role not in session.participants:
            return False

        decision_record = {
            "agent": agent_role,
            "participants": [session.creator] + list(session.participants.keys()),
            "timestamp": datetime.now().isoformat(),
            "decision": decision
        }

        session.decisions.append(decision_record)
        session.last_activity = datetime.now().isoformat()
        logger.info(f"Agent {agent_role} recorded decision in session {session_id}")
        return True

    async def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of the collaborative session.

        Args:
            session_id: Session ID

        Returns:
            Dict containing session summary
        """
        session = self.collaborative_sessions.get(session_id)
        if not session:
            return None

        return {
            "session_id": session.session_id,
            "topic": session.topic,
            "creator": session.creator,
            "participant_count": len(session.participants) + 1,
            "insights_count": len(session.insights),
            "decisions_count": len(session.decisions),
            "status": session.status,
            "participants": session.participants,
            "session_data": session.session_data,
            "is_expired": session.is_expired()
        }

    async def archive_session(self, session_id: str, agent_role: Optional[str] = None) -> bool:
        """
        Archive a collaborative session.

        Args:
            session_id: Session ID
            agent_role: Agent archiving (optional, defaults to allow if None)

        Returns:
            bool: Success status
        """
        session = self.collaborative_sessions.get(session_id)
        if not session or session.status != "active":
            return False

        if agent_role is None:
            if session.participants:
                return False
        elif agent_role != session.creator:
            return False

        session.status = "archived"
        session.archived_at = datetime.now().isoformat()
        # Keep in dict for potential retrieval, active_sessions filters
        logger.info(f"Session {session_id} archived by {agent_role or 'system'}")
        return True

    @property
    def active_sessions(self) -> Dict[str, CollaborativeSession]:
        """
        Get active collaborative sessions.

        Returns:
            Dict of active sessions
        """
        return {
            s.session_id: s for s in self.collaborative_sessions.values() if s.status == "active"
        }

    async def get_session_insights(self, session_id: str, agent_role: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get insights from a collaborative session.

        Args:
            session_id: Session ID
            agent_role: Filter by agent role (optional)

        Returns:
            List of insights
        """
        session = self.collaborative_sessions.get(session_id)
        if not session:
            return []

        insights = session.insights

        if agent_role:
            insights = [i for i in insights if i.get("agent") == agent_role]

        return insights

    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """
        List all active collaborative sessions.

        Returns:
            List of session information
        """
        return [
            {
                "session_id": s.session_id,
                "topic": s.topic,
                "participants": s.participants,
                "created_at": s.created_at
            } for s in self.collaborative_sessions.values() if s.status == "active"
        ]

    # Delegate other methods to A2A protocol
    def register_agent(self, agent_role: str, agent_info: Optional[Dict[str, Any]] = None):
        """Register an agent with the coordinator."""
        self.a2a_protocol.register_agent(agent_role, agent_info)

    def unregister_agent(self, agent_role: str):
        """Unregister an agent from the coordinator."""
        self.a2a_protocol.unregister_agent(agent_role)

    def get_registered_agents(self) -> List[str]:
        """Get list of registered agents."""
        return self.a2a_protocol.get_registered_agents()

    def get_namespace_info(self) -> Dict[str, Any]:
        """Get namespace information."""
        return self.a2a_protocol.get_namespace_info()


# Global instance
_multi_agent_coordinator = None

def get_multi_agent_coordinator() -> MultiAgentMemoryCoordinator:
    """
    Get global multi-agent memory coordinator instance.

    Returns:
        MultiAgentMemoryCoordinator: Global instance
    """
    global _multi_agent_coordinator
    if _multi_agent_coordinator is None:
        _multi_agent_coordinator = MultiAgentMemoryCoordinator()
    return _multi_agent_coordinator

# Convenience functions
async def share_memory_between_agents(from_agent: str, to_agent: str, namespace: str,
                                   key: str, data: Any) -> bool:
    """Share memory between agents."""
    return await get_multi_agent_coordinator().share_memory(from_agent, to_agent, namespace, key, data)

async def broadcast_coordination_signal(from_agent: str, signal_type: str,
                                      signal_data: Optional[Dict[str, Any]] = None):
    """Broadcast coordination signal."""
    await get_multi_agent_coordinator().broadcast_coordination_signal(from_agent, signal_type, signal_data)

# Convenience functions for collaborative sessions
async def create_collaborative_session(creator_agent: str, topic: str,
                                     max_participants: int = 10,
                                     session_timeout: int = 3600) -> Optional[str]:
    """Create a new collaborative session."""
    return await get_multi_agent_coordinator().create_collaborative_session(
        creator_agent, topic, max_participants, session_timeout
    )

async def join_collaborative_session(session_id: str, agent_role: str,
                                   agent_context: Optional[Dict[str, Any]] = None) -> bool:
    """Join an existing collaborative session."""
    return await get_multi_agent_coordinator().join_collaborative_session(
        session_id, agent_role, agent_context
    )

async def leave_collaborative_session(session_id: str, agent_role: str) -> bool:
    """Leave a collaborative session."""
    return await get_multi_agent_coordinator().leave_collaborative_session(
        session_id, agent_role
    )

async def contribute_to_session(session_id: str, agent_role: str,
                              insight: Dict[str, Any]) -> bool:
    """Contribute an insight to a collaborative session."""
    return await get_multi_agent_coordinator().contribute_to_session(
        session_id, agent_role, insight
    )

async def validate_session_insight(session_id: str, agent_role: str, insight_index: int,
                                 validation: Dict[str, Any]) -> bool:
    """Validate an insight in a collaborative session."""
    return await get_multi_agent_coordinator().validate_session_insight(
        session_id, agent_role, insight_index, validation
    )

async def get_session_insights(session_id: str, agent_role: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get insights from a collaborative session."""
    return await get_multi_agent_coordinator().get_session_insights(session_id, agent_role)

async def list_active_sessions() -> List[Dict[str, Any]]:
    """List all active collaborative sessions."""
    return get_multi_agent_coordinator().list_active_sessions()

async def archive_session(session_id: str, agent_role: Optional[str] = None) -> bool:
    """Archive a collaborative session."""
    return await get_multi_agent_coordinator().archive_session(session_id, agent_role)
