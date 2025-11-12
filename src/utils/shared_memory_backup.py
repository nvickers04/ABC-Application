# src/utils/shared_memory.py
# Purpose: Multi-agent memory sharing system
# Enables agents to share memories and coordinate through shared namespaces
# Implements A2A (Agent-to-Agent) protocols for memory exchange

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import json

from src.utils.advanced_memory import get_advanced_memory_manager
from src.utils.memory_persistence import get_memory_persistence

logger = logging.getLogger(__name__)

def sanitize_for_json(data: Any, max_depth: int = 3, current_depth: int = 0, seen_objects: set = None) -> Any:
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
    try:
        import numpy as np
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
    except ImportError:
        pass

    # Handle pandas DataFrames and Series
    try:
        import pandas as pd
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
    except ImportError:
        pass

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

    def __init__(self, namespace: str, access_control: Dict[str, List[str]] = None):
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

    def register_agent(self, agent_role: str, agent_info: Dict[str, Any]):
        """
        Register an agent with the A2A protocol.

        Args:
            agent_role: Agent role
            agent_info: Agent information and capabilities
        """
        self.agent_registrations[agent_role] = {
            **agent_info,
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

    def create_namespace(self, namespace: str, access_control: Dict[str, List[str]] = None) -> SharedMemoryNamespace:
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
                          payload: Dict[str, Any]):
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
            "payload": payload,
            "timestamp": datetime.now().isoformat(),
            "message_id": f"{from_agent}_{to_agent}_{datetime.now().timestamp()}"
        }

        await self.message_queue.put(message)
        logger.debug(f"Queued message from {from_agent} to {to_agent}: {message_type}")

    async def broadcast_message(self, from_agent: str, message_type: str,
                               payload: Dict[str, Any], target_roles: List[str] = None):
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

# Global instance
_multi_agent_coordinator = None

def get_multi_agent_coordinator() -> MultiAgentMemoryCoordinator:
    ""
    Get global multi-agent memory coordinator instance.

    Returns:
        MultiAgentMemoryCoordinator: Global instance
    ""
    global _multi_agent_coordinator
    if _multi_agent_coordinator is None:
        _multi_agent_coordinator = MultiAgentMemoryCoordinator()
    return _multi_agent_coordinator

# Convenience functions
async def share_memory_between_agents(from_agent: str, to_agent: str, namespace: str,
                                   key: str, data: Any) -> bool:
    ""Share memory between agents.""
    return await get_multi_agent_coordinator().share_memory(from_agent, to_agent, namespace, key, data)

async def broadcast_coordination_signal(from_agent: str, signal_type: str,
                                      signal_data: Dict[str, Any] = None):
    ""Broadcast coordination signal.""
    await get_multi_agent_coordinator().broadcast_coordination_signal(from_agent, signal_type, signal_data)

# Convenience functions for collaborative sessions
async def create_collaborative_session(creator_agent: str, topic: str,
                                     max_participants: int = 10,
                                     session_timeout: int = 3600) -> Optional[str]:
    ""Create a new collaborative session.""
    return await get_multi_agent_coordinator().create_collaborative_session(
        creator_agent, topic, max_participants, session_timeout
    )

async def join_collaborative_session(session_id: str, agent_role: str,
                                   agent_context: Dict[str, Any] = None) -> bool:
    ""Join an existing collaborative session.""
    return await get_multi_agent_coordinator().join_collaborative_session(
        session_id, agent_role, agent_context
    )

async def contribute_to_session(session_id: str, agent_role: str,
                              insight: Dict[str, Any]) -> bool:
    ""Contribute an insight to a collaborative session.""
    return await get_multi_agent_coordinator().contribute_to_session(
        session_id, agent_role, insight
    )

async def get_session_insights(session_id: str, agent_role: str = None) -> List[Dict[str, Any]]:
    ""Get insights from a collaborative session.""
    return await get_multi_agent_coordinator().get_session_insights(session_id, agent_role)

async def list_active_sessions() -> List[Dict[str, Any]]:
    ""List all active collaborative sessions.""
    return get_multi_agent_coordinator().list_active_sessions()

 #   G l o b a l   i n s t a n c e 
 _ m u l t i _ a g e n t _ c o o r d i n a t o r   =   N o n e 
 
 d e f   g e t _ m u l t i _ a g e n t _ c o o r d i n a t o r ( )   - >   M u l t i A g e n t M e m o r y C o o r d i n a t o r : 
         ' ' ' 
         G e t   g l o b a l   m u l t i - a g e n t   m e m o r y   c o o r d i n a t o r   i n s t a n c e . 
 
         R e t u r n s : 
                 M u l t i A g e n t M e m o r y C o o r d i n a t o r :   G l o b a l   i n s t a n c e 
         ' ' ' 
         g l o b a l   _ m u l t i _ a g e n t _ c o o r d i n a t o r 
         i f   _ m u l t i _ a g e n t _ c o o r d i n a t o r   i s   N o n e : 
                 _ m u l t i _ a g e n t _ c o o r d i n a t o r   =   M u l t i A g e n t M e m o r y C o o r d i n a t o r ( ) 
         r e t u r n   _ m u l t i _ a g e n t _ c o o r d i n a t o r 
 
 #   C o n v e n i e n c e   f u n c t i o n s 
 a s y n c   d e f   s h a r e _ m e m o r y _ b e t w e e n _ a g e n t s ( f r o m _ a g e n t :   s t r ,   t o _ a g e n t :   s t r ,   n a m e s p a c e :   s t r , 
                                                                       k e y :   s t r ,   d a t a :   A n y )   - >   b o o l : 
         ' ' ' S h a r e   m e m o r y   b e t w e e n   a g e n t s . ' ' ' 
         r e t u r n   a w a i t   g e t _ m u l t i _ a g e n t _ c o o r d i n a t o r ( ) . s h a r e _ m e m o r y ( f r o m _ a g e n t ,   t o _ a g e n t ,   n a m e s p a c e ,   k e y ,   d a t a ) 
 
 a s y n c   d e f   b r o a d c a s t _ c o o r d i n a t i o n _ s i g n a l ( f r o m _ a g e n t :   s t r ,   s i g n a l _ t y p e :   s t r , 
                                                                             s i g n a l _ d a t a :   D i c t [ s t r ,   A n y ]   =   N o n e ) : 
         ' ' ' B r o a d c a s t   c o o r d i n a t i o n   s i g n a l . ' ' ' 
         a w a i t   g e t _ m u l t i _ a g e n t _ c o o r d i n a t o r ( ) . b r o a d c a s t _ c o o r d i n a t i o n _ s i g n a l ( f r o m _ a g e n t ,   s i g n a l _ t y p e ,   s i g n a l _ d a t a ) 
 
 #   C o n v e n i e n c e   f u n c t i o n s   f o r   c o l l a b o r a t i v e   s e s s i o n s 
 a s y n c   d e f   c r e a t e _ c o l l a b o r a t i v e _ s e s s i o n ( c r e a t o r _ a g e n t :   s t r ,   t o p i c :   s t r , 
                                                                           m a x _ p a r t i c i p a n t s :   i n t   =   1 0 , 
                                                                           s e s s i o n _ t i m e o u t :   i n t   =   3 6 0 0 )   - >   O p t i o n a l [ s t r ] : 
         ' ' ' C r e a t e   a   n e w   c o l l a b o r a t i v e   s e s s i o n . ' ' ' 
         r e t u r n   a w a i t   g e t _ m u l t i _ a g e n t _ c o o r d i n a t o r ( ) . c r e a t e _ c o l l a b o r a t i v e _ s e s s i o n ( 
                 c r e a t o r _ a g e n t ,   t o p i c ,   m a x _ p a r t i c i p a n t s ,   s e s s i o n _ t i m e o u t 
         ) 
 
 a s y n c   d e f   j o i n _ c o l l a b o r a t i v e _ s e s s i o n ( s e s s i o n _ i d :   s t r ,   a g e n t _ r o l e :   s t r , 
                                                                       a g e n t _ c o n t e x t :   D i c t [ s t r ,   A n y ]   =   N o n e )   - >   b o o l : 
         ' ' ' J o i n   a n   e x i s t i n g   c o l l a b o r a t i v e   s e s s i o n . ' ' ' 
         r e t u r n   a w a i t   g e t _ m u l t i _ a g e n t _ c o o r d i n a t o r ( ) . j o i n _ c o l l a b o r a t i v e _ s e s s i o n ( 
                 s e s s i o n _ i d ,   a g e n t _ r o l e ,   a g e n t _ c o n t e x t 
         ) 
 
 a s y n c   d e f   c o n t r i b u t e _ t o _ s e s s i o n ( s e s s i o n _ i d :   s t r ,   a g e n t _ r o l e :   s t r , 
                                                             i n s i g h t :   D i c t [ s t r ,   A n y ] )   - >   b o o l : 
         ' ' ' C o n t r i b u t e   a n   i n s i g h t   t o   a   c o l l a b o r a t i v e   s e s s i o n . ' ' ' 
         r e t u r n   a w a i t   g e t _ m u l t i _ a g e n t _ c o o r d i n a t o r ( ) . c o n t r i b u t e _ t o _ s e s s i o n ( 
                 s e s s i o n _ i d ,   a g e n t _ r o l e ,   i n s i g h t 
         ) 
 
 a s y n c   d e f   g e t _ s e s s i o n _ i n s i g h t s ( s e s s i o n _ i d :   s t r ,   a g e n t _ r o l e :   s t r   =   N o n e )   - >   L i s t [ D i c t [ s t r ,   A n y ] ] : 
         ' ' ' G e t   i n s i g h t s   f r o m   a   c o l l a b o r a t i v e   s e s s i o n . ' ' ' 
         r e t u r n   a w a i t   g e t _ m u l t i _ a g e n t _ c o o r d i n a t o r ( ) . g e t _ s e s s i o n _ i n s i g h t s ( s e s s i o n _ i d ,   a g e n t _ r o l e ) 
 
 a s y n c   d e f   l i s t _ a c t i v e _ s e s s i o n s ( )   - >   L i s t [ D i c t [ s t r ,   A n y ] ] : 
         ' ' ' L i s t   a l l   a c t i v e   c o l l a b o r a t i v e   s e s s i o n s . ' ' ' 
         r e t u r n   g e t _ m u l t i _ a g e n t _ c o o r d i n a t o r ( ) . l i s t _ a c t i v e _ s e s s i o n s ( ) 
  
 