# src/utils/advanced_memory.py
# Purpose: Advanced memory management system with multiple storage backends
# Supports Redis, Mem0, vector databases, and semantic search capabilities
# Provides unified interface for different memory types and storage strategies

import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import os

# External dependencies
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    Memory = None

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None
    np = None

from src.utils.memory_persistence import get_memory_persistence
from src.utils.embeddings import get_embedding_manager
from src.utils.memory_security import get_secure_memory_manager

logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Environment variables loaded from .env file")
except ImportError:
    logger.warning("python-dotenv not available, environment variables may not be loaded")
from src.utils.embeddings import get_embedding_manager
from src.utils.memory_security import get_secure_memory_manager

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

class MemoryBackend:
    """Base class for memory storage backends."""

    async def store(self, key: str, data: Any, metadata: Dict[str, Any] = None) -> bool:
        """Store data with optional metadata."""
        raise NotImplementedError

    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data by key."""
        raise NotImplementedError

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant memories."""
        raise NotImplementedError

    async def delete(self, key: str) -> bool:
        """Delete data by key."""
        raise NotImplementedError

    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics."""
        raise NotImplementedError

class JSONBackend(MemoryBackend):
    """JSON file-based memory backend using existing persistence system."""

    def __init__(self):
        self.persistence = get_memory_persistence()

    async def store(self, key: str, data: Any, metadata: Dict[str, Any] = None) -> bool:
        try:
            # Sanitize data for JSON serialization to prevent recursion errors
            sanitized_data = sanitize_for_json(data)

            # For agent memory, use agent-specific storage
            if key.startswith("agent:"):
                agent_name = key.split(":")[1]
                return self.persistence.save_agent_memory(agent_name, sanitized_data)
            # For shared memory, use shared namespace
            else:
                # Try saving with the original namespace first. If it fails (e.g. invalid filename
                # chars on Windows), retry with a sanitized fallback that replaces colons and
                # slashes with underscores.
                try:
                    return self.persistence.save_shared_memory(key, sanitized_data)
                except Exception as e:
                    logger.warning(f"JSON backend initial save failed for '{key}', retrying with sanitized key: {e}")
                    safe_key = key.replace(":", "_").replace("/", "_")
                    try:
                        return self.persistence.save_shared_memory(safe_key, sanitized_data)
                    except Exception as e2:
                        logger.error(f"JSON backend fallback save also failed for '{safe_key}': {e2}")
                        return False
        except Exception as e:
            logger.error(f"JSON backend store failed for {key}: {e}")
            return False

    async def retrieve(self, key: str) -> Optional[Any]:
        try:
            if key.startswith("agent:"):
                agent_name = key.split(":")[1]
                return self.persistence.load_agent_memory(agent_name)
            else:
                # Sanitize key for filename (replace colons with underscores)
                safe_key = key.replace(":", "_").replace("/", "_")
                return self.persistence.load_shared_memory(safe_key)
        except Exception as e:
            logger.error(f"JSON backend retrieve failed for {key}: {e}")
            return None

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        # Basic keyword search in JSON files
        results = []
        try:
            # Search in agent memories
            for agent_name in self.persistence.list_agent_memories():
                memory = self.persistence.load_agent_memory(agent_name)
                if memory and self._matches_query(memory, query):
                    results.append({
                        "key": f"agent:{agent_name}",
                        "data": memory,
                        "type": "agent_memory",
                        "relevance": 0.8  # Basic relevance
                    })

            # Search in shared memories
            for namespace in self.persistence.list_shared_memories():
                memory = self.persistence.load_shared_memory(namespace)
                if memory and self._matches_query(memory, query):
                    results.append({
                        "key": namespace,
                        "data": memory,
                        "type": "shared_memory",
                        "relevance": 0.7
                    })

        except Exception as e:
            logger.error(f"JSON backend search failed: {e}")

        return results[:limit]

    def _matches_query(self, data: Any, query: str) -> bool:
        """Simple text matching for search."""
        query_lower = query.lower()
        if isinstance(data, dict):
            return any(query_lower in str(v).lower() for v in data.values())
        elif isinstance(data, list):
            return any(query_lower in str(item).lower() for item in data)
        else:
            return query_lower in str(data).lower()

    async def delete(self, key: str) -> bool:
        # JSON backend doesn't support deletion (files persist)
        logger.warning(f"JSON backend does not support deletion for {key}")
        return False

    def get_stats(self) -> Dict[str, Any]:
        return self.persistence.get_memory_stats()

class RedisBackend(MemoryBackend):
    """Redis-based memory backend for fast access and caching."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis")

        self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True,
                                       socket_connect_timeout=5, socket_timeout=5)
        self.ttl_seconds = 30 * 24 * 60 * 60  # 30 days default TTL

        # Test connection
        try:
            self.redis_client.ping()
        except (redis.ConnectionError, redis.TimeoutError) as e:
            raise ConnectionError(f"Failed to connect to Redis: {str(e)}")

    async def store(self, key: str, data: Any, metadata: Dict[str, Any] = None) -> bool:
        try:
            # Serialize data
            if isinstance(data, (dict, list)):
                serialized = json.dumps(data)
            else:
                serialized = str(data)

            # Store with TTL
            success = self.redis_client.setex(key, self.ttl_seconds, serialized)

            # Store metadata if provided
            if metadata and success:
                meta_key = f"{key}:meta"
                self.redis_client.setex(meta_key, self.ttl_seconds, json.dumps(metadata))

            return bool(success)
        except Exception as e:
            logger.error(f"Redis store failed for {key}: {e}")
            return False

    async def retrieve(self, key: str) -> Optional[Any]:
        try:
            data = self.redis_client.get(key)
            if data:
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    return data
            return None
        except Exception as e:
            logger.error(f"Redis retrieve failed for {key}: {e}")
            return None

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        # Redis doesn't have built-in semantic search
        # Return basic key pattern matching
        results = []
        try:
            keys = self.redis_client.keys(f"*{query}*")
            for key in keys[:limit]:
                data = await self.retrieve(key)
                if data:
                    results.append({
                        "key": key,
                        "data": data,
                        "type": "redis_cache",
                        "relevance": 0.5
                    })
        except Exception as e:
            logger.error(f"Redis search failed: {e}")

        return results

    async def delete(self, key: str) -> bool:
        try:
            # Delete main key and metadata
            deleted = self.redis_client.delete(key, f"{key}:meta")
            return deleted > 0
        except Exception as e:
            logger.error(f"Redis delete failed for {key}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        try:
            info = self.redis_client.info()
            return {
                "backend": "redis",
                "keys_count": self.redis_client.dbsize(),
                "memory_used": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "uptime_days": info.get("uptime_in_days", 0)
            }
        except (redis.ConnectionError, redis.TimeoutError) as e:
            return {"error": f"Redis unavailable or timed out: {str(e)}"}
        except Exception as e:
            return {"error": str(e)}

class Mem0Backend(MemoryBackend):
    """Mem0-based memory backend for self-improving recall."""

    def __init__(self, api_key: str = None):
        if not MEM0_AVAILABLE:
            raise ImportError("Mem0 not available. Install with: pip install mem0ai")

        self.api_key = api_key or os.getenv("MEM0_API_KEY")
        if not self.api_key:
            raise ValueError("Mem0 API key required")

        # Initialize Mem0 client
        self.memory = Memory()

    async def store(self, key: str, data: Any, metadata: Dict[str, Any] = None) -> bool:
        try:
            # Mem0 works with text/messages, so convert data to text
            if isinstance(data, dict):
                text_data = json.dumps(data)
            else:
                text_data = str(data)

            # Add to memory with user context
            user_id = metadata.get("user_id", "default") if metadata else "default"
            result = self.memory.add(text_data, user_id=user_id, metadata=metadata or {})

            return result is not None
        except Exception as e:
            logger.error(f"Mem0 store failed for {key}: {e}")
            return False

    async def retrieve(self, key: str) -> Optional[Any]:
        # Mem0 doesn't support direct key retrieval
        # Use search instead
        results = await self.search(key, limit=1)
        return results[0]["data"] if results else None

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            # Search memories
            results = self.memory.search(query, limit=limit)

            formatted_results = []
            for result in results:
                formatted_results.append({
                    "key": result.get("id", "unknown"),
                    "data": result.get("memory", ""),
                    "metadata": result.get("metadata", {}),
                    "type": "mem0_memory",
                    "relevance": result.get("score", 0.5)
                })

            return formatted_results
        except Exception as e:
            logger.error(f"Mem0 search failed: {e}")
            return []

    async def delete(self, key: str) -> bool:
        # Mem0 doesn't support direct deletion
        logger.warning(f"Mem0 backend does not support deletion for {key}")
        return False

    def get_stats(self) -> Dict[str, Any]:
        # Mem0 doesn't provide stats API
        return {"backend": "mem0", "status": "active"}

class VectorBackend(MemoryBackend):
    """Vector database backend using ChromaDB for semantic search."""

    def __init__(self, persist_directory: str = "./data/vector_db"):
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")

        if not EMBEDDINGS_AVAILABLE:
            raise ImportError("Sentence transformers not available. Install with: pip install sentence-transformers")

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get embedding manager
        self.embedding_manager = get_embedding_manager()

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="agent_memories",
            metadata={"description": "Agent memory storage with vector search"}
        )

    async def store(self, key: str, data: Any, metadata: Dict[str, Any] = None) -> bool:
        try:
            # Convert data to text for embedding
            if isinstance(data, dict):
                text_data = json.dumps(data)
            else:
                text_data = str(data)

            # Generate embedding using embedding manager
            embedding = self.embedding_manager.encode_text(text_data).tolist()

            # Prepare metadata
            doc_metadata = metadata or {}
            doc_metadata.update({
                "timestamp": datetime.now().isoformat(),
                "data_type": type(data).__name__,
                "original_key": key
            })

            # Add to collection
            self.collection.add(
                documents=[text_data],
                embeddings=[embedding],
                metadatas=[doc_metadata],
                ids=[key]
            )

            return True
        except Exception as e:
            logger.error(f"Vector store failed for {key}: {e}")
            return False

    async def retrieve(self, key: str) -> Optional[Any]:
        try:
            results = self.collection.get(ids=[key])
            if results and results['documents']:
                # Parse back to original format if possible
                text_data = results['documents'][0]
                try:
                    return json.loads(text_data)
                except json.JSONDecodeError:
                    return text_data
            return None
        except Exception as e:
            logger.error(f"Vector retrieve failed for {key}: {e}")
            return None

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            # Generate query embedding using embedding manager
            query_embedding = self.embedding_manager.encode_text(query).tolist()

            # Search collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit
            )

            formatted_results = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 1.0

                    # Convert distance to relevance score (lower distance = higher relevance)
                    relevance = max(0, 1 - distance)

                    formatted_results.append({
                        "key": results['ids'][0][i],
                        "data": doc,
                        "metadata": metadata,
                        "type": "vector_memory",
                        "relevance": relevance
                    })

            return formatted_results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def delete(self, key: str) -> bool:
        try:
            self.collection.delete(ids=[key])
            return True
        except Exception as e:
            logger.error(f"Vector delete failed for {key}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        try:
            count = self.collection.count()
            return {
                "backend": "chromadb",
                "collection": self.collection.name,
                "documents_count": count,
                "persist_directory": str(self.persist_directory)
            }
        except Exception as e:
            return {"error": str(e)}

class AdvancedMemoryManager:
    """
    Advanced memory manager with async operations, comprehensive memory management,
    and automatic failover redundancy across multiple storage backends.
    """

    def __init__(self):
        self.memory_store: Dict[str, Any] = {}
        self.cleanup_count = 0

        # Initialize multiple backends for redundancy
        self.backends = self._initialize_backends()
        self.backend_health = {name: True for name in self.backends.keys()}

        # Primary backend order (fastest/most reliable first)
        self.primary_order = ["redis", "json", "vector", "mem0"]

    def _initialize_backends(self) -> Dict[str, MemoryBackend]:
        """Initialize all available memory backends."""
        backends = {}

        # Always available JSON backend (fallback) - initialize first
        try:
            backends["json"] = JSONBackend()
            logger.info("JSON backend initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize JSON backend: {e}")
            # JSON backend is critical, if it fails we have bigger problems
            raise RuntimeError(f"Critical failure: JSON backend could not be initialized: {e}")

        # Redis backend (fast, but requires Redis server)
        if REDIS_AVAILABLE:
            try:
                redis_host = os.getenv("REDIS_HOST", "localhost")
                redis_port = int(os.getenv("REDIS_PORT", "6379"))
                backends["redis"] = RedisBackend(host=redis_host, port=redis_port)
                logger.info("Redis backend initialized successfully")
            except Exception as e:
                logger.warning(f"Redis backend not available (expected if Redis not running): {e}")
                logger.info("Falling back to JSON/in-memory storage for robustness")
                # Don't add to backends if it fails

        # Vector backend (semantic search, persistent)
        if CHROMA_AVAILABLE and EMBEDDINGS_AVAILABLE:
            try:
                vector_dir = os.getenv("VECTOR_DB_DIR", "./data/vector_db")
                backends["vector"] = VectorBackend(persist_directory=vector_dir)
                logger.info("Vector backend initialized successfully")
            except Exception as e:
                logger.warning(f"Vector backend not available: {e}")
                # Don't add to backends if it fails

        # Mem0 backend (AI-powered memory, requires API key) - DISABLED due to model issues
        # TODO: Re-enable when Mem0 fixes gpt-4o-mini model access
        if False and MEM0_AVAILABLE:  # Temporarily disabled
            try:
                mem0_key = os.getenv("MEM0_API_KEY")
                if mem0_key:
                    backends["mem0"] = Mem0Backend(api_key=mem0_key)
                    logger.info("Mem0 backend initialized successfully")
                else:
                    logger.info("Mem0 API key not found, skipping Mem0 backend")
            except Exception as e:
                logger.warning(f"Mem0 backend not available: {e}")
                # Don't add to backends if it fails

        # Ensure we have at least the JSON backend
        if not backends:
            raise RuntimeError("No memory backends available - JSON backend should always be available")

        logger.info(f"Initialized {len(backends)} memory backends: {list(backends.keys())}")
        return backends

    async def _try_backends_operation(self, operation: str, *args, **kwargs) -> Any:
        """
        Try an operation across multiple backends with automatic failover.

        Args:
            operation: Name of the backend method to call
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Result from first successful backend, or None if all fail
        """
        errors = []

        # Try backends in primary order
        for backend_name in self.primary_order:
            if backend_name not in self.backends or not self.backend_health.get(backend_name, True):
                continue

            backend = self.backends[backend_name]
            try:
                method = getattr(backend, operation)
                if asyncio.iscoroutinefunction(method):
                    result = await method(*args, **kwargs)
                else:
                    result = method(*args, **kwargs)

                # Mark backend as healthy
                self.backend_health[backend_name] = True
                return result

            except Exception as e:
                error_msg = f"{backend_name} backend failed for {operation}: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)

                # Mark backend as unhealthy
                self.backend_health[backend_name] = False

        # If all backends failed, log critical error
        if errors:
            logger.error(f"All backends failed for {operation}: {errors}")

        return None

    async def _sync_to_backends(self, key: str, data: Any, metadata: Dict[str, Any] = None):
        """
        Synchronize data to all healthy backends for redundancy.

        Args:
            key: Memory key
            data: Data to store
            metadata: Optional metadata
        """
        sync_errors = []

        for backend_name, backend in self.backends.items():
            if not self.backend_health.get(backend_name, True):
                continue

            try:
                if asyncio.iscoroutinefunction(backend.store):
                    success = await backend.store(key, data, metadata)
                else:
                    success = backend.store(key, data, metadata)

                if success:
                    logger.debug(f"Successfully synced {key} to {backend_name} backend")
                else:
                    sync_errors.append(f"{backend_name} store returned False")

            except Exception as e:
                sync_errors.append(f"{backend_name} sync failed: {e}")
                self.backend_health[backend_name] = False

        if sync_errors:
            logger.warning(f"Some backends failed to sync {key}: {sync_errors}")

    def get_memory_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status of all memory backends.

        Returns:
            Dict with health status for each backend and overall system health
        """
        try:
            backend_status = {}
            healthy_backends = 0
            total_backends = len(self.backends)

            for backend_name, backend in self.backends.items():
                try:
                    # Get backend stats
                    stats = backend.get_stats()
                    is_healthy = self.backend_health.get(backend_name, True)

                    backend_status[backend_name] = {
                        "healthy": is_healthy,
                        "stats": stats,
                        "last_check": datetime.now().isoformat()
                    }

                    if is_healthy:
                        healthy_backends += 1

                except Exception as e:
                    backend_status[backend_name] = {
                        "healthy": False,
                        "error": str(e),
                        "last_check": datetime.now().isoformat()
                    }

            # Overall system health
            overall_healthy = healthy_backends > 0  # At least one backend must be healthy
            redundancy_level = "full" if healthy_backends == total_backends else "partial" if healthy_backends > 1 else "minimal" if healthy_backends == 1 else "critical"

            return {
                "overall_healthy": overall_healthy,
                "redundancy_level": redundancy_level,
                "healthy_backends": healthy_backends,
                "total_backends": total_backends,
                "backends": backend_status,
                "primary_order": self.primary_order,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get memory health status: {e}")
            return {
                "overall_healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def store(self, key: str, data: Any) -> bool:
        """
        Asynchronously store data in memory with redundancy across backends.

        Args:
            key: Unique key for the data
            data: Data to store

        Returns:
            bool: True if at least one backend succeeded
        """
        try:
            # Try to store in primary backend first
            primary_success = await self._try_backends_operation("store", key, data)

            if primary_success:
                # Sync to other backends in background for redundancy
                asyncio.create_task(self._sync_to_backends(key, data))
                return True
            else:
                logger.error(f"Failed to store {key} in any backend")
                return False

        except Exception as e:
            logger.error(f"Failed to store data: {e}")
            return False

    async def retrieve(self, key: str) -> Optional[Any]:
        """
        Asynchronously retrieve data from memory with failover across backends.

        Args:
            key: Key to retrieve

        Returns:
            Data if found, None otherwise
        """
        try:
            return await self._try_backends_operation("retrieve", key)
        except Exception as e:
            logger.error(f"Failed to retrieve data for key {key}: {e}")
            return None

    def store_sync(self, key: str, data: Any) -> bool:
        """
        Synchronously store data in memory with redundancy.

        Args:
            key: Unique key for the data
            data: Data to store

        Returns:
            bool: True if successful
        """
        try:
            # Use asyncio to run async operation in sync context
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.store(key, data))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Failed to store data synchronously: {e}")
            return False

    async def store_memory(self, key: str, data: Any, memory_type: str = "long_term",
                           metadata: Dict[str, Any] = None, user: str = None) -> bool:
        """
        Store memory with type classification and metadata across redundant backends.

        Args:
            key: Unique key for the memory
            data: Data to store
            memory_type: Type of memory (long_term, short_term, working, etc.)
            metadata: Additional metadata
            user: User/context identifier

        Returns:
            bool: True if successful
        """
        try:
            memory_entry = {
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'memory_type': memory_type,
                'metadata': metadata or {},
                'user': user,
                'size': len(str(data)) if data else 0
            }

            # Store in redundant backends
            success = await self.store(key, memory_entry)
            if success:
                logger.debug(f"Stored {memory_type} memory: {key}")
            return success

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return False

    async def retrieve_memory(self, key: str, memory_type: str = None) -> Optional[Any]:
        """
        Retrieve memory by key with failover, optionally filtering by type.

        Args:
            key: Key to retrieve
            memory_type: Optional type filter

        Returns:
            Memory data if found and matches type filter, None otherwise
        """
        try:
            data = await self.retrieve(key)
            if data and isinstance(data, dict):
                # Check type filter if specified
                if memory_type is None or data.get('memory_type') == memory_type:
                    return data['data']
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve memory: {e}")
            return None

    async def search_memories(self, query: str, memory_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search memories by query string across all backends.

        Args:
            query: Search query
            memory_type: Optional type filter
            limit: Maximum results to return

        Returns:
            List of matching memory entries
        """
        try:
            all_results = []

            # Search across all healthy backends
            for backend_name in self.primary_order:
                if backend_name not in self.backends or not self.backend_health.get(backend_name, True):
                    continue

                backend = self.backends[backend_name]
                try:
                    if asyncio.iscoroutinefunction(backend.search):
                        results = await backend.search(query, limit)
                    else:
                        results = backend.search(query, limit)

                    # Add backend info to results
                    for result in results:
                        result['backend'] = backend_name
                        result['backend_health'] = 'healthy'
                        all_results.append(result)

                except Exception as e:
                    logger.warning(f"Search failed on {backend_name} backend: {e}")
                    self.backend_health[backend_name] = False

            # Sort by relevance and limit results
            all_results.sort(key=lambda x: x.get('relevance', 0), reverse=True)
            return all_results[:limit]

        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []

    def get_memory_health_status(self) -> Dict[str, Any]:
        """
        Get health status of all memory backends.

        Returns:
            Dict with backend health information
        """
        status = {
            "overall_health": "healthy" if any(self.backend_health.values()) else "critical",
            "backends": {},
            "redundancy_level": sum(1 for healthy in self.backend_health.values() if healthy),
            "total_backends": len(self.backends)
        }

        for name, backend in self.backends.items():
            try:
                stats = backend.get_stats()
                status["backends"][name] = {
                    "healthy": self.backend_health.get(name, False),
                    "stats": stats
                }
            except Exception as e:
                status["backends"][name] = {
                    "healthy": False,
                    "error": str(e)
                }

        return status

    async def repair_memory_consistency(self, key: str) -> bool:
        """
        Repair memory consistency by syncing data from healthy backends to failed ones.

        Args:
            key: Memory key to repair

        Returns:
            bool: True if repair successful
        """
        try:
            # Find data from a healthy backend
            data = None
            source_backend = None

            for backend_name in self.primary_order:
                if backend_name not in self.backends or not self.backend_health.get(backend_name, True):
                    continue

                backend = self.backends[backend_name]
                try:
                    if asyncio.iscoroutinefunction(backend.retrieve):
                        retrieved_data = await backend.retrieve(key)
                    else:
                        retrieved_data = backend.retrieve(key)

                    if retrieved_data is not None:
                        data = retrieved_data
                        source_backend = backend_name
                        break
                except Exception as e:
                    logger.debug(f"Failed to retrieve from {backend_name} during repair: {e}")

            if data is None:
                logger.warning(f"Cannot repair {key}: no healthy backend has the data")
                return False

            # Sync to unhealthy backends
            repaired_count = 0
            for backend_name, backend in self.backends.items():
                if backend_name == source_backend or self.backend_health.get(backend_name, True):
                    continue  # Skip source or already healthy backends

                try:
                    if asyncio.iscoroutinefunction(backend.store):
                        success = await backend.store(key, data)
                    else:
                        success = backend.store(key, data)

                    if success:
                        self.backend_health[backend_name] = True
                        repaired_count += 1
                        logger.info(f"Repaired {key} on {backend_name} backend")

                except Exception as e:
                    logger.debug(f"Failed to repair {key} on {backend_name}: {e}")

            return repaired_count > 0

        except Exception as e:
            logger.error(f"Failed to repair memory consistency for {key}: {e}")
            return False

# Singleton instance
_memory_manager_instance = None

def get_memory_manager() -> AdvancedMemoryManager:
    """
    Get singleton instance of AdvancedMemoryManager.

    Returns:
        AdvancedMemoryManager: Singleton instance
    """
    global _memory_manager_instance
    if _memory_manager_instance is None:
        _memory_manager_instance = AdvancedMemoryManager()
    return _memory_manager_instance

# Global instance
_advanced_memory_manager = None

def get_advanced_memory_manager() -> AdvancedMemoryManager:
    """
    Get global advanced memory manager instance.

    Returns:
        AdvancedMemoryManager: Global instance
    """
    global _advanced_memory_manager
    if _advanced_memory_manager is None:
        _advanced_memory_manager = AdvancedMemoryManager()
    return _advanced_memory_manager

# Convenience functions
async def store_memory(key: str, data: Any, memory_type: str = "long_term",
                      metadata: Dict[str, Any] = None) -> bool:
    """Convenience function to store memory."""
    return await get_advanced_memory_manager().store_memory(key, data, memory_type, metadata)

async def retrieve_memory(key: str, memory_type: str = None) -> Optional[Any]:
    """Convenience function to retrieve memory."""
    return await get_advanced_memory_manager().retrieve_memory(key, memory_type)

async def search_memories(query: str, memory_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
    """Convenience function to search memories."""
    return await get_advanced_memory_manager().search_memories(query, memory_type, limit)

def get_memory_health_status() -> Dict[str, Any]:
    """Get health status of memory backends."""
    return get_advanced_memory_manager().get_memory_health_status()

async def repair_memory_consistency(key: str) -> bool:
    """Repair memory consistency across backends."""
    return await get_advanced_memory_manager().repair_memory_consistency(key)