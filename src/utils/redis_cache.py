# src/utils/redis_cache.py
# Purpose: Redis-based caching layer for high-performance data caching
# Provides distributed caching with TTL, size limits, and automatic cleanup
# Used for sentiment analysis, predictive analytics, and other expensive operations

import json
import logging
import asyncio
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta
import hashlib
import os
import time

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from .alert_manager import get_alert_manager

alert_manager = get_alert_manager()

logger = logging.getLogger(__name__)

class RedisCacheManager:
    """
    Redis-based caching manager for high-performance distributed caching.
    Provides TTL-based expiration, size limits, and automatic cleanup.
    """

    def __init__(self, host: str = None, port: int = None, db: int = 0,
                 password: Optional[str] = None, max_memory_mb: int = 512,
                 default_ttl_seconds: int = 3600):
        """
        Initialize Redis cache manager.

        Args:
            host: Redis server host (defaults to localhost or REDIS_HOST env var)
            port: Redis server port (defaults to 6379 or REDIS_PORT env var)
            db: Redis database number
            password: Redis password (if required)
            max_memory_mb: Maximum memory usage in MB
            default_ttl_seconds: Default TTL for cache entries
        """
        # Use environment variables if available, otherwise defaults
        self.host = host or os.getenv('REDIS_HOST', 'localhost')
        self.port = port or int(os.getenv('REDIS_PORT', '6379'))
        self.db = db
        self.password = password or os.getenv('REDIS_PASSWORD')
        self.max_memory_mb = max_memory_mb
        self.default_ttl_seconds = default_ttl_seconds

        self.redis_client = None
        self._connect()

        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.sets = 0

        logger.info(f"Redis cache manager initialized (host={host}:{port}, db={db})")

    def _connect(self) -> bool:
        """Establish connection to Redis server."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available - install with: pip install redis")
            return False

        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )

            # Test connection
            self.redis_client.ping()
            logger.info("Successfully connected to Redis")

            # Configure Redis memory management
            self._configure_memory_management()

            return True

        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
            return False
        except Exception as e:
            logger.error(f"Redis initialization error: {e}")
            self.redis_client = None
            return False

    def _configure_memory_management(self):
        """Configure Redis memory management settings."""
        try:
            # Set maximum memory
            max_memory_bytes = self.max_memory_mb * 1024 * 1024
            self.redis_client.config_set('maxmemory', max_memory_bytes)

            # Set eviction policy (LRU - Least Recently Used)
            self.redis_client.config_set('maxmemory-policy', 'allkeys-lru')

            logger.info(f"Redis memory management configured: {self.max_memory_mb}MB max, LRU eviction")

        except Exception as e:
            logger.warning(f"Failed to configure Redis memory management: {e}")

    def _generate_key(self, namespace: str, key: str) -> str:
        """Generate a Redis key with namespace."""
        # Create a hash of the key for consistent length
        key_hash = hashlib.md5(key.encode()).hexdigest()[:8]
        return f"{namespace}:{key_hash}"

    def _serialize_value(self, value: Any) -> str:
        """Serialize value for Redis storage."""
        try:
            # Add metadata
            data = {
                'value': value,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }
            return json.dumps(data, default=str)
        except Exception as e:
            logger.error(f"Failed to serialize value: {e}")
            return json.dumps({'error': 'serialization_failed', 'timestamp': datetime.now().isoformat()})

    def _deserialize_value(self, data: str) -> Optional[Any]:
        """Deserialize value from Redis storage."""
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict) and 'value' in parsed:
                return parsed['value']
            return parsed
        except Exception as e:
            logger.error(f"Failed to deserialize value: {e}")
            return None

    def get(self, namespace: str, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            namespace: Cache namespace (e.g., 'sentiment', 'predictive')
            key: Cache key

        Returns:
            Cached value or None if not found/expired

        Raises:
            RuntimeError: If Redis is not available
        """
        if not self.redis_client:
            raise RuntimeError("Redis cache is not available - Redis connection failed")

        redis_key = self._generate_key(namespace, key)

        try:
            data = self.redis_client.get(redis_key)
            if data:
                self.hits += 1
                value = self._deserialize_value(data)
                logger.debug(f"Cache hit for {namespace}:{key}")
                return value
            else:
                self.misses += 1
                logger.debug(f"Cache miss for {namespace}:{key}")
                return None

        except Exception as e:
            logger.error(f"Redis get error for key {redis_key}: {e}")
            return None

    def set(self, namespace: str, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """
        Set value in cache with TTL.

        Args:
            namespace: Cache namespace
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds (uses default if None)

        Returns:
            True if successful

        Raises:
            RuntimeError: If Redis is not available
        """
        if not self.redis_client:
            raise RuntimeError("Redis cache is not available - Redis connection failed")

        redis_key = self._generate_key(namespace, key)
        ttl = ttl_seconds or self.default_ttl_seconds

        try:
            serialized = self._serialize_value(value)
            success = self.redis_client.setex(redis_key, ttl, serialized)

            if success:
                self.sets += 1
                logger.debug(f"Cached {namespace}:{key} for {ttl}s")
                return True
            else:
                logger.warning(f"Failed to cache {namespace}:{key}")
                return False

        except Exception as e:
            logger.error(f"Redis set error for key {redis_key}: {e}")
            return False

    def delete(self, namespace: str, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            namespace: Cache namespace
            key: Cache key

        Returns:
            True if deleted, False otherwise
        """
        if not self.redis_client:
            return False

        redis_key = self._generate_key(namespace, key)

        try:
            result = self.redis_client.delete(redis_key)
            return result > 0

        except Exception as e:
            logger.error(f"Redis delete error for key {redis_key}: {e}")
            return False

    def clear_namespace(self, namespace: str) -> int:
        """
        Clear all entries in a namespace.

        Args:
            namespace: Cache namespace to clear

        Returns:
            Number of keys deleted
        """
        if not self.redis_client:
            return 0

        try:
            # Get all keys in namespace
            pattern = f"{namespace}:*"
            keys = self.redis_client.keys(pattern)

            if keys:
                result = self.redis_client.delete(*keys)
                logger.info(f"Cleared {result} keys from namespace {namespace}")
                return result
            else:
                logger.info(f"No keys found in namespace {namespace}")
                return 0

        except Exception as e:
            logger.error(f"Redis clear namespace error for {namespace}: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

        stats = {
            'hits': self.hits,
            'misses': self.misses,
            'sets': self.sets,
            'hit_rate_percent': round(hit_rate, 2),
            'total_requests': total_requests,
            'redis_connected': self.redis_client is not None
        }

        # Add Redis info if connected
        if self.redis_client:
            try:
                info = self.redis_client.info()
                stats.update({
                    'redis_memory_used': info.get('used_memory_human', 'unknown'),
                    'redis_memory_peak': info.get('used_memory_peak_human', 'unknown'),
                    'redis_keys_total': info.get('db0', {}).get('keys', 0) if 'db0' in info else 0,
                    'redis_uptime_days': round(info.get('uptime_in_days', 0), 1)
                })
            except Exception as e:
                logger.warning(f"Failed to get Redis info: {e}")

        return stats

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Redis connection."""
        health = {
            'service': 'redis_cache',
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'details': {}
        }

        if not REDIS_AVAILABLE:
            health['details']['error'] = 'Redis library not available'
            return health

        if not self.redis_client:
            health['details']['error'] = 'Redis client not initialized'
            return health

        try:
            # Test basic operations
            self.redis_client.ping()
            test_key = 'health_check_test'
            self.redis_client.setex(test_key, 10, 'test_value')
            value = self.redis_client.get(test_key)
            self.redis_client.delete(test_key)

            if value == 'test_value':
                health['status'] = 'healthy'
                health['details']['latency_ms'] = 'ok'
                health['details']['operations'] = 'working'
            else:
                health['details']['error'] = 'Basic operations failed'

        except Exception as e:
            health['details']['error'] = str(e)

        return health

    async def cleanup_expired(self):
        """Async cleanup of expired keys (Redis handles this automatically with TTL)."""
        # Redis automatically handles TTL expiration
        # This method is here for potential future enhancements
        pass

# Global cache manager instance
_cache_manager = None

def get_redis_cache_manager() -> Optional[RedisCacheManager]:
    """Get global Redis cache manager instance."""
    global _cache_manager

    if _cache_manager is None:
        # Try to connect to Redis
        _cache_manager = RedisCacheManager()

        # If connection failed, _cache_manager.redis_client will be None
        if not _cache_manager.redis_client:
            logger.warning("Redis cache manager created but not connected")
            # Don't set to None, keep the instance for fallback behavior

    return _cache_manager

# Convenience functions for common operations
def cache_get(namespace: str, key: str) -> Optional[Any]:
    """Convenience function to get from cache."""
    try:
        manager = get_redis_cache_manager()
        if manager:
            return manager.get(namespace, key)
        else:
            alert_manager.warning(
                "Cache manager not available for get operation",
                {"namespace": namespace, "key": key},
                "redis_cache"
            )
            return None
    except Exception as e:
        alert_manager.error(
            f"Cache get operation failed: {namespace}:{key}",
            {"namespace": namespace, "key": key, "error": str(e)},
            "redis_cache"
        )
        return None

def cache_set(namespace: str, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
    """Convenience function to set in cache."""
    try:
        manager = get_redis_cache_manager()
        if manager:
            return manager.set(namespace, key, value, ttl_seconds)
        else:
            alert_manager.warning(
                "Cache manager not available for set operation",
                {"namespace": namespace, "key": key, "ttl_seconds": ttl_seconds},
                "redis_cache"
            )
            return False
    except Exception as e:
        alert_manager.error(
            f"Cache set operation failed: {namespace}:{key}",
            {"namespace": namespace, "key": key, "ttl_seconds": ttl_seconds, "error": str(e)},
            "redis_cache"
        )
        return False

def cache_delete(namespace: str, key: str) -> bool:
    """Convenience function to delete from cache."""
    manager = get_redis_cache_manager()
    return manager.delete(namespace, key) if manager else False

def cache_clear_namespace(namespace: str) -> int:
    """Convenience function to clear namespace."""
    manager = get_redis_cache_manager()
    return manager.clear_namespace(namespace) if manager else 0

def get_cache_stats() -> Dict[str, Any]:
    """Convenience function to get cache stats."""
    manager = get_redis_cache_manager()
    return manager.get_stats() if manager else {'error': 'Cache manager not available'}