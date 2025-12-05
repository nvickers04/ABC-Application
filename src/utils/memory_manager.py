#!/usr/bin/env python3
"""
Advanced Memory Management for ABC Application System
Implements memory pooling, lazy loading, and efficient data structures.
"""

import asyncio
import logging
import time
import weakref
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from contextlib import asynccontextmanager
import gc
import sys
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class MemoryPoolConfig:
    """Configuration for memory pool management."""
    max_pool_size: int = 100
    cleanup_interval: float = 60.0  # seconds
    memory_threshold_mb: float = 256
    enable_gc_optimization: bool = True
    lazy_loading_enabled: bool = True

@dataclass
class MemoryObject:
    """Wrapper for objects in memory pool."""
    obj: Any
    ref_count: int = 0
    last_accessed: float = 0.0
    size_bytes: int = 0

class MemoryPool:
    """
    Memory pool for efficient object reuse and management.
    """

    def __init__(self, config: Optional[MemoryPoolConfig] = None):
        self.config = config or MemoryPoolConfig()
        self.pool: Dict[str, MemoryObject] = {}
        self.access_history: List[str] = []
        self.cleanup_task: Optional[asyncio.Task] = None

    async def start_cleanup_task(self) -> None:
        """Start the periodic cleanup task."""
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def stop_cleanup_task(self) -> None:
        """Stop the periodic cleanup task."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            # Don't set to None immediately - let the test check if it's done first
            # self.cleanup_task = None

    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of unused objects."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_unused_objects()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Memory pool cleanup error: {e}")

    async def _cleanup_unused_objects(self) -> None:
        """Clean up objects that haven't been used recently."""
        current_time = asyncio.get_running_loop().time()
        to_remove = []

        for key, mem_obj in self.pool.items():
            # Remove objects not accessed in the last 2 cleanup intervals
            if current_time - mem_obj.last_accessed > self.config.cleanup_interval * 2:
                to_remove.append(key)

        for key in to_remove:
            del self.pool[key]
            logger.debug(f"Cleaned up unused object: {key}")

        # Force garbage collection if pool is getting large
        if len(self.pool) > self.config.max_pool_size * 0.8:
            gc.collect()
            logger.debug(f"Garbage collection performed, pool size: {len(self.pool)}")

    def get(self, key: str) -> Optional[Any]:
        """Get an object from the memory pool."""
        if key in self.pool:
            mem_obj = self.pool[key]
            mem_obj.ref_count += 1
            mem_obj.last_accessed = time.time()
            self.access_history.append(key)
            return mem_obj.obj
        return None

    def put(self, key: str, obj: Any, size_bytes: int = 0) -> None:
        """Put an object into the memory pool."""
        if len(self.pool) >= self.config.max_pool_size:
            # Remove least recently used item
            if self.access_history:
                lru_key = self.access_history.pop(0)
                if lru_key in self.pool:
                    del self.pool[lru_key]

        self.pool[key] = MemoryObject(
            obj=obj,
            ref_count=1,
            last_accessed=time.time(),
            size_bytes=size_bytes
        )

    def remove(self, key: str) -> None:
        """Remove an object from the memory pool."""
        if key in self.pool:
            del self.pool[key]

    def add_object(self, key: str, obj: Any, size_bytes: int = 0) -> bool:
        """
        Add an object to the memory pool.

        Args:
            key: Unique key for the object
            obj: Object to add
            size_bytes: Size of the object in bytes (optional)

        Returns:
            bool: True if successful
        """
        try:
            if len(self.pool) >= self.config.max_pool_size:
                # Remove oldest accessed object
                oldest_key = min(self.pool.keys(),
                               key=lambda k: self.pool[k].last_accessed)
                del self.pool[oldest_key]

            self.pool[key] = MemoryObject(
                obj=obj,
                ref_count=1,
                last_accessed=time.time(),
                size_bytes=size_bytes if size_bytes > 0 else sys.getsizeof(obj)
            )
            return True
        except Exception as e:
            logger.error(f"Failed to add object to pool: {e}")
            return False

    def get_object(self, key: str) -> Optional[Any]:
        """
        Get an object from the memory pool.

        Args:
            key: Key of the object to retrieve

        Returns:
            Object if found, None otherwise
        """
        if key in self.pool:
            obj_wrapper = self.pool[key]
            obj_wrapper.last_accessed = time.time()
            obj_wrapper.ref_count += 1
            self.access_history.append(key)  # Add to access history
            return obj_wrapper.obj
        return None

    def remove_object(self, key: str) -> bool:
        """
        Remove an object from the memory pool.

        Args:
            key: Key of the object to remove

        Returns:
            bool: True if object was removed, False if not found
        """
        if key in self.pool:
            del self.pool[key]
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        total_size = sum(obj.size_bytes for obj in self.pool.values())
        total_refs = sum(obj.ref_count for obj in self.pool.values())

        return {
            'pool_size': len(self.pool),
            'total_size_bytes': total_size,
            'total_references': total_refs,
            'max_pool_size': self.config.max_pool_size,
            'utilization_percent': (len(self.pool) / self.config.max_pool_size) * 100
        }

class LazyDataLoader:
    """
    Lazy loading wrapper for expensive data operations.
    """

    def __init__(self, loader_func: Callable, cache_key: Optional[str] = None):
        self.loader_func = loader_func
        self.cache_key = cache_key
        self._data = None
        self._loaded = False
        self._loading = False

    async def load(self) -> Any:
        """Load data lazily."""
        if self._loaded:
            return self._data

        if self._loading:
            # Wait for ongoing load
            while self._loading:
                await asyncio.sleep(0.1)
            return self._data

        self._loading = True
        try:
            self._data = await self.loader_func()
            self._loaded = True
            return self._data
        finally:
            self._loading = False

    def is_loaded(self) -> bool:
        """Check if data is loaded."""
        return self._loaded

    def unload(self) -> None:
        """Unload data to free memory."""
        self._data = None
        self._loaded = False
        self._loading = False

class DataFramePool:
    """
    Specialized pool for pandas DataFrames with memory optimization.
    """

    def __init__(self, max_frames: int = 20):
        self.max_frames = max_frames
        self.frames: Dict[str, weakref.ref] = {}
        self.access_order: List[str] = []

    def store(self, key: str, df) -> None:
        """Store a DataFrame with memory optimization."""
        # Convert object columns to category if beneficial
        df = self._optimize_dataframe(df)

        # Store weak reference to allow garbage collection
        self.frames[key] = weakref.ref(df, lambda ref: self._cleanup_frame(key))
        self.access_order.append(key)

        # Maintain size limit
        while len(self.frames) > self.max_frames:
            oldest_key = self.access_order.pop(0)
            if oldest_key in self.frames:
                del self.frames[oldest_key]

    def get(self, key: str):
        """Get a DataFrame from the pool."""
        if key in self.frames:
            df_ref = self.frames[key]
            df = df_ref()
            if df is not None:
                # Move to end of access order (most recently used)
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                return df
            else:
                # Reference is dead, clean up
                self._cleanup_frame(key)

        return None

    def _optimize_dataframe(self, df):
        """Optimize DataFrame memory usage."""
        # Convert object columns to category dtype where beneficial
        for col in df.select_dtypes(include=['object']):
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')

        # Downcast numeric types
        for col in df.select_dtypes(include=['int64']):
            df[col] = df[col].astype('int32')

        for col in df.select_dtypes(include=['float64']):
            df[col] = df[col].astype('float32')

        return df

    def _cleanup_frame(self, key: str) -> None:
        """Clean up a frame reference."""
        if key in self.frames:
            del self.frames[key]
        if key in self.access_order:
            self.access_order.remove(key)

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        total_frames = len([k for k, ref in self.frames.items() if ref() is not None])

        return {
            'active_frames': total_frames,
            'max_frames': self.max_frames,
            'utilization_percent': (total_frames / self.max_frames) * 100
        }

class MemoryPoolManager:
    """
    Advanced memory manager coordinating all memory optimization features.
    """

    def __init__(self):
        self.memory_pool = MemoryPool()
        self.dataframe_pool = DataFramePool()
        self.lazy_loaders: Dict[str, LazyDataLoader] = {}
        self.memory_stats: Dict[str, Any] = {
            'peak_usage_mb': 0,
            'current_usage_mb': 0,
            'gc_collections': 0,
            'pool_hits': 0,
            'pool_misses': 0
        }
        self.memory_store: Dict[str, Any] = {}
        self.cleanup_count = 0

    async def initialize(self) -> None:
        """Initialize the memory manager."""
        await self.memory_pool.start_cleanup_task()
        logger.info("Advanced memory manager initialized")

    async def cleanup(self) -> None:
        """Clean up all memory management resources."""
        await self.memory_pool.stop_cleanup_task()

        # Clear pools
        self.memory_pool.pool.clear()
        self.dataframe_pool.frames.clear()
        self.lazy_loaders.clear()

        # Force final garbage collection
        gc.collect()
        logger.info("Advanced memory manager cleaned up")

    @asynccontextmanager
    async def memory_efficient_context(self, operation_name: str = "operation"):
        """Context manager for memory-efficient operations."""
        start_memory = self._get_memory_usage_mb()

        try:
            yield
        finally:
            end_memory = self._get_memory_usage_mb()
            memory_delta = end_memory - start_memory

            self.memory_stats['peak_usage_mb'] = max(
                self.memory_stats['peak_usage_mb'], end_memory
            )

            if memory_delta > 50:  # Significant memory increase
                logger.warning(f"High memory usage in {operation_name}: +{memory_delta:.1f}MB")

            # Trigger cleanup if memory usage is high
            if end_memory > 400:  # 400MB threshold
                await self._emergency_cleanup()

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    async def _emergency_cleanup(self) -> None:
        """Perform emergency memory cleanup."""
        logger.warning("Performing emergency memory cleanup")

        # Clear lazy loaders that aren't in use
        to_remove = []
        for key, loader in self.lazy_loaders.items():
            if not loader.is_loaded():
                to_remove.append(key)

        for key in to_remove:
            del self.lazy_loaders[key]

        # Force garbage collection
        collected = gc.collect()
        self.memory_stats['gc_collections'] += 1

        logger.info(f"Emergency cleanup completed: {collected} objects collected")

    def create_lazy_loader(self, key: str, loader_func: Callable) -> LazyDataLoader:
        """Create a lazy data loader."""
        loader = LazyDataLoader(loader_func, key)
        self.lazy_loaders[key] = loader
        return loader

    def get_lazy_loader(self, key: str) -> Optional[LazyDataLoader]:
        """Get a lazy loader by key."""
        return self.lazy_loaders.get(key)

    def store_dataframe(self, key: str, df) -> None:
        """Store a DataFrame in the optimized pool."""
        self.dataframe_pool.store(key, df)

    def get_dataframe(self, key: str):
        """Get a DataFrame from the optimized pool."""
        return self.dataframe_pool.get(key)

    def pool_get(self, key: str) -> Optional[Any]:
        """Get an object from the memory pool."""
        obj = self.memory_pool.get(key)
        if obj is not None:
            self.memory_stats['pool_hits'] += 1
        else:
            self.memory_stats['pool_misses'] += 1
        return obj

    def pool_put(self, key: str, obj: Any, size_bytes: int = 0) -> None:
        """Put an object in the memory pool."""
        self.memory_pool.put(key, obj, size_bytes)

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        return {
            'memory_stats': self.memory_stats,
            'pool_stats': self.memory_pool.get_stats(),
            'dataframe_stats': self.dataframe_pool.get_memory_usage(),
            'lazy_loaders_count': len(self.lazy_loaders),
            'active_lazy_loaders': len([l for l in self.lazy_loaders.values() if l.is_loaded()])
        }

    async def store(self, key: str, data: Any) -> bool:
        """
        Asynchronously store data in memory.

        Args:
            key: Unique key for the data
            data: Data to store

        Returns:
            bool: True if successful
        """
        try:
            self.memory_store[key] = {
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'size': len(str(data)) if data else 0
            }
            return True
        except Exception as e:
            logger.error(f"Failed to store data: {e}")
            return False

    async def retrieve(self, key: str) -> Optional[Any]:
        """
        Asynchronously retrieve data from memory.

        Args:
            key: Key to retrieve

        Returns:
            Data if found, None otherwise
        """
        if key in self.memory_store:
            return self.memory_store[key]['data']
        return None

    def store_sync(self, key: str, data: Any) -> bool:
        """
        Synchronously store data in memory.

        Args:
            key: Unique key for the data
            data: Data to store

        Returns:
            bool: True if successful
        """
        try:
            self.memory_store[key] = {
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'size': len(str(data)) if data else 0
            }
            return True
        except Exception as e:
            logger.error(f"Failed to store data synchronously: {e}")
            return False

    def cleanup_memory(self) -> int:
        """
        Clean up old or unnecessary memory entries.

        Returns:
            int: Number of entries cleaned up
        """
        try:
            # Simple cleanup: remove entries older than 1 hour
            cutoff_time = datetime.now() - timedelta(hours=1)
            keys_to_remove = []

            for key, entry in self.memory_store.items():
                entry_time = datetime.fromisoformat(entry['timestamp'])
                if entry_time < cutoff_time:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.memory_store[key]

            self.cleanup_count += len(keys_to_remove)
            return len(keys_to_remove)
        except Exception as e:
            logger.error(f"Failed to cleanup memory: {e}")
            return 0

# Global memory manager instance
_memory_manager = None

def get_memory_pool_manager() -> MemoryPoolManager:
    """Get the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryPoolManager()
    return _memory_manager

async def initialize_memory_management() -> None:
    """Initialize global memory management."""
    manager = get_memory_manager()
    await manager.initialize()

async def cleanup_memory_management() -> None:
    """Clean up global memory management."""
    manager = get_memory_manager()
    manager.cleanup_memory()