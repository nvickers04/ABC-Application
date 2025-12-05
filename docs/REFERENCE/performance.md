# Performance Optimization Guide

## ⚡ Performance Optimization Strategies

This guide covers techniques for optimizing ABC-Application performance across different components and workloads.

## 📊 Performance Monitoring

### Key Metrics to Monitor
- **Response Time**: API call latency, agent processing time
- **Throughput**: Requests per second, trades per minute
- **Resource Usage**: CPU, memory, disk I/O, network
- **Error Rates**: Failed requests, timeout rates
- **Cache Hit Rates**: Redis cache effectiveness

### Monitoring Setup
```python
# src/utils/performance_monitor.py
import time
import psutil
import threading
from typing import Dict, Any, List
from collections import deque
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor application performance metrics."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.response_times = deque(maxlen=window_size)
        self.cpu_usage = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self._monitoring = False
        self._monitor_thread = None

    def start_monitoring(self):
        """Start background performance monitoring."""
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("Performance monitoring stopped")

    def _monitor_loop(self):
        """Background monitoring loop."""
        process = psutil.Process()
        while self._monitoring:
            # CPU usage
            cpu_percent = process.cpu_percent(interval=1)
            self.cpu_usage.append(cpu_percent)

            # Memory usage
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            self.memory_usage.append(memory_mb)

            time.sleep(5)  # Monitor every 5 seconds

    def record_response_time(self, response_time: float):
        """Record API response time."""
        self.response_times.append(response_time)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            'response_time': {
                'avg': sum(self.response_times) / len(self.response_times) if self.response_times else 0,
                'min': min(self.response_times) if self.response_times else 0,
                'max': max(self.response_times) if self.response_times else 0,
                'p95': self._percentile(self.response_times, 95) if self.response_times else 0
            },
            'cpu_usage': {
                'current': self.cpu_usage[-1] if self.cpu_usage else 0,
                'avg': sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
            },
            'memory_usage': {
                'current': self.memory_usage[-1] if self.memory_usage else 0,
                'avg': sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
            },
            'sample_size': len(self.response_times)
        }

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile from data."""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

# Global monitor instance
performance_monitor = PerformanceMonitor()

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    return performance_monitor
```

## 🚀 Agent Performance Optimization

### Agent Processing Optimization
```python
# src/agents/base.py - Optimized agent processing
import asyncio
import time
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

class OptimizedBaseAgent:
    """Base agent with performance optimizations."""

    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._processing_times = []
        self._cache = {}  # Simple in-memory cache

    async def process_input_optimized(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimized input processing with caching and parallel execution."""
        start_time = time.time()

        # Check cache first
        cache_key = self._generate_cache_key(input_data)
        if cache_key in self._cache:
            logger.debug("Cache hit for input processing")
            return self._cache[cache_key]

        try:
            # Parallel processing for independent tasks
            tasks = []

            # Data validation task
            if hasattr(self, '_validate_input'):
                tasks.append(self._validate_input_async(input_data))

            # Main processing task
            tasks.append(self._process_main_async(input_data))

            # Post-processing task
            if hasattr(self, '_post_process'):
                tasks.append(self._post_process_async(input_data))

            # Execute tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Combine results
            result = self._combine_results(results)

            # Cache result
            self._cache[cache_key] = result

            processing_time = time.time() - start_time
            self._processing_times.append(processing_time)

            # Log performance
            if processing_time > 1.0:
                logger.warning(f"Slow agent processing: {processing_time:.2f}s")

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Agent processing failed after {processing_time:.2f}s: {e}")
            raise

    def _generate_cache_key(self, input_data: Dict[str, Any]) -> str:
        """Generate cache key from input data."""
        # Simple key generation - can be made more sophisticated
        key_parts = []
        for k, v in sorted(input_data.items()):
            if isinstance(v, (str, int, float, bool)):
                key_parts.append(f"{k}:{v}")
        return "|".join(key_parts)

    async def _validate_input_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Async input validation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._validate_input, input_data)

    async def _process_main_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Async main processing."""
        return await self.process_input(input_data)  # Override in subclasses

    async def _post_process_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Async post-processing."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._post_process, input_data)

    def _combine_results(self, results: list) -> Dict[str, Any]:
        """Combine results from parallel tasks."""
        combined = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed: {result}")
                continue
            if isinstance(result, dict):
                combined.update(result)
        return combined

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics."""
        if not self._processing_times:
            return {'samples': 0}

        return {
            'samples': len(self._processing_times),
            'avg_time': sum(self._processing_times) / len(self._processing_times),
            'min_time': min(self._processing_times),
            'max_time': max(self._processing_times),
            'cache_size': len(self._cache)
        }
```

### Agent Caching Strategies
```python
# src/utils/agent_cache.py
import asyncio
import hashlib
from typing import Dict, Any, Optional
import redis.asyncio as redis
import json
import logging

logger = logging.getLogger(__name__)

class AgentCache:
    """Redis-backed agent result caching."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self._ttl = 3600  # 1 hour default TTL

    def _generate_key(self, agent_name: str, input_data: Dict[str, Any]) -> str:
        """Generate cache key from agent name and input."""
        # Create deterministic key from input
        input_str = json.dumps(input_data, sort_keys=True)
        input_hash = hashlib.md5(input_str.encode()).hexdigest()[:16]
        return f"agent_cache:{agent_name}:{input_hash}"

    async def get(self, agent_name: str, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached result if available."""
        key = self._generate_key(agent_name, input_data)
        try:
            cached_data = await self.redis.get(key)
            if cached_data:
                result = json.loads(cached_data)
                logger.debug(f"Cache hit for {agent_name}")
                return result
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        return None

    async def set(self, agent_name: str, input_data: Dict[str, Any], result: Dict[str, Any], ttl: Optional[int] = None):
        """Cache result with TTL."""
        key = self._generate_key(agent_name, input_data)
        try:
            data = json.dumps(result)
            await self.redis.setex(key, ttl or self._ttl, data)
            logger.debug(f"Cached result for {agent_name}")
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

    async def invalidate_pattern(self, pattern: str):
        """Invalidate cache keys matching pattern."""
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache keys matching {pattern}")
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            info = await self.redis.info()
            return {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory_human', '0B'),
                'total_keys': await self.redis.dbsize(),
                'hit_rate': info.get('keyspace_hits', 0) / max(1, info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0))
            }
        except Exception as e:
            logger.warning(f"Cache stats retrieval failed: {e}")
            return {}
```

## 💾 Memory Optimization

### Memory-Efficient Data Processing
```python
# src/utils/memory_efficient_processor.py
import pandas as pd
import numpy as np
from typing import Iterator, Dict, Any, List
import gc
import logging

logger = logging.getLogger(__name__)

class MemoryEfficientProcessor:
    """Process large datasets with minimal memory usage."""

    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size

    def process_large_dataframe(self, file_path: str, processor_func) -> Iterator[Dict[str, Any]]:
        """Process large CSV/Parquet files in chunks."""
        if file_path.endswith('.csv'):
            reader = pd.read_csv(file_path, chunksize=self.chunk_size)
        elif file_path.endswith('.parquet'):
            # For Parquet, we need to read in chunks differently
            reader = self._parquet_chunk_reader(file_path, self.chunk_size)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        for chunk in reader:
            try:
                # Process chunk
                result = processor_func(chunk)

                # Force garbage collection between chunks
                del chunk
                gc.collect()

                yield result

            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                continue

    def _parquet_chunk_reader(self, file_path: str, chunk_size: int) -> Iterator[pd.DataFrame]:
        """Read Parquet file in chunks."""
        # This is a simplified implementation
        # In practice, you'd use libraries like fastparquet or pyarrow
        df = pd.read_parquet(file_path)

        for i in range(0, len(df), chunk_size):
            yield df.iloc[i:i + chunk_size].copy()

    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        # Downcast numeric types
        for col in df.select_dtypes(include=['int64']):
            df[col] = pd.to_numeric(df[col], downcast='integer')

        for col in df.select_dtypes(include=['float64']):
            df[col] = pd.to_numeric(df[col], downcast='float')

        # Convert object columns to category if appropriate
        for col in df.select_dtypes(include=['object']):
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')

        # Remove duplicate rows
        df = df.drop_duplicates()

        return df

    def streaming_aggregation(self, data_stream: Iterator[Dict[str, Any]], group_by: str, agg_func) -> Dict[str, Any]:
        """Perform streaming aggregation to minimize memory usage."""
        groups = {}

        for item in data_stream:
            key = item.get(group_by)
            if key not in groups:
                groups[key] = []

            groups[key].append(item)

            # Process groups that exceed memory threshold
            if len(groups[key]) > self.chunk_size:
                groups[key] = [agg_func(groups[key])]

        # Final aggregation
        result = {}
        for key, items in groups.items():
            result[key] = agg_func(items)

        return result
```

### Memory Pool Management
```python
# src/utils/memory_pool.py
import threading
import gc
from typing import Dict, Any, List
import psutil
import logging

logger = logging.getLogger(__name__)

class MemoryPool:
    """Manage memory usage across the application."""

    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.lock = threading.Lock()
        self.memory_usage = 0
        self.allocations = []

    def allocate(self, size_mb: float, description: str = "") -> bool:
        """Request memory allocation."""
        with self.lock:
            if self.memory_usage + size_mb > self.max_memory_mb:
                logger.warning(f"Memory allocation denied: would exceed limit ({self.memory_usage + size_mb:.1f}MB > {self.max_memory_mb}MB)")
                return False

            self.memory_usage += size_mb
            self.allocations.append({
                'size': size_mb,
                'description': description,
                'timestamp': threading.current_thread().ident
            })

            logger.debug(f"Memory allocated: {size_mb:.1f}MB for {description}")
            return True

    def deallocate(self, size_mb: float):
        """Release memory allocation."""
        with self.lock:
            self.memory_usage = max(0, self.memory_usage - size_mb)

            # Remove matching allocation
            for i, alloc in enumerate(self.allocations):
                if alloc['size'] == size_mb:
                    del self.allocations[i]
                    break

            logger.debug(f"Memory deallocated: {size_mb:.1f}MB")

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        actual_memory = process.memory_info().rss / 1024 / 1024

        return {
            'pool_limit': self.max_memory_mb,
            'pool_used': self.memory_usage,
            'pool_available': self.max_memory_mb - self.memory_usage,
            'actual_memory': actual_memory,
            'allocations': len(self.allocations),
            'allocation_details': self.allocations.copy()
        }

    def force_cleanup(self):
        """Force garbage collection and cleanup."""
        with self.lock:
            # Aggressive garbage collection
            gc.collect()
            gc.collect()  # Second pass

            # Clear any cached objects
            self.allocations.clear()
            self.memory_usage = 0

            logger.info("Memory pool cleanup completed")

# Global memory pool instance
memory_pool = MemoryPool()

def get_memory_pool() -> MemoryPool:
    """Get global memory pool instance."""
    return memory_pool
```

## ⚡ API Performance Optimization

### Connection Pooling
```python
# src/utils/connection_pool.py
import asyncio
import aiohttp
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class APIConnectionPool:
    """Manage HTTP connections for API calls."""

    def __init__(self, max_connections: int = 20, timeout: int = 30):
        self.max_connections = max_connections
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None

    async def initialize(self):
        """Initialize connection pool."""
        if self._session is None:
            self._connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections // 2,
                ttl_dns_cache=300,  # 5 minutes
                use_dns_cache=True,
                keepalive_timeout=60,
                enable_cleanup_closed=True
            )

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=timeout,
                trust_env=True  # Use environment proxy settings
            )

            logger.info(f"API connection pool initialized with {self.max_connections} max connections")

    async def close(self):
        """Close connection pool."""
        if self._session:
            await self._session.close()
        if self._connector:
            await self._connector.close()
        logger.info("API connection pool closed")

    async def request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make HTTP request using connection pool."""
        if self._session is None:
            await self.initialize()

        try:
            async with self._session.request(method, url, **kwargs) as response:
                # Clone response for return (since context manager will close it)
                return await self._clone_response(response)
        except Exception as e:
            logger.error(f"API request failed: {method} {url} - {e}")
            raise

    async def _clone_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """Clone response data for return."""
        return {
            'status': response.status,
            'headers': dict(response.headers),
            'text': await response.text(),
            'json': await response.json() if response.content_type == 'application/json' else None
        }

# Global connection pool
api_pool = APIConnectionPool()

async def get_api_pool() -> APIConnectionPool:
    """Get global API connection pool."""
    return api_pool
```

### Request Batching and Parallelization
```python
# src/utils/request_batcher.py
import asyncio
import time
from typing import List, Dict, Any, Callable, Awaitable
import logging

logger = logging.getLogger(__name__)

class RequestBatcher:
    """Batch and parallelize API requests."""

    def __init__(self, batch_size: int = 10, max_concurrent: int = 5):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def batch_execute(
        self,
        requests: List[Dict[str, Any]],
        executor: Callable[[Dict[str, Any]], Awaitable[Any]]
    ) -> List[Any]:
        """Execute requests in batches with concurrency control."""
        results = []

        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]

            # Execute batch concurrently
            batch_tasks = []
            for request in batch:
                task = self._execute_with_semaphore(executor, request)
                batch_tasks.append(task)

            # Wait for batch to complete
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)

            # Small delay between batches to avoid overwhelming APIs
            await asyncio.sleep(0.1)

        return results

    async def _execute_with_semaphore(
        self,
        executor: Callable[[Dict[str, Any]], Awaitable[Any]],
        request: Dict[str, Any]
    ) -> Any:
        """Execute request with concurrency control."""
        async with self.semaphore:
            try:
                start_time = time.time()
                result = await executor(request)
                duration = time.time() - start_time

                if duration > 5.0:  # Log slow requests
                    logger.warning(f"Slow request: {duration:.2f}s")

                return result

            except Exception as e:
                logger.error(f"Request execution failed: {e}")
                return {'error': str(e), 'request': request}

    async def adaptive_batch_execute(
        self,
        requests: List[Dict[str, Any]],
        executor: Callable[[Dict[str, Any]], Awaitable[Any]],
        target_latency: float = 2.0
    ) -> List[Any]:
        """Execute requests with adaptive batch sizing based on performance."""
        results = []
        current_batch_size = self.batch_size

        for i in range(0, len(requests), current_batch_size):
            batch = requests[i:i + current_batch_size]

            start_time = time.time()
            batch_results = await self.batch_execute(batch, executor)
            batch_duration = time.time() - start_time

            results.extend(batch_results)

            # Adjust batch size based on performance
            avg_latency = batch_duration / len(batch)
            if avg_latency > target_latency:
                current_batch_size = max(1, current_batch_size // 2)
                logger.info(f"Reducing batch size to {current_batch_size} due to high latency")
            elif avg_latency < target_latency * 0.5:
                current_batch_size = min(len(requests), current_batch_size * 2)
                logger.info(f"Increasing batch size to {current_batch_size} due to good performance")

        return results
```

## 🗄️ Database Optimization

### Query Optimization
```python
# src/utils/query_optimizer.py
import asyncio
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class QueryMetrics:
    """Metrics for database query performance."""
    query: str
    execution_time: float
    rows_returned: int
    timestamp: float

class QueryOptimizer:
    """Optimize database queries and cache results."""

    def __init__(self):
        self.query_cache = {}
        self.metrics = []
        self.max_cache_size = 1000

    async def execute_optimized(
        self,
        query_func: callable,
        query_key: str,
        cache_ttl: int = 300
    ) -> Any:
        """Execute query with caching and metrics."""
        # Check cache first
        if query_key in self.query_cache:
            cache_entry = self.query_cache[query_key]
            if time.time() - cache_entry['timestamp'] < cache_ttl:
                logger.debug(f"Query cache hit: {query_key}")
                return cache_entry['result']

        # Execute query
        start_time = time.time()
        try:
            result = await query_func()
            execution_time = time.time() - start_time

            # Record metrics
            metrics = QueryMetrics(
                query=query_key,
                execution_time=execution_time,
                rows_returned=len(result) if hasattr(result, '__len__') else 1,
                timestamp=time.time()
            )
            self.metrics.append(metrics)

            # Cache result
            self.query_cache[query_key] = {
                'result': result,
                'timestamp': time.time()
            }

            # Maintain cache size
            if len(self.query_cache) > self.max_cache_size:
                # Remove oldest entries
                sorted_cache = sorted(
                    self.query_cache.items(),
                    key=lambda x: x[1]['timestamp']
                )
                self.query_cache = dict(sorted_cache[-self.max_cache_size:])

            # Log slow queries
            if execution_time > 1.0:
                logger.warning(f"Slow query: {query_key} took {execution_time:.2f}s")

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query failed after {execution_time:.2f}s: {query_key} - {e}")
            raise

    def get_query_stats(self) -> Dict[str, Any]:
        """Get query performance statistics."""
        if not self.metrics:
            return {'total_queries': 0}

        execution_times = [m.execution_time for m in self.metrics]

        return {
            'total_queries': len(self.metrics),
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'max_execution_time': max(execution_times),
            'min_execution_time': min(execution_times),
            'cache_size': len(self.query_cache),
            'cache_hit_rate': self._calculate_cache_hit_rate()
        }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate from recent metrics."""
        if not self.metrics:
            return 0.0

        # Simplified calculation - in practice you'd track hits vs misses
        return min(1.0, len(self.query_cache) / max(1, len(self.metrics) * 0.1))
```

## 🔄 Async Optimization

### Async Task Scheduling
```python
# src/utils/async_scheduler.py
import asyncio
import heapq
from typing import Callable, Any, List, Tuple
import time
import logging

logger = logging.getLogger(__name__)

class AsyncScheduler:
    """Asynchronous task scheduler with priority queuing."""

    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.task_queue = []
        self.running = False
        self.task_counter = 0

    def schedule_task(
        self,
        coro: Callable[[], Any],
        priority: int = 0,
        delay: float = 0
    ) -> str:
        """Schedule a task with priority and optional delay."""
        task_id = f"task_{self.task_counter}"
        self.task_counter += 1

        scheduled_time = time.time() + delay

        # Add to priority queue (negative priority for min-heap behavior)
        heapq.heappush(
            self.task_queue,
            (-priority, scheduled_time, task_id, coro)
        )

        logger.debug(f"Scheduled task {task_id} with priority {priority}, delay {delay}s")
        return task_id

    async def start_scheduler(self):
        """Start the task scheduler."""
        self.running = True
        logger.info("Async scheduler started")

        while self.running:
            try:
                # Check for due tasks
                current_time = time.time()

                due_tasks = []
                while self.task_queue and self.task_queue[0][1] <= current_time:
                    priority, scheduled_time, task_id, coro = heapq.heappop(self.task_queue)
                    due_tasks.append((task_id, coro))

                # Execute due tasks concurrently
                if due_tasks:
                    tasks = []
                    for task_id, coro in due_tasks:
                        task = self._execute_task(task_id, coro)
                        tasks.append(task)

                    await asyncio.gather(*tasks, return_exceptions=True)

                # Small sleep to prevent busy waiting
                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(1)

    async def _execute_task(self, task_id: str, coro: Callable[[], Any]):
        """Execute a task with concurrency control."""
        async with self.semaphore:
            try:
                start_time = time.time()
                result = await coro()
                duration = time.time() - start_time

                if duration > 5.0:  # Log slow tasks
                    logger.warning(f"Slow task {task_id}: {duration:.2f}s")

                logger.debug(f"Task {task_id} completed in {duration:.2f}s")
                return result

            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                raise

    def stop_scheduler(self):
        """Stop the task scheduler."""
        self.running = False
        logger.info("Async scheduler stopped")

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get scheduler queue statistics."""
        return {
            'queued_tasks': len(self.task_queue),
            'running': self.running,
            'max_concurrent': self.max_concurrent
        }

# Global scheduler instance
scheduler = AsyncScheduler()

def get_scheduler() -> AsyncScheduler:
    """Get global async scheduler instance."""
    return scheduler
```

## 📈 Performance Profiling

### Comprehensive Profiling
```python
# src/utils/performance_profiler.py
import cProfile
import pstats
import io
import time
from functools import wraps
from typing import Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)

class PerformanceProfiler:
    """Comprehensive performance profiling utilities."""

    def __init__(self):
        self.profiles = {}

    def profile_function(self, name: Optional[str] = None):
        """Decorator to profile function performance."""
        def decorator(func: Callable) -> Callable:
            profile_name = name or f"{func.__module__}.{func.__name__}"

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                profiler = cProfile.Profile()
                profiler.enable()

                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    profiler.disable()
                    execution_time = time.time() - start_time

                    # Store profile
                    self.profiles[profile_name] = {
                        'profile': profiler,
                        'execution_time': execution_time,
                        'timestamp': time.time()
                    }

                    # Log performance
                    if execution_time > 1.0:
                        logger.warning(f"Profiled slow function: {profile_name} took {execution_time:.2f}s")

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                profiler = cProfile.Profile()
                profiler.enable()

                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    profiler.disable()
                    execution_time = time.time() - start_time

                    self.profiles[profile_name] = {
                        'profile': profiler,
                        'execution_time': execution_time,
                        'timestamp': time.time()
                    }

                    if execution_time > 1.0:
                        logger.warning(f"Profiled slow function: {profile_name} took {execution_time:.2f}s")

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

    def get_profile_report(self, name: str, sort_by: str = 'cumulative', lines: int = 20) -> str:
        """Get profiling report for a function."""
        if name not in self.profiles:
            return f"No profile found for {name}"

        profile_data = self.profiles[name]
        profiler = profile_data['profile']

        # Generate report
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats(sort_by)
        stats.print_stats(lines)

        report = stream.getvalue()
        execution_time = profile_data['execution_time']

        return f"Profile Report for {name}\nExecution Time: {execution_time:.2f}s\n\n{report}"

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all profiled functions."""
        if not self.profiles:
            return {'total_profiles': 0}

        execution_times = [p['execution_time'] for p in self.profiles.values()]

        return {
            'total_profiles': len(self.profiles),
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'max_execution_time': max(execution_times),
            'min_execution_time': min(execution_times),
            'slow_functions': [
                name for name, data in self.profiles.items()
                if data['execution_time'] > 1.0
            ]
        }

    def clear_profiles(self, older_than: Optional[float] = None):
        """Clear old profiles."""
        if older_than is None:
            self.profiles.clear()
            return

        current_time = time.time()
        to_remove = [
            name for name, data in self.profiles.items()
            if current_time - data['timestamp'] > older_than
        ]

        for name in to_remove:
            del self.profiles[name]

        logger.info(f"Cleared {len(to_remove)} old profiles")

# Global profiler instance
profiler = PerformanceProfiler()

def get_profiler() -> PerformanceProfiler:
    """Get global performance profiler instance."""
    return profiler
```

---

*These optimization techniques can significantly improve ABC-Application's performance. Monitor metrics regularly and adjust parameters based on your specific workload and infrastructure.*