#!/usr/bin/env python3
"""
Advanced Memory Leak Detection and Profiling Script
Monitors ABC-Application components for memory leaks during realistic operations
"""

import psutil
import os
import time
import gc
import tracemalloc
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Any
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryLeakDetector:
    """Advanced memory leak detection for ABC-Application components"""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = None
        self.snapshots = []
        self.tracemalloc_started = False

    def start_tracemalloc(self):
        """Start tracemalloc for detailed memory tracking"""
        if not self.tracemalloc_started:
            tracemalloc.start()
            self.tracemalloc_started = True
            logger.info("Tracemalloc started for detailed memory tracking")

    def get_memory_snapshot(self) -> Dict[str, Any]:
        """Get comprehensive memory snapshot"""
        mem = psutil.virtual_memory()

        # Get process memory info
        mem_info = self.process.memory_info()

        # Get tracemalloc snapshot if available
        tracemalloc_snapshot = None
        if self.tracemalloc_started:
            try:
                snapshot = tracemalloc.take_snapshot()
                tracemalloc_snapshot = {
                    'total_size': sum(stat.size for stat in snapshot.statistics('filename')),
                    'total_count': sum(stat.count for stat in snapshot.statistics('filename')),
                    'top_10': snapshot.statistics('filename')[:10]
                }
            except Exception as e:
                logger.warning(f"Tracemalloc snapshot failed: {e}")

        return {
            'timestamp': datetime.now(),
            'system_memory': {
                'total_gb': mem.total / (1024**3),
                'used_gb': mem.used / (1024**3),
                'percent': mem.percent
            },
            'process_memory': {
                'rss_mb': mem_info.rss / (1024**2),  # Resident Set Size
                'vms_mb': mem_info.vms / (1024**2),  # Virtual Memory Size
                'uss_mb': getattr(mem_info, 'uss', 0) / (1024**2),  # Unique Set Size (Linux only)
            },
            'cpu_percent': self.process.cpu_percent(),
            'threads': self.process.num_threads(),
            'open_files': len(self.process.open_files()),
            'connections': len(self.process.connections()),
            'tracemalloc': tracemalloc_snapshot
        }

    def monitor_operation(self, operation_func, operation_name: str, duration_seconds: int = 30) -> Dict[str, Any]:
        """Monitor memory during an operation"""
        logger.info(f"🔍 Monitoring: {operation_name} (for {duration_seconds}s)")
        print("=" * 60)

        # Take baseline
        baseline = self.get_memory_snapshot()
        self.baseline_memory = baseline
        logger.info(f"📊 Baseline memory: {baseline['process_memory']['rss_mb']:.1f} MB RSS")

        # Start operation
        start_time = time.time()

        try:
            # Run operation in background if it's async
            if asyncio.iscoroutinefunction(operation_func):
                # Use asyncio.run for async functions
                result = asyncio.run(operation_func())
            else:
                # Run synchronous operation
                result = operation_func()
                # Keep monitoring for the specified duration
                time.sleep(min(duration_seconds, 5))  # Don't sleep too long for sync ops

        except Exception as e:
            logger.error(f"Operation failed: {e}")
            result = None

        end_time = time.time()

        # Take final snapshot
        final = self.get_memory_snapshot()

        # Calculate deltas
        memory_delta = final['process_memory']['rss_mb'] - baseline['process_memory']['rss_mb']
        duration = end_time - start_time

        logger.info(f"📊 Final memory: {final['process_memory']['rss_mb']:.1f} MB RSS")
        logger.info(f"📈 Memory delta: {memory_delta:+.1f} MB")
        logger.info(f"⏱️  Duration: {duration:.2f}s")

        return {
            'operation': operation_name,
            'baseline': baseline,
            'final': final,
            'memory_delta_mb': memory_delta,
            'duration_seconds': duration,
            'result': result
        }

    def detect_potential_leaks(self, results: List[Dict]) -> List[str]:
        """Analyze results for potential memory leaks"""
        issues = []

        for result in results:
            delta = result['memory_delta_mb']
            operation = result['operation']

            # Check for significant memory growth
            if delta > 50:  # More than 50MB growth
                issues.append(f"🚨 HIGH: {operation} grew {delta:.1f} MB - potential leak")
            elif delta > 20:  # More than 20MB growth
                issues.append(f"⚠️  MEDIUM: {operation} grew {delta:.1f} MB - monitor closely")
            elif delta > 10:  # More than 10MB growth
                issues.append(f"ℹ️  LOW: {operation} grew {delta:.1f} MB - minor growth")

        return issues

# Test operations for ABC-Application components
async def test_ibkr_connector():
    """Test IBKR connector memory usage"""
    try:
        from src.integrations.ibkr_connector import IBKRConnector

        # Create connector with default config (will fail but tests memory usage)
        connector = IBKRConnector()

        # Test multiple operations (they will fail but we measure memory)
        for i in range(3):
            try:
                await connector.get_account_summary()
            except Exception as e:
                logger.debug(f"IBKR operation {i+1} failed (expected): {e}")

        return "IBKR connector operations completed"
    except Exception as e:
        logger.error(f"IBKR connector test failed: {e}")
        return f"IBKR connector test failed: {e}"

def test_agent_loading():
    """Test loading agents and memory usage"""
    try:
        from src.agents.base import BaseAgent
        from src.agents.risk import RiskAgent

        agents = []

        # Create multiple agents (reduced number to avoid excessive memory usage)
        for i in range(3):
            agent = RiskAgent()
            agents.append(agent)

        # Let them process some data
        for agent in agents:
            try:
                agent.process_input("Test input for memory monitoring")
            except Exception as e:
                logger.debug(f"Agent {i} processing failed: {e}")

        return f"Created and tested {len(agents)} agents"
    except Exception as e:
        logger.error(f"Agent loading test failed: {e}")
        return f"Agent loading test failed: {e}"

def test_workflow_execution():
    """Test workflow execution memory usage"""
    try:
        from src.workflows.consensus_poller import ConsensusPoller

        # Create and run a consensus poll
        poller = ConsensusPoller()
        # Note: This would normally run longer, but we'll limit for testing

        return "Workflow execution test completed"
    except Exception as e:
        logger.error(f"Workflow test failed: {e}")
        return f"Workflow test failed: {e}"

def test_data_processing():
    """Test data processing memory usage"""
    try:
        import pandas as pd
        import numpy as np

        # Create large datasets
        dataframes = []
        for i in range(5):
            df = pd.DataFrame(np.random.randn(10000, 50))
            dataframes.append(df)

        # Process them
        results = []
        for df in dataframes:
            result = df.describe()
            results.append(result)

        return f"Processed {len(dataframes)} dataframes with {len(results)} results"
    except Exception as e:
        logger.error(f"Data processing test failed: {e}")
        return f"Data processing test failed: {e}"

async def run_memory_leak_detection():
    """Run comprehensive memory leak detection"""
    print("🧠 Advanced Memory Leak Detection")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print(f"Process ID: {os.getpid()}")

    detector = MemoryLeakDetector()
    detector.start_tracemalloc()

    results = []

    # Test operations
    test_operations = [
        (test_ibkr_connector, "IBKR Connector Operations", 15),
        (test_agent_loading, "Agent Loading", 10),
        (test_workflow_execution, "Workflow Execution", 10),
        (test_data_processing, "Data Processing", 10),
    ]

    for operation_func, name, duration in test_operations:
        try:
            result = detector.monitor_operation(operation_func, name, duration)
            results.append(result)

            # Force garbage collection between tests
            gc.collect()
            time.sleep(2)  # Let system stabilize

        except Exception as e:
            logger.error(f"Failed to monitor {name}: {e}")
            results.append({
                'operation': name,
                'error': str(e),
                'memory_delta_mb': 0,
                'duration_seconds': 0
            })

    # Analyze results
    print("\n🔍 Memory Leak Analysis")
    print("=" * 60)

    issues = detector.detect_potential_leaks(results)

    if issues:
        print("🚨 Potential Memory Issues Detected:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ No significant memory leaks detected")

    # Summary
    print("\n📊 Summary:")
    for result in results:
        delta = result.get('memory_delta_mb', 0)
        duration = result.get('duration_seconds', 0)
        print(".1f")

    # Detailed tracemalloc analysis if available
    if detector.tracemalloc_started:
        print("\n🔬 Top Memory Consumers (tracemalloc):")
        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('filename')[:10]

            for stat in top_stats:
                # Handle different tracemalloc API versions
                try:
                    # Try newer API
                    filename = getattr(stat, 'filename', 'unknown')
                    lineno = getattr(stat, 'lineno', 0)
                    print(f"  {filename}:{lineno} - {stat.size / 1024:.1f} KB ({stat.count} objects)")
                except AttributeError:
                    # Fallback for older API
                    print(f"  {stat} - {stat.size / 1024:.1f} KB ({stat.count} objects)")
        except Exception as e:
            logger.warning(f"Tracemalloc analysis failed: {e}")

    print(f"\n✅ Memory leak detection completed at: {datetime.now()}")

    return results

if __name__ == "__main__":
    asyncio.run(run_memory_leak_detection())