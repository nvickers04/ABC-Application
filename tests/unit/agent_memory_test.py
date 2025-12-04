#!/usr/bin/env python3
"""
Targeted Agent Memory Leak Test
Tests agent loading and unloading to identify memory leaks
"""

import psutil
import os
import time
import gc
import sys
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_memory_mb():
    """Get current process memory in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2)

def test_agent_lifecycle():
    """Test creating and destroying agents to check for memory leaks"""
    print("🔬 Agent Memory Lifecycle Test")
    print("=" * 50)

    results = []

    # Test 1: Baseline memory
    gc.collect()
    baseline = get_memory_mb()
    print(".1f")
    # Test 2: Import agent module
    start_mem = get_memory_mb()
    try:
        from src.agents.risk import RiskAgent
        import_mem = get_memory_mb()
        print(".1f")
        # Test 3: Create single agent
        agent_start = get_memory_mb()
        agent = RiskAgent()
        agent_created = get_memory_mb()
        print(".1f")
        # Test 4: Delete agent and force GC
        del agent
        gc.collect()
        agent_deleted = get_memory_mb()
        print(".1f")
        # Test 5: Create multiple agents sequentially
        multi_start = get_memory_mb()
        agents = []
        for i in range(3):
            agent = RiskAgent()
            agents.append(agent)

        multi_created = get_memory_mb()
        print(".1f")
        # Clean up
        del agents
        gc.collect()
        multi_deleted = get_memory_mb()
        print(".1f")
        results.append({
            'test': 'agent_lifecycle',
            'baseline': baseline,
            'import_delta': import_mem - start_mem,
            'single_agent_delta': agent_created - agent_start,
            'cleanup_delta': agent_deleted - agent_created,
            'multi_agent_delta': multi_created - multi_start,
            'multi_cleanup_delta': multi_deleted - multi_created,
            'final_memory': multi_deleted
        })

    except Exception as e:
        logger.error(f"Agent lifecycle test failed: {e}")
        results.append({'test': 'agent_lifecycle', 'error': str(e)})

    return results

def test_module_unloading():
    """Test if modules can be unloaded"""
    print("\n🔬 Module Unloading Test")
    print("=" * 50)

    # Check modules before
    modules_before = len(sys.modules)
    memory_before = get_memory_mb()
    print(f"Modules before: {modules_before}, Memory: {memory_before:.1f} MB")

    # Import agent module
    import src.agents.risk
    modules_after_import = len(sys.modules)
    memory_after_import = get_memory_mb()
    print(f"Modules after import: {modules_after_import}, Memory: {memory_after_import:.1f} MB")

    # Try to remove the module (this usually doesn't work well in Python)
    try:
        # This won't actually remove the module but let's see
        del sys.modules['src.agents.risk']
        modules_after_del = len(sys.modules)
        memory_after_del = get_memory_mb()
        print(f"Modules after del: {modules_after_del}, Memory: {memory_after_del:.1f} MB")
    except Exception as e:
        print(f"Could not delete module: {e}")

    return {
        'modules_before': modules_before,
        'modules_after_import': modules_after_import,
        'modules_after_del': modules_after_del if 'modules_after_del' in locals() else modules_after_import,
        'memory_before': memory_before,
        'memory_after_import': memory_after_import,
        'memory_after_del': memory_after_del if 'memory_after_del' in locals() else memory_after_import
    }

def analyze_memory_growth(results):
    """Analyze the memory growth patterns"""
    print("\n🔍 Memory Analysis")
    print("=" * 50)

    if not results or 'error' in results[0]:
        print("❌ Test failed, cannot analyze")
        return

    data = results[0]

    print("Memory Growth Breakdown:")
    print(".1f")
    print(".1f")
    print(".1f")
    print(".1f")
    print(".1f")
    # Check for concerning patterns
    issues = []

    if data['single_agent_delta'] > 100:  # More than 100MB per agent
        issues.append(f"🚨 HIGH: Single agent consumes {data['single_agent_delta']:.1f} MB")

    if data['cleanup_delta'] > 50:  # Memory not properly released
        issues.append(f"⚠️  MEDIUM: {abs(data['cleanup_delta']):.1f} MB not released after agent deletion")

    if data['multi_agent_delta'] > 500:  # Excessive memory for multiple agents
        issues.append(f"🚨 HIGH: Multiple agents consume {data['multi_agent_delta']:.1f} MB")

    if data['multi_cleanup_delta'] > 100:  # Memory not properly released
        issues.append(f"⚠️  MEDIUM: {abs(data['multi_cleanup_delta']):.1f} MB not released after multiple agent deletion")

    if issues:
        print("\n🚨 Memory Issues Detected:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ No major memory issues detected")

    return issues

if __name__ == "__main__":
    print("🧠 Agent Memory Leak Analysis")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")

    # Run lifecycle test
    lifecycle_results = test_agent_lifecycle()

    # Run module unloading test
    module_results = test_module_unloading()

    # Analyze results
    issues = analyze_memory_growth(lifecycle_results)

    print(f"\n📊 Module Analysis:")
    print(f"  Modules loaded: {module_results['modules_after_import'] - module_results['modules_before']}")
    print(".1f")
    print(f"\n✅ Analysis completed at: {datetime.now()}")

    # Summary
    if issues:
        print(f"\n🚨 SUMMARY: {len(issues)} memory issues detected")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ SUMMARY: Memory usage appears normal for ML application")