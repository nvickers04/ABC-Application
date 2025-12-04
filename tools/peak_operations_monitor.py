#!/usr/bin/env python3
"""
Peak Operations Resource Monitor

Monitors system resource usage during simulated peak trading operations
to identify performance bottlenecks and optimization opportunities.
"""

import asyncio
import time
import psutil
import os
import math
from src.integrations.nautilus_ibkr_bridge import NautilusIBKRBridge, BridgeConfig, BridgeMode

async def simulate_trading_operations():
    """Simulate realistic trading operations to monitor resource usage"""
    print('🚀 Starting peak operations monitoring...')
    print('Simulating realistic trading workflow with multiple operations')

    # Initialize bridge
    config = BridgeConfig(mode=BridgeMode.IB_INSYNC_ONLY)  # Use simpler mode for testing
    bridge = NautilusIBKRBridge(config)

    # Track metrics
    operations = []

    def log_operation(name, start_time, end_time, cpu_before, cpu_after, mem_before, mem_after):
        duration = end_time - start_time
        cpu_delta = cpu_after - cpu_before
        mem_delta = mem_after - mem_before
        operations.append({
            'operation': name,
            'duration': duration,
            'cpu_delta': cpu_delta,
            'mem_delta': mem_delta,
            'cpu_after': cpu_after,
            'mem_after': mem_after
        })
        print(f'📊 {name}: {duration:.3f}s, CPU: {cpu_before:.1f}% → {cpu_after:.1f}%, Mem: {mem_before:.1f}% → {mem_after:.1f}%')

    # Get baseline
    baseline_cpu = psutil.cpu_percent(interval=1)
    baseline_mem = psutil.virtual_memory().percent
    print(f'📈 Baseline - CPU: {baseline_cpu:.1f}%, Memory: {baseline_mem:.1f}%')

    try:
        # Operation 1: Initialize bridge
        start = time.time()
        cpu_before = psutil.cpu_percent()
        mem_before = psutil.virtual_memory().percent

        await bridge.initialize()

        end = time.time()
        cpu_after = psutil.cpu_percent()
        mem_after = psutil.virtual_memory().percent
        log_operation('Bridge Initialization', start, end, cpu_before, cpu_after, mem_before, mem_after)

        # Operation 2: Multiple market data requests
        for i in range(5):
            start = time.time()
            cpu_before = psutil.cpu_percent()
            mem_before = psutil.virtual_memory().percent

            # Simulate market data requests for different symbols
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
            symbol = symbols[i % len(symbols)]

            # This will use the mock/fallback since we don't have real IBKR connection
            result = await bridge.get_market_data(symbol)

            end = time.time()
            cpu_after = psutil.cpu_percent()
            mem_after = psutil.virtual_memory().percent
            log_operation(f'Market Data {symbol}', start, end, cpu_before, cpu_after, mem_before, mem_after)

            await asyncio.sleep(0.1)  # Small delay between requests

        # Operation 3: Account summary requests
        for i in range(3):
            start = time.time()
            cpu_before = psutil.cpu_percent()
            mem_before = psutil.virtual_memory().percent

            result = await bridge.get_account_summary()

            end = time.time()
            cpu_after = psutil.cpu_percent()
            mem_after = psutil.virtual_memory().percent
            log_operation(f'Account Summary {i+1}', start, end, cpu_before, cpu_after, mem_before, mem_after)

        # Operation 4: Position retrieval
        start = time.time()
        cpu_before = psutil.cpu_percent()
        mem_before = psutil.virtual_memory().percent

        positions = await bridge.get_positions()

        end = time.time()
        cpu_after = psutil.cpu_percent()
        mem_after = psutil.virtual_memory().percent
        log_operation('Position Retrieval', start, end, cpu_before, cpu_after, mem_before, mem_after)

        # Operation 5: Risk calculations (simulate position sizing)
        for i in range(3):
            start = time.time()
            cpu_before = psutil.cpu_percent()
            mem_before = psutil.virtual_memory().percent

            # Simulate risk calculation workload
            for j in range(10000):
                _ = math.sqrt(j) * math.sin(j)  # CPU intensive calculation

            end = time.time()
            cpu_after = psutil.cpu_percent()
            mem_after = psutil.virtual_memory().percent
            log_operation(f'Risk Calculation {i+1}', start, end, cpu_before, cpu_after, mem_before, mem_after)

        await bridge.disconnect()

    except Exception as e:
        print(f'❌ Error during operations: {e}')
        import traceback
        traceback.print_exc()

    # Analyze results
    print('\n📈 Peak Operations Analysis:')
    if operations:
        total_duration = sum(op['duration'] for op in operations)
        avg_cpu = sum(op['cpu_after'] for op in operations) / len(operations)
        max_cpu = max(op['cpu_after'] for op in operations)
        max_mem = max(op['mem_after'] for op in operations)
        min_mem = min(op['mem_after'] for op in operations)

        print(f'  Total operations time: {total_duration:.2f}s')
        print(f'  Average CPU during operations: {avg_cpu:.1f}%')
        print(f'  Peak CPU: {max_cpu:.1f}%')
        print(f'  Memory range: {min_mem:.1f}% - {max_mem:.1f}%')

        # Check for concerning patterns
        high_cpu_ops = [op for op in operations if op['cpu_after'] > 50]
        if high_cpu_ops:
            print(f'⚠️  High CPU operations (>50%): {len(high_cpu_ops)}')
            for op in high_cpu_ops:
                print(f'    - {op["operation"]}: {op["cpu_after"]:.1f}%')

        mem_increase_ops = [op for op in operations if op['mem_delta'] > 1]
        if mem_increase_ops:
            print(f'⚠️  Memory increasing operations (>1%): {len(mem_increase_ops)}')
            for op in mem_increase_ops:
                print(f'    - {op["operation"]}: +{op["mem_delta"]:.1f}%')

        # Performance recommendations
        print('\n💡 Performance Recommendations:')
        if max_cpu > 70:
            print('  - High CPU usage detected: Consider optimizing computationally intensive operations')
        if any(op['duration'] > 1.0 for op in operations):
            slow_ops = [op for op in operations if op['duration'] > 1.0]
            print(f'  - Slow operations detected ({len(slow_ops)}): Review timing for {", ".join(op["operation"] for op in slow_ops)}')
        if max_mem - min_mem > 5:
            print('  - Memory fluctuations detected: Monitor for potential leaks')

    print('\n✅ Peak operations monitoring completed!')

if __name__ == '__main__':
    asyncio.run(simulate_trading_operations())