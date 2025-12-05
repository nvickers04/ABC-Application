#!/usr/bin/env python3
"""
Performance testing script for system resource monitoring
"""

import time
import psutil
import os

def get_system_stats():
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'memory_used_gb': psutil.virtual_memory().used / (1024**3),
        'memory_available_gb': psutil.virtual_memory().available / (1024**3)
    }

def main():
    print('Testing system performance under load...')
    print('Baseline system stats:')
    baseline = get_system_stats()
    print(f'CPU: {baseline["cpu_percent"]}%, Memory: {baseline["memory_percent"]}% ({baseline["memory_used_gb"]:.2f}GB used)')

    # Simulate some load
    print('\nSimulating computational load...')
    start_time = time.time()
    for i in range(1000000):
        _ = i ** 2  # Simple computation
    end_time = time.time()

    load_stats = get_system_stats()
    print(f'After load - CPU: {load_stats["cpu_percent"]}%, Memory: {load_stats["memory_percent"]}% ({load_stats["memory_used_gb"]:.2f}GB used)')
    print(f'Computation time: {end_time - start_time:.2f} seconds')

    # Test memory allocation
    print('\nTesting memory allocation...')
    large_list = []
    for i in range(100000):
        large_list.append([0] * 100)  # Allocate memory

    alloc_stats = get_system_stats()
    print(f'After memory allocation - Memory: {alloc_stats["memory_percent"]}% ({alloc_stats["memory_used_gb"]:.2f}GB used)')

    # Clean up
    del large_list

    cleanup_stats = get_system_stats()
    print(f'After cleanup - Memory: {cleanup_stats["memory_percent"]}% ({cleanup_stats["memory_used_gb"]:.2f}GB used)')

    print('\nPerformance test completed.')

if __name__ == '__main__':
    main()