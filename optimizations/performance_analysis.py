#!/usr/bin/env python3
"""
Performance Analysis Script for ABC Application System
"""

import time
import psutil
import os
import sys
import subprocess
from pathlib import Path

def memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def run_performance_test():
    """Run comprehensive performance analysis"""
    print('=== PERFORMANCE ANALYSIS ===')

    # Initial memory
    initial_memory = memory_usage()
    print(f'Initial memory usage: {initial_memory:.1f} MB')

    # Test data processing performance
    print('\n--- Testing Data Processing Performance ---')
    start_time = time.time()

    # Run comprehensive test
    result = subprocess.run([sys.executable, 'comprehensive_test.py'],
                          capture_output=True, text=True, cwd=Path(__file__).parent)

    data_processing_time = time.time() - start_time
    peak_memory = memory_usage()

    print(f'Data processing time: {data_processing_time:.2f} seconds')
    print(f'Peak memory usage: {peak_memory:.1f} MB')
    print(f'Memory increase: {peak_memory - initial_memory:.1f} MB')

    # Analyze performance from logs
    print('\n--- Performance Insights ---')
    lines = result.stdout.split('\n')
    processing_times = []
    data_sizes = []

    for line in lines:
        if 'INFO' in line:
            # Extract timing information
            if 'processing input' in line.lower():
                print(f'Processing step: {line.strip()}')
            elif 'completed' in line.lower() or 'validated' in line.lower():
                print(f'Completion step: {line.strip()}')
            elif 'shape' in line and '(' in line:
                # Extract dataframe shapes
                try:
                    shape_part = line.split('(')[1].split(')')[0]
                    if ',' in shape_part:
                        rows, cols = map(int, shape_part.split(','))
                        data_sizes.append((rows, cols))
                        print(f'Data size: {rows} rows × {cols} columns')
                except:
                    pass

    if data_sizes:
        total_rows = sum(rows for rows, cols in data_sizes)
        total_cols = sum(cols for rows, cols in data_sizes)
        print(f'\nTotal data processed: {total_rows} rows, {total_cols} columns')

    # Performance recommendations
    print('\n--- Performance Recommendations ---')

    if data_processing_time > 30:
        print('⚠️  SLOW: Data processing takes >30 seconds - consider parallelization')
    else:
        print('✅ FAST: Data processing under 30 seconds')

    if peak_memory - initial_memory > 500:
        print('⚠️  HIGH MEMORY: >500MB increase - optimize data structures')
    else:
        print('✅ LOW MEMORY: Reasonable memory usage')

    if len(data_sizes) > 0:
        avg_cols = total_cols / len(data_sizes)
        if avg_cols > 50:
            print('⚠️  WIDE DATA: Many columns - consider feature selection')
        else:
            print('✅ NARROW DATA: Reasonable column count')

    print(f'\nTotal execution time: {data_processing_time:.2f} seconds')
    print(f'Memory efficiency: {(peak_memory - initial_memory) / data_processing_time:.1f} MB/sec')

if __name__ == '__main__':
    run_performance_test()