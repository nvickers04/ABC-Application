#!/usr/bin/env python3
"""
Basic memory profiling and monitoring script
"""

import psutil
import os
import time
from datetime import datetime

def get_memory_info():
    """Get detailed memory information"""
    mem = psutil.virtual_memory()
    process = psutil.Process(os.getpid())

    return {
        'system_total': mem.total / (1024**3),  # GB
        'system_used': mem.used / (1024**3),   # GB
        'system_percent': mem.percent,
        'process_rss': process.memory_info().rss / (1024**2),  # MB
        'process_vms': process.memory_info().vms / (1024**2),  # MB
        'cpu_percent': psutil.cpu_percent(interval=1)
    }

def monitor_operation(operation_func, operation_name="Operation"):
    """Monitor memory and CPU during an operation"""
    print(f"\n🔍 Monitoring: {operation_name}")
    print("=" * 50)

    # Baseline
    baseline = get_memory_info()
    print("📊 Baseline:")
    print(".1f")
    print(".1f")

    # Run operation
    start_time = time.time()
    result = operation_func()
    end_time = time.time()

    # After operation
    after = get_memory_info()
    print("\n📊 After operation:")
    print(".1f")
    print(".1f")

    # Calculate differences
    mem_diff = after['process_rss'] - baseline['process_rss']
    cpu_avg = (baseline['cpu_percent'] + after['cpu_percent']) / 2

    print("\n📈 Changes:")
    print(".1f")
    print(".2f")

    return {
        'result': result,
        'memory_delta': mem_diff,
        'cpu_avg': cpu_avg,
        'duration': end_time - start_time
    }

# Example operations to monitor
def test_import_operations():
    """Test importing various modules"""
    import sys
    modules_before = len(sys.modules)

    # Import some heavy modules
    import pandas as pd
    import numpy as np
    import torch

    modules_after = len(sys.modules)
    return f"Imported {modules_after - modules_before} modules"

def test_data_operations():
    """Test data processing operations"""
    import numpy as np
    import pandas as pd

    # Create some test data
    data = np.random.randn(10000, 100)
    df = pd.DataFrame(data)

    # Perform operations
    result = df.mean().sum()
    return f"Processed {len(df)} rows, result: {result:.2f}"

if __name__ == "__main__":
    print("🧠 Memory Profiling Session")
    print("=" * 50)
    print(f"Started at: {datetime.now()}")

    # Monitor import operations
    monitor_operation(test_import_operations, "Module Imports")

    # Monitor data operations
    monitor_operation(test_data_operations, "Data Processing")

    print("\n✅ Memory profiling completed")
    print(f"Finished at: {datetime.now()}")