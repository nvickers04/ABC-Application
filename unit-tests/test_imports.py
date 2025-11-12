#!/usr/bin/env python3
"""Test script for TensorFlow and Nautilus Trader imports"""

import os
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
import io

# Set TensorFlow logging level to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO, WARNING, and ERROR messages

# Suppress Python warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow_probability')
warnings.filterwarnings('ignore', category=UserWarning, module='tf_keras')
warnings.filterwarnings('ignore', category=DeprecationWarning)

print("Testing TensorFlow and Nautilus Trader imports...")

# Capture stderr to suppress TensorFlow warnings
stderr_capture = io.StringIO()

with redirect_stderr(stderr_capture):
    # Test TensorFlow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")

    # Test TensorFlow Probability
    try:
        import tensorflow_probability as tfp
        print("✅ TensorFlow Probability imported successfully")
    except ImportError as e:
        print(f"❌ TensorFlow Probability import failed: {e}")

# Test Nautilus Trader
try:
    from nautilus_trader.core.nautilus_pyo3 import AccountBalance, AccountId
    print("✅ Nautilus Trader core imports working")
except ImportError as e:
    print(f"❌ Nautilus Trader core import failed: {e}")

# Test Nautilus Trader adapters
try:
    from nautilus_trader.adapters.interactive_brokers import InteractiveBrokersLiveExecutionClient
    print("✅ Nautilus Trader IBKR adapter imported successfully")
except ImportError as e:
    print(f"❌ Nautilus Trader IBKR adapter import failed: {e}")

print("Import testing complete.")