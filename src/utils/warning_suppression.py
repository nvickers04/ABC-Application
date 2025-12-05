"""
Warning suppression utilities for ABC-Application.

This module provides centralized warning suppression to reduce noise from third-party libraries
while maintaining important application warnings.
"""

import os
import warnings
import logging

def suppress_third_party_warnings():
    """
    Suppress noisy warnings from third-party libraries that don't affect functionality.
    Call this at the start of main application entry points.
    """
    # Environment variables for TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

    # Suppress common third-party warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow.*')
    warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow_probability.*')
    warnings.filterwarnings('ignore', category=UserWarning, module='tf_keras.*')
    warnings.filterwarnings('ignore', category=UserWarning, module='gym.*')
    warnings.filterwarnings('ignore', message='.*oneDNN.*', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*Gym.*', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*distutils.*', category=DeprecationWarning)

    # Reduce logging noise from some libraries
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('tensorflow_probability').setLevel(logging.ERROR)
    logging.getLogger('tf_keras').setLevel(logging.ERROR)

def restore_warnings():
    """
    Restore default warning behavior if needed for debugging.
    """
    warnings.resetwarnings()
    os.environ.pop('TF_CPP_MIN_LOG_LEVEL', None)
    os.environ.pop('TF_ENABLE_ONEDNN_OPTS', None)