#!/usr/bin/env python3
"""
Centralized Logging Configuration for ABC Application
Provides consistent error logging across all components.
"""

import logging
import logging.handlers
import sys
from typing import Optional
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True
) -> logging.Logger:
    """
    Setup centralized logging configuration for the ABC Application.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Setup file handler
    if enable_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_file or str(log_dir / "abc_application.log")

        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)

    # Setup console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(numeric_level)
        root_logger.addHandler(console_handler)

    # Quiet noisy libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('discord').setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    operation: str,
    component: Optional[str] = None,
    **extra_context
):
    """Log an error with structured context."""
    context = {
        'operation': operation,
        'component': component or logger.name.split('.')[-1],
        'error_type': type(error).__name__,
        'error_message': str(error)
    }
    context.update(extra_context)

    logger.error(f"Error in {operation}: {error}", extra=context)