#!/usr/bin/env python3
"""
Unified Workflow Starter Script
Provides easy command-line interface to run the Unified Workflow Orchestrator in different modes.
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.unified_workflow_orchestrator import (
    UnifiedWorkflowOrchestrator,
    WorkflowMode,
    run_analysis_mode,
    run_execution_mode,
    run_hybrid_mode
)
from src.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


def parse_symbols(symbol_string: str) -> list:
    """Parse comma-separated symbol string into list."""
    return [s.strip().upper() for s in symbol_string.split(',') if s.strip()]


async def main():
    """Main entry point for the unified workflow starter."""
    parser = argparse.ArgumentParser(
        description="Unified Workflow Orchestrator - Consolidated Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run hybrid mode (analysis + automated execution)
  python tools/start_unified_workflow.py --mode hybrid --symbols SPY,QQQ

  # Run analysis-only mode for research
  python tools/start_unified_workflow.py --mode analysis --symbols AAPL,MSFT,GOOGL

  # Run automated execution only (during market hours)
  python tools/start_unified_workflow.py --mode execution --symbols SPY

  # Run without Discord integration
  python tools/start_unified_workflow.py --mode hybrid --no-discord
        """
    )

    parser.add_argument(
        '--mode',
        choices=['analysis', 'execution', 'hybrid'],
        default='hybrid',
        help='Operating mode (default: hybrid)'
    )

    parser.add_argument(
        '--symbols',
        type=parse_symbols,
        default=['SPY'],
        help='Comma-separated list of symbols to trade/analyze (default: SPY)'
    )

    parser.add_argument(
        '--no-discord',
        action='store_true',
        help='Disable Discord integration'
    )

    parser.add_argument(
        '--no-health-monitoring',
        action='store_true',
        help='Disable health monitoring'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="INFO", enable_console=True, enable_file=True)
    logger.info("🚀 Starting Unified Workflow Orchestrator")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Symbols: {', '.join(args.symbols)}")
    logger.info(f"Discord: {'Disabled' if args.no_discord else 'Enabled'}")
    logger.info(f"Health Monitoring: {'Disabled' if args.no_health_monitoring else 'Enabled'}")

    # Map string modes to enum
    mode_map = {
        'analysis': WorkflowMode.ANALYSIS,
        'execution': WorkflowMode.EXECUTION,
        'hybrid': WorkflowMode.HYBRID
    }

    try:
        # Create and initialize orchestrator
        orchestrator = UnifiedWorkflowOrchestrator(
            mode=mode_map[args.mode],
            enable_discord=not args.no_discord,
            enable_health_monitoring=not args.no_health_monitoring,
            symbols=args.symbols
        )

        # Initialize
        if not await orchestrator.initialize():
            logger.error("❌ Failed to initialize orchestrator")
            return 1

        # Start
        await orchestrator.start()
        return 0

    except KeyboardInterrupt:
        logger.info("🛑 Received shutdown signal")
        if 'orchestrator' in locals():
            await orchestrator.stop()
        return 0
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        if 'orchestrator' in locals():
            await orchestrator.stop()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)