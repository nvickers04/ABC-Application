#!/usr/bin/env python3
"""
Test script for batch analytics and memory monitoring dashboard.
Tests the extended batch analytics functionality and memory monitoring features.
"""

import asyncio
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.optimized_pipeline import OptimizedPipelineProcessor
from src.agents.data import DataAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_batch_analytics():
    """Test batch analytics functionality."""
    logger.info("Testing batch analytics functionality...")

    try:
        # Create pipeline processor
        pipeline = OptimizedPipelineProcessor()

        # Test data for batch processing
        batch_data = [
            {
                'symbol': 'SPY',
                'dataframe': None,  # Would be populated with actual data
                'sentiment': {'score': 0.6, 'source': 'test'},
                'news': {'headlines': ['SPY shows bullish momentum'], 'source': 'test'},
                'economic': {'indicators': {'GDP': 0.5}, 'source': 'test'}
            },
            {
                'symbol': 'AAPL',
                'dataframe': None,
                'sentiment': {'score': 0.7, 'source': 'test'},
                'news': {'headlines': ['AAPL reports strong earnings'], 'source': 'test'},
                'economic': {'indicators': {'GDP': 0.5}, 'source': 'test'}
            }
        ]

        # Execute batch analytics
        results = await pipeline._execute_batch_predictive_analytics(batch_data)

        logger.info(f"Batch analytics completed: {len(results)} results")
        for result in results:
            logger.info(f"Symbol {result.get('symbol')}: {result.get('short_term_direction')}")

        return True

    except Exception as e:
        logger.error(f"Batch analytics test failed: {e}")
        return False

async def test_memory_monitoring_dashboard():
    """Test memory monitoring dashboard."""
    logger.info("Testing memory monitoring dashboard...")

    try:
        # Create pipeline processor
        pipeline = OptimizedPipelineProcessor()

        # Get memory dashboard
        dashboard = pipeline.get_memory_monitoring_dashboard()

        logger.info("Memory monitoring dashboard generated successfully")
        logger.info(f"Current memory usage: {dashboard['current_stats']['used_mb']:.1f} MB")
        logger.info(f"Memory utilization: {dashboard['current_stats']['utilization_percent']:.1f}%")
        logger.info(f"Active alerts: {len(dashboard['alerts'])}")
        logger.info(f"Recommendations: {len(dashboard['recommendations'])}")

        # Log some key metrics
        trends = dashboard['trends']
        logger.info(f"Memory trend: {trends['usage_trend']}")
        logger.info(f"Efficiency score: {trends['efficiency_score']:.2f}")

        return True

    except Exception as e:
        logger.error(f"Memory monitoring dashboard test failed: {e}")
        return False

async def test_cache_warming():
    """Test cache warming functionality."""
    logger.info("Testing cache warming functionality...")

    try:
        # Create pipeline processor
        pipeline = OptimizedPipelineProcessor()

        # Test cache warming with a subset of symbols
        test_symbols = ['SPY', 'AAPL']
        await pipeline.warmup_cache(test_symbols)

        logger.info("Cache warming completed successfully")
        return True

    except Exception as e:
        logger.error(f"Cache warming test failed: {e}")
        return False

async def test_full_pipeline_with_batch():
    """Test full pipeline processing with batch analytics."""
    logger.info("Testing full pipeline with batch analytics...")

    try:
        # Create DataAgent
        data_agent = DataAgent()

        # Process multiple symbols to trigger batch analytics
        input_data = {'symbols': ['SPY', 'AAPL']}
        result = await data_agent.process_input(input_data)

        if result and 'dataframe' in result:
            logger.info("Full pipeline processing completed successfully")
            logger.info(f"Processed symbols: {result.get('symbols_processed', 0)}")
            logger.info(f"DataFrame shape: {result['dataframe'].shape}")
            return True
        else:
            logger.error("Pipeline processing failed or returned no data")
            return False

    except Exception as e:
        logger.error(f"Full pipeline test failed: {e}")
        return False

async def main():
    """Run all tests."""
    logger.info("Starting comprehensive batch analytics and memory monitoring tests...")

    tests = [
        ("Batch Analytics", test_batch_analytics),
        ("Memory Monitoring Dashboard", test_memory_monitoring_dashboard),
        ("Cache Warming", test_cache_warming),
        ("Full Pipeline with Batch", test_full_pipeline_with_batch)
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name}...")
        try:
            success = await test_func()
            results.append((test_name, success))
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY:")
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1

    logger.info(f"\nOverall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        logger.info("🎉 All tests passed! Batch analytics and memory monitoring are working correctly.")
        return True
    else:
        logger.warning("⚠️  Some tests failed. Check the logs above for details.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)