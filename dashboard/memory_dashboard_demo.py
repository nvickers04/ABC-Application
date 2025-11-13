#!/usr/bin/env python3
"""
Memory Monitoring Dashboard Demo
Demonstrates the comprehensive memory monitoring dashboard functionality.
"""

import asyncio
import logging
import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.optimized_pipeline import OptimizedPipelineProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demonstrate_memory_dashboard():
    """Demonstrate the memory monitoring dashboard."""
    logger.info("🚀 Memory Monitoring Dashboard Demo")
    logger.info("=" * 50)

    # Create pipeline processor
    pipeline = OptimizedPipelineProcessor()

    # Simulate some memory usage by running a few operations
    logger.info("📊 Generating memory usage data...")

    # Get initial dashboard
    dashboard = pipeline.get_memory_monitoring_dashboard()

    logger.info("📈 MEMORY MONITORING DASHBOARD")
    logger.info("=" * 50)

    # Display current stats
    stats = dashboard['current_stats']
    logger.info("💾 Current Memory Statistics:")
    logger.info(f"   Used: {stats['used_mb']:.1f} MB")
    logger.info(f"   Peak: {stats['peak_mb']:.1f} MB")
    logger.info(f"   Available: {stats['available_mb']:.1f} MB")
    logger.info(f"   Utilization: {stats['utilization_percent']:.1f}%")
    logger.info(f"   GC Collections: {stats['gc_collections']}")
    logger.info(f"   Active Objects: {stats['active_objects']}")

    # Display trends
    trends = dashboard['trends']
    logger.info("\n📊 Memory Trends:")
    logger.info(f"   Usage Trend: {trends['usage_trend']}")
    logger.info(f"   Efficiency Score: {trends['efficiency_score']:.2f}")
    logger.info(f"   Leak Indicators: {len(trends['leak_indicators'])} detected")
    if trends['leak_indicators']:
        for indicator in trends['leak_indicators']:
            logger.info(f"      ⚠️  {indicator}")
    logger.info(f"   Optimization Opportunities: {len(trends['optimization_opportunities'])}")
    if trends['optimization_opportunities']:
        for opp in trends['optimization_opportunities']:
            logger.info(f"      💡 {opp}")

    # Display alerts
    alerts = dashboard['alerts']
    logger.info(f"\n🚨 Active Alerts: {len(alerts)}")
    if alerts:
        for alert in alerts:
            severity_icon = "🔴" if alert['severity'] == 'critical' else "🟡" if alert['severity'] == 'warning' else "ℹ️"
            logger.info(f"   {severity_icon} {alert['severity'].upper()}: {alert['message']}")
            logger.info(f"      Recommendation: {alert['recommendation']}")
    else:
        logger.info("   ✅ No active alerts")

    # Display recommendations
    recommendations = dashboard['recommendations']
    logger.info(f"\n💡 Memory Optimization Recommendations: {len(recommendations)}")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"   {i}. {rec}")

    # Display performance metrics
    metrics = dashboard['performance_metrics']
    logger.info(f"\n⚡ Performance Metrics:")
    logger.info(f"   Avg Processing Time: {metrics['avg_processing_time']:.1f}s per symbol")
    logger.info(f"   Memory per Symbol: {metrics['memory_per_symbol']:.1f} MB")
    logger.info(f"   Cache Hit Rate: {metrics['cache_hit_rate']:.1%}")

    # Show JSON output option
    logger.info(f"\n📄 Raw Dashboard Data (JSON):")
    logger.info(json.dumps(dashboard, indent=2, default=str)[:500] + "...")

    logger.info("\n✅ Memory monitoring dashboard demonstration complete!")
    logger.info("This dashboard provides real-time insights into memory usage,")
    logger.info("performance trends, and optimization recommendations.")

if __name__ == "__main__":
    asyncio.run(demonstrate_memory_dashboard())