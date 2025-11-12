#!/usr/bin/env python3
"""
Memory Cleanup Script for ABC Application Trading System
Cleans up accumulated memory files to optimize system performance.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.memory_persistence import get_memory_persistence
from src.utils.advanced_memory import get_advanced_memory_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def cleanup_memory_system():
    """
    Clean up the memory system by removing old backups and expired memories.
    """
    logger.info("Starting memory cleanup process...")

    try:
        # Get memory managers
        persistence = get_memory_persistence()
        advanced_memory = get_advanced_memory_manager()

        # Get stats before cleanup
        logger.info("Getting pre-cleanup memory statistics...")
        pre_stats = persistence.get_memory_stats()
        logger.info(f"Pre-cleanup stats: {pre_stats}")

        # Clean up old backups (keeps only 10 most recent per file)
        logger.info("Cleaning up old backup files...")
        backup_cleanup_count = persistence.cleanup_old_backups(max_backups=10)
        logger.info(f"Cleaned up {backup_cleanup_count} old backup files")

        # Clean up expired memories (30+ days old)
        logger.info("Cleaning up expired memories...")
        expired_cleanup_count = await advanced_memory.cleanup_expired_memories(days_old=30)
        logger.info(f"Cleaned up {expired_cleanup_count} expired memories")

        # Get stats after cleanup
        logger.info("Getting post-cleanup memory statistics...")
        post_stats = persistence.get_memory_stats()
        logger.info(f"Post-cleanup stats: {post_stats}")

        # Calculate cleanup summary
        total_cleaned = backup_cleanup_count + expired_cleanup_count
        logger.info(f"Memory cleanup completed successfully!")
        logger.info(f"Total files/memories cleaned up: {total_cleaned}")

        return {
            "success": True,
            "backup_files_cleaned": backup_cleanup_count,
            "expired_memories_cleaned": expired_cleanup_count,
            "total_cleaned": total_cleaned,
            "pre_stats": pre_stats,
            "post_stats": post_stats
        }

    except Exception as e:
        logger.error(f"Memory cleanup failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def main():
    """
    Main entry point for the memory cleanup script.
    """
    logger.info("ABC Application Memory Cleanup Script")
    logger.info("=" * 50)

    # Run the async cleanup
    result = asyncio.run(cleanup_memory_system())

    if result["success"]:
        logger.info("✅ Memory cleanup completed successfully!")
        logger.info(f"   - Backup files cleaned: {result['backup_files_cleaned']}")
        logger.info(f"   - Expired memories cleaned: {result['expired_memories_cleaned']}")
        logger.info(f"   - Total cleaned: {result['total_cleaned']}")
    else:
        logger.error("❌ Memory cleanup failed!")
        logger.error(f"   Error: {result['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()