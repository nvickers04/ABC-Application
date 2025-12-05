"""
Service Manager for ABC-Application.

Automatically starts required services (Redis, etc.) when the application starts.
Provides graceful fallback when services cannot be started.
"""

import asyncio
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class ServiceManager:
    """
    Manages automatic startup of required services for ABC-Application.
    """

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.services = {
            'redis': {
                'name': 'Redis Server',
                'check_command': self._check_redis,
                'start_command': self._start_redis,
                'required': False,  # System works without it
                'process': None
            }
        }

    async def start_required_services(self) -> Dict[str, bool]:
        """
        Start all required services automatically.

        Returns:
            Dict mapping service names to startup success status
        """
        logger.info("🔄 Checking and starting required services...")

        results = {}
        for service_name, service_config in self.services.items():
            try:
                logger.info(f"Checking {service_config['name']}...")
                is_running = await service_config['check_command']()

                if is_running:
                    logger.info(f"✅ {service_config['name']} is already running")
                    results[service_name] = True
                else:
                    logger.info(f"🚀 Starting {service_config['name']}...")
                    success = await service_config['start_command']()
                    results[service_name] = success

                    if success:
                        logger.info(f"✅ {service_config['name']} started successfully")
                    else:
                        if service_config['required']:
                            logger.error(f"❌ Failed to start required service: {service_config['name']}")
                        else:
                            logger.warning(f"⚠️ Failed to start optional service: {service_config['name']} - using fallback")

            except Exception as e:
                logger.error(f"Error managing service {service_name}: {e}")
                results[service_name] = False

        return results

    async def stop_services(self):
        """
        Stop any services that were started by this manager.
        """
        logger.info("🛑 Stopping managed services...")

        for service_name, service_config in self.services.items():
            if service_config['process'] and service_config['process'].poll() is None:
                try:
                    logger.info(f"Stopping {service_config['name']}...")
                    service_config['process'].terminate()
                    await asyncio.wait_for(service_config['process'].wait(), timeout=10.0)
                    logger.info(f"✅ {service_config['name']} stopped")
                except Exception as e:
                    logger.warning(f"Error stopping {service_config['name']}: {e}")

    async def _check_redis(self) -> bool:
        """
        Check if Redis is running.
        """
        try:
            import redis
            client = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=2)
            client.ping()
            return True
        except Exception:
            return False

    async def _start_redis(self) -> bool:
        """
        Start Redis server using the bundled Redis-on-Windows.
        """
        try:
            redis_exe = self.project_root / "redis" / "redis-server.exe"
            redis_config = self.project_root / "redis" / "minimal.conf"

            if not redis_exe.exists():
                logger.warning(f"Redis executable not found at {redis_exe}")
                return False

            if not redis_config.exists():
                logger.warning(f"Redis config not found at {redis_config}")
                return False

            # Start Redis as background process
            process = await asyncio.create_subprocess_exec(
                str(redis_exe),
                str(redis_config),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=str(self.project_root / "redis")
            )

            # Store process reference for cleanup
            self.services['redis']['process'] = process

            # Wait a moment for Redis to start
            await asyncio.sleep(2)

            # Verify it started
            return await self._check_redis()

        except Exception as e:
            logger.error(f"Failed to start Redis: {e}")
            return False

# Global service manager instance
_service_manager = None

def get_service_manager() -> ServiceManager:
    """
    Get the global service manager instance.
    """
    global _service_manager
    if _service_manager is None:
        _service_manager = ServiceManager()
    return _service_manager

async def start_required_services() -> Dict[str, bool]:
    """
    Convenience function to start all required services.
    """
    manager = get_service_manager()
    return await manager.start_required_services()

async def stop_services():
    """
    Convenience function to stop managed services.
    """
    manager = get_service_manager()
    await manager.stop_services()