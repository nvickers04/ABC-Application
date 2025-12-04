# [LABEL:INTEGRATION:acontext] [LABEL:COMPONENT:sop_storage] [LABEL:FRAMEWORK:asyncio]
# [LABEL:AUTHOR:GitHub Copilot] [LABEL:UPDATED:2024-12-04] [LABEL:REVIEWED:pending]
#
# Purpose: Acontext integration for SOP storage, retrieval, and cross-agent propagation
# Dependencies: acontext>=0.0.6, pyyaml
# Related: config/acontext_config.yaml, src/agents/learning.py
#
"""
Acontext Integration Module

Provides functionality for:
- SOP storage and retrieval via Acontext API
- Session logging for trading decisions
- Artifact upload for ML models and backtest results
- Cross-agent directive propagation with priority queuing
- Graceful fallback when Acontext is unavailable
"""

import os
import sys
import logging
import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import uuid

# Set up path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.utils import load_yaml

logger = logging.getLogger(__name__)

# Try to import acontext with graceful fallback
ACONTEXT_AVAILABLE = False
AcontextClient = None
AcontextAsyncClient = None

try:
    from acontext import AcontextClient, AcontextAsyncClient
    ACONTEXT_AVAILABLE = True
    logger.info("Acontext SDK available for SOP storage integration")
except ImportError as e:
    logger.warning(f"Acontext SDK not available: {e}. Using fallback storage.")


@dataclass
class TradingDirective:
    """Represents a trading directive/SOP that can be stored and propagated."""
    id: str
    category: str
    name: str
    description: str
    content: Dict[str, Any]
    applies_to: List[str]  # List of agent roles this directive applies to
    source: str  # Source agent that created this directive
    priority: str  # Priority level: critical, high, medium, low, background
    sop_id: Optional[str] = None  # Acontext SOP ID (set after storage)
    created_at: Optional[str] = None
    expires_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingDirective':
        """Create from dictionary."""
        return cls(**data)


class AcontextIntegration:
    """
    Manages Acontext integration for SOP storage, retrieval, and cross-agent propagation.
    Provides graceful fallback when Acontext is unavailable.
    """

    def __init__(self, config_path: str = "config/acontext_config.yaml"):
        """
        Initialize Acontext integration.

        Args:
            config_path: Path to acontext configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.client = None
        self.async_client = None
        self._initialized = False
        self._fallback_mode = False
        self._directive_queue: List[TradingDirective] = []
        self._consecutive_failures = 0
        
        # Initialize fallback storage directory
        fallback_path = self.config.get('fallback', {}).get('local_storage_path', 'data/acontext_fallback')
        self.fallback_dir = Path(fallback_path)
        if self.config.get('fallback', {}).get('use_local_storage', True):
            self.fallback_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> Dict[str, Any]:
        """Load Acontext configuration from YAML file."""
        try:
            project_root = Path(__file__).parent.parent.parent
            config_file = project_root / self.config_path
            if config_file.exists():
                full_config = load_yaml(str(config_file))
                return full_config.get('acontext', {})
            else:
                logger.warning(f"Acontext config not found at {config_file}, using defaults")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading Acontext config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if config file is unavailable."""
        return {
            'api': {
                'base_url': 'https://api.acontext.dev',
                'timeout_seconds': 30,
                'max_retries': 3,
            },
            'space': {
                'name': 'abc-trading-sops',
            },
            'sop': {
                'id_prefix': 'trading_directive',
                'default_ttl_days': 90,
                'priority_levels': {
                    'critical': 100,
                    'high': 75,
                    'medium': 50,
                    'low': 25,
                    'background': 10,
                }
            },
            'fallback': {
                'enabled': True,
                'use_local_storage': True,
                'local_storage_path': 'data/acontext_fallback',
            },
            'propagation': {
                'enabled': True,
                'target_agents': ['strategy', 'risk', 'execution', 'learning'],
                'priority_queuing_enabled': True,
            }
        }

    async def initialize(self) -> bool:
        """
        Initialize Acontext client connection.

        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return not self._fallback_mode

        try:
            if not ACONTEXT_AVAILABLE:
                logger.warning("Acontext SDK not available, using fallback mode")
                self._fallback_mode = True
                self._initialized = True
                return False

            api_key = os.getenv("ACONTEXT_API_KEY")
            if not api_key:
                logger.warning("ACONTEXT_API_KEY not set, using fallback mode")
                self._fallback_mode = True
                self._initialized = True
                return False

            api_config = self.config.get('api', {})
            base_url = os.getenv("ACONTEXT_BASE_URL", api_config.get('base_url'))
            timeout = api_config.get('timeout_seconds', 30)

            # Initialize async client
            self.async_client = AcontextAsyncClient(
                api_key=api_key,
                base_url=base_url,
                timeout=float(timeout)
            )

            # Test connection
            try:
                await asyncio.wait_for(
                    self.async_client.ping(),
                    timeout=10.0
                )
                logger.info("Acontext connection established successfully")
                self._initialized = True
                self._fallback_mode = False
                return True
            except asyncio.TimeoutError:
                logger.warning("Acontext connection timed out, using fallback mode")
                self._fallback_mode = True
                self._initialized = True
                return False

        except Exception as e:
            logger.error(f"Failed to initialize Acontext client: {e}")
            self._fallback_mode = True
            self._initialized = True
            return False

    def _generate_directive_id(self, category: str) -> str:
        """Generate a unique directive ID."""
        prefix = self.config.get('sop', {}).get('id_prefix', 'trading_directive')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        return f"{prefix}_{category}_{timestamp}_{unique_id}"

    async def store_sop(self, directive: TradingDirective) -> Optional[str]:
        """
        Store a trading directive/SOP in Acontext.

        Args:
            directive: TradingDirective to store

        Returns:
            SOP ID if successful, None otherwise
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Ensure directive has required fields
            if not directive.id:
                directive.id = self._generate_directive_id(directive.category)
            if not directive.created_at:
                directive.created_at = datetime.now().isoformat()

            # Calculate expiration
            ttl_days = self.config.get('sop', {}).get('default_ttl_days', 90)
            if ttl_days > 0 and not directive.expires_at:
                directive.expires_at = (datetime.now() + timedelta(days=ttl_days)).isoformat()

            if self._fallback_mode:
                return await self._store_sop_fallback(directive)

            # Store in Acontext
            sop_content = json.dumps(directive.to_dict())
            
            # Use blocks API to store SOP content
            result = await self.async_client.blocks.create(
                content=sop_content,
                metadata={
                    'directive_id': directive.id,
                    'category': directive.category,
                    'priority': directive.priority,
                    'applies_to': ','.join(directive.applies_to),
                    'source': directive.source,
                }
            )

            sop_id = result.get('id') if isinstance(result, dict) else str(result)
            directive.sop_id = sop_id
            
            logger.info(f"Stored SOP {directive.id} in Acontext with ID: {sop_id}")
            self._consecutive_failures = 0
            
            # Recovery from fallback mode if operation succeeded
            if self._fallback_mode:
                logger.info("Successfully recovered from fallback mode")
                self._fallback_mode = False
                
            return sop_id

        except Exception as e:
            logger.error(f"Failed to store SOP in Acontext: {e}")
            self._consecutive_failures += 1
            
            # Check if we should switch to fallback
            threshold = self.config.get('monitoring', {}).get('alert_threshold_failures', 3)
            if self._consecutive_failures >= threshold:
                logger.warning(f"Switching to fallback mode after {self._consecutive_failures} failures")
                self._fallback_mode = True

            # Try fallback storage
            if self.config.get('fallback', {}).get('enabled', True):
                return await self._store_sop_fallback(directive)
            return None

    async def attempt_recovery_from_fallback(self) -> bool:
        """
        Attempt to recover from fallback mode by testing Acontext connectivity.
        
        Returns:
            True if recovery successful, False otherwise
        """
        if not self._fallback_mode:
            return True  # Already not in fallback mode
            
        try:
            if not ACONTEXT_AVAILABLE or not self.async_client:
                return False
                
            # Try to ping the service
            await asyncio.wait_for(
                self.async_client.ping(),
                timeout=5.0
            )
            
            # If successful, exit fallback mode
            logger.info("Recovery from fallback mode successful")
            self._fallback_mode = False
            self._consecutive_failures = 0
            return True
            
        except Exception as e:
            logger.debug(f"Recovery attempt failed: {e}")
            return False

    async def _store_sop_fallback(self, directive: TradingDirective) -> Optional[str]:
        """Store SOP in local fallback storage."""
        try:
            if not self.fallback_dir.exists():
                self.fallback_dir.mkdir(parents=True, exist_ok=True)

            # Generate fallback ID
            fallback_id = f"local_{directive.id}"
            directive.sop_id = fallback_id

            # Store as JSON file
            file_path = self.fallback_dir / f"{directive.id}.json"
            with open(file_path, 'w') as f:
                json.dump(directive.to_dict(), f, indent=2)

            if self.config.get('fallback', {}).get('log_fallback_events', True):
                logger.info(f"Stored SOP {directive.id} in fallback storage: {file_path}")

            return fallback_id

        except Exception as e:
            logger.error(f"Failed to store SOP in fallback storage: {e}")
            return None

    async def retrieve_sop(self, sop_id: str) -> Optional[TradingDirective]:
        """
        Retrieve a trading directive/SOP from Acontext.

        Args:
            sop_id: SOP ID to retrieve

        Returns:
            TradingDirective if found, None otherwise
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Check if it's a local fallback ID
            if sop_id.startswith('local_'):
                return await self._retrieve_sop_fallback(sop_id)

            if self._fallback_mode:
                return await self._retrieve_sop_fallback(sop_id)

            # Retrieve from Acontext
            result = await self.async_client.blocks.retrieve(sop_id)
            
            if result:
                content = result.get('content') if isinstance(result, dict) else str(result)
                directive_data = json.loads(content) if isinstance(content, str) else content
                directive = TradingDirective.from_dict(directive_data)
                logger.info(f"Retrieved SOP {sop_id} from Acontext")
                return directive

            return None

        except Exception as e:
            logger.error(f"Failed to retrieve SOP from Acontext: {e}")
            # Try fallback
            return await self._retrieve_sop_fallback(sop_id)

    async def _retrieve_sop_fallback(self, sop_id: str) -> Optional[TradingDirective]:
        """Retrieve SOP from local fallback storage."""
        try:
            # Extract directive ID from sop_id
            directive_id = sop_id.replace('local_', '') if sop_id.startswith('local_') else sop_id

            file_path = self.fallback_dir / f"{directive_id}.json"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                return TradingDirective.from_dict(data)

            # Try to find by pattern matching
            for file in self.fallback_dir.glob("*.json"):
                if directive_id in file.name:
                    with open(file, 'r') as f:
                        data = json.load(f)
                    return TradingDirective.from_dict(data)

            return None

        except Exception as e:
            logger.error(f"Failed to retrieve SOP from fallback storage: {e}")
            return None

    async def query_sops(self, category: Optional[str] = None, 
                        applies_to: Optional[str] = None,
                        priority: Optional[str] = None,
                        limit: int = 50) -> List[TradingDirective]:
        """
        Query SOPs based on filters.

        Args:
            category: Filter by category
            applies_to: Filter by target agent
            priority: Filter by priority level
            limit: Maximum number of results

        Returns:
            List of matching TradingDirectives
        """
        if not self._initialized:
            await self.initialize()

        try:
            if self._fallback_mode:
                return await self._query_sops_fallback(category, applies_to, priority, limit)

            # Build query filters
            filters = {}
            if category:
                filters['category'] = category
            if applies_to:
                filters['applies_to'] = applies_to
            if priority:
                filters['priority'] = priority

            # Query Acontext blocks
            result = await self.async_client.blocks.list(
                limit=limit,
                **filters
            )

            directives = []
            items = result.get('items', []) if isinstance(result, dict) else result
            for item in items:
                try:
                    content = item.get('content') if isinstance(item, dict) else str(item)
                    directive_data = json.loads(content) if isinstance(content, str) else content
                    directives.append(TradingDirective.from_dict(directive_data))
                except Exception as e:
                    logger.warning(f"Failed to parse SOP item: {e}")

            logger.info(f"Retrieved {len(directives)} SOPs from Acontext")
            return directives

        except Exception as e:
            logger.error(f"Failed to query SOPs from Acontext: {e}")
            return await self._query_sops_fallback(category, applies_to, priority, limit)

    async def _query_sops_fallback(self, category: Optional[str] = None,
                                   applies_to: Optional[str] = None,
                                   priority: Optional[str] = None,
                                   limit: int = 50) -> List[TradingDirective]:
        """Query SOPs from local fallback storage."""
        try:
            directives = []
            
            for file in self.fallback_dir.glob("*.json"):
                if len(directives) >= limit:
                    break
                    
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                    directive = TradingDirective.from_dict(data)

                    # Apply filters
                    if category and directive.category != category:
                        continue
                    if applies_to and applies_to not in directive.applies_to:
                        continue
                    if priority and directive.priority != priority:
                        continue

                    directives.append(directive)

                except Exception as e:
                    logger.warning(f"Failed to parse fallback SOP file {file}: {e}")

            return directives

        except Exception as e:
            logger.error(f"Failed to query SOPs from fallback storage: {e}")
            return []

    async def log_session(self, session_data: Dict[str, Any]) -> Optional[str]:
        """
        Log a trading session.

        Args:
            session_data: Session data to log

        Returns:
            Session ID if successful, None otherwise
        """
        if not self._initialized:
            await self.initialize()

        session_config = self.config.get('session', {})
        if not session_config.get('log_all_sessions', True):
            return None

        try:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            session_data['session_id'] = session_id
            session_data['timestamp'] = datetime.now().isoformat()

            if self._fallback_mode:
                return await self._log_session_fallback(session_data)

            # Log to Acontext sessions
            result = await self.async_client.sessions.create(
                data=json.dumps(session_data),
                metadata={
                    'session_id': session_id,
                    'type': 'trading_session',
                }
            )

            logger.info(f"Logged trading session: {session_id}")
            return session_id

        except Exception as e:
            logger.error(f"Failed to log session: {e}")
            if self.config.get('fallback', {}).get('enabled', True):
                return await self._log_session_fallback(session_data)
            return None

    async def _log_session_fallback(self, session_data: Dict[str, Any]) -> Optional[str]:
        """Log session to local fallback storage."""
        try:
            sessions_dir = self.fallback_dir / "sessions"
            sessions_dir.mkdir(parents=True, exist_ok=True)

            session_id = session_data.get('session_id', f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            file_path = sessions_dir / f"{session_id}.json"

            with open(file_path, 'w') as f:
                json.dump(session_data, f, indent=2)

            return session_id

        except Exception as e:
            logger.error(f"Failed to log session to fallback: {e}")
            return None

    async def upload_artifact(self, artifact_type: str, artifact_data: bytes, 
                             metadata: Dict[str, Any]) -> Optional[str]:
        """
        Upload an artifact (ML model, backtest result, etc.).

        Args:
            artifact_type: Type of artifact
            artifact_data: Binary artifact data
            metadata: Artifact metadata

        Returns:
            Artifact ID if successful, None otherwise
        """
        if not self._initialized:
            await self.initialize()

        artifacts_config = self.config.get('artifacts', {})
        if not artifacts_config.get('enabled', True):
            logger.info("Artifact uploads disabled in configuration")
            return None

        # Check artifact type
        allowed_types = artifacts_config.get('types', [])
        if allowed_types and artifact_type not in allowed_types:
            logger.warning(f"Artifact type '{artifact_type}' not in allowed types: {allowed_types}")
            return None

        # Check size limit
        max_size = artifacts_config.get('max_size_mb', 100) * 1024 * 1024
        if len(artifact_data) > max_size:
            logger.warning(f"Artifact exceeds max size: {len(artifact_data)} > {max_size}")
            return None

        try:
            artifact_id = f"artifact_{artifact_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            metadata['artifact_id'] = artifact_id
            metadata['artifact_type'] = artifact_type
            metadata['timestamp'] = datetime.now().isoformat()
            metadata['size_bytes'] = len(artifact_data)

            if self._fallback_mode:
                return await self._upload_artifact_fallback(artifact_id, artifact_data, metadata)

            # Upload to Acontext disks
            result = await self.async_client.disks.artifacts.upload(
                data=artifact_data,
                metadata=metadata
            )

            logger.info(f"Uploaded artifact: {artifact_id}")
            return artifact_id

        except Exception as e:
            logger.error(f"Failed to upload artifact: {e}")
            if self.config.get('fallback', {}).get('enabled', True):
                return await self._upload_artifact_fallback(artifact_id, artifact_data, metadata)
            return None

    async def _upload_artifact_fallback(self, artifact_id: str, artifact_data: bytes,
                                        metadata: Dict[str, Any]) -> Optional[str]:
        """Upload artifact to local fallback storage."""
        try:
            artifacts_dir = self.fallback_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            # Save artifact data
            artifact_path = artifacts_dir / f"{artifact_id}.bin"
            with open(artifact_path, 'wb') as f:
                f.write(artifact_data)

            # Save metadata
            metadata_path = artifacts_dir / f"{artifact_id}.meta.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Uploaded artifact to fallback: {artifact_id}")
            return artifact_id

        except Exception as e:
            logger.error(f"Failed to upload artifact to fallback: {e}")
            return None

    def queue_directive(self, directive: TradingDirective) -> None:
        """
        Add directive to the priority queue for asynchronous propagation.

        Args:
            directive: TradingDirective to queue
        """
        if not self.config.get('propagation', {}).get('priority_queuing_enabled', True):
            logger.info("Priority queuing disabled")
            return

        self._directive_queue.append(directive)
        # Sort by priority (higher priority first)
        priority_levels = self.config.get('sop', {}).get('priority_levels', {})
        self._directive_queue.sort(
            key=lambda d: priority_levels.get(d.priority, 50),
            reverse=True
        )
        logger.debug(f"Queued directive {directive.id} with priority {directive.priority}")

    async def process_directive_queue(self) -> List[str]:
        """
        Process queued directives.

        Returns:
            List of processed directive IDs
        """
        processed = []
        
        while self._directive_queue:
            directive = self._directive_queue.pop(0)
            try:
                sop_id = await self.store_sop(directive)
                if sop_id:
                    processed.append(directive.id)
                    logger.info(f"Processed queued directive: {directive.id}")
            except Exception as e:
                logger.error(f"Failed to process queued directive {directive.id}: {e}")
                # Re-queue for retry (with lower priority)
                directive.priority = 'background'
                self._directive_queue.append(directive)

        return processed

    def get_queue_status(self) -> Dict[str, Any]:
        """Get status of the directive queue."""
        priority_counts = {}
        for directive in self._directive_queue:
            priority_counts[directive.priority] = priority_counts.get(directive.priority, 0) + 1

        return {
            'total_queued': len(self._directive_queue),
            'by_priority': priority_counts,
            'fallback_mode': self._fallback_mode,
            'consecutive_failures': self._consecutive_failures,
        }

    async def close(self) -> None:
        """Close Acontext client connections."""
        try:
            if self.async_client:
                await self.async_client.aclose()
                logger.info("Acontext async client closed")
        except Exception as e:
            logger.warning(f"Error closing Acontext client: {e}")


# Singleton instance
_acontext_integration: Optional[AcontextIntegration] = None


def get_acontext_integration() -> AcontextIntegration:
    """Get the singleton AcontextIntegration instance."""
    global _acontext_integration
    if _acontext_integration is None:
        _acontext_integration = AcontextIntegration()
    return _acontext_integration


async def initialize_acontext() -> bool:
    """Initialize the Acontext integration."""
    integration = get_acontext_integration()
    return await integration.initialize()
