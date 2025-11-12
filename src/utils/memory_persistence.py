# src/utils/memory_persistence.py
# Purpose: Basic JSON-based memory persistence for agents
# Provides save/load functionality for agent memory across restarts
# Foundation for more advanced memory systems (Redis, Mem0, etc.)

import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

class MemoryPersistence:
    """
    Basic JSON-based memory persistence system.
    Provides foundation for agent memory storage and retrieval.
    """

    def __init__(self, base_dir: str = "data/memory"):
        """
        Initialize memory persistence system.

        Args:
            base_dir: Base directory for memory storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different memory types
        self.agent_memory_dir = self.base_dir / "agents"
        self.shared_memory_dir = self.base_dir / "shared"
        self.backups_dir = self.base_dir / "backups"

        for dir_path in [self.agent_memory_dir, self.shared_memory_dir, self.backups_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Memory persistence initialized at {self.base_dir}")

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a string to be safe for use as a filename.
        Replaces invalid characters with underscores.

        Args:
            filename: Original filename string

        Returns:
            str: Sanitized filename
        """
        import re
        # Replace invalid characters for Windows filenames
        # Invalid chars: < > : " | ? * \ /
        # Also replace spaces and other potentially problematic chars
        sanitized = re.sub(r'[<>:"|*\\\/\s]', '_', filename)
        # Handle ? separately - replace with empty string at end
        sanitized = re.sub(r'\?+$', '', sanitized)
        # Replace multiple underscores with single
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores and spaces
        sanitized = sanitized.strip('_ ')
        # Ensure it's not empty
        if not sanitized:
            sanitized = "_"
        # Handle length limit
        if len(sanitized) > 255:
            sanitized = sanitized[:255]
        return sanitized

    def save_agent_memory(self, agent_role: str, memory_data: Dict[str, Any],
                         create_backup: bool = True) -> bool:
        """
        Save agent memory to JSON file.

        Args:
            agent_role: Role/name of the agent (e.g., 'reflection', 'strategy')
            memory_data: Memory dictionary to save
            create_backup: Whether to create a backup of previous memory

        Returns:
            bool: Success status
        """
        try:
            # Create backup if requested and file exists
            memory_file = self.agent_memory_dir / f"{agent_role}_memory.json"
            if create_backup and memory_file.exists():
                self.create_backup(agent_role)

            # Add metadata
            memory_with_meta = {
                "agent_role": agent_role,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0",
                "data": memory_data
            }

            # Save to file
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_with_meta, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved memory for agent '{agent_role}' to {memory_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save memory for agent '{agent_role}': {e}")
            return False

    def load_agent_memory(self, agent_role: str) -> Optional[Dict[str, Any]]:
        """
        Load agent memory from JSON file.

        Args:
            agent_role: Role/name of the agent

        Returns:
            Optional[Dict]: Memory data or None if not found/error
        """
        try:
            memory_file = self.agent_memory_dir / f"{agent_role}_memory.json"

            if not memory_file.exists():
                logger.info(f"No saved memory found for agent '{agent_role}'")
                return None

            with open(memory_file, 'r', encoding='utf-8') as f:
                memory_with_meta = json.load(f)

            # Validate structure
            if "data" not in memory_with_meta:
                logger.error(f"Invalid memory file structure for agent '{agent_role}'")
                return None

            logger.info(f"Loaded memory for agent '{agent_role}' from {memory_file}")
            return memory_with_meta["data"]

        except Exception as e:
            logger.error(f"Failed to load memory for agent '{agent_role}': {e}")
            return None

    def save_shared_memory(self, namespace: str, memory_data: Dict[str, Any],
                          create_backup: bool = True) -> bool:
        """
        Save shared memory to JSON file.

        Args:
            namespace: Shared memory namespace (e.g., 'user_prefs', 'market_data')
            memory_data: Shared memory data
            create_backup: Whether to create backup

        Returns:
            bool: Success status
        """
        try:
            # Sanitize namespace for filename
            safe_namespace = self._sanitize_filename(namespace)

            memory_file = self.shared_memory_dir / f"{safe_namespace}_shared.json"
            if create_backup and memory_file.exists():
                self.create_backup(f"shared_{safe_namespace}")

            memory_with_meta = {
                "namespace": namespace,  # Store original namespace
                "safe_namespace": safe_namespace,  # Store sanitized version
                "timestamp": datetime.now().isoformat(),
                "version": "1.0",
                "data": memory_data
            }

            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_with_meta, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved shared memory for namespace '{namespace}' to {memory_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save shared memory for namespace '{namespace}': {e}")
            return False

    def load_shared_memory(self, namespace: str) -> Optional[Dict[str, Any]]:
        """
        Load shared memory from JSON file.

        Args:
            namespace: Shared memory namespace

        Returns:
            Optional[Dict]: Shared memory data or None
        """
        try:
            # Try direct filename first (for backward compatibility)
            memory_file = self.shared_memory_dir / f"{namespace}_shared.json"

            # If file doesn't exist, try sanitized version
            if not memory_file.exists():
                safe_namespace = self._sanitize_filename(namespace)
                memory_file = self.shared_memory_dir / f"{safe_namespace}_shared.json"

            if not memory_file.exists():
                logger.info(f"No shared memory found for namespace '{namespace}'")
                return None

            with open(memory_file, 'r', encoding='utf-8') as f:
                memory_with_meta = json.load(f)

            if "data" not in memory_with_meta:
                logger.error(f"Invalid shared memory file structure for namespace '{namespace}'")
                return None

            logger.info(f"Loaded shared memory for namespace '{namespace}'")
            return memory_with_meta["data"]

        except Exception as e:
            logger.error(f"Failed to load shared memory for namespace '{namespace}': {e}")
            return None

    def list_agent_memories(self) -> list[str]:
        """
        List all saved agent memory files.

        Returns:
            list[str]: List of agent roles with saved memory
        """
        try:
            memory_files = list(self.agent_memory_dir.glob("*_memory.json"))
            agent_roles = [f.stem.replace("_memory", "") for f in memory_files]
            return agent_roles
        except Exception as e:
            logger.error(f"Failed to list agent memories: {e}")
            return []

    def list_shared_memories(self) -> list[str]:
        """
        List all saved shared memory namespaces.

        Returns:
            list[str]: List of shared memory namespaces (original names)
        """
        try:
            memory_files = list(self.shared_memory_dir.glob("*_shared.json"))
            namespaces = []

            for memory_file in memory_files:
                try:
                    with open(memory_file, 'r', encoding='utf-8') as f:
                        memory_with_meta = json.load(f)

                    # Use original namespace if available, otherwise derive from filename
                    original_namespace = memory_with_meta.get("namespace")
                    if original_namespace:
                        namespaces.append(original_namespace)
                    else:
                        # Fallback: remove _shared suffix
                        namespaces.append(memory_file.stem.replace("_shared", ""))
                except Exception as e:
                    logger.warning(f"Could not read namespace from {memory_file}: {e}")
                    # Fallback to filename
                    namespaces.append(memory_file.stem.replace("_shared", ""))

            return namespaces
        except Exception as e:
            logger.error(f"Failed to list shared memories: {e}")
            return []

    def cleanup_old_backups(self, max_backups: int = 10) -> int:
        """
        Clean up old backup files, keeping only the most recent ones.

        Args:
            max_backups: Maximum number of backups to keep per memory file

        Returns:
            int: Number of backups deleted
        """
        try:
            deleted_count = 0
            backup_files = list(self.backups_dir.glob("*.json"))

            # Group by original filename
            backup_groups = {}
            for backup_file in backup_files:
                # Extract original filename from backup name
                parts = backup_file.stem.split("_backup_")
                if len(parts) >= 2:
                    original_name = "_backup_".join(parts[:-1])
                    timestamp = parts[-1]

                    if original_name not in backup_groups:
                        backup_groups[original_name] = []
                    backup_groups[original_name].append((backup_file, timestamp))

            # Clean up each group
            for original_name, backups in backup_groups.items():
                if len(backups) > max_backups:
                    # Sort by timestamp (newest first) and delete oldest
                    backups.sort(key=lambda x: x[1], reverse=True)
                    to_delete = backups[max_backups:]

                    for backup_file, _ in to_delete:
                        backup_file.unlink()
                        deleted_count += 1

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old backup files")

            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")
            return 0

    def create_backup(self, agent_id: str) -> Optional[Path]:
        """
        Create a backup of an agent's memory.

        Args:
            agent_id: Agent identifier

        Returns:
            Optional[Path]: Path to backup file if successful, None otherwise
        """
        try:
            memory_file = self.agent_memory_dir / f"{agent_id}_memory.json"
            if not memory_file.exists():
                logger.warning(f"No memory file found for agent '{agent_id}' to backup")
                return None

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{agent_id}_backup_{timestamp}.json"
            backup_file = self.backups_dir / backup_filename

            # Copy original to backup
            import shutil
            shutil.copy2(memory_file, backup_file)

            logger.info(f"Created backup: {backup_file}")
            return backup_file

        except Exception as e:
            logger.error(f"Failed to create backup for agent '{agent_id}': {e}")
            return None

    def restore_from_backup(self, agent_id: str, backup_path: Path) -> bool:
        """
        Restore agent memory from a backup file.

        Args:
            agent_id: Agent identifier
            backup_path: Path to backup file

        Returns:
            bool: Success status
        """
        try:
            if not backup_path.exists():
                logger.error(f"Backup file does not exist: {backup_path}")
                return False

            # Copy backup to current memory file
            memory_file = self.agent_memory_dir / f"{agent_id}_memory.json"
            import shutil
            shutil.copy2(backup_path, memory_file)

            logger.info(f"Restored memory for agent '{agent_id}' from backup: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore backup for agent '{agent_id}': {e}")
            return False

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored memories.

        Returns:
            Dict: Memory statistics
        """
        try:
            agent_memories = self.list_agent_memories()
            shared_memories = self.list_shared_memories()

            # Calculate total size
            total_size = 0
            for memory_file in self.agent_memory_dir.glob("*_memory.json"):
                total_size += memory_file.stat().st_size
            for memory_file in self.shared_memory_dir.glob("*_shared.json"):
                total_size += memory_file.stat().st_size

            backup_files = list(self.backups_dir.glob("*.json"))
            backup_size = sum(f.stat().st_size for f in backup_files)

            return {
                "agent_memories_count": len(agent_memories),
                "shared_memories_count": len(shared_memories),
                "total_memory_size_kb": round(total_size / 1024, 2),
                "backup_files_count": len(backup_files),
                "backup_size_kb": round(backup_size / 1024, 2),
                "agent_roles": agent_memories,
                "shared_namespaces": shared_memories,
                "total_agents": len(agent_memories),
                "total_shared_items": len(shared_memories)  # Add missing key
            }

        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}

    def load_long_term_memory(self) -> Optional[Dict[str, Any]]:
        """
        Load long-term memory data.
        This is a convenience method that loads from a special 'long_term' namespace.

        Returns:
            Optional[Dict]: Long-term memory data or None
        """
        return self.load_shared_memory("long_term")

    def load_agent_memory_spaces(self) -> Optional[Dict[str, Any]]:
        """
        Load all agent memory spaces.
        This loads from a special 'agent_spaces' namespace.

        Returns:
            Optional[Dict]: Agent memory spaces data or None
        """
        return self.load_shared_memory("agent_spaces")

    def load_positions_memory(self) -> Optional[Dict[str, Any]]:
        """
        Load positions memory data.
        This loads from a special 'positions' namespace.

        Returns:
            Optional[Dict]: Positions memory data or None
        """
        return self.load_shared_memory("positions")

# Global instance for easy access
_memory_persistence = None

def get_memory_persistence() -> MemoryPersistence:
    """
    Get global memory persistence instance.

    Returns:
        MemoryPersistence: Global instance
    """
    global _memory_persistence
    if _memory_persistence is None:
        _memory_persistence = MemoryPersistence()
    return _memory_persistence

# Convenience functions for easy use
def save_agent_memory(agent_role: str, memory_data: Dict[str, Any]) -> bool:
    """Convenience function to save agent memory."""
    return get_memory_persistence().save_agent_memory(agent_role, memory_data)

def load_agent_memory(agent_role: str) -> Optional[Dict[str, Any]]:
    """Convenience function to load agent memory."""
    return get_memory_persistence().load_agent_memory(agent_role)

def save_shared_memory(namespace: str, memory_data: Dict[str, Any]) -> bool:
    """Convenience function to save shared memory."""
    return get_memory_persistence().save_shared_memory(namespace, memory_data)

def load_shared_memory(namespace: str) -> Optional[Dict[str, Any]]:
    """Convenience function to load shared memory."""
    return get_memory_persistence().load_shared_memory(namespace)