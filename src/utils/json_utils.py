"""
JSON utility functions to avoid circular imports.
"""

import json
import logging
from typing import Any, Optional, Set

logger = logging.getLogger(__name__)

def sanitize_for_json(data: Any, max_depth: int = 3, current_depth: int = 0, seen_objects: Optional[Set] = None) -> Any:
    """
    Recursively sanitize data for JSON serialization.

    Args:
        data: The data to sanitize
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth
        seen_objects: Set of object IDs already seen (for circular reference detection)

    Returns:
        JSON-serializable version of the data
    """
    if seen_objects is None:
        seen_objects = set()

    # Prevent infinite recursion
    if current_depth >= max_depth:
        return f"<truncated at depth {max_depth}>"

    # Handle circular references
    obj_id = id(data)
    if obj_id in seen_objects:
        return f"<circular reference to {type(data).__name__}>"
    seen_objects.add(obj_id)

    try:
        if data is None:
            return None
        elif isinstance(data, (str, int, float, bool)):
            return data
        elif isinstance(data, (list, tuple)):
            return [sanitize_for_json(item, max_depth, current_depth + 1, seen_objects) for item in data]
        elif isinstance(data, dict):
            return {str(k): sanitize_for_json(v, max_depth, current_depth + 1, seen_objects) for k, v in data.items()}
        elif hasattr(data, 'to_dict') and callable(getattr(data, 'to_dict')):
            # Pandas DataFrame/Series or similar objects
            try:
                dict_data = data.to_dict()
                return sanitize_for_json(dict_data, max_depth, current_depth + 1, seen_objects)
            except Exception:
                return str(data)
        elif hasattr(data, '__dict__'):
            # Custom objects - convert to dict
            try:
                obj_dict = {}
                for attr in dir(data):
                    if not attr.startswith('_'):
                        try:
                            value = getattr(data, attr)
                            if not callable(value):
                                obj_dict[attr] = sanitize_for_json(value, max_depth, current_depth + 1, seen_objects)
                        except Exception:
                            obj_dict[attr] = f"<error accessing {attr}>"
                return obj_dict
            except Exception:
                return str(data)
        elif hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
            # Other iterables
            try:
                return [sanitize_for_json(item, max_depth, current_depth + 1, seen_objects) for item in data]
            except Exception:
                return str(data)
        else:
            # Fallback to string representation
            return str(data)
    finally:
        seen_objects.discard(obj_id)