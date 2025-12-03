# src/agents/data_analyzers/base_data_analyzer.py
# Base Data Analyzer Class
# Provides common data exploration and analysis functionality for all data analyzers
# Ensures consistent data exploration patterns across different data sources

import asyncio
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from abc import ABC, abstractmethod

from src.agents.base import BaseAgent
from src.utils.redis_cache import get_redis_cache_manager

logger = logging.getLogger(__name__)

class BaseDataAnalyzer(BaseAgent, ABC):
    """
    Base class for all data analyzers providing common data exploration functionality.

    This class provides:
    - Standardized data exploration planning
    - Common data validation and quality checks
    - Consistent memory storage patterns
    - Abstract methods for data source-specific implementations
    """

    def __init__(self, role: str):
        super().__init__(role=role)
        self.cache_manager = get_redis_cache_manager()
        self.data_quality_threshold = 0.8
        self.max_concurrent_requests = 5

    async def _process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standard process_input method for all data analyzers.
        This method can be overridden by subclasses for custom behavior.

        Args:
            input_data: Dictionary containing analysis parameters

        Returns:
            Dictionary with consolidated analysis results
        """
        try:
            # Get exploration plan - try different signatures for compatibility
            try:
                exploration_plan = await self._plan_data_exploration(input_data)
            except TypeError:
                # Fallback for analyzers with different method signatures
                symbols = input_data.get("symbols", [])
                exploration_plan = await self._plan_data_exploration(symbols, input_data)

            # Execute data fetching - try different signatures for compatibility
            try:
                raw_data = await self._execute_data_exploration(exploration_plan)
            except TypeError:
                # Fallback for analyzers with different method signatures
                symbols = input_data.get("symbols", [])
                raw_data = await self._execute_data_exploration(symbols, exploration_plan)

            # Validate data quality
            validated_data = self._validate_data_quality(raw_data)

            # Enhance with analysis
            enhanced_data = await self._enhance_data(validated_data)

            # Store in memory
            await self._store_analysis_results(input_data, enhanced_data)

            return {
                "consolidated_data": enhanced_data,
                "data_quality": self._calculate_data_quality_score(validated_data),
                "exploration_plan": exploration_plan,
                "enhanced": True
            }

        except Exception as e:
            logger.error(f"{self.__class__.__name__} process_input failed: {e}")
            return {
                "error": str(e),
                "enhanced": False,
                "data_quality": 0.0
            }

    @abstractmethod
    async def _plan_data_exploration(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Plan the data exploration strategy.

        Args:
            *args, **kwargs: Flexible arguments for different analyzer implementations

        Returns:
            Dictionary with exploration plan
        """
        pass

    @abstractmethod
    async def _execute_data_exploration(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute the data exploration plan.

        Args:
            *args, **kwargs: Flexible arguments for different analyzer implementations

        Returns:
            Raw data from exploration
        """
        pass

    def _validate_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the quality of fetched data.

        Args:
            data: Raw data to validate

        Returns:
            Validated data with quality metrics
        """
        # Basic validation - can be overridden by subclasses
        validated = {}

        for key, value in data.items():
            if isinstance(value, dict):
                # Keep dicts as-is (they are assumed to be structured data)
                validated[key] = value
            else:
                validated[key] = {"data": value, "quality_score": 0.8}

        return validated

    @abstractmethod
    async def _enhance_data(self, validated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance validated data with additional analysis.

        Args:
            validated_data: Data from _validate_data_quality

        Returns:
            Enhanced data with analysis
        """
        pass

    async def _store_analysis_results(self, input_data: Dict[str, Any], enhanced_data: Dict[str, Any]) -> None:
        """
        Store analysis results in memory.

        Args:
            input_data: Original input data
            enhanced_data: Enhanced analysis results
        """
        try:
            # Create a key based on input data
            key_components = []
            if 'symbols' in input_data and input_data['symbols']:
                key_components.append(f"symbols:{','.join(input_data['symbols'][:3])}")
            elif 'symbol' in input_data:
                key_components.append(f"symbol:{input_data['symbol']}")
            else:
                key_components.append("general")

            key_components.append(f"analyzer:{self.__class__.__name__}")
            key_components.append(f"timestamp:{datetime.now().isoformat()}")

            memory_key = ":".join(key_components)

            await self.memory_manager.store_memory(
                memory_key,
                {
                    "input_data": input_data,
                    "enhanced_data": enhanced_data,
                    "analyzer": self.__class__.__name__,
                    "timestamp": datetime.now().isoformat()
                },
                "working_memory"
            )

        except Exception as e:
            logger.error(f"Failed to store analysis results: {e}")

    def _calculate_data_quality_score(self, validated_data: Dict[str, Any]) -> float:
        """
        Calculate overall data quality score.

        Args:
            validated_data: Validated data with quality scores

        Returns:
            Overall quality score (0.0 to 1.0)
        """
        if not validated_data:
            return 0.0

        quality_scores = []
        for item in validated_data.values():
            if isinstance(item, dict) and 'quality_score' in item:
                quality_scores.append(item['quality_score'])
            elif isinstance(item, dict) and 'data' in item:
                quality_scores.append(0.8)  # Default quality for data items

        if not quality_scores:
            return 0.0

        return sum(quality_scores) / len(quality_scores)

    async def _get_exploration_plan_template(self, data_sources: List[str], data_types: List[str],
                                           symbols: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a template exploration plan that can be customized by subclasses.

        Args:
            data_sources: List of data sources to use
            data_types: Types of data to fetch
            symbols: Symbols/assets to analyze
            context: Additional context

        Returns:
            Template exploration plan
        """
        return {
            "data_sources": data_sources,
            "data_types": data_types,
            "symbols": symbols,
            "context": context,
            "strategy": "concurrent_fetch",
            "max_concurrent": self.max_concurrent_requests,
            "quality_threshold": self.data_quality_threshold,
            "timestamp": datetime.now().isoformat()
        }