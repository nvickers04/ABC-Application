# src/utils/llm_factory.py
# [LABEL:COMPONENT:llm_factory] [LABEL:FRAMEWORK:langchain]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-12-04] [LABEL:REVIEWED:pending]
#
# Purpose: Factory class for LLM initialization, extracted from BaseAgent
# Dependencies: langchain libraries, asyncio
# Related: src/agents/base.py, docs/architecture.md

import asyncio
import logging
from typing import Optional, Any, Dict
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

class LLMFactory:
    """
    Factory for initializing various LLM providers with consistent configuration.
    """
    
    @staticmethod
    def create_xai_llm(api_key: str, model: str, config: Dict[str, Any]) -> Optional[Any]:
        """Create XAI Chat model."""
        try:
            from langchain_xai import ChatXAI
            return ChatXAI(
                api_key=api_key,
                model=model,
                temperature=config.get('temperature', 0.1),
                max_tokens=config.get('max_tokens', 32768),
                timeout=config.get('timeout_seconds', 30)
            )
        except Exception as e:
            logger.debug(f"XAI {model} initialization failed: {e}")
            return None

    @staticmethod
    def create_openai_llm(api_key: str, model: str, config: Dict[str, Any]) -> Optional[Any]:
        """Create OpenAI Chat model."""
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=config.get('temperature', 0.1),
                max_tokens=config.get('max_tokens', 32768),
                timeout=config.get('timeout_seconds', 30)
            )
        except Exception as e:
            logger.debug(f"OpenAI {model} initialization failed: {e}")
            return None

    @staticmethod
    def create_anthropic_llm(api_key: str, model: str, config: Dict[str, Any]) -> Optional[Any]:
        """Create Anthropic Claude Chat model."""
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                api_key=api_key,
                model=model,
                temperature=config.get('temperature', 0.1),
                max_tokens=config.get('max_tokens', 32768),
                timeout=config.get('timeout_seconds', 30)
            )
        except Exception as e:
            logger.debug(f"Anthropic {model} initialization failed: {e}")
            return None

    @staticmethod
    def create_google_llm(api_key: str, model: str, config: Dict[str, Any]) -> Optional[Any]:
        """Create Google Gemini Chat model."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                api_key=api_key,
                model=model,
                temperature=config.get('temperature', 0.1),
                max_tokens=config.get('max_tokens', 32768),
                timeout=config.get('timeout_seconds', 30)
            )
        except Exception as e:
            logger.debug(f"Google {model} initialization failed: {e}")
            return None

    @staticmethod
    async def test_llm_connection(llm, timeout: int = 30) -> bool:
        """Test LLM connectivity."""
        try:
            test_prompt = "Respond with 'OK' if you can understand this message."
            response = await asyncio.wait_for(
                llm.ainvoke(test_prompt),
                timeout=timeout
            )
            return response and hasattr(response, 'content') and 'OK' in str(response.content).upper()
        except Exception as e:
            logger.debug(f"LLM connectivity test failed: {e}")
            return False