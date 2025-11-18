# src/agents/base.py
# Purpose: Base class for all agents in the AI Portfolio Manager, providing common structure (e.g., YAML/prompt loading, logging).
# This is abstract—subclasses like RiskAgent implement process_input.
# Structural Reasoning: Ensures consistency across agents (e.g., fresh YAML loads for constraints); backs funding with traceable logs (e.g., every decision audited).
# Ties to code-skeleton.md: Implements BaseAgent with init/tools/memory stubs; async for scalability in loops/pings.
# For legacy wealth: Robust error-handling preserves capital (e.g., defaults on failures, enforcing <5% drawdown).
# Update: Dynamic sys.path insert to fix 'No module named src' on direct runs (prepends project root for absolute imports).

import abc
import logging
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path
import os
from dotenv import load_dotenv
import uuid
import pandas as pd
import asyncio

# Load environment variables
load_dotenv()

# Dynamic path setup for robust imports (works from any subdir)
project_root = Path(__file__).parent.parent.parent  # From src/agents/base.py -> project root
sys.path.insert(0, str(project_root))

from src.utils.utils import load_yaml, load_prompt_template  # Absolute from src/utils.py (now discoverable).
from langchain_core.tools import BaseTool

# Lazy imports for heavy dependencies to avoid startup failures
_langchain_openai = None
_langchain_xai = None
_langchain_anthropic = None
_langchain_google = None
_memory_persistence = None
_advanced_memory = None
_shared_memory = None
_api_health_monitor = None

def _get_langchain_openai():
    global _langchain_openai
    if _langchain_openai is None:
        try:
            from langchain_openai import ChatOpenAI
            _langchain_openai = ChatOpenAI
        except ImportError as e:
            logger.warning(f"Failed to import langchain_openai: {e}")
            _langchain_openai = None
    return _langchain_openai

def _get_langchain_xai():
    global _langchain_xai
    if _langchain_xai is None:
        try:
            from langchain_xai import ChatXAI
            _langchain_xai = ChatXAI
        except ImportError as e:
            logger.warning(f"Failed to import langchain_xai: {e}")
            _langchain_xai = None
    return _langchain_xai

def _get_langchain_anthropic():
    global _langchain_anthropic
    if _langchain_anthropic is None:
        try:
            from langchain_anthropic import ChatAnthropic
            _langchain_anthropic = ChatAnthropic
        except ImportError as e:
            logger.warning(f"Failed to import langchain_anthropic: {e}")
            _langchain_anthropic = None
    return _langchain_anthropic

def _get_langchain_google():
    global _langchain_google
    if _langchain_google is None:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            _langchain_google = ChatGoogleGenerativeAI
        except ImportError as e:
            logger.warning(f"Failed to import langchain_google_genai: {e}")
            _langchain_google = None
    return _langchain_google

def _get_api_health_monitor():
    """Get the centralized API health monitor instance"""
    global _api_health_monitor
    if _api_health_monitor is None:
        try:
            from src.utils.api_health_monitor import get_health_monitor
            _api_health_monitor = get_health_monitor()
        except ImportError as e:
            logger.warning(f"Failed to import API health monitor: {e}")
            _api_health_monitor = None
    return _api_health_monitor

def _get_memory_persistence():
    global _memory_persistence
    if _memory_persistence is None:
        try:
            from src.utils.memory_persistence import get_memory_persistence
            _memory_persistence = get_memory_persistence()
        except ImportError as e:
            logger.warning(f"Failed to import memory_persistence: {e}")
            _memory_persistence = None
    return _memory_persistence

def _get_advanced_memory_manager():
    global _advanced_memory
    if _advanced_memory is None:
        try:
            from src.utils.advanced_memory import get_advanced_memory_manager
            _advanced_memory = get_advanced_memory_manager()
        except ImportError as e:
            logger.warning(f"Failed to import advanced_memory: {e}")
            _advanced_memory = None
    return _advanced_memory

def _get_multi_agent_coordinator():
    global _shared_memory
    if _shared_memory is None:
        try:
            from src.utils.shared_memory import get_multi_agent_coordinator
            _shared_memory = get_multi_agent_coordinator()
        except ImportError as e:
            logger.warning(f"Failed to import shared_memory: {e}")
            _shared_memory = None
    return _shared_memory

# Setup logging (shared across agents for audits)
logger = logging.getLogger(__name__)

class BaseAgent(abc.ABC):
    """
    Abstract base class for agents.
    Reasoning: Provides init with prompt/YAML loading; abstract process_input for role-specific logic (e.g., Risk vets proposals).
    """
    def _try_initialize_xai(self, api_key: str, model: str) -> Optional[Any]:
        """Try to initialize XAI Chat model."""
        try:
            ChatXAI = _get_langchain_xai()
            if not ChatXAI:
                return None

            return ChatXAI(
                api_key=api_key,
                model=model,
                temperature=0.1,  # Lower temperature for more consistent responses
                max_tokens=4096,
                timeout=30
            )
        except Exception as e:
            logger.debug(f"XAI {model} initialization failed: {e}")
            return None

    def _try_initialize_openai(self, api_key: str, model: str) -> Optional[Any]:
        """Try to initialize OpenAI Chat model."""
        try:
            ChatOpenAI = _get_langchain_openai()
            if not ChatOpenAI:
                return None

            return ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=0.1,
                max_tokens=4096,
                timeout=30
            )
        except Exception as e:
            logger.debug(f"OpenAI {model} initialization failed: {e}")
            return None

    def _try_initialize_anthropic(self, api_key: str, model: str) -> Optional[Any]:
        """Try to initialize Anthropic Claude Chat model."""
        try:
            ChatAnthropic = _get_langchain_anthropic()
            if not ChatAnthropic:
                return None

            return ChatAnthropic(
                api_key=api_key,
                model=model,
                temperature=0.1,
                max_tokens=4096,
                timeout=30
            )
        except Exception as e:
            logger.debug(f"Anthropic {model} initialization failed: {e}")
            return None

    def _try_initialize_google(self, api_key: str, model: str) -> Optional[Any]:
        """Try to initialize Google Gemini Chat model."""
        try:
            ChatGoogleGenerativeAI = _get_langchain_google()
            if not ChatGoogleGenerativeAI:
                return None

            return ChatGoogleGenerativeAI(
                api_key=api_key,
                model=model,
                temperature=0.1,
                max_tokens=4096,
                timeout=30
            )
        except Exception as e:
            logger.debug(f"Google {model} initialization failed: {e}")
            return None

    async def _test_llm_connection(self, llm) -> bool:
        """Test LLM connectivity with a simple query."""
        try:
            test_prompt = "Respond with 'OK' if you can understand this message."
            response = await asyncio.wait_for(
                llm.ainvoke(test_prompt),
                timeout=30  # Increased timeout from 10 to 30 seconds
            )
            if response and hasattr(response, 'content') and 'OK' in str(response.content).upper():
                return True
            return False
        except Exception as e:
            logger.debug(f"LLM connectivity test failed: {e}")
            return False

    def _check_system_health_for_operation(self, operation_name: str) -> bool:
        """
        Check if system health allows the specified operation to proceed.
        Uses centralized health monitoring to make proactive decisions.

        Args:
            operation_name: Name of the operation being checked

        Returns:
            True if operation should proceed, False if it should be deferred/skipped
        """
        try:
            health_monitor = _get_api_health_monitor()
            if not health_monitor:
                logger.warning("Health monitor not available - allowing operation to proceed")
                return True

            # Get current health summary
            health_summary = health_monitor.get_health_summary()
            summary = health_summary.get('summary', {})

            # Define health thresholds for different operations
            health_thresholds = {
                'llm_reasoning': {'max_unhealthy': 2, 'max_degraded': 5},
                'data_fetching': {'max_unhealthy': 3, 'max_degraded': 6},
                'memory_operations': {'max_unhealthy': 1, 'max_degraded': 3},
                'tool_execution': {'max_unhealthy': 2, 'max_degraded': 4}
            }

            thresholds = health_thresholds.get(operation_name, {'max_unhealthy': 2, 'max_degraded': 5})

            unhealthy_count = summary.get('unhealthy', 0)
            degraded_count = summary.get('degraded', 0)

            # Check if operation should proceed
            if unhealthy_count > thresholds['max_unhealthy']:
                logger.warning(f"Operation '{operation_name}' blocked due to {unhealthy_count} unhealthy APIs (threshold: {thresholds['max_unhealthy']})")
                return False

            if unhealthy_count + degraded_count > thresholds['max_degraded']:
                logger.warning(f"Operation '{operation_name}' blocked due to {unhealthy_count + degraded_count} unhealthy/degraded APIs (threshold: {thresholds['max_degraded']})")
                return False

            # Check specific API health for operation-specific requirements
            if operation_name == 'llm_reasoning':
                # Require at least one LLM API to be healthy
                grok_status = health_monitor.get_api_status('grok_api')
                if grok_status and grok_status.get('status') in ['healthy', 'degraded']:
                    return True
                logger.warning("No healthy LLM APIs available for reasoning operation")
                return False

            elif operation_name == 'data_fetching':
                # Require at least yfinance to be available for data operations
                yfinance_status = health_monitor.get_api_status('yfinance')
                if yfinance_status and yfinance_status.get('status') == 'healthy':
                    return True
                logger.warning("YFinance API not healthy - data fetching may be unreliable")
                # Allow operation but log warning
                return True

            elif operation_name == 'memory_operations':
                # Check memory backend health
                try:
                    from src.utils.advanced_memory import get_memory_health_status
                    memory_health = get_memory_health_status()
                    redundancy_level = memory_health.get('redundancy_level', 0)

                    if redundancy_level == 0:
                        logger.error("No memory backends available - memory operations blocked")
                        return False
                    elif redundancy_level == 1:
                        logger.warning("Memory operations proceeding with no redundancy")
                        return True
                    else:
                        return True
                except Exception as e:
                    logger.warning(f"Could not check memory health for operation: {e}")
                    return True  # Allow operation if health check fails

            return True

        except Exception as e:
            logger.warning(f"Error checking system health for {operation_name}: {e}")
            # Allow operation to proceed if health check fails
            return True

    def get_system_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system health status for decision making.
        Integrates API health, circuit breaker status, and system metrics.

        Returns:
            Dict with comprehensive health status
        """
        try:
            health_status = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'overall_health': 'unknown',
                'can_operate': True,
                'api_health': {},
                'circuit_breakers': {},
                'recommendations': []
            }

            # Get API health monitor status
            health_monitor = _get_api_health_monitor()
            if health_monitor:
                api_summary = health_monitor.get_health_summary()
                health_status['api_health'] = api_summary

                summary = api_summary.get('summary', {})
                unhealthy_count = summary.get('unhealthy', 0)
                degraded_count = summary.get('degraded', 0)

                # Determine overall health
                if unhealthy_count > 3:
                    health_status['overall_health'] = 'critical'
                    health_status['can_operate'] = False
                    health_status['recommendations'].append("System health critical - defer non-essential operations")
                elif unhealthy_count > 1 or degraded_count > 4:
                    health_status['overall_health'] = 'degraded'
                    health_status['recommendations'].append("System health degraded - use caution with critical operations")
                else:
                    health_status['overall_health'] = 'healthy'
            else:
                health_status['recommendations'].append("API health monitor not available")

            # Get memory health status
            try:
                from src.utils.advanced_memory import get_memory_health_status
                memory_health = get_memory_health_status()
                health_status['memory_health'] = memory_health

                # Check memory redundancy
                redundancy_level = memory_health.get('redundancy_level', 0)
                if redundancy_level == 0:
                    health_status['overall_health'] = 'critical'
                    health_status['can_operate'] = False
                    health_status['recommendations'].append("No memory backends available - system cannot operate")
                elif redundancy_level == 1:
                    health_status['recommendations'].append("Only one memory backend available - no redundancy")
                elif redundancy_level < len(memory_health.get('backends', {})):
                    health_status['recommendations'].append(f"Memory redundancy degraded: {redundancy_level}/{len(memory_health.get('backends', {}))} backends healthy")

            except Exception as e:
                logger.warning(f"Could not get memory health status: {e}")
                health_status['memory_health'] = {'error': str(e)}
                health_status['recommendations'].append("Memory health status unavailable")

            # Get circuit breaker status

                # Check if critical APIs have open circuits
                critical_open = any(
                    status.get('state') == 'OPEN'
                    for api, status in circuit_status.items()
                    if api in ['yfinance', 'marketdataapp_api']
                )

                if critical_open:
                    health_status['can_operate'] = False
                    health_status['recommendations'].append("Critical API circuits open - trading not allowed")

            except Exception as e:
                logger.warning(f"Could not get circuit breaker status: {e}")
                health_status['recommendations'].append("Circuit breaker status unavailable")

            return health_status

        except Exception as e:
            logger.error(f"Error getting system health status: {e}")
            return {
                'timestamp': pd.Timestamp.now().isoformat(),
                'overall_health': 'unknown',
                'can_operate': True,  # Allow operation if health check fails
                'error': str(e),
                'recommendations': ["Health status check failed - proceeding with caution"]
            }

    def _execute_with_graceful_degradation(self, operation_func: callable, operation_name: str, 
                                          critical: bool = False, fallback_result: Any = None) -> Any:
        """
        Execute an operation with graceful degradation for non-critical components.
        
        Args:
            operation_func: Function to execute
            operation_name: Name of the operation for logging
            critical: Whether this operation is critical for system operation
            fallback_result: Result to return if operation fails and is non-critical
            
        Returns:
            Operation result or fallback result
        """
        try:
            # Check system health before attempting operation
            if not self._check_system_health_for_operation(operation_name):
                if critical:
                    logger.error(f"Critical operation '{operation_name}' blocked due to system health")
                    raise Exception(f"System health prevents critical operation: {operation_name}")
                else:
                    logger.warning(f"Non-critical operation '{operation_name}' deferred due to system health - using fallback")
                    return fallback_result
            
            # Execute the operation
            result = operation_func()
            return result
            
        except Exception as e:
            if critical:
                logger.error(f"Critical operation '{operation_name}' failed: {e}")
                raise  # Re-raise for critical operations
            else:
                logger.warning(f"Non-critical operation '{operation_name}' failed: {e} - using graceful degradation")
                return fallback_result

    def _get_component_health_status(self, component_name: str) -> Dict[str, Any]:
        """
        Get health status of a specific component for graceful degradation decisions.
        
        Args:
            component_name: Name of the component to check
            
        Returns:
            Dict with component health status
        """
        try:
            health_status = self.get_system_health_status()
            
            # Component-specific health checks
            component_health = {
                'memory': health_status.get('memory_health', {}),
                'llm': {'healthy': self.llm is not None},
                'api_monitor': {'healthy': _get_api_health_monitor() is not None},
                'shared_memory': {'healthy': self.shared_memory_coordinator is not None}
            }
            
            return component_health.get(component_name, {'healthy': True, 'status': 'unknown'})
            
        except Exception as e:
            logger.warning(f"Could not get health status for component '{component_name}': {e}")
            return {'healthy': True, 'status': 'unknown', 'error': str(e)}  # Default to healthy to avoid blocking

    def __init__(self, role: str, config_paths: Dict[str, str] = None, prompt_paths: Dict[str, str] = None, tools: List[Any] = None, a2a_protocol: Any = None):
        """
        Initializes the agent with role, configs, prompts, and tools.
        Args:
            role (str): Agent role (e.g., 'risk').
            config_paths (Dict): Paths to YAMLs (e.g., {'risk': 'config/risk-constraints.yaml'}—relative to project root).
            prompt_paths (Dict): Paths to prompts (e.g., {'base': 'base_prompt.txt', 'role': 'risk-agent-prompt.md'}—relative to root).
            tools (List[BaseTool]): List of Langchain tools for the agent.
            a2a_protocol (Any): A2A protocol instance for inter-agent communication.
        Reasoning: Loads fresh for each init (e.g., constraints for discipline); integrates Langchain tools for tool calling.
        """
        self.role = role
        self.a2a_protocol = a2a_protocol  # Store A2A protocol reference for monitored communication
        
        self.configs = {}
        if config_paths:
            for key, path in config_paths.items():
                full_path = project_root / path  # Resolve relative to root
                self.configs[key] = load_yaml(str(full_path))
        
        self.prompt = load_prompt_template(role, base_path=str(project_root / prompt_paths.get('base', '')) if prompt_paths.get('base') else '', 
                                           role_path=str(project_root / prompt_paths.get('role', '')) if prompt_paths.get('role') else '') if prompt_paths else "Default prompt."
        
        # Langchain integration with lazy loading
        self.tools = tools or []
        xai_api_key = os.getenv("GROK_API_KEY")

        # Initialize LLM with robust retry and circuit breaker logic
        self.llm = None  # Initialize as None, will be set up asynchronously

        # Note: LLM initialization is deferred to async_initialize_llm() method
        # This allows agents to be created in synchronous contexts

        self.agent = None

        # Memory systems with lazy loading and error handling
        try:
            self.memory_persistence = _get_memory_persistence()
        except Exception as e:
            logger.warning(f"Memory persistence initialization failed: {e}, using in-memory fallback")
            self.memory_persistence = None

        try:
            self.advanced_memory = _get_advanced_memory_manager()
        except Exception as e:
            logger.warning(f"Advanced memory initialization failed: {e}, using basic memory fallback")
            self.advanced_memory = None

        try:
            self.shared_memory_coordinator = _get_multi_agent_coordinator()
        except Exception as e:
            logger.warning(f"Shared memory coordinator initialization failed: {e}, operating without coordination")
            self.shared_memory_coordinator = None

        # Initialize memory structures safely
        self.memory = {}
        if self.memory_persistence:
            try:
                loaded_memory = self.memory_persistence.load_agent_memory(role)
                if loaded_memory and isinstance(loaded_memory, dict):
                    self.memory = loaded_memory
                logger.info("Memory persistence initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to load persistent memory: {e}, starting with empty memory")
        else:
            logger.info("Operating without memory persistence")

        # Register with shared memory coordinator if available
        if self.shared_memory_coordinator:
            try:
                self.shared_memory_coordinator.a2a_protocol.register_agent(self.role, {"capabilities": ["memory_sharing", "coordination"]})
                logger.info("Shared memory coordinator initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to register with shared memory coordinator: {e}")

        logger.info(f"Initialized {self.role} Agent with configs: {list(self.configs.keys())}, tools: {[t.name for t in self.tools]}")

    async def async_initialize_llm(self) -> bool:
        """
        Asynchronously initialize the LLM for this agent.
        This method should be called after agent creation to set up the LLM.

        Returns:
            bool: True if LLM was successfully initialized, False otherwise
        """
        try:
            logger.info(f"Starting async LLM initialization for {self.role} agent")
            self.llm = await self._initialize_llm_with_resilience_async()
            if self.llm:
                logger.info(f"Successfully initialized LLM for {self.role} agent")
                return True
            else:
                logger.warning(f"Failed to initialize LLM for {self.role} agent")
                return False
        except Exception as e:
            logger.error(f"Error during async LLM initialization for {self.role}: {e}")
            return False

    async def _initialize_llm_with_resilience_async(self) -> Optional[Any]:
        """
        Initialize LLM with comprehensive retry logic, circuit breakers, and fallback strategies.
        Uses centralized API health monitoring for proactive failure prevention.

        Returns:
            Initialized LLM instance or None if all methods fail
        """
        import asyncio
        import time

        # Retry configuration
        max_retries = 3
        base_delay = 1.0  # seconds
        max_delay = 30.0  # seconds

        # Get centralized health monitor
        health_monitor = _get_api_health_monitor()
        if health_monitor:
            # Check overall system health before attempting initialization
            health_summary = health_monitor.get_health_summary()
            unhealthy_count = health_summary['summary'].get('unhealthy', 0)

            if unhealthy_count > 3:  # Too many unhealthy APIs
                logger.warning(f"System health critical ({unhealthy_count} unhealthy APIs) - deferring LLM initialization")
                return None

        xai_api_key = os.getenv("GROK_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY")

        # Check if any API keys are available
        has_any_keys = bool(xai_api_key or openai_api_key or anthropic_api_key or google_api_key)

        if not has_any_keys:
            logger.warning("No LLM API keys found (GROK_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY). System will operate in limited mode.")
            logger.warning("For full functionality, set environment variables: GROK_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, and/or GOOGLE_API_KEY")
            # In development/testing mode, allow operation but log warnings
            dev_mode = os.getenv("DEV_MODE", "false").lower() == "true"
            if dev_mode:
                logger.info("DEV_MODE enabled - allowing agent initialization without LLM for testing")
                return None
            else:
                logger.error("No API keys available and DEV_MODE not enabled - cannot initialize LLM")
                return None

        # Primary initialization strategies with priority and health checks and health checks
        strategies = []

        if xai_api_key:
            # Check XAI API health before attempting
            xai_healthy = True
            if health_monitor:
                xai_status = health_monitor.get_api_status('grok_api')
                # Only skip if explicitly unhealthy and we've tried before
                # If no status exists, assume healthy (first attempt)
                if xai_status and xai_status.get('status') == 'unhealthy' and xai_status.get('attempts', 0) > 0:
                    xai_healthy = False
                    logger.warning("XAI API marked as unhealthy by health monitor - skipping XAI strategies")
                else:
                    logger.info("XAI API healthy or not yet tested - proceeding with XAI strategies")

            if xai_healthy:
                strategies.extend([
                    ("ChatXAI_grok4", lambda: self._try_initialize_xai(xai_api_key, "grok-4-fast-reasoning")),
                ])

        if openai_api_key:
            # Check OpenAI API health (if monitored)
            openai_healthy = True
            # Note: OpenAI API health monitoring would need to be added to api_health_monitor.py

            if openai_healthy:
                strategies.extend([
                    ("OpenAI_gpt4", lambda: self._try_initialize_openai(openai_api_key, "gpt-4")),
                    ("OpenAI_gpt35", lambda: self._try_initialize_openai(openai_api_key, "gpt-3.5-turbo")),
                ])

        if anthropic_api_key:
            # Check Anthropic API health (if monitored)
            anthropic_healthy = True
            # Note: Anthropic API health monitoring would need to be added to api_health_monitor.py

            if anthropic_healthy:
                strategies.extend([
                    ("Anthropic_claude3", lambda: self._try_initialize_anthropic(anthropic_api_key, "claude-3-sonnet-20240229")),
                    ("Anthropic_claude2", lambda: self._try_initialize_anthropic(anthropic_api_key, "claude-2.1")),
                ])

        if google_api_key:
            # Check Google API health (if monitored)
            google_healthy = True
            # Note: Google API health monitoring would need to be added to api_health_monitor.py

            if google_healthy:
                strategies.extend([
                    ("Google_gemini_pro", lambda: self._try_initialize_google(google_api_key, "gemini-pro")),
                    ("Google_gemini_1", lambda: self._try_initialize_google(google_api_key, "gemini-1.5-flash")),
                ])

        # Try each strategy with exponential backoff
        for strategy_name, init_func in strategies:
            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempting LLM initialization with {strategy_name} (attempt {attempt + 1}/{max_retries})")
                    llm = init_func()

                    if llm:
                        # Test the LLM with a simple query
                        test_success = False
                        try:
                            test_response = await self._test_llm_connection(llm)
                            test_success = test_response
                        except asyncio.CancelledError:
                            logger.warning(f"LLM test cancelled for {strategy_name}, proceeding without test")
                            test_success = True  # Assume it's working if we got this far
                        except Exception as e:
                            logger.warning(f"LLM test failed for {strategy_name}: {e}, proceeding anyway")
                            test_success = True  # Allow initialization to continue

                        if test_success:
                            logger.info(f"Successfully initialized and tested LLM with {strategy_name}")
                            # Update health monitor on success
                            if health_monitor and 'grok' in strategy_name.lower():
                                # Mark XAI API as healthy after successful connection
                                health_monitor._update_metrics('grok_api', True, 1.0)  # Assume 1 second response
                            return llm
                        else:
                            logger.warning(f"LLM {strategy_name} initialized but failed connectivity test")
                            # Update health monitor on failure
                            if health_monitor and 'grok' in strategy_name.lower():
                                health_monitor._update_metrics('grok_api', False, 0.0, "Connectivity test failed")

                except Exception as e:
                    logger.warning(f"LLM initialization attempt {attempt + 1} with {strategy_name} failed: {e}")
                    # Update health monitor on failure
                    if health_monitor and 'grok' in strategy_name.lower():
                        health_monitor._update_metrics('grok_api', False, 0.0, str(e))

                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.info(f"Retrying {strategy_name} in {delay} seconds...")
                        await asyncio.sleep(delay)

            # Strategy failed completely
            logger.error(f"All attempts failed for {strategy_name}")

        logger.error(f"All LLM initialization strategies failed after {len(strategies)} strategies and {max_retries} retries each")
        return None

    async def reason_with_llm(self, context: str, question: str, options: Dict[str, Any] = None) -> str:
        """
        Use LLM for reasoning on complex decisions while leveraging hardcoded foundation logic.
        This implements the hybrid approach: foundation code + LLM reasoning.

        Args:
            context: Background information and foundation analysis
            question: The specific decision or analysis needed
            options: Additional context options

        Returns:
            LLM reasoning response
        """
        # Check system health before proceeding with LLM operations
        if not self._check_system_health_for_operation('llm_reasoning'):
            logger.error("System health check failed - deferring LLM reasoning operation")
            raise Exception("System health does not allow LLM reasoning operations at this time")

        # Handle DEV_MODE without LLM
        dev_mode = os.getenv("DEV_MODE", "false").lower() == "true"
        if not self.llm:
            if dev_mode:
                logger.warning("DEV_MODE: No LLM available - returning default reasoning response")
                return f"DEV_MODE: Default reasoning response for {self.role} agent. Context: {context[:100]}... Question: {question}"
            else:
                logger.error(f"CRITICAL FAILURE: No LLM available for {self.role} agent - cannot perform AI reasoning")
                raise Exception(f"LLM required for {self.role} agent reasoning - no foundation-only fallback allowed")

        try:
            # Sanitize inputs to prevent prompt injection
            sanitized_context = self._sanitize_llm_input(context)
            sanitized_question = self._sanitize_llm_input(question)
            sanitized_options = self._sanitize_llm_options(options) if options else None

            # Build comprehensive prompt with foundation context
            full_prompt = f"""
{self.prompt}

FOUNDATION ANALYSIS CONTEXT:
{sanitized_context}

DECISION REQUIRED:
{sanitized_question}

ADDITIONAL CONTEXT:
{sanitized_options or 'No additional context provided'}

Please provide your reasoning and recommendation based on the foundation analysis above.
Consider market conditions, risk factors, and alignment with our goals (10-20% monthly ROI, <5% drawdown).
"""

            # Use LLM for reasoning
            response = await self.llm.ainvoke(full_prompt)

            logger.info(f"LLM reasoning completed for {self.role} agent")
            return response.content if hasattr(response, 'content') else str(response)

        except Exception as e:
            logger.error(f"LLM reasoning failed for {self.role}: {e}")
            return f"LLM_ERROR: {str(e)}"

    def _sanitize_llm_input(self, input_text: str) -> str:
        """
        Sanitize input text for LLM prompts to prevent injection attacks.
        Removes or escapes potentially harmful content.
        """
        if not isinstance(input_text, str):
            return str(input_text)

        # Remove or escape common injection patterns
        sanitized = input_text

        # Remove system prompt override attempts
        sanitized = sanitized.replace("SYSTEM:", "").replace("system:", "")
        sanitized = sanitized.replace("ASSISTANT:", "").replace("assistant:", "")
        sanitized = sanitized.replace("USER:", "").replace("user:", "")

        # Remove prompt injection markers
        injection_markers = [
            "###", "---", "```", "IGNORE PREVIOUS",
            "FORGET INSTRUCTIONS", "NEW INSTRUCTIONS",
            "SYSTEM PROMPT", "You are now", "FROM NOW ON",
            "HENCEFORTH", "STARTING NOW", "BEGINNING NOW"
        ]

        for marker in injection_markers:
            sanitized = sanitized.replace(marker, "[FILTERED]")

        # Remove attempts to override model behavior
        behavior_overrides = [
            "ignore all previous", "disregard above", "forget everything",
            "new persona", "act as if", "pretend to be", "role-play as"
        ]

        for override in behavior_overrides:
            sanitized = sanitized.replace(override, "[FILTERED]")

        # Limit input length to prevent excessive token usage and abuse
        max_length = 8000  # Reasonable limit for context
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "...[TRUNCATED FOR SECURITY]"

        # Remove excessive whitespace and control characters
        import re
        sanitized = re.sub(r'\n\s*\n\s*\n+', '\n\n', sanitized)  # Multiple newlines to double
        sanitized = re.sub(r'\s+', ' ', sanitized.strip())  # Multiple spaces to single
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)  # Remove control characters

        return sanitized

    def _sanitize_llm_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize options dictionary for LLM prompts.
        """
        if not isinstance(options, dict):
            return {}

        sanitized_options = {}
        for key, value in options.items():
            if isinstance(key, str):
                sanitized_key = self._sanitize_llm_input(key)
            else:
                sanitized_key = str(key)

            if isinstance(value, str):
                sanitized_value = self._sanitize_llm_input(value)
            elif isinstance(value, (int, float, bool)):
                sanitized_value = value
            elif isinstance(value, (list, dict)):
                # Convert complex types to string and sanitize
                sanitized_value = self._sanitize_llm_input(str(value))
            else:
                sanitized_value = str(value)

            sanitized_options[sanitized_key] = sanitized_value

        return sanitized_options

    def _should_use_tool(self, tool: BaseTool, query_lower: str) -> bool:
        """Determine if a tool should be used based on query keywords."""
        tool_name = tool.name.lower()
        
        # Yfinance tool
        if "yfinance" in tool_name and any(word in query_lower for word in ["fetch", "data", "stock", "symbol", "price"]):
            return True
        # Sentiment tool
        elif "sentiment" in tool_name and any(word in query_lower for word in ["sentiment", "analyze", "market", "mood"]):
            return True
        # News tool
        elif "news" in tool_name and any(word in query_lower for word in ["news", "headlines", "articles"]):
            return True
        # Economic tool
        elif "economic" in tool_name and any(word in query_lower for word in ["economic", "indicators", "fed", "gdp", "inflation"]):
            return True
        
        return False

    def _execute_tool_with_params(self, tool: BaseTool, query: str) -> str:
        """Execute a tool with extracted parameters."""
        tool_name = tool.name.lower()
        
        try:
            if "yfinance" in tool_name:
                # Extract symbol and period
                symbol = self._extract_symbol(query)
                period = self._extract_period(query)
                return tool.run({"symbol": symbol, "period": period})
            elif "sentiment" in tool_name:
                # Extract text to analyze
                text = self._extract_text(query)
                return tool.run({"text": text})
            elif "news" in tool_name:
                # Extract symbol
                symbol = self._extract_symbol(query)
                return tool.run({"symbol": symbol})
            elif "economic" in tool_name:
                return tool.run({})  # No params needed
            else:
                return tool.run({})  # Default no params
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {e}")
            return f"Tool execution failed: {e}"

    def _extract_symbol(self, query: str) -> str:
        """Extract stock symbol from query."""
        # Look for common symbols
        symbols = ['SPY', 'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        query_upper = query.upper()
        for symbol in symbols:
            if symbol in query_upper:
                return symbol
        return "SPY"  # Default

    def _extract_period(self, query: str) -> str:
        """Extract period from query."""
        query_lower = query.lower()
        if "5d" in query_lower or "5 days" in query_lower:
            return "5d"
        elif "1mo" in query_lower or "month" in query_lower:
            return "1mo"
        elif "3mo" in query_lower or "quarter" in query_lower:
            return "3mo"
        elif "6mo" in query_lower or "6 months" in query_lower:
            return "6mo"
        elif "1y" in query_lower or "year" in query_lower:
            return "1y"
        elif "2y" in query_lower or "2 years" in query_lower:
            return "2y"
        elif "5y" in query_lower or "5 years" in query_lower:
            return "5y"
        elif "10y" in query_lower or "10 years" in query_lower:
            return "10y"
        elif "max" in query_lower or "maximum" in query_lower:
            return "max"
        return "2y"  # Default to 2 years for robust analysis

    def _extract_text(self, query: str) -> str:
        """Extract text for analysis from query."""
        # Look for text after keywords
        keywords = ["analyze", "sentiment", "for"]
        query_lower = query.lower()
        
        for keyword in keywords:
            idx = query_lower.find(keyword)
            if idx != -1:
                text = query[idx + len(keyword):].strip()
                if text:
                    return text
        
        return query  # Return whole query if no specific text found

    @abc.abstractmethod
    async def process_input(self, input_data: Any) -> Dict[str, Any]:
        """
        Abstract method for processing input (e.g., proposals for Risk).
        Returns: Dict with output (e.g., {'approved': True, 'adjustments': {...}}).
        Reasoning: Async for parallel sims/pings; subclasses implement ReAct-like logic with reflections.
        """
        pass

    def save_memory(self, create_backup: bool = True) -> bool:
        """
        Save current memory state to persistent storage.

        Args:
            create_backup: Whether to create a backup of previous memory

        Returns:
            bool: Success status
        """
        # Check system health before memory operations
        if not self._check_system_health_for_operation('memory_operations'):
            logger.warning("System health check failed - skipping memory save operation")
            return False

        try:
            if self.memory_persistence:
                success = self.memory_persistence.save_agent_memory(
                    self.role, self.memory, create_backup=create_backup
                )
                if success:
                    logger.debug(f"Memory saved for {self.role} agent")
                return success
            else:
                logger.debug(f"Memory persistence not available, skipping save for {self.role} agent")
                return False
        except Exception as e:
            logger.error(f"Failed to save memory for {self.role}: {e}")
            return False
    
    def load_memory(self) -> bool:
        """
        Load memory from persistent storage.
        
        Returns:
            bool: Success status
        """
        try:
            loaded_memory = self.memory_persistence.load_agent_memory(self.role)
            if loaded_memory is not None:
                # Ensure loaded memory is a dict
                if isinstance(loaded_memory, dict):
                    self.memory = loaded_memory
                else:
                    # If loaded memory is not a dict, initialize as empty dict
                    # and log a warning
                    logger.warning(f"Loaded memory for {self.role} is not a dict (type: {type(loaded_memory)}, value: {loaded_memory}), initializing empty memory")
                    self.memory = {}
                logger.debug(f"Memory loaded for {self.role} agent")
                return True
            else:
                logger.debug(f"No saved memory found for {self.role} agent")
                return False
        except Exception as e:
            logger.error(f"Failed to load memory for {self.role}: {e}")
            # Initialize empty memory on error
            self.memory = {}
            return False
    
    def update_memory(self, key: str, value: Any, save_immediately: bool = True) -> None:
        """
        Update a specific memory key and optionally save.
        
        Args:
            key: Memory key to update
            value: Value to store
            save_immediately: Whether to save to disk immediately
        """
        self.memory[key] = value
        if save_immediately:
            self.save_memory()
    
    def get_memory(self, key: str, default: Any = None) -> Any:
        """
        Get a value from memory.
        
        Args:
            key: Memory key to retrieve
            default: Default value if key not found
            
        Returns:
            Any: Memory value or default
        """
        return self.memory.get(key, default)
    
    def append_to_memory_list(self, key: str, item: Any, max_items: int = None, 
                            save_immediately: bool = True) -> None:
        """
        Append an item to a memory list, with optional size limiting.
        
        Args:
            key: Memory key (should be a list)
            item: Item to append
            max_items: Maximum items to keep (None for unlimited)
            save_immediately: Whether to save immediately
        """
        if key not in self.memory:
            self.memory[key] = []
        
        if not isinstance(self.memory[key], list):
            logger.warning(f"Memory key '{key}' is not a list, converting to list")
            self.memory[key] = [self.memory[key]]
        
        self.memory[key].append(item)
        
        # Trim if needed
        if max_items and len(self.memory[key]) > max_items:
            self.memory[key] = self.memory[key][-max_items:]
        
        if save_immediately:
            self.save_memory()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics for this agent.
        
        Returns:
            Dict: Memory statistics
        """
        try:
            stats = {
                "agent_role": self.role,
                "memory_keys": list(self.memory.keys()),
                "total_keys": len(self.memory),
                "memory_size_estimate": self._estimate_memory_size(),
                "last_saved": None  # Would need to track this in future
            }
            return stats
        except Exception as e:
            logger.error(f"Failed to get memory stats for {self.role}: {e}")
            return {"error": str(e)}
    
    def _estimate_memory_size(self) -> str:
        """
        Estimate the memory size for this agent.
        
        Returns:
            str: Estimated size
        """
        try:
            import sys
            size_bytes = sys.getsizeof(self.memory)
            for key, value in self.memory.items():
                size_bytes += sys.getsizeof(key) + sys.getsizeof(value)
            
            # Convert to human readable
            if size_bytes < 1024:
                return f"{size_bytes} bytes"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            else:
                return f"{size_bytes / (1024 * 1024):.1f} MB"
        except Exception:
            return "unknown"
    
    async def store_advanced_memory(self, key: str, data: Any, memory_type: str = "long_term",
                                   metadata: Dict[str, Any] = None) -> bool:
        """
        Store memory using advanced memory system.
        
        Args:
            key: Memory key
            data: Data to store
            memory_type: Type of memory (short_term, long_term, episodic, semantic, procedural)
            metadata: Additional metadata
            
        Returns:
            bool: Success status
        """
        try:
            # Add agent context to metadata
            if metadata is None:
                metadata = {}
            metadata.update({
                "agent_role": self.role,
                "agent_type": "ai_portfolio_manager"
            })
            
            full_key = f"agent:{self.role}:{key}"
            return await self.advanced_memory.store_memory(full_key, data, memory_type, metadata, user=self.role)
        except Exception as e:
            logger.error(f"Advanced memory store failed for {self.role}:{key}: {e}")
            return False
    
    async def retrieve_advanced_memory(self, key: str, memory_type: str = None) -> Optional[Any]:
        """
        Retrieve memory from advanced memory system.
        
        Args:
            key: Memory key
            memory_type: Memory type hint
            
        Returns:
            Retrieved data or None
        """
        try:
            full_key = f"agent:{self.role}:{key}"
            return await self.advanced_memory.retrieve_memory(full_key, memory_type, user=self.role)
        except Exception as e:
            logger.error(f"Advanced memory retrieve failed for {self.role}:{key}: {e}")
            return None
    
    async def search_advanced_memory(self, query: str, memory_type: str = None, 
                                   limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search memories in advanced memory system.
        
        Args:
            query: Search query
            memory_type: Filter by memory type
            limit: Maximum results
            
        Returns:
            List of search results
        """
        try:
            # Add agent context to query
            agent_query = f"{self.role} {query}"
            results = await self.advanced_memory.search_memories(agent_query, memory_type, limit)
            
            # Filter results to this agent
            agent_results = []
            for result in results:
                if f"agent:{self.role}:" in result.get("key", ""):
                    agent_results.append(result)
            
            return agent_results
        except Exception as e:
            logger.error(f"Advanced memory search failed for {self.role}: {e}")
            return []
    
    async def share_memory_with_agent(self, target_agent: str, namespace: str, 
                                    key: str, data: Any) -> bool:
        """
        Share memory with another agent through shared namespace.
        
        Args:
            target_agent: Role of target agent
            namespace: Shared namespace
            key: Memory key
            data: Data to share
            
        Returns:
            bool: Success status
        """
        try:
            return await self.shared_memory_coordinator.share_memory(
                self.role, target_agent, namespace, key, data
            )
        except Exception as e:
            logger.error(f"Failed to share memory with {target_agent}: {e}")
            return False
    
    async def request_memory_from_agent(self, target_agent: str, namespace: str, 
                                      key: str) -> Optional[Any]:
        """
        Request memory from another agent.
        
        Args:
            target_agent: Role of target agent
            namespace: Shared namespace
            key: Memory key
            
        Returns:
            Requested data or None
        """
        try:
            return await self.shared_memory_coordinator.request_memory(
                self.role, target_agent, namespace, key
            )
        except Exception as e:
            logger.error(f"Failed to request memory from {target_agent}: {e}")
            return None
    
    async def store_shared_memory(self, namespace: str, key: str, data: Any) -> bool:
        """
        Store data in a shared namespace accessible to multiple agents.
        
        Args:
            namespace: Shared namespace
            key: Memory key
            data: Data to store
            
        Returns:
            bool: Success status
        """
        try:
            shared_ns = self.shared_memory_coordinator.a2a_protocol.get_namespace(namespace)
            if not shared_ns:
                # Create namespace if it doesn't exist
                shared_ns = self.shared_memory_coordinator.a2a_protocol.create_namespace(namespace)
            
            return await shared_ns.store_shared_memory(key, data, self.role)
        except Exception as e:
            logger.error(f"Failed to store shared memory in {namespace}:{key}: {e}")
            return False
    
    async def retrieve_shared_memory(self, namespace: str, key: str) -> Optional[Any]:
        """
        Retrieve data from a shared namespace.
        
        Args:
            namespace: Shared namespace
            key: Memory key
            
        Returns:
            Retrieved data or None
        """
        try:
            shared_ns = self.shared_memory_coordinator.a2a_protocol.get_namespace(namespace)
            if not shared_ns:
                logger.warning(f"Shared namespace {namespace} does not exist")
                return None
            
            return await shared_ns.retrieve_shared_memory(key, self.role)
        except Exception as e:
            logger.error(f"Failed to retrieve shared memory from {namespace}:{key}: {e}")
            return None
    
    async def search_shared_memory(self, namespace: str, query: str, 
                                 limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search memories in a shared namespace.
        
        Args:
            namespace: Shared namespace
            query: Search query
            limit: Maximum results
            
        Returns:
            List of search results
        """
        try:
            shared_ns = self.shared_memory_coordinator.a2a_protocol.get_namespace(namespace)
            if not shared_ns:
                logger.warning(f"Shared namespace {namespace} does not exist")
                return []
            
            return await shared_ns.search_shared_memories(query, self.role, limit)
        except Exception as e:
            logger.error(f"Failed to search shared memory in {namespace}: {e}")
            return []
    
    async def broadcast_coordination_signal(self, signal_type: str, 
                                          signal_data: Dict[str, Any] = None) -> bool:
        """
        Broadcast a coordination signal to all registered agents.
        
        Args:
            signal_type: Type of coordination signal
            signal_data: Additional signal data
            
        Returns:
            bool: Success status
        """
        try:
            await self.shared_memory_coordinator.broadcast_coordination_signal(
                self.role, signal_type, signal_data
            )
            return True
        except Exception as e:
            logger.error(f"Failed to broadcast coordination signal {signal_type}: {e}")
            return False
    
    def get_shared_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about shared memory usage.
        
        Returns:
            Dict: Shared memory statistics
        """
        try:
            # Try to get coordinator stats if method exists
            if hasattr(self.shared_memory_coordinator, 'get_coordinator_stats'):
                coordinator_stats = self.shared_memory_coordinator.get_coordinator_stats()
                return {
                    "agent_role": self.role,
                    "coordinator_stats": coordinator_stats,
                    "shared_namespaces": list(coordinator_stats.get("namespaces", {}).keys()),
                    "registered_agents": coordinator_stats.get("registered_agents", [])
                }
            else:
                # Fallback: provide basic stats without coordinator details
                return {
                    "agent_role": self.role,
                    "coordinator_available": self.shared_memory_coordinator is not None,
                    "shared_namespaces": "unknown",
                    "registered_agents": "unknown"
                }
        except Exception as e:
            logger.error(f"Failed to get shared memory stats for {self.role}: {e}")
            return {"error": str(e)}
    
    def reflect(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Base reflection method for agents to override.
        Provides basic reflection logic that subclasses can extend.
        
        Args:
            metrics (Dict): Performance metrics for reflection
            
        Returns:
            Dict: Adjustments based on reflection
        """
        adjustments = {}
        
        # Basic reflection logic - can be overridden by subclasses
        if 'sd_variance' in metrics and metrics['sd_variance'] > self.configs.get('risk', {}).get('constraints', {}).get('variance_sd_threshold', 1.0):
            adjustments['pop_floor'] = 0.65  # Example tighten
            logger.info(f"Base reflection for {self.role}: Adjusted based on metrics {metrics}")
        
        return adjustments

    async def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the agent.
        Returns comprehensive status information for monitoring and Discord integration.
        
        Returns:
            Dict: Agent status including health, memory, and recent activity
        """
        try:
            # Get system health status
            health_status = self.get_system_health_status()
            
            # Get memory statistics
            memory_stats = self.get_memory_stats()
            
            # Get shared memory stats if available
            shared_memory_stats = {}
            if self.shared_memory_coordinator:
                shared_memory_stats = self.get_shared_memory_stats()
            
            # Get recent activity from memory
            recent_activity = []
            if hasattr(self, 'memory') and isinstance(self.memory, dict):
                # Look for recent entries in memory (this is agent-specific)
                for key, value in self.memory.items():
                    if isinstance(value, list) and len(value) > 0:
                        # Get last few entries
                        recent_entries = value[-3:] if len(value) > 3 else value
                        recent_activity.extend([f"{key}: {entry}" for entry in recent_entries])
            
            status = {
                'agent_role': self.role,
                'timestamp': pd.Timestamp.now().isoformat(),
                'health_status': health_status,
                'memory_stats': memory_stats,
                'shared_memory_stats': shared_memory_stats,
                'recent_activity': recent_activity[-5:],  # Last 5 activities
                'llm_available': self.llm is not None,
                'tools_count': len(self.tools),
                'config_files': list(self.configs.keys()),
                'active_sessions': self.list_my_sessions() if hasattr(self, 'list_my_sessions') else []
            }
            
            logger.debug(f"Status retrieved for {self.role} agent")
            return status
            
        except Exception as e:
            logger.error(f"Error getting status for {self.role}: {e}")
            return {
                'agent_role': self.role,
                'timestamp': pd.Timestamp.now().isoformat(),
                'error': str(e),
                'health_status': 'error'
            }

    async def analyze(self, data: Any) -> Dict[str, Any]:
        """
        Perform analysis on provided data.
        This is a base implementation that can be overridden by subclasses for specific analysis types.
        
        Args:
            data: Data to analyze (can be a dict or string query)
            
        Returns:
            Dict: Analysis results
        """
        try:
            # Handle string queries (from Discord commands)
            if isinstance(data, str):
                data = {'query': data, 'analysis_type': 'general'}
            
            analysis_type = data.get('analysis_type', 'general')
            
            # Use LLM for analysis if available
            if self.llm:
                context = f"""
AGENT ANALYSIS REQUEST:
Role: {self.role}
Analysis Type: {analysis_type}
Data Provided: {data}

Please provide analysis based on your role and expertise.
"""
                
                question = data.get('question', data.get('query', f"What insights can you provide about this {analysis_type} data?"))
                
                llm_response = await self.reason_with_llm(context, question)
                
                analysis_result = {
                    'agent_role': self.role,
                    'analysis_type': analysis_type,
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'llm_analysis': llm_response,
                    'data_summary': self._summarize_analysis_data(data),
                    'confidence_level': 'medium'  # Base confidence
                }
            else:
                # Fallback analysis without LLM
                analysis_result = {
                    'agent_role': self.role,
                    'analysis_type': analysis_type,
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'fallback_analysis': f"Analysis of {analysis_type} data without LLM assistance",
                    'data_summary': self._summarize_analysis_data(data),
                    'confidence_level': 'low'
                }
            
            # Store analysis in memory
            await self.store_advanced_memory('analysis_history', {
                'timestamp': analysis_result['timestamp'],
                'type': analysis_type,
                'result': analysis_result
            })
            
            logger.info(f"Analysis completed by {self.role} agent for {analysis_type}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in analysis for {self.role}: {e}")
            return {
                'agent_role': self.role,
                'analysis_type': data.get('analysis_type', 'unknown') if isinstance(data, dict) else 'unknown',
                'timestamp': pd.Timestamp.now().isoformat(),
                'error': str(e),
                'confidence_level': 'none'
            }

    def get_recent_memories(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent memory entries for Discord display.
        
        Args:
            limit: Maximum number of memories to return
            
        Returns:
            List of recent memory entries
        """
        try:
            recent_memories = []
            
            # Get memories from agent memory
            if hasattr(self, 'memory') and isinstance(self.memory, dict):
                for key, value in self.memory.items():
                    if isinstance(value, list):
                        # Get recent entries from this memory list
                        recent_entries = value[-limit:] if len(value) > limit else value
                        for entry in recent_entries:
                            memory_item = {
                                'key': key,
                                'content': str(entry),
                                'timestamp': pd.Timestamp.now().isoformat()  # Default timestamp
                            }
                            recent_memories.append(memory_item)
            
            # Sort by timestamp (most recent first) and limit
            recent_memories.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            recent_memories = recent_memories[:limit]
            
            # If no memories found, return a default message
            if not recent_memories:
                recent_memories = [{
                    'key': 'system',
                    'content': f'No recent memories available for {self.role} agent',
                    'timestamp': pd.Timestamp.now().isoformat()
                }]
            
            return recent_memories
            
        except Exception as e:
            logger.error(f"Error getting recent memories for {self.role}: {e}")
            return [{
                'key': 'error',
                'content': f'Error retrieving memories: {str(e)}',
                'timestamp': pd.Timestamp.now().isoformat()
            }]

    def _summarize_analysis_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of the analysis data for reporting.
        
        Args:
            data: Raw analysis data
            
        Returns:
            Dict: Summarized data
        """
        try:
            summary = {
                'data_keys': list(data.keys()),
                'data_types': {k: type(v).__name__ for k, v in data.items()},
                'has_numeric_data': any(isinstance(v, (int, float)) for v in data.values()),
                'has_text_data': any(isinstance(v, str) for v in data.values()),
                'has_list_data': any(isinstance(v, list) for v in data.values()),
                'data_size_estimate': self._estimate_data_size(data)
            }
            
            return summary
            
        except Exception as e:
            return {'error': f'Summary failed: {str(e)}'}

    def _estimate_data_size(self, data: Dict[str, Any]) -> str:
        """
        Estimate the size of the data for reporting.
        
        Args:
            data: Data to estimate size for
            
        Returns:
            str: Size estimate
        """
        try:
            import sys
            size_bytes = sys.getsizeof(data)
            
            # Add sizes of nested structures
            for key, value in data.items():
                size_bytes += sys.getsizeof(key)
                if isinstance(value, (list, dict)):
                    size_bytes += sys.getsizeof(value)
            
            # Convert to human readable
            if size_bytes < 1024:
                return f"{size_bytes} bytes"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            else:
                return f"{size_bytes / (1024 * 1024):.1f} MB"
                
        except Exception:
            return "unknown"

    async def create_collaborative_session(self, topic: str, max_participants: int = 10,
                                         session_timeout: int = 3600) -> Optional[str]:
        """
        Create a new collaborative session.

        Args:
            topic: Session topic/purpose
            max_participants: Maximum number of participants
            session_timeout: Session timeout in seconds

        Returns:
            Session ID or None if failed
        """
        try:
            session_id = await self.shared_memory_coordinator.create_collaborative_session(
                self.role, topic, max_participants, session_timeout
            )
            if session_id:
                logger.info(f"Agent {self.role} created collaborative session: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Failed to create collaborative session for {self.role}: {e}")
            return None

    async def join_collaborative_session(self, session_id: str,
                                       agent_context: Dict[str, Any] = None) -> bool:
        """
        Join an existing collaborative session.

        Args:
            session_id: Session ID to join
            agent_context: Agent's context/knowledge for the session

        Returns:
            bool: Success status
        """
        try:
            success = await self.shared_memory_coordinator.join_collaborative_session(
                session_id, self.role, agent_context
            )
            if success:
                logger.info(f"Agent {self.role} joined collaborative session: {session_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to join collaborative session {session_id} for {self.role}: {e}")
            return False

    async def leave_collaborative_session(self, session_id: str) -> bool:
        """
        Leave a collaborative session.

        Args:
            session_id: Session ID to leave

        Returns:
            bool: Success status
        """
        try:
            success = await self.shared_memory_coordinator.leave_collaborative_session(
                session_id, self.role
            )
            if success:
                logger.info(f"Agent {self.role} left collaborative session: {session_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to leave collaborative session {session_id} for {self.role}: {e}")
            return False

    async def contribute_session_insight(self, session_id: str, insight: Dict[str, Any]) -> bool:
        """
        Contribute an insight to a collaborative session.

        Args:
            session_id: Session ID
            insight: Insight data (type, content, confidence, evidence, etc.)

        Returns:
            bool: Success status
        """
        try:
            success = await self.shared_memory_coordinator.contribute_to_session(
                session_id, self.role, insight
            )
            if success:
                logger.info(f"Agent {self.role} contributed insight to session: {session_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to contribute insight to session {session_id} for {self.role}: {e}")
            return False

    async def validate_session_insight(self, session_id: str, insight_index: int,
                                     validation: Dict[str, Any]) -> bool:
        """
        Validate or challenge an insight in a collaborative session.

        Args:
            session_id: Session ID
            insight_index: Index of insight to validate
            validation: Validation data (agreement, disagreement, reasoning, evidence)

        Returns:
            bool: Success status
        """
        try:
            success = await self.shared_memory_coordinator.validate_session_insight(
                session_id, self.role, insight_index, validation
            )
            if success:
                logger.info(f"Agent {self.role} validated insight {insight_index} in session: {session_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to validate insight in session {session_id} for {self.role}: {e}")
            return False

    async def record_session_decision(self, session_id: str, decision: Dict[str, Any]) -> bool:
        """
        Record a collaborative decision in a session.

        Args:
            session_id: Session ID
            decision: Decision data (conclusion, rationale, confidence, trade_details)

        Returns:
            bool: Success status
        """
        try:
            success = await self.shared_memory_coordinator.record_session_decision(
                session_id, self.role, decision
            )
            if success:
                logger.info(f"Agent {self.role} recorded decision in session: {session_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to record decision in session {session_id} for {self.role}: {e}")
            return False

    async def update_session_context(self, session_id: str, context_key: str,
                                   context_value: Any) -> bool:
        """
        Update shared context in a collaborative session.

        Args:
            session_id: Session ID
            context_key: Context key (market_data, risk_metrics, strategy_params, etc.)
            context_value: Context value

        Returns:
            bool: Success status
        """
        try:
            success = await self.shared_memory_coordinator.update_session_context(
                session_id, self.role, context_key, context_value
            )
            if success:
                logger.info(f"Agent {self.role} updated context '{context_key}' in session: {session_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to update session context in {session_id} for {self.role}: {e}")
            return False

    async def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get summary of a collaborative session.

        Args:
            session_id: Session ID

        Returns:
            Session summary or None
        """
        try:
            return await self.shared_memory_coordinator.get_session_summary(session_id)
        except Exception as e:
            logger.error(f"Failed to get session summary for {session_id}: {e}")
            return None

    async def get_session_insights(self, session_id: str, agent_filter: str = None) -> List[Dict[str, Any]]:
        """
        Get insights from a collaborative session.

        Args:
            session_id: Session ID
            agent_filter: Filter insights by specific agent (optional)

        Returns:
            List of insights
        """
        try:
            return await self.shared_memory_coordinator.get_session_insights(session_id, agent_filter)
        except Exception as e:
            logger.error(f"Failed to get session insights for {session_id}: {e}")
            return []

    async def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get shared context from a collaborative session.

        Args:
            session_id: Session ID

        Returns:
            Shared context dictionary
        """
        try:
            return await self.shared_memory_coordinator.get_session_context(session_id)
        except Exception as e:
            logger.error(f"Failed to get session context for {session_id}: {e}")
            return {}

    def list_my_sessions(self) -> List[Dict[str, Any]]:
        """
        List collaborative sessions this agent is participating in.

        Returns:
            List of session summaries
        """
        try:
            all_sessions = self.shared_memory_coordinator.list_active_sessions()
            my_sessions = []

            for session in all_sessions:
                if self.role in session.get("participants", []):
                    my_sessions.append(session)

            return my_sessions
        except Exception as e:
            logger.error(f"Failed to list sessions for {self.role}: {e}")
            return []

    async def archive_session(self, session_id: str) -> bool:
        """
        Archive a collaborative session (creator only).

        Args:
            session_id: Session ID

        Returns:
            bool: Success status
        """
        try:
            # Check if this agent is the creator
            summary = await self.get_session_summary(session_id)
            if not summary or summary.get("creator") != self.role:
                logger.warning(f"Agent {self.role} is not the creator of session {session_id}")
                return False

            success = await self.shared_memory_coordinator.archive_session(session_id)
            if success:
                logger.info(f"Agent {self.role} archived session: {session_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to archive session {session_id} for {self.role}: {e}")
            return False

    async def submit_optimization_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit an optimization proposal to the LearningAgent for evaluation and potential implementation.
        
        Args:
            proposal: Proposal dictionary containing:
                - proposal_type: Type of optimization (strategy, risk, execution, data, macro)
                - description: Human-readable description
                - changes: Specific changes proposed
                - expected_impact: Expected performance impact
                - evidence: Supporting data/evidence
                - confidence_score: Confidence in the proposal (0-1)
                - test_requirements: Requirements for testing the proposal
                
        Returns:
            Dict with submission status and proposal ID
        """
        try:
            # Validate proposal structure
            required_fields = ['proposal_type', 'description', 'changes', 'expected_impact']
            for field in required_fields:
                if field not in proposal:
                    return {
                        'submitted': False,
                        'error': f'Missing required field: {field}',
                        'proposal_id': None
                    }
            
            # Generate unique proposal ID
            proposal_id = f"proposal_{self.role}_{uuid.uuid4().hex[:8]}"
            
            # Enrich proposal with metadata
            enriched_proposal = {
                **proposal,
                'proposal_id': proposal_id,
                'submitted_by': self.role,
                'submitted_at': pd.Timestamp.now().isoformat(),
                'status': 'submitted',
                'evaluation_status': 'pending',
                'implementation_status': 'pending',
                'performance_tracking': {
                    'baseline_metrics': {},
                    'post_implementation_metrics': {},
                    'rollback_available': False
                }
            }
            
            # Submit via A2A protocol to LearningAgent
            if self.shared_memory_coordinator:
                try:
                    # Use A2A protocol to send proposal to LearningAgent
                    success = await self.shared_memory_coordinator.share_memory(
                        self.role, 'learning_agent', 'optimization_proposals', 
                        proposal_id, enriched_proposal
                    )
                    
                    if success:
                        logger.info(f"Agent {self.role} submitted optimization proposal: {proposal_id}")
                        return {
                            'submitted': True,
                            'proposal_id': proposal_id,
                            'message': 'Proposal submitted successfully to LearningAgent'
                        }
                    else:
                        logger.warning(f"Failed to submit proposal via A2A protocol: {proposal_id}")
                        return {
                            'submitted': False,
                            'error': 'A2A protocol submission failed',
                            'proposal_id': proposal_id
                        }
                        
                except Exception as e:
                    logger.error(f"A2A protocol error submitting proposal: {e}")
                    return {
                        'submitted': False,
                        'error': f'A2A protocol error: {str(e)}',
                        'proposal_id': proposal_id
                    }
            else:
                # Fallback: store in local memory for LearningAgent to pick up
                logger.warning("A2A protocol not available, using fallback storage")
                
                # Store proposal in shared namespace if available
                try:
                    success = await self.store_shared_memory(
                        'optimization_proposals', proposal_id, enriched_proposal
                    )
                    
                    if success:
                        logger.info(f"Agent {self.role} stored optimization proposal locally: {proposal_id}")
                        return {
                            'submitted': True,
                            'proposal_id': proposal_id,
                            'message': 'Proposal stored locally (A2A fallback)'
                        }
                    else:
                        return {
                            'submitted': False,
                            'error': 'Local storage failed',
                            'proposal_id': None
                        }
                        
                except Exception as e:
                    logger.error(f"Local storage error: {e}")
                    return {
                        'submitted': False,
                        'error': f'Storage error: {str(e)}',
                        'proposal_id': None
                    }
                    
        except Exception as e:
            logger.error(f"Error submitting optimization proposal from {self.role}: {e}")
            return {
                'submitted': False,
                'error': f'Unexpected error: {str(e)}',
                'proposal_id': None
            }

    async def send_a2a_message(self, message_type: str, receiver: str, data: Dict[str, Any], reply_to: Optional[str] = None) -> Optional[str]:
        """
        Send a monitored A2A message to another agent.
        
        Args:
            message_type: Type of message (e.g., 'proposal', 'data_request', 'status_update')
            receiver: Target agent role or 'all' for broadcast
            data: Message payload
            reply_to: Optional message ID this is replying to
            
        Returns:
            Message ID if sent successfully, None otherwise
        """
        if not self.a2a_protocol:
            logger.warning(f"A2A protocol not available for {self.role} - cannot send message")
            return None
            
        try:
            from src.utils.a2a_protocol import BaseMessage
            
            message = BaseMessage(
                type=message_type,
                sender=self.role,
                receiver=receiver,
                timestamp="",  # Will be auto-filled by protocol
                data=data,
                id="",  # Will be auto-filled by protocol
                reply_to=reply_to
            )
            
            message_id = await self.a2a_protocol.send_message(message)
            logger.info(f"Agent {self.role} sent A2A message {message_type} to {receiver}")
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to send A2A message from {self.role}: {e}")
            return None

    async def receive_a2a_message(self) -> Optional[Dict[str, Any]]:
        """
        Receive a monitored A2A message.
        
        Returns:
            Message dict if available, None if no messages
        """
        if not self.a2a_protocol:
            logger.warning(f"A2A protocol not available for {self.role} - cannot receive messages")
            return None
            
        try:
            message = await self.a2a_protocol.receive_message(self.role)
            if message:
                logger.info(f"Agent {self.role} received A2A message: {message.type}")
                return message.dict() if hasattr(message, 'dict') else message.__dict__
            return None
            
        except Exception as e:
            logger.error(f"Failed to receive A2A message for {self.role}: {e}")
            return None

    async def request_a2a_data(self, target_agent: str, data_type: str, parameters: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Request data from another agent via A2A protocol.
        
        Args:
            target_agent: Agent to request data from
            data_type: Type of data requested
            parameters: Additional parameters for the request
            
        Returns:
            Response data or None if failed
        """
        if not self.a2a_protocol:
            logger.warning(f"A2A protocol not available for {self.role} - cannot request data")
            return None
            
        try:
            # Send data request
            request_data = {
                'data_type': data_type,
                'parameters': parameters or {},
                'request_timestamp': pd.Timestamp.now().isoformat()
            }
            
            message_id = await self.send_a2a_message('data_request', target_agent, request_data)
            if not message_id:
                return None
                
            # Wait for response (with timeout)
            timeout = 30  # seconds
            start_time = asyncio.get_event_loop().time()
            
            while asyncio.get_event_loop().time() - start_time < timeout:
                response = await self.receive_a2a_message()
                if response and response.get('reply_to') == message_id:
                    logger.info(f"Agent {self.role} received data response from {target_agent}")
                    return response.get('data')
                    
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            logger.warning(f"Timeout waiting for data response from {target_agent}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to request A2A data from {target_agent}: {e}")
            return None

    async def share_a2a_data(self, target_agent: str, data_type: str, data: Any, context: Dict[str, Any] = None) -> bool:
        """
        Share data with another agent via A2A protocol.
        
        Args:
            target_agent: Agent to share data with
            data_type: Type of data being shared
            data: The data to share
            context: Additional context information
            
        Returns:
            True if shared successfully, False otherwise
        """
        if not self.a2a_protocol:
            logger.warning(f"A2A protocol not available for {self.role} - cannot share data")
            return False
            
        try:
            share_data = {
                'data_type': data_type,
                'data': data,
                'context': context or {},
                'share_timestamp': pd.Timestamp.now().isoformat()
            }
            
            message_id = await self.send_a2a_message('data_share', target_agent, share_data)
            return message_id is not None
            
        except Exception as e:
            logger.error(f"Failed to share A2A data with {target_agent}: {e}")
            return False

    async def broadcast_a2a_status(self, status_type: str, status_data: Dict[str, Any]) -> bool:
        """
        Broadcast status update to all agents via A2A protocol.
        
        Args:
            status_type: Type of status update
            status_data: Status information
            
        Returns:
            True if broadcast successfully, False otherwise
        """
        if not self.a2a_protocol:
            logger.warning(f"A2A protocol not available for {self.role} - cannot broadcast status")
            return False
            
        try:
            status_payload = {
                'status_type': status_type,
                'status_data': status_data,
                'broadcast_timestamp': pd.Timestamp.now().isoformat()
            }
            
            message_id = await self.send_a2a_message('status_broadcast', 'all', status_payload)
            return message_id is not None
            
        except Exception as e:
            logger.error(f"Failed to broadcast A2A status from {self.role}: {e}")
            return False