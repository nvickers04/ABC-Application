import importlib
import os
from pathlib import Path
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def check_dependencies() -> Dict[str, str]:
    """Check key Python dependencies."""
    deps = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'yfinance': 'yfinance',
        'scikit-learn': 'sklearn',
        'tensorflow': 'tensorflow',
        'torch': 'torch',
        'transformers': 'transformers',
        'langchain_core': 'langchain_core',
        'python-dotenv': 'dotenv',
        'pytest-asyncio': 'pytest_asyncio',
    }
    results = {}
    for name, module in deps.items():
        try:
            importlib.import_module(module)
            results[name] = 'PASS'
        except ImportError as e:
            results[name] = f'FAIL: {str(e)}'
    return results

def check_imports() -> Dict[str, str]:
    """Check imports of key modules."""
    modules = [
        'src.agents.base',
        'src.agents.data',
        'src.agents.risk',
        'src.agents.strategy',
        'src.agents.execution',
        'src.agents.learning',
        'src.agents.reflection',
        'src.utils.shared_memory',
        'src.utils.a2a_protocol',
        'src.integrations.ibkr_connector',
    ]
    results = {}
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            results[module_name] = 'PASS'
        except Exception as e:
            results[module_name] = f'FAIL: {str(e)}'
    return results

def check_agents() -> Dict[str, str]:
    """Attempt to instantiate agents."""
    from src.agents.base import BaseAgent  # Assuming base is importable
    agents = {
        'DataAgent': 'src.agents.data.DataAgent',
        'RiskAgent': 'src.agents.risk.RiskAgent',
        'StrategyAgent': 'src.agents.strategy.StrategyAgent',
        'ExecutionAgent': 'src.agents.execution.ExecutionAgent',
        'LearningAgent': 'src.agents.learning.LearningAgent',
        'ReflectionAgent': 'src.agents.reflection.ReflectionAgent',
    }
    results = {}
    for name, path in agents.items():
        try:
            module = importlib.import_module(path.rsplit('.', 1)[0])
            agent_class = getattr(module, name)
            agent_class()  # Instantiate without args if possible
            results[name] = 'PASS'
        except Exception as e:
            results[name] = f'FAIL: {str(e)}'
    return results

def check_integrations() -> Dict[str, str]:
    """Check API keys and directories."""
    results = {}
    # API keys (check existence in env)
    keys = ['OPENAI_API_KEY', 'GROK_API_KEY', 'IBKR_ACCOUNT_ID']
    for key in keys:
        if os.getenv(key):
            results[key] = 'PASS'
        else:
            results[key] = 'MISSING'
    
    # Directories
    dirs = ['src', 'src/agents', 'src/utils', 'config', 'data']
    for d in dirs:
        if Path(d).exists():
            results[f'dir_{d}'] = 'PASS'
        else:
            results[f'dir_{d}'] = 'FAIL'
    return results

def main():
    logger.info("🏥 New ABC Application System Health Check")
    logger.info("Dependencies:")
    for dep, status in check_dependencies().items():
        logger.info(f"  {dep}: {status}")
    
    logger.info("\nImports:")
    for mod, status in check_imports().items():
        logger.info(f"  {mod}: {status}")
    
    logger.info("\nAgents:")
    for agent, status in check_agents().items():
        logger.info(f"  {agent}: {status}")
    
    logger.info("\nIntegrations:")
    for item, status in check_integrations().items():
        logger.info(f"  {item}: {status}")

if __name__ == "__main__":
    main()
