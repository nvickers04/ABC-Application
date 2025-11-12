#!/usr/bin/env python3
"""
System Health Check for ABC Application
Comprehensive validation of all components, dependencies, and integrations
"""

import sys
import os
import importlib
import subprocess
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

class SystemHealthChecker:
    """Comprehensive system health validation"""

    def __init__(self):
        self.results = {
            'dependencies': {},
            'imports': {},
            'agents': {},
            'integrations': {},
            'performance': {}
        }
        self.errors = []

    def log_result(self, category: str, component: str, status: str, details: Dict[str, Any] = None):
        """Log test result"""
        if category not in self.results:
            self.results[category] = {}

        self.results[category][component] = {
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }

        status_icon = '✅' if status == 'PASS' else '❌' if status == 'FAIL' else '⚠️'
        print(f"{status_icon} {category}.{component}: {status}")

        if details and 'error' in details:
            print(f"   Error: {details['error']}")

    def check_dependencies(self):
        """Check all required Python packages"""
        print("🔍 Checking Python Dependencies")
        print("=" * 50)

        # Core dependencies to check
        core_deps = [
            'pandas',
            'numpy',
            'asyncio',
            'yfinance',
            'requests',
            'aiohttp',
            'streamlit',
            'plotly',
            'matplotlib',
            'seaborn',
            'scikit-learn',
            'tensorflow',
            'torch',
            'transformers',
            'openai',
            'anthropic',
            'redis',
            'pymongo',
            'sqlalchemy',
            'fastapi',
            'uvicorn',
            'pydantic',
            'python-dotenv',
            'pytest',
            'pytest-asyncio'
        ]

        # Optional dependencies
        optional_deps = [
            'ta',  # Technical analysis
            'financedatabase',
            'alpha_vantage',
            'newsapi',
            'tweepy',  # Twitter API
            'textblob',  # Sentiment analysis
            'vaderSentiment',
            'spacy',
            'nltk',
            'beautifulsoup4',
            'selenium',
            'webdriver_manager'
        ]

        for dep in core_deps:
            self._check_package(dep, required=True)

        for dep in optional_deps:
            self._check_package(dep, required=False)

    def _check_package(self, package_name: str, required: bool = True):
        """Check if a package can be imported"""
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', 'unknown')

            # Try to get version more specifically for some packages
            if package_name == 'tensorflow':
                try:
                    import tensorflow as tf
                    version = tf.__version__
                except:
                    pass
            elif package_name == 'torch':
                try:
                    import torch
                    version = torch.__version__
                except:
                    pass

            self.log_result('dependencies', package_name, 'PASS',
                          {'version': version, 'required': required})

        except ImportError as e:
            status = 'FAIL' if required else 'MISSING'
            self.log_result('dependencies', package_name, status,
                          {'error': str(e), 'required': required})

            if required:
                self.errors.append(f"Required dependency missing: {package_name}")

    def check_imports(self):
        """Check all ABC Application imports"""
        print("\n🔍 Checking ABC Application Imports")
        print("=" * 50)

        # Set up path - src is at the same level as examples
        grok_root = Path(__file__).parent.parent  # Go up one level from examples to project root
        sys.path.insert(0, str(grok_root / 'src'))

        # Core modules to test
        core_modules = [
            'src.agents.base',
            'src.agents.data',
            'src.agents.risk',
            'src.agents.strategy',
            'src.agents.execution',
            'src.agents.learning',
            'src.agents.reflection',
            'src.agents.portfolio_dashboard',
            'src.utils.historical_simulation_engine',
            'src.utils.shared_memory',
            'src.utils.a2a_protocol',
            'src.utils.utils',
            'src.integrations.ibkr_connector'
        ]

        for module_name in core_modules:
            self._check_import(module_name)

    def _check_import(self, module_name: str):
        """Check if a module can be imported"""
        try:
            module = importlib.import_module(module_name)
            self.log_result('imports', module_name, 'PASS',
                          {'module': str(module)})

        except ImportError as e:
            self.log_result('imports', module_name, 'FAIL',
                          {'error': str(e)})
            self.errors.append(f"Import failed: {module_name} - {str(e)}")

        except Exception as e:
            self.log_result('imports', module_name, 'ERROR',
                          {'error': str(e)})
            self.errors.append(f"Import error: {module_name} - {str(e)}")

    def check_agents(self):
        """Check agent instantiation"""
        print("\n🔍 Checking Agent Instantiation")
        print("=" * 50)

        # Set up environment - src is at the same level as examples
        grok_root = Path(__file__).parent.parent  # Go up one level from examples to project root
        sys.path.insert(0, str(grok_root / 'src'))

        # Load environment variables if .env exists
        env_file = grok_root / '.env'
        if env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_file)
                print("✅ Environment variables loaded from .env")
            except ImportError:
                print("⚠️  python-dotenv not available, skipping .env loading")

        agents_to_test = [
            ('DataAgent', 'src.agents.data', 'DataAgent'),
            ('RiskAgent', 'src.agents.risk', 'RiskAgent'),
            ('StrategyAgent', 'src.agents.strategy', 'StrategyAgent'),
            ('ExecutionAgent', 'src.agents.execution', 'ExecutionAgent'),
            ('LearningAgent', 'src.agents.learning', 'LearningAgent'),
            ('ReflectionAgent', 'src.agents.reflection', 'ReflectionAgent'),
        ]

        for agent_name, module_name, class_name in agents_to_test:
            try:
                module = importlib.import_module(module_name)
                agent_class = getattr(module, class_name)
                agent_instance = agent_class()

                self.log_result('agents', agent_name, 'PASS',
                              {'class': class_name, 'module': module_name})

            except Exception as e:
                self.log_result('agents', agent_name, 'FAIL',
                              {'error': str(e), 'class': class_name, 'module': module_name})
                self.errors.append(f"Agent instantiation failed: {agent_name} - {str(e)}")

    def check_integrations(self):
        """Check external integrations"""
        print("\n🔍 Checking External Integrations")
        print("=" * 50)

        # Check Redis
        self._check_redis()

        # Check API keys
        self._check_api_keys()

        # Check file system
        self._check_file_system()

    def _check_redis(self):
        """Check Redis connectivity"""
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=1)
            r.ping()
            self.log_result('integrations', 'redis', 'PASS',
                          {'host': 'localhost', 'port': 6379})
        except ImportError:
            self.log_result('integrations', 'redis', 'MISSING',
                          {'error': 'redis package not installed'})
        except Exception as e:
            self.log_result('integrations', 'redis', 'FAIL',
                          {'error': str(e), 'note': 'Redis server may not be running'})

    def _check_api_keys(self):
        """Check for required API keys"""
        required_keys = [
            'OPENAI_API_KEY',
            'ANTHROPIC_API_KEY',
            'GROK_API_KEY',
            'ALPHA_VANTAGE_API_KEY',
            'NEWSAPI_KEY',
            'TWITTER_BEARER_TOKEN',
            'FINNHUB_API_KEY',
            'IBKR_ACCOUNT_ID',
            'IBKR_PASSWORD'
        ]

        optional_keys = [
            'REDIS_URL',
            'MONGODB_URL',
            'DATABASE_URL'
        ]

        for key in required_keys:
            value = os.getenv(key)
            if value and len(value.strip()) > 10:  # Basic validation
                self.log_result('integrations', f'api_key_{key.lower()}', 'PASS',
                              {'configured': True})
            else:
                self.log_result('integrations', f'api_key_{key.lower()}', 'MISSING',
                              {'configured': False, 'note': 'API key not set or too short'})

        for key in optional_keys:
            value = os.getenv(key)
            if value:
                self.log_result('integrations', f'api_key_{key.lower()}', 'PASS',
                              {'configured': True})
            else:
                self.log_result('integrations', f'api_key_{key.lower()}', 'OPTIONAL_MISSING',
                              {'configured': False, 'note': 'Optional API key not set'})

    def _check_file_system(self):
        """Check required directories and files"""
        grok_root = Path(__file__).parent.parent  # Go up one level from examples to project root

        required_dirs = [
            'src',
            'src/agents',
            'src/utils',
            'src/integrations',
            'config',
            'data',
            'examples',
            'docs'
        ]

        required_files = [
            'requirements.txt',
            'README.md',
            'config/risk-constraints.yaml',
            'config/profitability-targets.yaml'
        ]

        for dir_path in required_dirs:
            full_path = grok_root / dir_path
            if full_path.exists() and full_path.is_dir():
                self.log_result('integrations', f'dir_{dir_path.replace("/", "_")}', 'PASS')
            else:
                self.log_result('integrations', f'dir_{dir_path.replace("/", "_")}', 'FAIL',
                              {'error': f'Directory not found: {full_path}'})
                self.errors.append(f"Required directory missing: {dir_path}")

        for file_path in required_files:
            full_path = grok_root / file_path
            if full_path.exists() and full_path.is_file():
                self.log_result('integrations', f'file_{file_path.replace("/", "_").replace(".", "_")}', 'PASS')
            else:
                self.log_result('integrations', f'file_{file_path.replace("/", "_").replace(".", "_")}', 'FAIL',
                              {'error': f'File not found: {full_path}'})
                self.errors.append(f"Required file missing: {file_path}")

    def check_basic_functionality(self):
        """Check basic functionality of core components"""
        print("\n🔍 Checking Basic Functionality")
        print("=" * 50)

        grok_root = Path(__file__).parent.parent  # Go up one level from examples to project root
        sys.path.insert(0, str(grok_root / 'src'))

        # Test basic data agent functionality
        try:
            from src.agents.data import DataAgent
            data_agent = DataAgent()

            # Test basic input processing
            test_input = {'symbols': ['SPY'], 'period': '1mo'}
            result = asyncio.run(data_agent.process_input(test_input))

            if result and 'symbols_processed' in result:
                self.log_result('performance', 'data_agent_basic', 'PASS',
                              {'symbols_processed': result['symbols_processed']})
            else:
                self.log_result('performance', 'data_agent_basic', 'FAIL',
                              {'result': result})

        except Exception as e:
            self.log_result('performance', 'data_agent_basic', 'ERROR',
                          {'error': str(e)})

        # Test basic simulation engine
        try:
            from src.utils.historical_simulation_engine import run_historical_portfolio_simulation

            result = run_historical_portfolio_simulation(
                symbols=['SPY'],
                start_date='2024-01-01',
                end_date='2024-01-15',
                initial_capital=10000
            )

            if 'error' not in result:
                self.log_result('performance', 'simulation_engine_basic', 'PASS',
                              {'result_keys': list(result.keys())})
            else:
                self.log_result('performance', 'simulation_engine_basic', 'FAIL',
                              {'error': result.get('error')})

        except Exception as e:
            self.log_result('performance', 'simulation_engine_basic', 'ERROR',
                          {'error': str(e)})

    def generate_report(self):
        """Generate comprehensive health report"""
        print("\n📊 SYSTEM HEALTH REPORT")
        print("=" * 60)

        total_checks = 0
        passed_checks = 0
        failed_checks = 0
        error_checks = 0
        missing_checks = 0

        for category, components in self.results.items():
            print(f"\n📂 {category.upper()}:")
            for component, result in components.items():
                status = result['status']
                total_checks += 1

                if status == 'PASS':
                    passed_checks += 1
                    print(f"  ✅ {component}")
                elif status == 'FAIL':
                    failed_checks += 1
                    print(f"  ❌ {component}")
                    if 'error' in result.get('details', {}):
                        print(f"     └─ {result['details']['error']}")
                elif status == 'ERROR':
                    error_checks += 1
                    print(f"  💥 {component}")
                    if 'error' in result.get('details', {}):
                        print(f"     └─ {result['details']['error']}")
                elif status == 'MISSING':
                    missing_checks += 1
                    print(f"  ⚠️  {component} (missing)")
                else:
                    print(f"  ❓ {component}: {status}")

        print(f"\n📈 SUMMARY:")
        print(f"   Total Checks: {total_checks}")
        if total_checks > 0:
            print(f"   Passed: {passed_checks} ({passed_checks/total_checks*100:.1f}%)")
            print(f"   Failed: {failed_checks} ({failed_checks/total_checks*100:.1f}%)")
            print(f"   Errors: {error_checks} ({error_checks/total_checks*100:.1f}%)")
            print(f"   Missing: {missing_checks} ({missing_checks/total_checks*100:.1f}%)")

        # Overall health assessment
        success_rate = passed_checks / total_checks if total_checks > 0 else 0
        if success_rate >= 0.9:
            print("   🎉 OVERALL HEALTH: EXCELLENT")
        elif success_rate >= 0.75:
            print("   👍 OVERALL HEALTH: GOOD")
        elif success_rate >= 0.6:
            print("   ⚠️  OVERALL HEALTH: FAIR - Some issues to address")
        elif success_rate >= 0.4:
            print("   ❌ OVERALL HEALTH: POOR - Significant issues")
        else:
            print("   💀 OVERALL HEALTH: CRITICAL - Major components failing")

        if self.errors:
            print(f"\n🚨 CRITICAL ISSUES ({len(self.errors)}):")
            for error in self.errors[:10]:  # Show first 10
                print(f"   • {error}")
            if len(self.errors) > 10:
                print(f"   ... and {len(self.errors) - 10} more")

        return self.results

    def install_missing_dependencies(self):
        """Attempt to install missing dependencies"""
        print("\n🔧 Attempting to Install Missing Dependencies")
        print("=" * 60)

        # Check if we have pip
        try:
            import pip
        except ImportError:
            print("❌ pip not available - cannot install dependencies")
            return

        # Install from requirements.txt if it exists
        grok_root = Path(__file__).parent.parent  # Go up one level from examples to project root
        req_file = grok_root / 'requirements.txt'
        if req_file.exists():
            print("📦 Installing from requirements.txt...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(req_file)])
                print("✅ Requirements installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install requirements: {e}")
        else:
            print("⚠️  requirements.txt not found")

def main():
    """Run comprehensive system health check"""
    print("🏥 ABC Application SYSTEM HEALTH CHECK")
    print("=" * 70)
    print("Comprehensive validation of all components and dependencies")
    print()

    checker = SystemHealthChecker()

    # Run all checks
    checker.check_dependencies()
    checker.check_imports()
    checker.check_agents()
    checker.check_integrations()
    checker.check_basic_functionality()

    # Generate report
    results = checker.generate_report()

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'system_health_check_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Detailed results saved to: {filename}")

    # Offer to install missing dependencies
    if checker.errors:
        response = input("\n🔧 Would you like to attempt installing missing dependencies? (y/N): ")
        if response.lower() in ['y', 'yes']:
            checker.install_missing_dependencies()

    print("\n🏁 SYSTEM HEALTH CHECK COMPLETE!")

if __name__ == '__main__':
    main()
