#!/usr/bin/env python3
"""
Comprehensive Validation Test for Optimization Proposal Protocols
Tests all agents' optimization proposal implementations across the system.
"""

import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizationProposalValidator:
    """Comprehensive validator for optimization proposal implementations."""

    def __init__(self):
        self.agents = {}
        self.test_results = {}
        self.agent_modules = {
            'learning': 'src.agents.learning',
            'strategy': 'src.agents.strategy',
            'risk': 'src.agents.risk',
            'data': 'src.agents.data',
            'execution': 'src.agents.execution',
            'macro': 'src.agents.macro'
        }

    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        logger.info("🚀 Starting Comprehensive Optimization Proposal Validation")

        try:
            # Phase 1: Import Validation
            await self._validate_imports()

            # Phase 2: Method Availability
            await self._validate_method_availability()

            # Phase 3: Basic Functionality Tests
            await self._validate_basic_functionality()

            # Phase 4: Inter-Agent Communication
            await self._validate_inter_agent_communication()

            # Phase 5: Performance Tests
            await self._validate_performance()

            # Generate final report
            return self._generate_validation_report()

        except Exception as e:
            logger.error(f"❌ Validation failed with error: {e}")
            return {
                'status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    async def _validate_imports(self) -> None:
        """Test that all agents can be imported successfully."""
        logger.info("📦 Phase 1: Validating Agent Imports")

        for agent_name, module_path in self.agent_modules.items():
            try:
                module = __import__(module_path, fromlist=[f'{agent_name.capitalize()}Agent'])
                agent_class = getattr(module, f'{agent_name.capitalize()}Agent')
                agent_instance = agent_class()

                self.agents[agent_name] = agent_instance
                self.test_results[f'{agent_name}_import'] = {'status': 'PASSED'}

                logger.info(f"✅ {agent_name.capitalize()}Agent imported successfully")

            except Exception as e:
                self.test_results[f'{agent_name}_import'] = {
                    'status': 'FAILED',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                logger.error(f"❌ Failed to import {agent_name.capitalize()}Agent: {e}")

    async def _validate_method_availability(self) -> None:
        """Test that all required optimization proposal methods are available."""
        logger.info("🔧 Phase 2: Validating Method Availability")

        required_methods = [
            'monitor_performance',  # Will be agent-specific like monitor_strategy_performance
            'evaluate_proposal',
            'test_proposal',
            'implement_proposal',
            'rollback_proposal',
            'submit_optimization_proposal'
        ]

        # Agent-specific monitor methods
        agent_specific_monitors = {
            'learning': 'monitor_strategy_performance',
            'strategy': 'monitor_strategy_performance',
            'risk': 'monitor_strategy_performance',
            'data': 'monitor_data_quality_performance',
            'execution': 'monitor_execution_performance',
            'macro': 'monitor_macro_performance'
        }

        for agent_name, agent_instance in self.agents.items():
            try:
                available_methods = [m for m in dir(agent_instance) if not m.startswith('_')]

                # Check agent-specific monitor method
                specific_monitor = agent_specific_monitors[agent_name]
                if specific_monitor in available_methods:
                    self.test_results[f'{agent_name}_monitor_method'] = {'status': 'PASSED'}
                else:
                    self.test_results[f'{agent_name}_monitor_method'] = {
                        'status': 'FAILED',
                        'error': f'Missing method: {specific_monitor}'
                    }

                # Check standard optimization proposal methods
                missing_methods = []
                for method in required_methods[1:]:  # Skip generic monitor_performance
                    if method not in available_methods:
                        missing_methods.append(method)

                if not missing_methods:
                    self.test_results[f'{agent_name}_proposal_methods'] = {'status': 'PASSED'}
                else:
                    self.test_results[f'{agent_name}_proposal_methods'] = {
                        'status': 'FAILED',
                        'error': f'Missing methods: {missing_methods}'
                    }

                if all(test.get('status') == 'PASSED' for test in [
                    self.test_results.get(f'{agent_name}_monitor_method', {}),
                    self.test_results.get(f'{agent_name}_proposal_methods', {})
                ]):
                    logger.info(f"✅ {agent_name.capitalize()}Agent has all required methods")
                else:
                    logger.error(f"❌ {agent_name.capitalize()}Agent missing required methods")

            except Exception as e:
                self.test_results[f'{agent_name}_method_validation'] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
                logger.error(f"❌ Error validating methods for {agent_name}: {e}")

    async def _validate_basic_functionality(self) -> None:
        """Test basic functionality of optimization proposal methods."""
        logger.info("⚙️ Phase 3: Validating Basic Functionality")

        # Test proposal template
        test_proposal = {
            'id': 'test_proposal_001',
            'title': 'Test Optimization Proposal',
            'description': 'A test proposal for validation',
            'changes': {'parameter': 'test_param', 'value': 0.5},
            'expected_benefits': {'improvement': 0.1},
            'estimated_costs': {'development_time': '1_day'},
            'implementation_complexity': 'low',
            'estimated_implementation_time': '1_day',
            'success_metrics': ['performance_improvement'],
            'risk_assessment': {'technical': 'low', 'business': 'low'}
        }

        for agent_name, agent_instance in self.agents.items():
            try:
                # Test evaluate_proposal
                eval_result = await agent_instance.evaluate_proposal(test_proposal)
                if eval_result and 'recommendation' in eval_result:
                    self.test_results[f'{agent_name}_evaluate'] = {'status': 'PASSED'}
                else:
                    self.test_results[f'{agent_name}_evaluate'] = {
                        'status': 'FAILED',
                        'error': 'Invalid evaluation result'
                    }

                # Test test_proposal
                test_result = await agent_instance.test_proposal(test_proposal)
                if test_result and 'test_passed' in test_result:
                    self.test_results[f'{agent_name}_test'] = {'status': 'PASSED'}
                else:
                    self.test_results[f'{agent_name}_test'] = {
                        'status': 'FAILED',
                        'error': 'Invalid test result'
                    }

                # Test monitor performance (agent-specific)
                agent_specific_monitors = {
                    'learning': 'monitor_strategy_performance',
                    'strategy': 'monitor_strategy_performance',
                    'risk': 'monitor_strategy_performance',
                    'data': 'monitor_data_quality_performance',
                    'execution': 'monitor_execution_performance',
                    'macro': 'monitor_macro_performance'
                }
                
                monitor_method_name = agent_specific_monitors.get(agent_name)
                if monitor_method_name:
                    monitor_method = getattr(agent_instance, monitor_method_name, None)
                    if monitor_method:
                        monitor_result = await monitor_method()
                        if monitor_result and 'performance_metrics' in monitor_result:
                            self.test_results[f'{agent_name}_monitor'] = {'status': 'PASSED'}
                        else:
                            self.test_results[f'{agent_name}_monitor'] = {
                                'status': 'FAILED',
                                'error': 'Invalid monitor result'
                            }
                    else:
                        self.test_results[f'{agent_name}_monitor'] = {
                            'status': 'FAILED',
                            'error': f'Monitor method {monitor_method_name} not found'
                        }
                else:
                    self.test_results[f'{agent_name}_monitor'] = {
                        'status': 'FAILED',
                        'error': 'No monitor method defined for agent'
                    }

                # Check if all basic tests passed
                basic_tests = [
                    self.test_results.get(f'{agent_name}_evaluate', {}),
                    self.test_results.get(f'{agent_name}_test', {}),
                    self.test_results.get(f'{agent_name}_monitor', {})
                ]

                if all(test.get('status') == 'PASSED' for test in basic_tests):
                    logger.info(f"✅ {agent_name.capitalize()}Agent basic functionality validated")
                else:
                    logger.error(f"❌ {agent_name.capitalize()}Agent basic functionality failed")

            except Exception as e:
                self.test_results[f'{agent_name}_basic_functionality'] = {
                    'status': 'FAILED',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                logger.error(f"❌ Error in basic functionality test for {agent_name}: {e}")

    async def _validate_inter_agent_communication(self) -> None:
        """Test inter-agent communication for proposal submission."""
        logger.info("📡 Phase 4: Validating Inter-Agent Communication")

        # Test proposal submission from each agent
        test_proposal = {
            'proposal_type': 'test_communication',
            'description': 'Test inter-agent communication',
            'changes': {'test': 'communication'},
            'expected_impact': {'communication': 'validated'},
            'evidence': 'validation_test'
        }

        for agent_name, agent_instance in self.agents.items():
            try:
                # Test submit_optimization_proposal
                submit_result = await agent_instance.submit_optimization_proposal(test_proposal)

                if submit_result and submit_result.get('submitted', False):
                    self.test_results[f'{agent_name}_communication'] = {'status': 'PASSED'}
                    logger.info(f"✅ {agent_name.capitalize()}Agent communication validated")
                else:
                    self.test_results[f'{agent_name}_communication'] = {
                        'status': 'FAILED',
                        'error': 'Proposal submission failed',
                        'result': submit_result
                    }
                    logger.error(f"❌ {agent_name.capitalize()}Agent communication failed")

            except Exception as e:
                self.test_results[f'{agent_name}_communication'] = {
                    'status': 'FAILED',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                logger.error(f"❌ Error in communication test for {agent_name}: {e}")

    async def _validate_performance(self) -> None:
        """Test performance characteristics of optimization proposal methods."""
        logger.info("⚡ Phase 5: Validating Performance Characteristics")

        import time

        test_proposal = {
            'id': 'perf_test_001',
            'title': 'Performance Test Proposal',
            'description': 'Testing method performance',
            'changes': {'param': 'test'},
            'expected_benefits': {'speed': 0.1}
        }

        for agent_name, agent_instance in self.agents.items():
            try:
                # Test evaluation performance
                start_time = time.time()
                eval_result = await agent_instance.evaluate_proposal(test_proposal)
                eval_time = time.time() - start_time

                # Test should complete within reasonable time (5 seconds)
                if eval_time < 5.0 and eval_result:
                    self.test_results[f'{agent_name}_performance'] = {
                        'status': 'PASSED',
                        'eval_time': eval_time
                    }
                    logger.info(f"✅ {agent_name.capitalize()}Agent performance validated ({eval_time:.2f}s)")
                else:
                    self.test_results[f'{agent_name}_performance'] = {
                        'status': 'FAILED',
                        'error': f'Too slow or invalid result: {eval_time:.2f}s',
                        'eval_time': eval_time
                    }
                    logger.error(f"❌ {agent_name.capitalize()}Agent performance test failed ({eval_time:.2f}s)")

            except Exception as e:
                self.test_results[f'{agent_name}_performance'] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
                logger.error(f"❌ Error in performance test for {agent_name}: {e}")

    def _generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        logger.info("📊 Generating Validation Report")

        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'PASSED')
        failed_tests = total_tests - passed_tests

        # Categorize results by agent
        agent_results = {}
        for agent_name in self.agent_modules.keys():
            agent_tests = {k: v for k, v in self.test_results.items() if k.startswith(agent_name)}
            agent_passed = sum(1 for result in agent_tests.values() if result.get('status') == 'PASSED')
            agent_total = len(agent_tests)
            agent_results[agent_name] = {
                'total_tests': agent_total,
                'passed_tests': agent_passed,
                'failed_tests': agent_total - agent_passed,
                'success_rate': agent_passed / agent_total if agent_total > 0 else 0
            }

        # Overall status
        overall_status = 'PASSED' if failed_tests == 0 else 'FAILED'

        report = {
            'status': overall_status,
            'timestamp': '2025-11-11T14:35:00Z',  # Current date
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0
            },
            'agent_results': agent_results,
            'detailed_results': self.test_results,
            'recommendations': self._generate_recommendations()
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        # Check for failed imports
        failed_imports = [k for k, v in self.test_results.items()
                         if k.endswith('_import') and v.get('status') == 'FAILED']
        if failed_imports:
            recommendations.append(f"Fix import issues for agents: {[k.replace('_import', '') for k in failed_imports]}")

        # Check for missing methods
        missing_methods = [k for k, v in self.test_results.items()
                          if 'method' in k and v.get('status') == 'FAILED']
        if missing_methods:
            recommendations.append("Implement missing optimization proposal methods")

        # Check for functionality failures
        failed_functionality = [k for k, v in self.test_results.items()
                               if k.endswith(('_evaluate', '_test', '_monitor')) and v.get('status') == 'FAILED']
        if failed_functionality:
            recommendations.append("Debug and fix optimization proposal method implementations")

        # Check for communication failures
        failed_communication = [k for k, v in self.test_results.items()
                               if k.endswith('_communication') and v.get('status') == 'FAILED']
        if failed_communication:
            recommendations.append("Fix inter-agent communication issues")

        # Check for performance issues
        slow_performance = [k for k, v in self.test_results.items()
                           if k.endswith('_performance') and v.get('status') == 'FAILED']
        if slow_performance:
            recommendations.append("Optimize method performance for faster execution")

        if not recommendations:
            recommendations.append("All validation tests passed! System is ready for production use.")

        return recommendations


async def main():
    """Main validation execution."""
    validator = OptimizationProposalValidator()
    results = await validator.run_comprehensive_validation()

    # Print summary
    print("\n" + "="*80)
    print("🎯 OPTIMIZATION PROPOSAL VALIDATION RESULTS")
    print("="*80)
    print(f"Status: {results['status']}")
    print(f"Total Tests: {results['summary']['total_tests']}")
    print(f"Passed: {results['summary']['passed_tests']}")
    print(f"Failed: {results['summary']['failed_tests']}")
    print(".1f")
    print()

    print("📈 AGENT-BY-AGENT RESULTS:")
    for agent, stats in results['agent_results'].items():
        status = "✅" if stats['failed_tests'] == 0 else "❌"
        print(".1f")

    print()
    print("💡 RECOMMENDATIONS:")
    for rec in results['recommendations']:
        print(f"• {rec}")

    print("\n" + "="*80)

    # Return exit code based on results
    return 0 if results['status'] == 'PASSED' else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)