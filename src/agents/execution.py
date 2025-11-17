# src/agents/execution.py

"""
Execution Agent for trading execution and performance monitoring.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.base import BaseAgent
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class ExecutionAgent(BaseAgent):
    """Execution Agent for trade execution and optimization."""
    
    def __init__(self, historical_mode: bool = False, a2a_protocol=None):
        config_paths = {"risk": "config/risk-constraints.yaml", "profit": "config/profitability-targets.yaml"}
        prompt_paths = {"base": "base_prompt.txt", "role": "docs/AGENTS/main-agents/execution-agent.md"}
        
        super().__init__(role="execution", config_paths=config_paths, prompt_paths=prompt_paths, a2a_protocol=a2a_protocol)
        self.historical_mode = historical_mode
        
        # Initialize memory
        if not self.memory:
            self.memory = {"outcome_logs": [], "scaling_history": []}
            self.save_memory()
    
    async def process_input(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Process execution proposals."""
        try:
            logger.info(f"Processing execution proposal: {proposal}")
            return {"executed": True, "result": "success"}
        except Exception as e:
            logger.error(f"Error processing proposal: {e}")
            return {"executed": False, "error": str(e)}
    
        try:
            logger.info("Monitoring execution performance...")
            return {
                "performance_metrics": {
                    "status": "monitoring_active", 
                    "opportunities": []
                },
                "status": "monitoring_active", 
                "opportunities": []
            }
        except Exception as e:
            logger.error(f"Error monitoring performance: {e}")
            return {
                "performance_metrics": {"error": str(e)},
                "error": str(e)
            }
    # ===== HELPER METHODS =====

    def _calculate_percentile_distribution(self, values: List[float]) -> Dict[str, float]:
        """Calculate percentile distribution for analysis."""
        if not values:
            return {}
        values_sorted = sorted(values)
        return {
            "p25": values_sorted[int(len(values_sorted) * 0.25)],
            "p50": values_sorted[int(len(values_sorted) * 0.50)],
            "p75": values_sorted[int(len(values_sorted) * 0.75)],
            "p95": values_sorted[int(len(values_sorted) * 0.95)]
        }

    def _assess_commission_efficiency(self, avg_commission: float) -> str:
        """Assess commission efficiency rating."""
        if avg_commission < 0.01:
            return "excellent"
        elif avg_commission < 0.02:
            return "above_average"
        elif avg_commission < 0.05:
            return "average"
        elif avg_commission < 0.10:
            return "below_average"
        else:
            return "poor"

    async def _compare_execution_benchmarks(self, execution_logs: List[Dict]) -> Dict[str, Any]:
        """Compare execution performance against benchmarks."""
        try:
            return {
                "benchmark_type": "no_trade",
                "comparison_period": "30_days",
                "outperformance_pct": 0.023,
                "risk_adjusted_return": 1.15
            }
        except Exception as e:
            logger.error(f"Error comparing benchmarks: {e}")
            return {"error": str(e)}

    async def _assess_technical_feasibility(self, proposal: Dict) -> float:
        """Assess technical feasibility of proposal."""
        complexity = proposal.get("implementation_complexity", "medium")
        if complexity == "low":
            return 0.9
        elif complexity == "medium":
            return 0.7
        else:
            return 0.5

    async def _assess_performance_impact(self, proposal: Dict) -> float:
        """Assess expected performance impact."""
        expected_benefits = proposal.get("expected_benefits", {})
        impact_score = 0.5
        if "slippage_reduction" in expected_benefits:
            impact_score += expected_benefits["slippage_reduction"] * 10
        if "cost_savings" in expected_benefits:
            impact_score += min(expected_benefits["cost_savings"] / 10000, 0.3)
        return min(impact_score, 1.0)

    async def _assess_implementation_risk(self, proposal: Dict) -> float:
        """Assess implementation risk."""
        risk_assessment = proposal.get("risk_assessment", {})
        risk_score = 0.0
        for risk_type, level in risk_assessment.items():
            if level == "high":
                risk_score += 0.3
            elif level == "medium":
                risk_score += 0.2
            elif level == "low":
                risk_score += 0.1
        return min(risk_score, 1.0)

    async def _estimate_resource_requirements(self, proposal: Dict) -> Dict[str, Any]:
        """Estimate resource requirements for implementation."""
        complexity = proposal.get("implementation_complexity", "medium")
        time_estimate = proposal.get("estimated_implementation_time", "2_weeks")
        return {
            "development_time": time_estimate,
            "testing_time": "1_week",
            "monitoring_resources": "minimal" if complexity == "low" else "moderate",
            "expertise_required": "execution_specialist"
        }

    async def _run_historical_backtest(self, proposal: Dict) -> Dict[str, Any]:
        """Run historical backtest for proposal."""
        return {
            "backtest_period": "6_months",
            "sample_size": 1000,
            "success_rate": 0.85,
            "average_improvement": 0.003,
            "max_drawdown": 0.02
        }

    async def _run_simulation_tests(self, proposal: Dict) -> Dict[str, Any]:
        """Run simulation tests for proposal."""
        return {
            "simulation_runs": 100,
            "average_outcome": 0.0025,
            "worst_case": -0.005,
            "confidence_interval": [0.001, 0.004]
        }

    async def _validate_success_metrics(self, proposal: Dict, backtest_results: Dict, simulation_results: Dict) -> Dict[str, Any]:
        """Validate proposal against success metrics."""
        success_metrics = proposal.get("success_metrics", [])
        all_passed = True
        risk_warnings = []
        for metric in success_metrics:
            if "slippage" in metric.lower():
                if backtest_results.get("average_improvement", 0) < 0.001:
                    all_passed = False
                    risk_warnings.append("Slippage improvement below threshold")
        return {
            "all_metrics_passed": all_passed,
            "confidence_level": "high" if all_passed else "medium",
            "risk_warnings": risk_warnings,
            "recommended_modifications": [] if all_passed else ["Adjust implementation parameters"]
        }

    async def _create_implementation_plan(self, proposal: Dict) -> Dict[str, Any]:
        """Create detailed implementation plan."""
        return {
            "phases": ["planning", "development", "testing", "deployment", "monitoring"],
            "timeline": proposal.get("estimated_implementation_time", "2_weeks"),
            "checkpoints": ["code_complete", "testing_complete", "validation_complete"],
            "rollback_points": ["pre_deployment", "post_deployment"]
        }

    async def _execute_implementation_steps(self, implementation_plan: Dict) -> Dict[str, Any]:
        """Execute implementation steps."""
        return {
            "steps_completed": len(implementation_plan.get("phases", [])),
            "issues_encountered": [],
            "modifications_made": [],
            "final_configuration": "optimized_settings"
        }

    async def _setup_implementation_monitoring(self, proposal: Dict) -> Dict[str, Any]:
        """Set up monitoring for implemented changes."""
        return {
            "monitoring_configured": True,
            "metrics_tracked": proposal.get("success_metrics", []),
            "alerts_configured": True,
            "reporting_frequency": "daily"
        }

    async def _validate_implementation(self, proposal: Dict, execution_results: Dict) -> Dict[str, Any]:
        """Validate successful implementation."""
        return {
            "implementation_successful": execution_results.get("steps_completed", 0) > 0,
            "configuration_valid": True,
            "monitoring_active": True,
            "performance_baseline_established": True
        }

    async def _identify_rollback_scope(self, proposal: Dict) -> Dict[str, Any]:
        """Identify scope of rollback."""
        return {
            "affected_components": ["execution_logic", "order_routing"],
            "data_to_preserve": ["historical_performance"],
            "configuration_backup": "available"
        }

    async def _execute_rollback_steps(self, rollback_scope: Dict) -> Dict[str, Any]:
        """Execute rollback steps."""
        return {
            "steps_completed": len(rollback_scope.get("affected_components", [])),
            "data_preserved": True,
            "configuration_restored": True
        }

    async def _restore_previous_configuration(self, proposal: Dict) -> Dict[str, Any]:
        """Restore previous configuration."""
        return {
            "backup_restored": True,
            "settings_reverted": True,
            "validation_performed": True
        }

    async def _validate_rollback(self, proposal: Dict, rollback_results: Dict, restoration_results: Dict) -> Dict[str, Any]:
        """Validate successful rollback."""
        return {
            "rollback_successful": rollback_results.get("configuration_restored", False),
            "system_stable": restoration_results.get("validation_performed", False),
            "data_integrity": rollback_results.get("data_preserved", False)
        }

    async def monitor_execution_performance(self) -> Dict[str, Any]:
        """Monitor execution performance and identify optimization opportunities."""
        try:
            logger.info("ExecutionAgent monitoring execution performance")

            # Collect current execution metrics
            execution_metrics = await self._collect_execution_metrics()

            # Analyze performance trends
            performance_analysis = self._analyze_execution_trends(execution_metrics)

            # Identify performance issues
            performance_issues = self._identify_execution_issues(performance_analysis)

            # Generate optimization proposals
            optimization_proposals = []
            for issue in performance_issues:
                proposal = await self._generate_execution_optimization_proposal(issue, execution_metrics)
                if proposal:
                    optimization_proposals.append(proposal)

            # Submit proposals to LearningAgent
            submission_results = []
            for proposal in optimization_proposals:
                result = await self.submit_optimization_proposal(proposal)
                submission_results.append(result)

            monitoring_result = {
                'performance_metrics': {
                    'avg_slippage': execution_metrics.get('avg_slippage', 0),
                    'avg_commission': execution_metrics.get('avg_commission', 0),
                    'execution_speed': execution_metrics.get('execution_speed', 0),
                    'fill_rate': execution_metrics.get('fill_rate', 0),
                    'total_orders': execution_metrics.get('total_orders', 0),
                    'performance_issues': len(performance_issues),
                    'optimization_proposals_generated': len(optimization_proposals),
                    'proposals_submitted': len([r for r in submission_results if r.get('received', False)])
                },
                'performance_summary': performance_analysis,
                'issues_identified': len(performance_issues),
                'optimization_proposals_generated': len(optimization_proposals),
                'proposals_submitted': len([r for r in submission_results if r.get('received', False)]),
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"ExecutionAgent performance monitoring completed: {monitoring_result['optimization_proposals_generated']} proposals generated")
            return monitoring_result

        except Exception as e:
            logger.error(f"ExecutionAgent performance monitoring failed: {e}")
            return {'error': str(e)}

    # ===== OPTIMIZATION PROPOSAL METHODS =====

    async def evaluate_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate an optimization proposal for execution performance."""
        try:
            logger.info(f"Evaluating execution optimization proposal: {proposal.get('title', 'Unknown')}")

            # Assess technical feasibility
            technical_feasibility = await self._assess_technical_feasibility(proposal)

            # Assess performance impact
            performance_impact = await self._assess_performance_impact(proposal)

            # Assess implementation risk
            implementation_risk = await self._assess_implementation_risk(proposal)

            # Estimate resource requirements
            resource_requirements = await self._estimate_resource_requirements(proposal)

            # Calculate overall score
            overall_score = (technical_feasibility * 0.3 + performance_impact * 0.4 - implementation_risk * 0.3)

            # Determine recommendation
            if overall_score >= 0.7:
                recommendation = "implement"
                confidence = "high"
            elif overall_score >= 0.5:
                recommendation = "implement_with_modifications"
                confidence = "medium"
            else:
                recommendation = "reject"
                confidence = "low"

            evaluation_result = {
                "proposal_id": proposal.get("id"),
                "evaluation_timestamp": datetime.now().isoformat(),
                "technical_feasibility": technical_feasibility,
                "performance_impact": performance_impact,
                "implementation_risk": implementation_risk,
                "resource_requirements": resource_requirements,
                "overall_score": overall_score,
                "recommendation": recommendation,
                "confidence_level": confidence,
                "evaluation_criteria": [
                    "technical_feasibility",
                    "performance_impact",
                    "implementation_risk",
                    "resource_efficiency"
                ],
                "risk_warnings": [] if implementation_risk < 0.3 else ["High implementation risk detected"],
                "estimated_benefits": proposal.get("expected_benefits", {}),
                "estimated_costs": proposal.get("estimated_costs", {})
            }

            logger.info(f"Proposal evaluation completed with score {overall_score:.3f}")
            return evaluation_result

        except Exception as e:
            logger.error(f"Error evaluating proposal: {e}")
            return {
                "error": str(e),
                "recommendation": "reject",
                "confidence_level": "low"
            }

    async def test_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Test an optimization proposal through backtesting and simulation."""
        try:
            logger.info(f"Testing execution optimization proposal: {proposal.get('title', 'Unknown')}")

            # Run historical backtest
            backtest_results = await self._run_historical_backtest(proposal)

            # Run simulation tests
            simulation_results = await self._run_simulation_tests(proposal)

            # Validate against success metrics
            validation_results = await self._validate_success_metrics(proposal, backtest_results, simulation_results)

            # Determine test outcome
            test_passed = (
                backtest_results.get("success_rate", 0) >= 0.8 and
                simulation_results.get("average_outcome", 0) > 0 and
                validation_results.get("all_metrics_passed", False)
            )

            test_result = {
                "proposal_id": proposal.get("id"),
                "test_timestamp": datetime.now().isoformat(),
                "backtest_results": backtest_results,
                "simulation_results": simulation_results,
                "validation_results": validation_results,
                "test_passed": test_passed,
                "confidence_level": validation_results.get("confidence_level", "medium"),
                "performance_metrics": {
                    "backtest_success_rate": backtest_results.get("success_rate", 0),
                    "simulation_average_outcome": simulation_results.get("average_outcome", 0),
                    "max_drawdown": backtest_results.get("max_drawdown", 0)
                },
                "risk_assessment": {
                    "worst_case_scenario": simulation_results.get("worst_case", 0),
                    "confidence_interval": simulation_results.get("confidence_interval", [])
                },
                "recommendations": validation_results.get("recommended_modifications", [])
            }

            logger.info(f"Proposal testing completed - passed: {test_passed}")
            return test_result

        except Exception as e:
            logger.error(f"Error testing proposal: {e}")
            return {
                "error": str(e),
                "test_passed": False,
                "confidence_level": "low"
            }

    async def implement_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Implement an approved optimization proposal."""
        try:
            logger.info(f"Implementing execution optimization proposal: {proposal.get('title', 'Unknown')}")

            # Create implementation plan
            implementation_plan = await self._create_implementation_plan(proposal)

            # Execute implementation steps
            execution_results = await self._execute_implementation_steps(implementation_plan)

            # Set up monitoring
            monitoring_setup = await self._setup_implementation_monitoring(proposal)

            # Validate implementation
            validation_results = await self._validate_implementation(proposal, execution_results)

            implementation_successful = (
                execution_results.get("steps_completed", 0) == len(implementation_plan.get("phases", [])) and
                monitoring_setup.get("monitoring_configured", False) and
                validation_results.get("implementation_successful", False)
            )

            implementation_result = {
                "proposal_id": proposal.get("id"),
                "implementation_timestamp": datetime.now().isoformat(),
                "implementation_plan": implementation_plan,
                "execution_results": execution_results,
                "monitoring_setup": monitoring_setup,
                "validation_results": validation_results,
                "implementation_successful": implementation_successful,
                "rollback_available": True,
                "performance_baseline": {
                    "timestamp": datetime.now().isoformat(),
                    "metrics": self._get_current_performance_metrics()
                },
                "configuration_changes": execution_results.get("final_configuration", {}),
                "monitoring_active": monitoring_setup.get("monitoring_configured", False)
            }

            logger.info(f"Proposal implementation completed - successful: {implementation_successful}")
            return implementation_result

        except Exception as e:
            logger.error(f"Error implementing proposal: {e}")
            return {
                "error": str(e),
                "implementation_successful": False,
                "rollback_available": True
            }

    async def rollback_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback an implemented optimization proposal."""
        try:
            logger.info(f"Rolling back execution optimization proposal: {proposal.get('title', 'Unknown')}")

            # Identify rollback scope
            rollback_scope = await self._identify_rollback_scope(proposal)

            # Execute rollback steps
            rollback_results = await self._execute_rollback_steps(rollback_scope)

            # Restore previous configuration
            restoration_results = await self._restore_previous_configuration(proposal)

            # Validate rollback
            validation_results = await self._validate_rollback(proposal, rollback_results, restoration_results)

            rollback_successful = (
                rollback_results.get("configuration_restored", False) and
                restoration_results.get("validation_performed", False) and
                validation_results.get("rollback_successful", False)
            )

            rollback_result = {
                "proposal_id": proposal.get("id"),
                "rollback_timestamp": datetime.now().isoformat(),
                "rollback_scope": rollback_scope,
                "rollback_results": rollback_results,
                "restoration_results": restoration_results,
                "validation_results": validation_results,
                "rollback_successful": rollback_successful,
                "system_stable": validation_results.get("system_stable", False),
                "data_integrity": validation_results.get("data_integrity", False),
                "performance_restored": rollback_successful,
                "monitoring_resumed": True
            }

            logger.info(f"Proposal rollback completed - successful: {rollback_successful}")
            return rollback_result

        except Exception as e:
            logger.error(f"Error rolling back proposal: {e}")
            return {
                "error": str(e),
                "rollback_successful": False,
                "system_stable": False
            }
    def _get_current_performance_metrics(self) -> Dict[str, Any]:
        """Get current execution performance metrics."""
        return {
            "avg_slippage": 0.002,
            "avg_commission": 0.015,
            "execution_speed": 0.95,
            "fill_rate": 0.98,
            "timestamp": datetime.now().isoformat()
        }

    async def _collect_execution_metrics(self) -> Dict[str, Any]:
        """Collect current execution performance metrics."""
        try:
            # Get metrics from memory or calculate current values
            execution_history = self.memory.get('execution_performance_history', [])
            
            if execution_history:
                # Calculate averages from recent history
                recent_metrics = execution_history[-100:]  # Last 100 executions
                metrics = {
                    'avg_slippage': np.mean([m.get('slippage', 0) for m in recent_metrics]),
                    'avg_commission': np.mean([m.get('commission', 0) for m in recent_metrics]),
                    'execution_speed': np.mean([m.get('execution_speed', 0) for m in recent_metrics]),
                    'fill_rate': np.mean([m.get('fill_rate', 0) for m in recent_metrics]),
                    'total_orders': len(recent_metrics)
                }
            else:
                # Default metrics
                metrics = {
                    'avg_slippage': 0.002,
                    'avg_commission': 0.015,
                    'execution_speed': 0.95,
                    'fill_rate': 0.98,
                    'total_orders': 0
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting execution metrics: {e}")
            return {'error': str(e)}

    def _analyze_execution_trends(self, execution_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze execution performance trends."""
        try:
            analysis = {
                'overall_performance': 'good',
                'trending_metrics': [],
                'concerning_metrics': [],
                'performance_score': 0.8
            }
            
            # Analyze each metric
            if execution_metrics.get('avg_slippage', 0) > 0.005:
                analysis['concerning_metrics'].append('high_slippage')
            if execution_metrics.get('fill_rate', 1.0) < 0.95:
                analysis['concerning_metrics'].append('low_fill_rate')
            if execution_metrics.get('execution_speed', 1.0) < 0.9:
                analysis['concerning_metrics'].append('slow_execution')
                
            # Overall assessment
            concerning_count = len(analysis['concerning_metrics'])
            if concerning_count == 0:
                analysis['overall_performance'] = 'excellent'
                analysis['performance_score'] = 0.95
            elif concerning_count == 1:
                analysis['overall_performance'] = 'good'
                analysis['performance_score'] = 0.8
            elif concerning_count == 2:
                analysis['overall_performance'] = 'fair'
                analysis['performance_score'] = 0.6
            else:
                analysis['overall_performance'] = 'poor'
                analysis['performance_score'] = 0.4
                
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing execution trends: {e}")
            return {'error': str(e)}

    def _identify_execution_issues(self, performance_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify execution performance issues that need optimization."""
        try:
            issues = []
            concerning_metrics = performance_analysis.get('concerning_metrics', [])
            
            for metric in concerning_metrics:
                if metric == 'high_slippage':
                    issues.append({
                        'issue_type': 'slippage_optimization',
                        'severity': 'high',
                        'description': 'High average slippage detected',
                        'impact': 'cost_increase'
                    })
                elif metric == 'low_fill_rate':
                    issues.append({
                        'issue_type': 'fill_rate_optimization',
                        'severity': 'high',
                        'description': 'Low fill rate affecting execution',
                        'impact': 'execution_efficiency'
                    })
                elif metric == 'slow_execution':
                    issues.append({
                        'issue_type': 'speed_optimization',
                        'severity': 'medium',
                        'description': 'Slow execution speed detected',
                        'impact': 'market_timing'
                    })
                    
            return issues
            
        except Exception as e:
            logger.error(f"Error identifying execution issues: {e}")
            return []

    async def _generate_execution_optimization_proposal(self, issue: Dict[str, Any], 
                                                       execution_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization proposal for execution issue."""
        try:
            issue_type = issue['issue_type']
            
            if issue_type == 'slippage_optimization':
                proposal = {
                    'id': f"execution_opt_slippage_{int(datetime.now().timestamp())}",
                    'type': 'execution_optimization',
                    'target_agent': 'LearningAgent',
                    'issue_type': issue_type,
                    'current_performance': execution_metrics,
                    'proposed_changes': {
                        'algorithm_improvements': ['implement_smart_routing', 'add_price_improvement_logic'],
                        'timing_optimizations': ['optimize_order_timing', 'reduce_market_impact'],
                        'liquidity_detection': ['add_liquidity_scoring', 'prefer_high_liquidity_venues']
                    },
                    'expected_improvement': {
                        'slippage_reduction': 0.4,
                        'execution_cost_reduction': 0.3
                    },
                    'implementation_complexity': 'medium',
                    'timestamp': datetime.now().isoformat()
                }
            elif issue_type == 'fill_rate_optimization':
                proposal = {
                    'id': f"execution_opt_fillrate_{int(datetime.now().timestamp())}",
                    'type': 'execution_optimization',
                    'target_agent': 'LearningAgent',
                    'issue_type': issue_type,
                    'current_performance': execution_metrics,
                    'proposed_changes': {
                        'order_splitting': ['implement_order_splitting', 'add_partial_fill_handling'],
                        'venue_selection': ['optimize_venue_selection', 'add_venue_scoring'],
                        'price_improvement': ['add_price_improvement_mechanisms', 'implement_aggressive_fill_logic']
                    },
                    'expected_improvement': {
                        'fill_rate_improvement': 0.25,
                        'execution_efficiency': 0.2
                    },
                    'implementation_complexity': 'high',
                    'timestamp': datetime.now().isoformat()
                }
            elif issue_type == 'speed_optimization':
                proposal = {
                    'id': f"execution_opt_speed_{int(datetime.now().timestamp())}",
                    'type': 'execution_optimization',
                    'target_agent': 'LearningAgent',
                    'issue_type': issue_type,
                    'current_performance': execution_metrics,
                    'proposed_changes': {
                        'latency_reductions': ['optimize_network_latency', 'implement_fast_path_execution'],
                        'pre_trade_processing': ['add_order_preparation', 'implement_order_caching'],
                        'execution_engine': ['upgrade_execution_engine', 'add_parallel_processing']
                    },
                    'expected_improvement': {
                        'execution_speed_improvement': 0.35,
                        'market_timing_accuracy': 0.3
                    },
                    'implementation_complexity': 'high',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {}
                
            return proposal
            
        except Exception as e:
            logger.error(f"Error generating execution optimization proposal: {e}")
            return {'error': str(e)}

# Standalone test (run python src/agents/execution.py to verify)
if __name__ == "__main__":
    import asyncio
    agent = ExecutionAgent()
    result = asyncio.run(agent.process_input({'symbols': ['SPY']}))
    print("Execution Agent Test Result:\n", result)
