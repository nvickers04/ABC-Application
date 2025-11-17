#!/usr/bin/env python3
"""
Workflow Status Tracker
Tracks progress through the iterative reasoning workflow phases
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any

class WorkflowStatusTracker:
    """Tracks the status and progress of iterative reasoning workflows"""

    def __init__(self, workflow_file: str = "workflow_status.json"):
        self.workflow_file = workflow_file
        self.status = self.load_status()

    def load_status(self) -> Dict[str, Any]:
        """Load workflow status from file"""
        if os.path.exists(self.workflow_file):
            try:
                with open(self.workflow_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return self.initialize_new_workflow()

    def initialize_new_workflow(self) -> Dict[str, Any]:
        """Initialize a new workflow status structure"""
        return {
            'workflow_id': f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'start_time': datetime.now().isoformat(),
            'current_phase': 'not_started',
            'phases_completed': [],
            'phase_status': {
                'macro_foundation': {'status': 'pending', 'commands_sent': 0, 'responses_received': 0},
                'iteration_1_intelligence': {'status': 'pending', 'commands_sent': 0, 'responses_received': 0},
                'iteration_1_strategy': {'status': 'pending', 'commands_sent': 0, 'responses_received': 0},
                'iteration_1_debate': {'status': 'pending', 'commands_sent': 0, 'responses_received': 0},
                'iteration_1_risk': {'status': 'pending', 'commands_sent': 0, 'responses_received': 0},
                'iteration_1_consensus': {'status': 'pending', 'commands_sent': 0, 'responses_received': 0},
                'iteration_1_execution': {'status': 'pending', 'commands_sent': 0, 'responses_received': 0},
                'iteration_1_learning': {'status': 'pending', 'commands_sent': 0, 'responses_received': 0},
                'iteration_2_executive': {'status': 'pending', 'commands_sent': 0, 'responses_received': 0},
                'supreme_oversight': {'status': 'pending', 'commands_sent': 0, 'responses_received': 0}
            },
            'agent_responses': [],
            'key_insights': [],
            'decisions': [],
            'warnings': [],
            'completion_time': None,
            'total_duration_minutes': 0
        }

    def update_phase_status(self, phase: str, status: str, commands_sent: int = None, responses_received: int = None):
        """Update the status of a specific phase"""
        if phase in self.status['phase_status']:
            self.status['phase_status'][phase]['status'] = status
            if commands_sent is not None:
                self.status['phase_status'][phase]['commands_sent'] = commands_sent
            if responses_received is not None:
                self.status['phase_status'][phase]['responses_received'] = responses_received

            if status == 'completed' and phase not in self.status['phases_completed']:
                self.status['phases_completed'].append(phase)

        self.status['current_phase'] = phase
        self.save_status()

    def add_agent_response(self, agent: str, command: str, response: str, phase: str):
        """Add an agent response to the tracking"""
        response_entry = {
            'timestamp': datetime.now().isoformat(),
            'phase': phase,
            'agent': agent,
            'command': command,
            'response': response[:500]  # Truncate long responses
        }
        self.status['agent_responses'].append(response_entry)

        # Update response counts
        if phase in self.status['phase_status']:
            self.status['phase_status'][phase]['responses_received'] += 1

        self.save_status()

    def add_insight(self, insight: str, agent: str, phase: str):
        """Add a key insight from the workflow"""
        insight_entry = {
            'timestamp': datetime.now().isoformat(),
            'phase': phase,
            'agent': agent,
            'insight': insight
        }
        self.status['key_insights'].append(insight_entry)
        self.save_status()

    def add_decision(self, decision: str, agent: str, phase: str):
        """Add a decision made during the workflow"""
        decision_entry = {
            'timestamp': datetime.now().isoformat(),
            'phase': phase,
            'agent': agent,
            'decision': decision
        }
        self.status['decisions'].append(decision_entry)
        self.save_status()

    def add_warning(self, warning: str, agent: str, phase: str):
        """Add a warning or concern raised during the workflow"""
        warning_entry = {
            'timestamp': datetime.now().isoformat(),
            'phase': phase,
            'agent': agent,
            'warning': warning
        }
        self.status['warnings'].append(warning_entry)
        self.save_status()

    def complete_workflow(self):
        """Mark the workflow as completed"""
        self.status['completion_time'] = datetime.now().isoformat()
        start_time = datetime.fromisoformat(self.status['start_time'])
        completion_time = datetime.fromisoformat(self.status['completion_time'])
        duration = completion_time - start_time
        self.status['total_duration_minutes'] = duration.total_seconds() / 60
        self.save_status()

    def save_status(self):
        """Save the current status to file"""
        with open(self.workflow_file, 'w') as f:
            json.dump(self.status, f, indent=2, default=str)

    def get_status_summary(self) -> str:
        """Get a human-readable status summary"""
        summary = []
        summary.append("🤖 Iterative Reasoning Workflow Status")
        summary.append("=" * 50)
        summary.append(f"Workflow ID: {self.status['workflow_id']}")
        summary.append(f"Start Time: {self.status['start_time']}")
        summary.append(f"Current Phase: {self.status['current_phase']}")
        summary.append(f"Phases Completed: {len(self.status['phases_completed'])}/10")
        summary.append("")

        # Phase status
        summary.append("📊 Phase Status:")
        for phase, status in self.status['phase_status'].items():
            icon = "✅" if status['status'] == 'completed' else "⏳" if status['status'] == 'in_progress' else "❌"
            summary.append(f"  {icon} {phase.replace('_', ' ').title()}: {status['status']}")
            summary.append(f"     Commands: {status['commands_sent']}, Responses: {status['responses_received']}")

        summary.append("")
        summary.append(f"📝 Key Insights: {len(self.status['key_insights'])}")
        summary.append(f"🎯 Decisions: {len(self.status['decisions'])}")
        summary.append(f"⚠️  Warnings: {len(self.status['warnings'])}")
        summary.append(f"💬 Agent Responses: {len(self.status['agent_responses'])}")

        if self.status.get('completion_time'):
            summary.append("")
            summary.append(f"✅ Completed: {self.status['completion_time']}")
            summary.append(f"⏱️  Duration: {self.status['total_duration_minutes']:.1f} minutes")

        return "\n".join(summary)

    def get_next_recommended_commands(self) -> List[str]:
        """Get recommended next commands based on current status"""
        current_phase = self.status['current_phase']

        recommendations = {
            'not_started': [
                "!m analyze Assess current market regime, volatility levels, and macroeconomic trends. Identify top 5 sectors/assets with highest relative strength, momentum, and risk-adjusted returns."
            ],
            'macro_foundation': [
                "!d analyze Gather and validate multi-source market data for the identified opportunities",
                "!m analyze Provide market regime context and sector analysis",
                "!s analyze Begin forming initial hypotheses based on market data",
                "!r analyze Evaluate data quality and identify potential risk signals"
            ],
            'iteration_1_intelligence': [
                "!s analyze Develop comprehensive trading strategies informed by complete intelligence picture",
                "!d analyze Provide specific insights and validation for proposed approaches",
                "!r analyze Integrate risk constraints and probability assessments into strategy design",
                "!e analyze Evaluate practical feasibility and market impact considerations"
            ],
            'iteration_1_strategy': [
                '!m debate "Debate the proposed strategies considering current market regime and risk factors" strategy risk reflection execution'
            ],
            'iteration_1_debate': [
                "!r analyze Conduct comprehensive probabilistic analysis and risk assessment",
                "!s analyze Refine strategies based on risk analysis and agent feedback",
                "!m analyze Ensure strategies align with broader market regime analysis"
            ],
            'iteration_1_risk': [
                "!ref analyze Synthesize all inputs and mediate conflicts to reach consensus on optimal strategies"
            ],
            'iteration_1_consensus': [
                "!e analyze Validate practical feasibility, timing, and market impact",
                "!m analyze Final sanity checks and market timing validation"
            ],
            'iteration_1_execution': [
                "!l analyze Incorporate outcomes into future reasoning processes and identify improvement areas"
            ],
            'iteration_1_learning': [
                "!ref analyze Synthesize comprehensive inputs into cohesive strategic narratives with elevated perspective",
                "!r analyze Apply more conservative probability thresholds and heightened risk sensitivity",
                "!s analyze Consider broader market implications and systemic risks",
                "!e analyze Emphasize practical constraints and market impact",
                "!l analyze Provide deeper pattern recognition and historical precedent analysis"
            ],
            'iteration_2_executive': [
                "!ref analyze Conduct comprehensive audit of all data points and analysis from both iterations",
                "!ref analyze Evaluate decisions against multiple potential market scenarios and stress conditions",
                "!ref analyze Identify subtle warning signals, historical precedents, and potential catastrophic scenarios",
                "!ref analyze Ensure all conclusions follow from established premises and logical reasoning",
                "!ref analyze Render final decision with authority to veto any strategy or mandate additional iterations if concerning patterns emerge"
            ]
        }

        return recommendations.get(current_phase, ["Workflow status unclear. Check current phase."])

def main():
    """Command-line interface for workflow status tracking"""
    tracker = WorkflowStatusTracker()

    print("🤖 Workflow Status Tracker")
    print("=" * 30)
    print("Commands:")
    print("1. Show status summary")
    print("2. Get next recommended commands")
    print("3. Mark phase as completed")
    print("4. Add insight/decision/warning")
    print("5. Complete workflow")
    print("6. Reset workflow")

    while True:
        choice = input("\nEnter command (1-6, or 'q' to quit): ").strip()

        if choice == 'q':
            break
        elif choice == '1':
            print("\n" + tracker.get_status_summary())
        elif choice == '2':
            commands = tracker.get_next_recommended_commands()
            print(f"\n📋 Recommended next commands for phase '{tracker.status['current_phase']}':")
            for i, cmd in enumerate(commands, 1):
                print(f"{i}. {cmd}")
        elif choice == '3':
            phases = list(tracker.status['phase_status'].keys())
            print("\nAvailable phases:")
            for i, phase in enumerate(phases, 1):
                print(f"{i}. {phase.replace('_', ' ').title()}")
            phase_choice = input("Enter phase number to mark complete: ").strip()
            try:
                phase_idx = int(phase_choice) - 1
                if 0 <= phase_idx < len(phases):
                    tracker.update_phase_status(phases[phase_idx], 'completed')
                    print(f"✅ Marked {phases[phase_idx]} as completed")
                else:
                    print("❌ Invalid phase number")
            except ValueError:
                print("❌ Invalid input")
        elif choice == '4':
            entry_type = input("Type (insight/decision/warning): ").strip().lower()
            content = input("Content: ").strip()
            agent = input("Agent (optional): ").strip() or "manual"
            phase = tracker.status['current_phase']

            if entry_type == 'insight':
                tracker.add_insight(content, agent, phase)
                print("✅ Insight added")
            elif entry_type == 'decision':
                tracker.add_decision(content, agent, phase)
                print("✅ Decision added")
            elif entry_type == 'warning':
                tracker.add_warning(content, agent, phase)
                print("✅ Warning added")
            else:
                print("❌ Invalid type")
        elif choice == '5':
            tracker.complete_workflow()
            print("✅ Workflow marked as completed")
        elif choice == '6':
            confirm = input("Reset workflow? This will create a new workflow. (y/N): ").strip().lower()
            if confirm == 'y':
                tracker.status = tracker.initialize_new_workflow()
                tracker.save_status()
                print("🔄 Workflow reset")
        else:
            print("❌ Invalid command")

if __name__ == "__main__":
    main()