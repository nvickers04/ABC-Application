#!/usr/bin/env python3
"""
Command Registry for Discord Bot Commands
Maintains a centralized registry of all available commands for documentation and help systems.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class CommandType(Enum):
    """Types of commands"""
    SLASH = "slash"
    LEGACY = "legacy"
    BOTH = "both"


@dataclass
class CommandInfo:
    """Information about a command"""
    name: str
    description: str
    usage: str
    category: str
    command_type: CommandType
    aliases: Optional[List[str]] = None
    examples: Optional[List[str]] = None
    permissions: Optional[List[str]] = None


class CommandRegistry:
    """Registry for tracking all Discord commands"""

    def __init__(self):
        self.commands: Dict[str, CommandInfo] = {}
        self.categories: Dict[str, List[str]] = {}

        # Initialize with existing commands
        self._initialize_commands()

    def _initialize_commands(self):
        """Initialize the registry with all known commands"""

        # Core Workflows
        self.register_command(CommandInfo(
            name="start_workflow",
            description="Begin the full AI trading analysis workflow",
            usage="!start_workflow",
            category="Core Workflows",
            command_type=CommandType.LEGACY
        ))

        self.register_command(CommandInfo(
            name="start_market_open_execution",
            description="Fast-track execution workflow at market open",
            usage="!start_market_open_execution",
            category="Core Workflows",
            command_type=CommandType.LEGACY
        ))

        self.register_command(CommandInfo(
            name="start_trade_monitoring",
            description="Start monitoring active trading positions",
            usage="!start_trade_monitoring",
            category="Core Workflows",
            command_type=CommandType.LEGACY
        ))

        self.register_command(CommandInfo(
            name="start_premarket_analysis",
            description="Run premarket analysis workflow",
            usage="!start_premarket_analysis",
            category="Core Workflows",
            command_type=CommandType.LEGACY
        ))

        self.register_command(CommandInfo(
            name="pause_workflow",
            description="Pause the current running workflow",
            usage="!pause_workflow",
            category="Core Workflows",
            command_type=CommandType.LEGACY
        ))

        self.register_command(CommandInfo(
            name="resume_workflow",
            description="Resume a paused workflow",
            usage="!resume_workflow",
            category="Core Workflows",
            command_type=CommandType.LEGACY
        ))

        self.register_command(CommandInfo(
            name="stop_workflow",
            description="Stop the current workflow",
            usage="!stop_workflow",
            category="Core Workflows",
            command_type=CommandType.LEGACY
        ))

        self.register_command(CommandInfo(
            name="stop_monitoring",
            description="Stop position monitoring workflow",
            usage="!stop_monitoring",
            category="Core Workflows",
            command_type=CommandType.LEGACY
        ))

        self.register_command(CommandInfo(
            name="workflow_status",
            description="Get current workflow status and progress",
            usage="!workflow_status",
            category="Core Workflows",
            command_type=CommandType.LEGACY
        ))

        # Analysis Commands
        self.register_command(CommandInfo(
            name="analyze",
            description="Run analysis-only workflow for specific queries",
            usage="!analyze <query>",
            category="Analysis & Research",
            command_type=CommandType.LEGACY,
            examples=["!analyze What is the current market sentiment?", "!analyze Analyze AAPL fundamentals"]
        ))

        self.register_command(CommandInfo(
            name="share_news",
            description="Share news links for Data Agent processing",
            usage="!share_news <url> [description]",
            category="Analysis & Research",
            command_type=CommandType.LEGACY,
            examples=["!share_news https://example.com/news Important market news"]
        ))

        # Consensus Commands
        self.register_command(CommandInfo(
            name="consensus_status",
            description="Show current consensus poll status",
            usage="/consensus_status",
            category="Consensus Polling",
            command_type=CommandType.SLASH
        ))

        self.register_command(CommandInfo(
            name="poll_consensus",
            description="Start a new consensus poll",
            usage="/poll_consensus <question> <agents>",
            category="Consensus Polling",
            command_type=CommandType.BOTH,
            examples=["/poll_consensus Should we execute this trade? risk_agent strategy_agent"]
        ))

        # Monitoring Commands
        self.register_command(CommandInfo(
            name="status",
            description="Get system health and component status",
            usage="!status",
            category="Monitoring & Health",
            command_type=CommandType.LEGACY
        ))

        self.register_command(CommandInfo(
            name="health_check",
            description="Run comprehensive component health check",
            usage="!health_check",
            category="Monitoring & Health",
            command_type=CommandType.LEGACY
        ))

        self.register_command(CommandInfo(
            name="scheduler_status",
            description="Show status of scheduled tasks",
            usage="!scheduler_status",
            category="Monitoring & Health",
            command_type=CommandType.LEGACY
        ))

        self.register_command(CommandInfo(
            name="alert_test",
            description="Test the alert system",
            usage="!alert_test",
            category="Monitoring & Health",
            command_type=CommandType.LEGACY
        ))

        self.register_command(CommandInfo(
            name="alert_history",
            description="Show recent alert history",
            usage="!alert_history",
            category="Monitoring & Health",
            command_type=CommandType.LEGACY
        ))

        self.register_command(CommandInfo(
            name="alert_stats",
            description="Show alert statistics and trends",
            usage="!alert_stats",
            category="Monitoring & Health",
            command_type=CommandType.LEGACY
        ))

        self.register_command(CommandInfo(
            name="alert_dashboard",
            description="Show comprehensive alert monitoring dashboard",
            usage="!alert_dashboard",
            category="Monitoring & Health",
            command_type=CommandType.LEGACY
        ))

        # System Commands
        self.register_command(CommandInfo(
            name="commands",
            description="Show all available commands",
            usage="/commands or !commands",
            category="System & Utilities",
            command_type=CommandType.BOTH,
            aliases=["help"]
        ))

        self.register_command(CommandInfo(
            name="set_commands_channel",
            description="Configure the dedicated commands documentation channel",
            usage="!set_commands_channel <channel_id>",
            category="System & Utilities",
            command_type=CommandType.LEGACY,
            examples=["!set_commands_channel 1234567890123456789"]
        ))

    def register_command(self, command_info: CommandInfo):
        """Register a new command"""
        self.commands[command_info.name] = command_info

        # Add to category
        if command_info.category not in self.categories:
            self.categories[command_info.category] = []
        self.categories[command_info.category].append(command_info.name)

        # Register aliases
        if command_info.aliases:
            for alias in command_info.aliases:
                self.commands[alias] = command_info

    def get_command(self, name: str) -> Optional[CommandInfo]:
        """Get command information by name"""
        return self.commands.get(name)

    def get_commands_by_category(self, category: str) -> List[CommandInfo]:
        """Get all commands in a category"""
        if category not in self.categories:
            return []
        return [self.commands[name] for name in self.categories[category]]

    def get_all_categories(self) -> List[str]:
        """Get all command categories"""
        return list(self.categories.keys())

    def get_all_commands(self) -> List[CommandInfo]:
        """Get all unique commands (no duplicates for aliases)"""
        seen = set()
        unique_commands = []
        for cmd in self.commands.values():
            if cmd.name not in seen:
                unique_commands.append(cmd)
                seen.add(cmd.name)
        return unique_commands

    def generate_markdown_docs(self) -> str:
        """Generate markdown documentation for all commands"""
        docs = ["# Discord Commands Reference\n"]
        docs.append("Complete list of available Discord commands for ABC-Application.\n")

        for category in sorted(self.categories.keys()):
            docs.append(f"## {category}\n")
            commands = self.get_commands_by_category(category)

            for cmd in commands:
                docs.append(f"### `{cmd.usage}`")
                docs.append(f"{cmd.description}\n")

                if cmd.examples:
                    docs.append("**Examples:**")
                    for example in cmd.examples:
                        docs.append(f"- `{example}`")
                    docs.append("")

                if cmd.aliases:
                    docs.append(f"**Aliases:** {', '.join(cmd.aliases)}\n")

                docs.append("---\n")

        return "\n".join(docs)

    def generate_discord_embed_data(self) -> Dict[str, Any]:
        """Generate data for Discord embed display"""
        embed_data = {
            "title": "🤖 ABC-Application Discord Commands",
            "description": "Complete list of available commands. Visit #commands channel for detailed documentation.",
            "fields": []
        }

        for category in sorted(self.categories.keys()):
            commands = self.get_commands_by_category(category)
            field_value = ""

            for cmd in commands:
                prefix = "• " if cmd.command_type == CommandType.LEGACY else "/ " if cmd.command_type == CommandType.SLASH else "• "
                field_value += f"{prefix}`{cmd.usage}` - {cmd.description}\n"

            if field_value:
                embed_data["fields"].append({
                    "name": f"{self._get_category_emoji(category)} {category}",
                    "value": field_value[:1024],  # Discord field limit
                    "inline": False
                })

        return embed_data

    def _get_category_emoji(self, category: str) -> str:
        """Get emoji for category"""
        emoji_map = {
            "Core Workflows": "🤖",
            "Analysis & Research": "🔍",
            "Consensus Polling": "⚖️",
            "Monitoring & Health": "📊",
            "Trading Operations": "📈",
            "System & Utilities": "🛠️"
        }
        return emoji_map.get(category, "📋")


# Global registry instance
_command_registry = None

def get_command_registry() -> CommandRegistry:
    """Get the global command registry instance"""
    global _command_registry
    if _command_registry is None:
        _command_registry = CommandRegistry()
    return _command_registry