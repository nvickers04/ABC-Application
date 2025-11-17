#!/usr/bin/env python3
"""
Discord Agent Integration Setup Script
Helps configure and test the Discord agent system
"""

import os
import sys
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_env_template():
    """Create environment variables template"""
    env_template = """# Discord Bot Tokens
DISCORD_GUILD_ID=YOUR_GUILD_ID
DISCORD_MACRO_AGENT_TOKEN=YOUR_MACRO_BOT_TOKEN
DISCORD_DATA_AGENT_TOKEN=YOUR_DATA_BOT_TOKEN
DISCORD_STRATEGY_AGENT_TOKEN=YOUR_STRATEGY_BOT_TOKEN
DISCORD_RISK_AGENT_TOKEN=YOUR_RISK_BOT_TOKEN
DISCORD_REFLECTION_AGENT_TOKEN=YOUR_REFLECTION_BOT_TOKEN
DISCORD_EXECUTION_AGENT_TOKEN=YOUR_EXECUTION_BOT_TOKEN
DISCORD_LEARNING_AGENT_TOKEN=YOUR_LEARNING_BOT_TOKEN
"""

    template_path = Path(".env.template")
    if not template_path.exists():
        with open(template_path, 'w') as f:
            f.write(env_template)
        logger.info(f"Created .env template at {template_path}")

    return template_path

def validate_config():
    """Validate Discord configuration from environment variables"""
    logger.info("Validating Discord configuration from environment variables...")

    # Check required environment variables
    required_vars = [
        'DISCORD_GUILD_ID',
        'DISCORD_MACRO_AGENT_TOKEN',
        'DISCORD_DATA_AGENT_TOKEN',
        'DISCORD_STRATEGY_AGENT_TOKEN',
        'DISCORD_RISK_AGENT_TOKEN',
        'DISCORD_REFLECTION_AGENT_TOKEN',
        'DISCORD_EXECUTION_AGENT_TOKEN',
        'DISCORD_LEARNING_AGENT_TOKEN'
    ]

    missing_vars = []
    placeholder_vars = []

    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        elif value.startswith('YOUR_'):
            placeholder_vars.append(var)

    if missing_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        return False

    if placeholder_vars:
        logger.warning(f"Placeholder values found for: {', '.join(placeholder_vars)}")
        logger.warning("Please replace placeholder tokens with actual Discord bot tokens")

    # Validate token formats (basic check)
    token_vars = [var for var in required_vars if var.endswith('_TOKEN')]
    for var in token_vars:
        token = os.getenv(var)
        if token and not token.startswith('YOUR_'):
            # Discord tokens typically start with numbers and have specific format
            if not (token.startswith(('M', 'N', 'O')) and '.' in token):
                logger.warning(f"Token {var} may not be a valid Discord bot token format")

    logger.info("Discord configuration validation completed")
    return True

def test_agent_imports():
    """Test that all agent classes can be imported"""
    try:
        sys.path.insert(0, str(Path(__file__).parent))

        from src.agents.base import BaseAgent
        from src.agents.macro import MacroAgent
        from src.agents.data import DataAgent
        from src.agents.strategy import StrategyAgent
        from src.agents.risk import RiskAgent
        from src.agents.reflection import ReflectionAgent
        from src.agents.execution import ExecutionAgent
        from src.agents.learning import LearningAgent

        logger.info("All agent imports successful")
        return True

    except ImportError as e:
        logger.error(f"Failed to import agents: {e}")
        return False

def check_discord_dependencies():
    """Check if Discord.py is installed"""
    try:
        import discord
        from discord.ext import commands
        logger.info(f"Discord.py version: {discord.__version__}")
        return True
    except ImportError:
        logger.error("Discord.py not installed. Install with: pip install discord.py")
        return False

def generate_setup_instructions():
    """Generate setup instructions"""
    instructions = """# Discord Agent Integration Setup Instructions

## 1. Create Discord Applications
For each agent, create a separate Discord bot application:

1. Go to https://discord.com/developers/applications
2. Click "New Application" for each agent
3. Go to "Bot" section and create a bot
4. Copy the bot token
5. Enable all Privileged Gateway Intents

## 2. Configure Bot Permissions
In the Discord developer portal:
- Go to OAuth2 → URL Generator
- Select scopes: bot, applications.commands
- Select permissions: Send Messages, Use Slash Commands, Read Message History, Add Reactions

## 3. Invite Bots to Server
Use the generated URLs to invite each bot to your Discord server.

## 4. Create Channels
Create dedicated channels for each agent (e.g., #macro-agent, #data-agent, etc.)

## 5. Update Environment Variables
Edit your .env file with the bot tokens and server ID:
```
DISCORD_GUILD_ID=your_server_id_here
DISCORD_MACRO_AGENT_TOKEN=your_macro_bot_token
DISCORD_DATA_AGENT_TOKEN=your_data_bot_token
DISCORD_STRATEGY_AGENT_TOKEN=your_strategy_bot_token
DISCORD_RISK_AGENT_TOKEN=your_risk_bot_token
DISCORD_REFLECTION_AGENT_TOKEN=your_reflection_bot_token
DISCORD_EXECUTION_AGENT_TOKEN=your_execution_bot_token
DISCORD_LEARNING_AGENT_TOKEN=your_learning_bot_token
```

## 6. Run the System
python discord_agents.py

## 7. Test Commands
Try these commands in Discord:
- !status - Check agent status
- !debate "Test topic" - Start a debate
- !system_health - Check system status

## Troubleshooting
- Ensure all bot tokens are correct
- Check that bots have proper permissions
- Verify channel IDs are correct
- Check bot intents are enabled in Discord developer portal
- Make sure .env file is loaded (python-dotenv should handle this automatically)
"""
    return instructions

def main():
    """Main setup function"""
    print("🤖 Discord Agent Integration Setup")
    print("=" * 50)

    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_discord_dependencies():
        return

    if not test_agent_imports():
        return

    # Validate configuration from environment variables
    print("\n✅ Validating configuration...")
    if validate_config():
        print("Configuration is valid!")
    else:
        print("Configuration has issues. Please check the errors above.")

    # Generate instructions
    print("\n📖 Setup Instructions:")
    instructions = generate_setup_instructions()
    print(instructions)

    # Save instructions to file
    instructions_file = Path("DISCORD_SETUP_INSTRUCTIONS.md")
    with open(instructions_file, 'w', encoding='utf-8') as f:
        f.write(instructions)
    print(f"\n📄 Instructions also saved to {instructions_file}")

    print("\n🎉 Setup complete! Follow the instructions above to get started.")

if __name__ == "__main__":
    main()