# [LABEL:TEST:discord_integration] [LABEL:TEST:integration] [LABEL:FRAMEWORK:discord]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-11-17] [LABEL:REVIEWED:pending]
#
# Purpose: Integration test for Discord server access and bot connectivity
# Dependencies: discord.py, environment variables, Discord API
# Related: docs/IMPLEMENTATION/DISCORD_SETUP_INSTRUCTIONS.md, tools/discord/
#
import discord
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

GUILD_ID = int(os.getenv('DISCORD_GUILD_ID', '0'))

async def check_server_access(token, name):
    try:
        client = discord.Client(intents=discord.Intents.default())

        @client.event
        async def on_ready():
            print(f'{name}: Connected as {client.user}')

            # Check if bot is in the target guild
            guild = client.get_guild(GUILD_ID)
            if guild:
                print(f'{name}: Found target server "{guild.name}" (ID: {guild.id})')
                member = guild.get_member(client.user.id)
                if member:
                    print(f'{name}: Bot is a member of the server')
                else:
                    print(f'{name}: Bot is NOT a member of the server')
            else:
                print(f'{name}: Could not find target server (ID: {GUILD_ID})')
                print(f'{name}: Available servers: {[g.name for g in client.guilds]}')

            await client.close()

        await client.start(token)
    except Exception as e:
        print(f'{name}: Error: {e}')

async def main():
    print(f'Target Guild ID: {GUILD_ID}')
    tokens = {
        'Macro': os.getenv('DISCORD_MACRO_AGENT_TOKEN'),
        'Data': os.getenv('DISCORD_DATA_AGENT_TOKEN'),
        'Strategy': os.getenv('DISCORD_STRATEGY_AGENT_TOKEN'),
        'Risk': os.getenv('DISCORD_RISK_AGENT_TOKEN'),
        'Reflection': os.getenv('DISCORD_REFLECTION_AGENT_TOKEN'),
        'Execution': os.getenv('DISCORD_EXECUTION_AGENT_TOKEN'),
        'Learning': os.getenv('DISCORD_LEARNING_AGENT_TOKEN')
    }

    for name, token in tokens.items():
        if token:
            await check_server_access(token, name)
        else:
            print(f'{name}: No token found')

asyncio.run(main())