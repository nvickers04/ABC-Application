import asyncio
import discord
import os
from dotenv import load_dotenv

load_dotenv()

async def debug_channel_routing():
    token = os.getenv('DISCORD_ORCHESTRATOR_TOKEN')
    guild_id = int(os.getenv('DISCORD_GUILD_ID', '0'))

    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True

    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print('=== CHANNEL ROUTING DEBUG ===')
        print('Connected as:', client.user)

        guild = client.get_guild(guild_id)
        if guild:
            print(f'Guild: {guild.name}')

            # Check channel setup
            agent_channels = {}
            agent_channel_map = {
                'macro': 'macro',
                'data': 'data',
                'strategy': 'strategy',
                'risk': 'risk',
                'reflection': 'reflection',
                'execution': 'execution',
                'learning': 'learning',
                'debates': 'debates',
                'alerts': 'alerts'
            }

            print('\nChannel Detection:')
            for agent_type, channel_name in agent_channel_map.items():
                for ch in guild.text_channels:
                    if ch.name == channel_name:
                        agent_channels[agent_type] = ch
                        print(f'  ✅ {agent_type} → #{ch.name}')
                        break
                else:
                    print(f'  ❌ {agent_type} → #{channel_name} NOT FOUND')

            # Test command routing
            print('\nCommand Routing Test:')
            test_commands = [
                '!m analyze test macro command',
                '!d analyze test data command',
                '!s analyze test strategy command',
                '!m debate test debate command'
            ]

            for cmd in test_commands:
                # Simulate get_command_channel logic
                prefix_to_agent = {
                    '!m': 'macro',
                    '!d': 'data',
                    '!s': 'strategy',
                    '!r': 'risk',
                    '!ref': 'reflection',
                    '!exec': 'execution',
                    '!l': 'learning'
                }

                prefix = cmd.split()[0].lower()
                agent_type = prefix_to_agent.get(prefix)

                # Special handling for debate commands
                if 'debate' in cmd.lower():
                    agent_type = 'debates'

                if agent_type and agent_type in agent_channels:
                    target_channel = agent_channels[agent_type]
                    print(f'  "{cmd}" → #{target_channel.name}')
                else:
                    print(f'  "{cmd}" → #general (fallback)')

        await client.close()

    try:
        await client.start(token)
    except Exception as e:
        print('Failed:', e)

if __name__ == "__main__":
    asyncio.run(debug_channel_routing())