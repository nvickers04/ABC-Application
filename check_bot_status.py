import discord
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

GUILD_ID = int(os.getenv('DISCORD_GUILD_ID', '0'))

async def check_bot_status(token, name):
    try:
        client = discord.Client(intents=discord.Intents.default())

        @client.event
        async def on_ready():
            if client.user:
                print(f'{name}: Connected as {client.user} (ID: {client.user.id})')

                guild = client.get_guild(GUILD_ID)
                if guild:
                    member = guild.get_member(client.user.id)
                    if member:
                        status = member.status
                        print(f'{name}: Status - {status}')
                        if status == discord.Status.online:
                            print(f'{name}: ✅ ONLINE')
                        elif status == discord.Status.offline:
                            print(f'{name}: ❌ OFFLINE')
                        elif status == discord.Status.idle:
                            print(f'{name}: 😴 IDLE')
                        elif status == discord.Status.dnd:
                            print(f'{name}: 🚫 DO NOT DISTURB')
                        else:
                            print(f'{name}: ❓ UNKNOWN STATUS')
                    else:
                        print(f'{name}: Not a member of the server')
                else:
                    print(f'{name}: Server not found')
            else:
                print(f'{name}: Failed to connect')

            await client.close()

        await client.start(token)
    except Exception as e:
        print(f'{name}: Error - {e}')

async def main():
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
            await check_bot_status(token, name)
            await asyncio.sleep(1)  # Small delay between checks
        else:
            print(f'{name}: No token found')

asyncio.run(main())