import discord
import asyncio
import os
from dotenv import load_dotenv
from src.utils.vault_client import get_vault_secret

load_dotenv()

GUILD_ID = int(get_vault_secret('DISCORD_GUILD_ID') or '0')

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
    # Use orchestrator token for all agents
    orchestrator_token = get_vault_secret('DISCORD_ORCHESTRATOR_TOKEN')

    if orchestrator_token:
        print("🔍 Checking Discord orchestrator bot status...")
        await check_bot_status(orchestrator_token, 'Orchestrator')
        print("✅ All agents use the same orchestrator token")
    else:
        print('❌ DISCORD_ORCHESTRATOR_TOKEN not found')

asyncio.run(main())