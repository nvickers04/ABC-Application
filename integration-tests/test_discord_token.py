import discord
import os
import asyncio
from src.utils.vault_client import get_vault_secret

async def test_token():
    """Test if Discord token is valid"""
    try:
        token = get_vault_secret('DISCORD_ORCHESTRATOR_TOKEN')
        print(f"Token loaded, length: {len(token)}")

        # Try to create a client and test login
        intents = discord.Intents.default()
        client = discord.Client(intents=intents)

        @client.event
        async def on_ready():
            print(f"✅ Token is valid! Logged in as {client.user}")
            await client.close()

        @client.event
        async def on_error(event, *args, **kwargs):
            print(f"❌ Discord error: {event}")

        try:
            await client.start(token)
        except discord.LoginFailure:
            print("❌ Invalid token - LoginFailure")
        except Exception as e:
            print(f"❌ Token validation failed: {e}")

    except Exception as e:
        print(f"❌ Error getting token: {e}")

if __name__ == "__main__":
    asyncio.run(test_token())