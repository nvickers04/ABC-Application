import discord
import os
import asyncio
from src.utils.vault_client import get_vault_secret

async def test_full_connection():
    """Test full Discord connection and guild access"""
    try:
        token = get_vault_secret('DISCORD_ORCHESTRATOR_TOKEN')
        guild_id = int(os.getenv('DISCORD_GUILD_ID', '0'))
        print(f"Token loaded, guild_id: {guild_id}")

        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True

        client = discord.Client(intents=intents)

        @client.event
        async def on_ready():
            print(f"✅ Connected as {client.user}")
            print(f"Guilds: {len(client.guilds)}")

            if guild_id:
                guild = client.get_guild(guild_id)
                if guild:
                    print(f"✅ Found target guild: {guild.name} (ID: {guild.id})")
                    print(f"Bot permissions in guild: {guild.me.guild_permissions}")

                    # Check channels
                    text_channels = [ch for ch in guild.text_channels]
                    print(f"Text channels: {len(text_channels)}")
                    for ch in text_channels[:5]:  # First 5
                        print(f"  - #{ch.name} (ID: {ch.id})")
                else:
                    print(f"❌ Guild {guild_id} not found!")
                    print(f"Available guilds: {[(g.name, g.id) for g in client.guilds]}")
            else:
                print("❌ No guild_id set")

            await client.close()

        @client.event
        async def on_error(event, *args, **kwargs):
            print(f"❌ Discord error: {event}")

        try:
            print("Starting client...")
            await client.start(token)
        except discord.LoginFailure:
            print("❌ Invalid token")
        except Exception as e:
            print(f"❌ Connection failed: {e}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_full_connection())