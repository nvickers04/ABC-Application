#!/usr/bin/env python3
"""
Quick Workflow Test - Test individual phases of the iterative reasoning workflow
"""

import asyncio
import discord
import os
from dotenv import load_dotenv

load_dotenv()

async def test_macro_foundation():
    """Test just the macro foundation phase"""
    print("Testing Macro Foundation Phase...")

    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True

    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f"Connected as {client.user}")

        guild = client.get_guild(int(os.getenv('DISCORD_GUILD_ID', '0')))
        if guild:
            channel = None
            for ch in guild.text_channels:
                if ch.permissions_for(guild.me).send_messages:
                    channel = ch
                    break

            if channel:
                print(f"Sending macro foundation command to #{channel.name}")
                await channel.send("!m analyze Assess current market regime, volatility levels, and macroeconomic trends. Identify top 5 sectors/assets with highest relative strength, momentum, and risk-adjusted returns.")
                print("✅ Command sent! Check Discord for response.")
            else:
                print("❌ No suitable channel found")
        else:
            print("❌ Guild not found")

        await client.close()

    token = os.getenv('DISCORD_MACRO_AGENT_TOKEN')
    if token:
        await client.start(token)
    else:
        print("❌ No token found")

async def test_debate_phase():
    """Test the debate phase"""
    print("Testing Debate Phase...")

    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True

    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f"Connected as {client.user}")

        guild = client.get_guild(int(os.getenv('DISCORD_GUILD_ID', '0')))
        if guild:
            channel = None
            for ch in guild.text_channels:
                if ch.permissions_for(guild.me).send_messages:
                    channel = ch
                    break

            if channel:
                print(f"Sending debate command to #{channel.name}")
                await channel.send('!m debate "What are the current market opportunities and risks?" strategy risk reflection')
                print("✅ Debate started! Check Discord for multi-agent discussion.")
            else:
                print("❌ No suitable channel found")
        else:
            print("❌ Guild not found")

        await client.close()

    token = os.getenv('DISCORD_MACRO_AGENT_TOKEN')
    if token:
        await client.start(token)
    else:
        print("❌ No token found")

async def test_reflection_oversight():
    """Test reflection agent supreme oversight"""
    print("Testing Reflection Agent Supreme Oversight...")

    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True

    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f"Connected as {client.user}")

        guild = client.get_guild(int(os.getenv('DISCORD_GUILD_ID', '0')))
        if guild:
            channel = None
            for ch in guild.text_channels:
                if ch.permissions_for(guild.me).send_messages:
                    channel = ch
                    break

            if channel:
                print(f"Sending supreme oversight command to #{channel.name}")
                await channel.send("!ref analyze Conduct comprehensive audit of recent analysis and render final decision with veto authority if concerning patterns emerge")
                print("✅ Supreme oversight initiated! Check Discord for final judgment.")
            else:
                print("❌ No suitable channel found")
        else:
            print("❌ Guild not found")

        await client.close()

    token = os.getenv('DISCORD_MACRO_AGENT_TOKEN')
    if token:
        await client.start(token)
    else:
        print("❌ No token found")

def main():
    print("🤖 Iterative Reasoning Workflow - Quick Tests")
    print("=" * 50)
    print("Choose a test:")
    print("1. Macro Foundation (Phase 0)")
    print("2. Debate Phase (Phase 3)")
    print("3. Reflection Oversight (Supreme)")
    print("4. Run Complete Workflow (Advanced)")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        asyncio.run(test_macro_foundation())
    elif choice == "2":
        asyncio.run(test_debate_phase())
    elif choice == "3":
        asyncio.run(test_reflection_oversight())
    elif choice == "4":
        print("Running complete workflow...")
        # Import and run the full workflow
        import subprocess
        import sys
        try:
            result = subprocess.run([sys.executable, "iterative_reasoning_workflow.py"],
                                  capture_output=True, text=True, cwd=os.getcwd())
            print("Workflow completed!")
            print("Output:", result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
        except Exception as e:
            print(f"Error running workflow: {e}")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()