#!/usr/bin/env python3
"""
Test LLM initialization
"""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def test_grok():
    try:
        from langchain_xai import ChatXAI
        from langchain_core.messages import HumanMessage

        api_key = os.getenv('GROK_API_KEY')
        if not api_key:
            print("❌ No GROK_API_KEY found")
            return

        print(f"🔑 API Key found (length: {len(api_key)})")

        # Initialize Grok
        from langchain_core.utils import convert_to_secret_str
        chat = ChatXAI(
            api_key=convert_to_secret_str(api_key),
            model="grok-4-fast-reasoning",
            temperature=0.1,
            max_tokens=4096
        )

        print("🤖 Initialized ChatXAI client")

        # Test the connection
        message = HumanMessage(content="Respond with 'Grok is working!' if you can understand this message.")
        response = await chat.ainvoke([message])

        print(f"✅ Response received: {response.content}")

        if "Grok is working" in response.content:
            print("🎉 Grok LLM is working correctly!")
        else:
            print("⚠️ Grok responded but not as expected")

    except Exception as e:
        print(f"❌ Error testing Grok: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_grok())