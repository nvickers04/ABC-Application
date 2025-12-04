#!/usr/bin/env python3
"""
Unit tests for LangChain 1.x memory patterns in the LearningAgent.
Tests InMemoryChatMessageHistory implementation for conversation persistence.
"""

import pytest
from unittest.mock import Mock


class TestLangChain1xMemory:
    """Test cases for LangChain 1.x memory functionality."""

    @pytest.fixture
    def langchain_memory(self):
        """Create LangChain 1.x memory instance for testing."""
        try:
            from langchain_core.chat_history import InMemoryChatMessageHistory
            return InMemoryChatMessageHistory()
        except ImportError:
            pytest.skip("LangChain 1.x not available")

    @pytest.fixture
    def message_classes(self):
        """Get LangChain 1.x message classes for testing."""
        try:
            from langchain_core.messages import HumanMessage, AIMessage
            return {"HumanMessage": HumanMessage, "AIMessage": AIMessage}
        except ImportError:
            pytest.skip("LangChain 1.x message classes not available")

    def test_memory_initialization(self, langchain_memory):
        """Test InMemoryChatMessageHistory initialization."""
        assert langchain_memory is not None
        assert len(langchain_memory.messages) == 0

    def test_add_user_message(self, langchain_memory, message_classes):
        """Test adding user messages to memory."""
        langchain_memory.add_user_message("Hello, what's the market outlook?")
        
        assert len(langchain_memory.messages) == 1
        assert isinstance(langchain_memory.messages[0], message_classes["HumanMessage"])
        assert "market outlook" in langchain_memory.messages[0].content.lower()

    def test_add_ai_message(self, langchain_memory, message_classes):
        """Test adding AI messages to memory."""
        langchain_memory.add_ai_message("The market outlook is bullish with positive momentum.")
        
        assert len(langchain_memory.messages) == 1
        assert isinstance(langchain_memory.messages[0], message_classes["AIMessage"])
        assert "bullish" in langchain_memory.messages[0].content.lower()

    def test_conversation_flow(self, langchain_memory, message_classes):
        """Test a complete conversation flow."""
        # Add user message
        langchain_memory.add_user_message("What is the current risk level?")
        # Add AI response
        langchain_memory.add_ai_message("Current risk indicators are moderate. VaR is at 5%.")
        # Add follow-up
        langchain_memory.add_user_message("Should we adjust position sizes?")
        langchain_memory.add_ai_message("Yes, consider reducing position sizes by 10%.")
        
        assert len(langchain_memory.messages) == 4
        
        # Verify message types alternate
        assert isinstance(langchain_memory.messages[0], message_classes["HumanMessage"])
        assert isinstance(langchain_memory.messages[1], message_classes["AIMessage"])
        assert isinstance(langchain_memory.messages[2], message_classes["HumanMessage"])
        assert isinstance(langchain_memory.messages[3], message_classes["AIMessage"])

    def test_clear_memory(self, langchain_memory):
        """Test clearing conversation memory."""
        # Add some messages
        langchain_memory.add_user_message("Test message 1")
        langchain_memory.add_ai_message("Test response 1")
        langchain_memory.add_user_message("Test message 2")
        
        assert len(langchain_memory.messages) == 3
        
        # Clear memory
        langchain_memory.clear()
        
        # Verify memory is empty
        assert len(langchain_memory.messages) == 0

    def test_get_conversation_history_format(self, langchain_memory, message_classes):
        """Test formatting conversation history as readable string."""
        # Add test conversation
        langchain_memory.add_user_message("Analyze AAPL")
        langchain_memory.add_ai_message("AAPL shows strong momentum. Consider a long position.")
        
        # Format like learning.py get_conversation_history method
        messages = langchain_memory.messages
        history_text = ""
        for msg in messages[-10:]:
            if isinstance(msg, message_classes["HumanMessage"]):
                history_text += f"Human: {msg.content}\n"
            elif isinstance(msg, message_classes["AIMessage"]):
                history_text += f"Agent: {msg.content}\n"
        
        history = history_text.strip()
        
        assert "Human: Analyze AAPL" in history
        assert "Agent: AAPL shows strong momentum" in history

    def test_message_limit_in_history(self, langchain_memory, message_classes):
        """Test retrieving last N messages from history."""
        # Add 15 message pairs (30 total)
        for i in range(15):
            langchain_memory.add_user_message(f"Question {i}")
            langchain_memory.add_ai_message(f"Answer {i}")
        
        assert len(langchain_memory.messages) == 30
        
        # Get only last 10 messages like learning.py does
        messages = langchain_memory.messages[-10:]
        
        assert len(messages) == 10
        # Last 10 should be from questions 10-14
        assert "Question 10" in messages[0].content


class TestLearningAgentMemoryIntegration:
    """Test LangChain 1.x memory integration with LearningAgent-like behavior."""

    @pytest.fixture
    def agent_with_memory(self):
        """Create a mock agent with LangChain 1.x memory."""
        try:
            from langchain_core.chat_history import InMemoryChatMessageHistory
            from langchain_core.messages import HumanMessage, AIMessage
            
            class MockLearningAgent:
                def __init__(self):
                    self.conversation_memory = InMemoryChatMessageHistory()
                    self.memory_initialized = True
                    self.HumanMessage = HumanMessage
                    self.AIMessage = AIMessage
                
                def add_to_conversation_memory(self, user_input: str, agent_response: str):
                    """Add conversation to LangChain 1.x memory."""
                    if self.memory_initialized and self.conversation_memory:
                        self.conversation_memory.add_user_message(user_input)
                        self.conversation_memory.add_ai_message(agent_response)
                
                def get_conversation_history(self) -> str:
                    """Retrieve conversation history."""
                    if self.memory_initialized and self.conversation_memory:
                        messages = self.conversation_memory.messages
                        if messages:
                            history_text = ""
                            for msg in messages[-10:]:
                                if isinstance(msg, self.HumanMessage):
                                    history_text += f"Human: {msg.content}\n"
                                elif isinstance(msg, self.AIMessage):
                                    history_text += f"Agent: {msg.content}\n"
                            return history_text.strip()
                    return "No conversation history available"
                
                def clear_conversation_memory(self):
                    """Clear conversation memory."""
                    if self.memory_initialized and self.conversation_memory:
                        self.conversation_memory.clear()
            
            return MockLearningAgent()
        except ImportError:
            pytest.skip("LangChain 1.x not available")

    def test_agent_add_conversation(self, agent_with_memory):
        """Test adding conversation through agent interface."""
        agent_with_memory.add_to_conversation_memory(
            "What strategy should we use for NVDA?",
            "Based on momentum indicators, consider a covered call strategy."
        )
        
        history = agent_with_memory.get_conversation_history()
        
        assert "NVDA" in history
        assert "covered call" in history.lower()

    def test_agent_multiple_conversations(self, agent_with_memory):
        """Test multiple conversation turns."""
        agent_with_memory.add_to_conversation_memory(
            "Analyze the tech sector",
            "Tech sector showing strength with XLK up 2%."
        )
        agent_with_memory.add_to_conversation_memory(
            "What about risk exposure?",
            "Current portfolio has 35% tech exposure. Consider diversifying."
        )
        
        history = agent_with_memory.get_conversation_history()
        
        assert "Human: Analyze the tech sector" in history
        assert "Agent: Tech sector showing strength" in history
        assert "Human: What about risk exposure" in history
        assert "diversifying" in history.lower()

    def test_agent_clear_history(self, agent_with_memory):
        """Test clearing conversation through agent interface."""
        agent_with_memory.add_to_conversation_memory("Test", "Response")
        agent_with_memory.clear_conversation_memory()
        
        history = agent_with_memory.get_conversation_history()
        
        assert history == "No conversation history available"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
