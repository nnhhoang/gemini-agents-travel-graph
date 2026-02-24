"""Tests for conversation agent."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Set env var before imports
os.environ.setdefault("GEMINI_API_KEY", "test-key")

from travel_planner.agents.base import AgentConfig
from travel_planner.agents.conversation import ConversationAgent


@pytest.fixture
def mock_genai():
    with patch("travel_planner.agents.base.genai") as mock:
        mock_client = MagicMock()
        mock.Client.return_value = mock_client

        # Mock async generate_content
        mock_response = MagicMock()
        mock_response.text = "I recommend trying the local ramen shop nearby."
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )
        yield mock_client


def test_conversation_agent_init(mock_genai):
    agent = ConversationAgent()
    assert agent.name == "Conversation Agent"
    assert "tourism guide" in agent.instructions.lower()


async def test_conversation_agent_chat(mock_genai):
    agent = ConversationAgent()
    result = await agent.chat(
        message="Where should I eat?",
        system_prompt="You are a tourism guide.",
        history=[],
    )
    assert result is not None
    assert isinstance(result, str)
    mock_genai.aio.models.generate_content.assert_called_once()
