"""Tests for conversation service."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-key")

from travel_planner.data.conversation_models import MessageRole
from travel_planner.data.preferences import CuisineType, UserPreferences
from travel_planner.services.conversation_service import ConversationService


@pytest.fixture
def mock_repo():
    repo = MagicMock()
    repo.get_preferences.return_value = UserPreferences(
        cuisine_types=[CuisineType.JAPANESE],
    )
    repo.get_messages.return_value = []
    repo.get_conversation.return_value = None
    return repo


@pytest.fixture
def mock_agent():
    with patch("travel_planner.agents.base.genai"):
        agent = MagicMock()
        agent.chat = AsyncMock(return_value="Try the ramen at Ichiran!")
        return agent


@pytest.fixture
def service(mock_repo, mock_agent):
    return ConversationService(repo=mock_repo, agent=mock_agent)


async def test_handle_chat_new_conversation(service, mock_repo):
    result = await service.handle_chat(
        user_id="123",
        message="Where should I eat?",
    )
    assert result["response"] == "Try the ramen at Ichiran!"
    assert "conversation_id" in result
    assert "message_id" in result
    # Should save user message + assistant message
    assert mock_repo.save_message.call_count == 2


async def test_handle_chat_existing_conversation(service, mock_repo):
    result = await service.handle_chat(
        user_id="123",
        message="What about sushi?",
        conversation_id="existing-conv",
    )
    assert result["conversation_id"] == "existing-conv"


async def test_handle_chat_loads_preferences(service, mock_repo):
    await service.handle_chat(
        user_id="123",
        message="Recommend lunch",
    )
    mock_repo.get_preferences.assert_called_with("123")
