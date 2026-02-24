"""Tests for recommendation agent."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-key")

from travel_planner.agents.recommendation import RecommendationAgent
from travel_planner.data.preferences import (
    CuisineType,
    TravelStyle,
    UserPreferences,
)


@pytest.fixture
def mock_genai():
    with patch("travel_planner.agents.base.genai") as mock:
        mock_client = MagicMock()
        mock.Client.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = '{"places": [{"name": "Tsukiji Market"}]}'
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )
        yield mock_client


def test_recommendation_agent_init(mock_genai):
    agent = RecommendationAgent()
    assert agent.name == "Recommendation Agent"


async def test_recommend_with_preferences(mock_genai):
    prefs = UserPreferences(
        travel_styles=[TravelStyle.GOURMET],
        cuisine_types=[CuisineType.JAPANESE, CuisineType.SEAFOOD],
    )
    agent = RecommendationAgent()
    result = await agent.recommend(
        preferences=prefs,
        location={"lat": 35.6812, "lng": 139.7671},
        category="restaurant",
    )
    assert result is not None
    mock_genai.aio.models.generate_content.assert_called_once()
