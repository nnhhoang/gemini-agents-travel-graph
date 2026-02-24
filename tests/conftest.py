"""
Pytest configuration for the Travel Planner system tests.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

# Register asyncio marker
pytest.importorskip("pytest_asyncio")
pytest.mark.asyncio = pytest.mark.asyncio

# Import project modules after configuring pytest
from travel_planner.agents.base import AgentConfig  # noqa: E402
from travel_planner.config import (  # noqa: E402
    APIConfig,
    SystemConfig,
    TravelPlannerConfig,
)
from travel_planner.utils import LogLevel, setup_logging  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Set up logging for tests."""
    setup_logging(LogLevel.DEBUG)


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client for testing."""
    mock_client = MagicMock()

    # Mock the aio.models.generate_content method
    mock_response = MagicMock()
    mock_response.text = "Test response"

    mock_client.aio = MagicMock()
    mock_client.aio.models = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    return mock_client


@pytest.fixture
def test_agent_config():
    """Test agent configuration."""
    return AgentConfig(
        name="Test Agent",
        instructions="You are a test agent",
        model="gemini-2.5-flash",
        temperature=0.5,
    )


@pytest.fixture
def test_config():
    """Test application configuration."""
    return TravelPlannerConfig(
        api=APIConfig(
            gemini_api_key="test-key",
            aws_region="ap-northeast-1",
            dynamodb_table_name="travel-planner-test",
            tavily_api_key="test-key",
            firecrawl_api_key="test-key",
        ),
        system=SystemConfig(
            log_level=LogLevel.DEBUG,
            environment="test",
            max_concurrency=2,
            default_budget=1000,
            default_currency="USD",
        ),
    )
