"""
Mock implementations for Google Gemini to support testing.

This module provides mock classes and functions to simulate Gemini functionality
in tests without requiring the actual google-genai package. It should be imported
before any modules that depend on Google Gemini.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock


class MockGenerateContentResponse:
    """Mock for Gemini generate_content response."""

    def __init__(self, text="This is a mock response"):
        self.text = text


class MockAioModels:
    """Mock for client.aio.models."""

    def __init__(self):
        self.generate_content = AsyncMock(
            return_value=MockGenerateContentResponse()
        )


class MockAio:
    """Mock for client.aio."""

    def __init__(self):
        self.models = MockAioModels()


class MockGeminiClient(MagicMock):
    """Mock for the google.genai.Client."""

    def __init__(self, **kwargs):
        """
        Initialize mock Gemini client, ignoring API key requirements
        and all other parameters.
        """
        super().__init__()
        self.aio = MockAio()


class MockContent:
    """Mock for google.genai.types.Content."""

    def __init__(self, role="user", parts=None):
        self.role = role
        self.parts = parts or []


class MockPart:
    """Mock for google.genai.types.Part."""

    def __init__(self, text=""):
        self.text = text

    @staticmethod
    def from_text(text=""):
        return MockPart(text=text)


class MockGenerateContentConfig:
    """Mock for google.genai.types.GenerateContentConfig."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def setup_mock_gemini():
    """Set up the mock Gemini modules and classes."""
    # Set up environment for mocking
    if "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = "mock-gemini-api-key-for-testing"

    # Create mock types module
    mock_types = MagicMock()
    mock_types.Content = MockContent
    mock_types.Part = MockPart
    mock_types.Part.from_text = MockPart.from_text
    mock_types.GenerateContentConfig = MockGenerateContentConfig

    # Create mock genai module
    mock_genai = MagicMock()
    mock_genai.Client = MockGeminiClient
    mock_genai.types = mock_types

    # Create mock google module
    mock_google = MagicMock()
    mock_google.genai = mock_genai

    # Register in sys.modules to override any imports
    sys.modules["google"] = mock_google
    sys.modules["google.genai"] = mock_genai
    sys.modules["google.genai.types"] = mock_types

    return mock_genai
