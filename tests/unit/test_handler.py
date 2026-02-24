"""Tests for Lambda handler."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ["GEMINI_API_KEY"] = "test-key"
os.environ["DYNAMODB_TABLE_NAME"] = "test-table"
os.environ["DYNAMODB_ENDPOINT"] = "http://localhost:8000"


def test_route_chat():
    from handler import route_event

    event = {
        "action": "chat",
        "userId": "USER#123",
        "message": "Hello",
    }
    action, params = route_event(event)
    assert action == "chat"
    assert params["user_id"] == "123"
    assert params["message"] == "Hello"


def test_route_save_preferences():
    from handler import route_event

    event = {
        "action": "save_preferences",
        "userId": "USER#123",
        "preferences": {"travel_styles": ["GOURMET"]},
    }
    action, params = route_event(event)
    assert action == "save_preferences"


def test_route_unknown_action():
    from handler import route_event

    event = {"action": "unknown"}
    action, params = route_event(event)
    assert action == "unknown"


def test_extract_user_id():
    from handler import _extract_user_id

    assert _extract_user_id("USER#123") == "123"
    assert _extract_user_id("123") == "123"


def test_route_plan_trip():
    from handler import route_event

    event = {
        "action": "plan_trip",
        "userId": "USER#456",
        "query": "Visit Tokyo for a week",
        "origin": "Osaka",
        "budget": "50000-100000",
    }
    action, params = route_event(event)
    assert action == "plan_trip"
    assert params["user_id"] == "456"
    assert params["query"] == "Visit Tokyo for a week"
    assert params["origin"] == "Osaka"
    assert params["budget"] == "50000-100000"


def test_route_plan_trip_registered():
    from handler import _HANDLERS

    assert "plan_trip" in _HANDLERS
