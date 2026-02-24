"""Tests for DynamoDB repository."""

from unittest.mock import MagicMock

import pytest

from travel_planner.data.conversation_models import (
    Conversation,
    Message,
    MessageRole,
    User,
)
from travel_planner.data.preferences import (
    CuisineType,
    TravelStyle,
    UserPreferences,
)
from travel_planner.data.repository import DynamoDBRepository


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def repo(mock_db):
    return DynamoDBRepository(mock_db)


def test_save_user(repo, mock_db):
    user = User(
        user_id="123",
        email="test@example.com",
        name="Test",
    )
    repo.save_user(user)
    mock_db.put_item.assert_called_once()
    item = mock_db.put_item.call_args[0][0]
    assert item["PK"] == "USER#123"
    assert item["SK"] == "PROFILE"
    assert item["EntityType"] == "User"


def test_get_user(repo, mock_db):
    mock_db.get_item.return_value = {
        "PK": "USER#123",
        "SK": "PROFILE",
        "Data": {
            "user_id": "123",
            "email": "test@example.com",
            "name": "Test",
        },
        "EntityType": "User",
        "Version": 1,
    }
    user = repo.get_user("123")
    assert user is not None
    assert user.user_id == "123"
    mock_db.get_item.assert_called_with("USER#123", "PROFILE")


def test_get_user_not_found(repo, mock_db):
    mock_db.get_item.return_value = None
    user = repo.get_user("999")
    assert user is None


def test_save_preferences(repo, mock_db):
    prefs = UserPreferences(
        travel_styles=[TravelStyle.GOURMET],
        cuisine_types=[CuisineType.JAPANESE],
    )
    repo.save_preferences("123", prefs)
    mock_db.put_item.assert_called_once()
    item = mock_db.put_item.call_args[0][0]
    assert item["PK"] == "USER#123"
    assert item["SK"] == "PREFERENCES"


def test_get_preferences(repo, mock_db):
    mock_db.get_item.return_value = {
        "PK": "USER#123",
        "SK": "PREFERENCES",
        "Data": {
            "travel_styles": ["GOURMET"],
            "cuisine_types": ["JAPANESE"],
        },
    }
    prefs = repo.get_preferences("123")
    assert prefs is not None
    assert TravelStyle.GOURMET in prefs.travel_styles


def test_save_conversation(repo, mock_db):
    conv = Conversation(
        conversation_id="789",
        user_id="123",
        title="Lunch",
    )
    repo.save_conversation(conv)
    item = mock_db.put_item.call_args[0][0]
    assert item["PK"] == "USER#123#CONVERSATION"
    assert item["SK"] == "CONV#789"


def test_list_conversations(repo, mock_db):
    mock_db.query.return_value = [
        {
            "PK": "USER#123#CONVERSATION",
            "SK": "CONV#1",
            "Data": {
                "conversation_id": "1",
                "user_id": "123",
                "title": "Lunch",
            },
        },
    ]
    convs = repo.list_conversations("123")
    assert len(convs) == 1
    mock_db.query.assert_called_with(pk="USER#123#CONVERSATION")


def test_save_message(repo, mock_db):
    msg = Message(
        conversation_id="789",
        sequence=1,
        role=MessageRole.USER,
        content="Hello",
    )
    repo.save_message(msg)
    item = mock_db.put_item.call_args[0][0]
    assert item["PK"] == "CONVERSATION#789#MESSAGE"
    assert item["SK"] == "000001"


def test_get_messages(repo, mock_db):
    mock_db.query.return_value = [
        {
            "PK": "CONVERSATION#789#MESSAGE",
            "SK": "000001",
            "Data": {
                "conversation_id": "789",
                "sequence": 1,
                "role": "user",
                "content": "Hello",
            },
        },
    ]
    msgs = repo.get_messages("789")
    assert len(msgs) == 1
    mock_db.query.assert_called_with(pk="CONVERSATION#789#MESSAGE")
