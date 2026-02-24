"""Tests for conversation domain models."""

from travel_planner.data.conversation_models import (
    Conversation,
    Message,
    MessageRole,
    Place,
    Session,
    User,
)


def test_user_model():
    user = User(
        user_id="123",
        email="test@example.com",
        name="Test User",
        provider="google",
        account_type="member",
    )
    assert user.user_id == "123"
    assert user.pk == "USER#123"
    assert user.sk == "PROFILE"


def test_session_model():
    session = Session(
        session_id="789",
        user_id="123",
        device_id="abc",
    )
    assert session.pk == "SESSION#789"
    assert session.sk == "METADATA"
    assert session.gsi1pk == "USER#123#SESSION"


def test_conversation_model():
    conv = Conversation(
        conversation_id="789",
        user_id="123",
        title="Lunch recommendations",
    )
    assert conv.pk == "USER#123#CONVERSATION"
    assert conv.sk == "CONV#789"


def test_message_model():
    msg = Message(
        conversation_id="789",
        sequence=1,
        role=MessageRole.USER,
        content="Where should I eat?",
    )
    assert msg.pk == "CONVERSATION#789#MESSAGE"
    assert msg.sk == "000001"
    assert msg.role == MessageRole.USER


def test_message_sequence_padding():
    msg = Message(
        conversation_id="789",
        sequence=42,
        role=MessageRole.ASSISTANT,
        content="I recommend...",
    )
    assert msg.sk == "000042"


def test_place_model():
    place = Place(
        place_id="456",
        name="Restaurant ABC",
        category="food",
        lat=35.6812,
        lng=139.7671,
    )
    assert place.pk == "PLACE#456"
    assert place.sk == "METADATA"
    assert place.gsi1pk.startswith("GEOHASH#")
