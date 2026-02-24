"""Tests for context builder."""

from travel_planner.data.conversation_models import Message, MessageRole
from travel_planner.data.preferences import (
    BudgetPreference,
    CuisineType,
    TravelStyle,
    UserPreferences,
)
from travel_planner.prompts.context import ContextBuilder


def test_build_context_minimal():
    builder = ContextBuilder()
    ctx = builder.build(message="Where should I eat?")
    assert "Where should I eat?" in ctx["message"]
    assert ctx["preferences_text"] == "No preferences set"


def test_build_context_with_preferences():
    prefs = UserPreferences(
        travel_styles=[TravelStyle.GOURMET],
        cuisine_types=[CuisineType.JAPANESE, CuisineType.SEAFOOD],
        budget_preference=BudgetPreference.REASONABLE,
    )
    builder = ContextBuilder()
    ctx = builder.build(message="Recommend lunch", preferences=prefs)
    assert "gourmet" in ctx["preferences_text"]
    assert "Japanese" in ctx["preferences_text"]


def test_build_context_with_location():
    builder = ContextBuilder()
    ctx = builder.build(
        message="What's nearby?",
        location={"lat": 35.6812, "lng": 139.7671},
    )
    assert ctx["location"] == "lat=35.6812, lng=139.7671"


def test_build_context_with_history():
    history = [
        Message(
            conversation_id="1",
            sequence=1,
            role=MessageRole.USER,
            content="I like sushi",
        ),
        Message(
            conversation_id="1",
            sequence=2,
            role=MessageRole.ASSISTANT,
            content="Great choice!",
        ),
    ]
    builder = ContextBuilder()
    ctx = builder.build(message="What else?", history=history)
    assert len(ctx["history"]) == 2
    assert ctx["history"][0]["role"] == "user"


def test_build_context_time_awareness():
    builder = ContextBuilder()
    ctx = builder.build(
        message="What to do?",
        timestamp="2026-01-10T08:30:00Z",
    )
    assert "time_of_day" in ctx
    assert "day_of_week" in ctx


def test_build_system_prompt():
    prefs = UserPreferences(
        cuisine_types=[CuisineType.JAPANESE],
    )
    builder = ContextBuilder()
    prompt = builder.build_system_prompt(
        preferences=prefs,
        location={"lat": 35.6812, "lng": 139.7671},
        timestamp="2026-01-10T12:00:00Z",
    )
    assert "Japanese" in prompt
    assert "35.6812" in prompt
    assert "Trip" in prompt
    assert "<rules>" in prompt


def test_build_system_prompt_sections():
    prefs = UserPreferences(
        travel_styles=[TravelStyle.NATURE],
        cuisine_types=[CuisineType.LOCAL],
    )
    builder = ContextBuilder()
    prompt = builder.build_system_prompt(
        preferences=prefs,
        location={"lat": 34.6937, "lng": 135.5023},
        timestamp="2026-01-10T19:00:00Z",
    )
    assert "<user_preferences>" in prompt
    assert "<real_time_context>" in prompt
    assert "<thinking_process>" in prompt
    assert "<response_format>" in prompt
    assert "<rules>" in prompt
    assert "evening" in prompt


def test_build_system_prompt_has_preference_usage_instructions():
    prefs = UserPreferences(
        travel_styles=[TravelStyle.GOURMET],
        cuisine_types=[CuisineType.SEAFOOD],
    )
    builder = ContextBuilder()
    prompt = builder.build_system_prompt(preferences=prefs)
    assert "NEVER recommend something that violates dietary restrictions" in prompt
    assert "hidden gems" in prompt.lower() or "hard constraints" in prompt


def test_build_system_prompt_has_anti_hallucination():
    builder = ContextBuilder()
    prompt = builder.build_system_prompt()
    assert "NEVER invent" in prompt
    assert "verify" in prompt.lower()


def test_build_system_prompt_no_location():
    prefs = UserPreferences(
        travel_styles=[TravelStyle.GOURMET],
    )
    builder = ContextBuilder()
    prompt = builder.build_system_prompt(preferences=prefs)
    assert "Trip" in prompt
    assert "<user_preferences>" in prompt
    assert "<real_time_context>" not in prompt
    assert "<rules>" in prompt
