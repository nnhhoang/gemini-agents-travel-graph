"""Tests for user preference models."""

from travel_planner.data.preferences import (
    ActivityInterest,
    ActivityStyle,
    BeverageType,
    BudgetPreference,
    CuisineType,
    DietaryRestriction,
    TravelFrequency,
    TravelPurpose,
    TravelStyle,
    UserPreferences,
    _ja,
)


def test_travel_frequency_values():
    assert TravelFrequency.YEAR_ONCE == "YEAR_ONCE"
    assert TravelFrequency.CASUAL == "CASUAL"


def test_cuisine_type_values():
    assert CuisineType.JAPANESE == "JAPANESE"
    assert CuisineType.STREET_FOOD == "STREET_FOOD"
    assert CuisineType.B_GRADE == "B_GRADE"


def test_user_preferences_defaults():
    prefs = UserPreferences()
    assert prefs.travel_frequency is None
    assert prefs.travel_styles == []
    assert prefs.cuisine_types == []
    assert prefs.activity_interests == []
    assert prefs.custom_notes is None


def test_user_preferences_with_data():
    prefs = UserPreferences(
        travel_frequency=TravelFrequency.MONTH_ONCE,
        travel_styles=[TravelStyle.GOURMET, TravelStyle.NATURE],
        cuisine_types=[CuisineType.JAPANESE, CuisineType.SEAFOOD],
        budget_preference=BudgetPreference.BALANCED,
        dietary_restrictions=[DietaryRestriction.NO_SPICY],
        activity_interests=[ActivityInterest.PHOTO, ActivityInterest.ONSEN],
    )
    assert prefs.travel_frequency == TravelFrequency.MONTH_ONCE
    assert len(prefs.travel_styles) == 2
    assert CuisineType.SEAFOOD in prefs.cuisine_types


def test_user_preferences_serialization():
    prefs = UserPreferences(
        travel_styles=[TravelStyle.SOLO],
        cuisine_types=[CuisineType.LOCAL],
        beverage_types=[BeverageType.SAKE],
    )
    data = prefs.model_dump()
    assert data["travel_styles"] == ["SOLO"]
    assert data["cuisine_types"] == ["LOCAL"]
    restored = UserPreferences.model_validate(data)
    assert restored.travel_styles == [TravelStyle.SOLO]


def test_user_preferences_to_prompt_context():
    prefs = UserPreferences(
        travel_styles=[TravelStyle.GOURMET],
        cuisine_types=[CuisineType.JAPANESE, CuisineType.SEAFOOD],
        budget_preference=BudgetPreference.REASONABLE,
        dietary_restrictions=[DietaryRestriction.NO_SPICY],
        activity_interests=[ActivityInterest.ONSEN, ActivityInterest.PHOTO],
    )
    ctx = prefs.to_prompt_context()
    assert "gourmet" in ctx
    assert "Japanese" in ctx
    assert "seafood" in ctx
    assert "no spicy food" in ctx
    assert "hot springs" in ctx
    assert "photography" in ctx


def test_user_preferences_to_prompt_context_all_fields():
    prefs = UserPreferences(
        travel_frequency=TravelFrequency.MONTH_ONCE,
        travel_styles=[TravelStyle.GOURMET],
        travel_purposes=[TravelPurpose.HEALING],
        activity_style=ActivityStyle.QUIET,
        cuisine_types=[CuisineType.JAPANESE],
        budget_preference=BudgetPreference.REASONABLE,
    )
    ctx = prefs.to_prompt_context()
    assert "Travel frequency" in ctx
    assert "once a month" in ctx
    assert "Travel purpose" in ctx
    assert "relaxation" in ctx
    assert "Activity style" in ctx
    assert "relaxed" in ctx


def test_user_preferences_to_prompt_context_empty():
    prefs = UserPreferences()
    ctx = prefs.to_prompt_context()
    assert ctx == "No preferences set"


def test_ja_label_mapping():
    assert _ja(TravelStyle.GOURMET) == "gourmet"
    assert _ja(CuisineType.JAPANESE) == "Japanese"
    assert _ja(BudgetPreference.LUXURY) == "luxury"
    assert _ja(DietaryRestriction.HALAL) == "halal"
    assert _ja(ActivityInterest.ONSEN) == "hot springs (onsen)"


def test_ja_handles_overlapping_values():
    """CuisineType.LOCAL and DiningStyle.LOCAL have different labels."""
    from travel_planner.data.preferences import DiningStyle

    assert _ja(CuisineType.LOCAL) == "regional specialties"
    assert _ja(DiningStyle.LOCAL) == "local favorites"
