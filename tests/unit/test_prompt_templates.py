"""Tests for prompt template management."""

from travel_planner.prompts.templates import PromptTemplate, render_template


def test_render_template_basic():
    template = "Hello {name}, welcome to {place}!"
    result = render_template(template, name="Tanaka", place="Tokyo")
    assert result == "Hello Tanaka, welcome to Tokyo!"


def test_render_template_with_preferences():
    template = (
        "User preferences: {preferences}\n"
        "Location: {location}\n"
        "Question: {message}"
    )
    result = render_template(
        template,
        preferences="Cuisine: JAPANESE, SEAFOOD; Budget: REASONABLE",
        location="Shibuya, Tokyo",
        message="Where should I eat?",
    )
    assert "JAPANESE" in result
    assert "Shibuya" in result


def test_render_template_missing_var():
    template = "Hello {name}, you are in {place}!"
    result = render_template(template, name="Tanaka")
    assert "{place}" in result  # unresolved vars stay as-is


def test_prompt_template_model():
    pt = PromptTemplate(
        template_id="recommend_spot",
        version=1,
        template="Recommend a {category} near {location}",
        status="active",
    )
    assert pt.pk == "PROMPT#recommend_spot"
    assert pt.sk == "VERSION#001"
    assert pt.is_active


def test_prompt_template_render():
    pt = PromptTemplate(
        template_id="recommend_spot",
        version=1,
        template="Recommend a {category} near {location}",
    )
    result = pt.render(category="restaurant", location="Shinjuku")
    assert result == "Recommend a restaurant near Shinjuku"
