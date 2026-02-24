"""Tests for content moderation."""

import pytest

from travel_planner.prompts.moderation import (
    ModerationResult,
    moderate_input,
    moderate_output,
)


def test_valid_input():
    result = moderate_input("Where should I eat in Tokyo?")
    assert result.is_safe
    assert result.reason is None


def test_empty_input():
    result = moderate_input("")
    assert not result.is_safe
    assert "empty" in result.reason.lower()


def test_too_long_input():
    result = moderate_input("a" * 5001)
    assert not result.is_safe
    assert "long" in result.reason.lower()


def test_valid_output():
    result = moderate_output("I recommend Tsukiji Market for fresh sushi!")
    assert result.is_safe


def test_output_with_pii_email():
    result = moderate_output("Contact them at user@example.com for booking")
    assert not result.is_safe
    assert "PII" in result.reason
