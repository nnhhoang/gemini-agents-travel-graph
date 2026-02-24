"""Tests for log anonymization."""

from travel_planner.services.anonymizer import anonymize


def test_anonymize_email():
    text = "User email is john@example.com and he logged in"
    result = anonymize(text)
    assert "john@example.com" not in result
    assert "[EMAIL]" in result


def test_anonymize_phone():
    text = "Call us at 090-1234-5678 for support"
    result = anonymize(text)
    assert "090-1234-5678" not in result
    assert "[PHONE]" in result


def test_anonymize_no_pii():
    text = "User visited Tokyo Tower"
    result = anonymize(text)
    assert result == text


def test_anonymize_multiple():
    text = "john@test.com called 090-1234-5678"
    result = anonymize(text)
    assert "[EMAIL]" in result
    assert "[PHONE]" in result
