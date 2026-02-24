"""
Content moderation for AI conversation inputs and outputs.

Layered approach:
1. Gemini's built-in safety filters (handled by SDK)
2. Input validation (length, empty check)
3. Output check (PII detection)
"""

import re

from pydantic import BaseModel

MAX_INPUT_LENGTH = 5000


class ModerationResult(BaseModel):
    is_safe: bool
    reason: str | None = None


def moderate_input(text: str) -> ModerationResult:
    """Validate user input before sending to Gemini."""
    if not text or not text.strip():
        return ModerationResult(is_safe=False, reason="Input is empty")

    if len(text) > MAX_INPUT_LENGTH:
        return ModerationResult(
            is_safe=False,
            reason=f"Input too long ({len(text)} chars, max {MAX_INPUT_LENGTH})",
        )

    return ModerationResult(is_safe=True)


# Simple PII patterns
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_PATTERN = re.compile(r"\b\d{3}[-.]?\d{3,4}[-.]?\d{4}\b")


def moderate_output(text: str) -> ModerationResult:
    """Check AI output for PII or inappropriate content."""
    if _EMAIL_PATTERN.search(text):
        return ModerationResult(
            is_safe=False, reason="PII detected: email address"
        )
    if _PHONE_PATTERN.search(text):
        return ModerationResult(
            is_safe=False, reason="PII detected: phone number"
        )
    return ModerationResult(is_safe=True)
