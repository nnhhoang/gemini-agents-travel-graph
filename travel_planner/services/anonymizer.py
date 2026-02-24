"""
Log anonymization service.

Strips PII (emails, phone numbers) from log entries
before writing to CloudWatch or other log sinks.
"""

import re

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\b\d{2,4}[-.]?\d{3,4}[-.]?\d{4}\b")


def anonymize(text: str) -> str:
    """Replace PII patterns with placeholders."""
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    return text
