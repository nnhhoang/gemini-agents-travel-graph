"""Tests for token usage tracking."""

from datetime import date
from unittest.mock import MagicMock

from travel_planner.services.token_tracker import TokenTracker


def test_track_usage():
    mock_db = MagicMock()
    mock_db.get_item.return_value = None
    tracker = TokenTracker(mock_db)
    tracker.track("123", input_tokens=100, output_tokens=50, model="gemini-2.5-flash")
    mock_db.put_item.assert_called_once()
    item = mock_db.put_item.call_args[0][0]
    assert item["PK"] == "USER#123#TOKEN_USAGE"
    assert item["Data"]["input_tokens"] == 100


def test_track_usage_accumulates():
    mock_db = MagicMock()
    mock_db.get_item.return_value = {
        "PK": "USER#123#TOKEN_USAGE",
        "SK": date.today().isoformat(),
        "Data": {"input_tokens": 50, "output_tokens": 25, "requests": 1},
    }
    tracker = TokenTracker(mock_db)
    tracker.track("123", input_tokens=100, output_tokens=50)
    mock_db.update_item.assert_called_once()
