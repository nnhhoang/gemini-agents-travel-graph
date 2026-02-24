"""Tests for conversation caching."""

import time
from unittest.mock import MagicMock

from travel_planner.services.cache_service import CacheService


def test_cache_set_and_get():
    mock_db = MagicMock()
    mock_db.get_item.return_value = {
        "Data": {"value": "cached response"},
        "TTL": int(time.time()) + 3600,
    }
    cache = CacheService(mock_db, ttl=3600)
    cache.set("key1", "cached response")
    result = cache.get("key1")
    assert result == "cached response"


def test_cache_miss():
    mock_db = MagicMock()
    mock_db.get_item.return_value = None
    cache = CacheService(mock_db)
    result = cache.get("nonexistent")
    assert result is None


def test_cache_expired():
    mock_db = MagicMock()
    mock_db.get_item.return_value = {
        "Data": {"value": "old"},
        "TTL": int(time.time()) - 1,
    }
    cache = CacheService(mock_db)
    result = cache.get("expired_key")
    assert result is None
