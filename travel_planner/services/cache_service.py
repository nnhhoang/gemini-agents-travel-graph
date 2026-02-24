"""
Conversation caching service using DynamoDB with TTL.

Caches frequent recommendation results to reduce Gemini API calls.
"""

import hashlib
import time
from typing import Any

from travel_planner.data.dynamodb import DynamoDBClient


class CacheService:
    """Cache service backed by DynamoDB with TTL auto-expiry."""

    def __init__(self, db: DynamoDBClient, ttl: int = 3600):
        self.db = db
        self.ttl = ttl

    def _cache_key(self, key: str) -> tuple[str, str]:
        hashed = hashlib.sha256(key.encode()).hexdigest()[:16]
        return f"CACHE#{hashed}", "DATA"

    def set(self, key: str, value: Any) -> None:
        pk, sk = self._cache_key(key)
        self.db.put_item(
            {
                "PK": pk,
                "SK": sk,
                "EntityType": "Cache",
                "Data": {"value": value},
                "TTL": int(time.time()) + self.ttl,
            }
        )

    def get(self, key: str) -> Any | None:
        pk, sk = self._cache_key(key)
        item = self.db.get_item(pk, sk)
        if not item:
            return None
        if item.get("TTL", 0) < time.time():
            return None
        return item.get("Data", {}).get("value")
