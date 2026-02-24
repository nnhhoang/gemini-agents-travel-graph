"""
Prompt A/B testing service.

Assigns users to prompt variants deterministically (hash-based)
and tracks performance metrics in DynamoDB.
"""

import hashlib
import time
from typing import Any

from travel_planner.data.dynamodb import DynamoDBClient


class ABTestService:
    """A/B testing for prompt variants."""

    def __init__(self, db: DynamoDBClient):
        self.db = db

    def assign_variant(
        self, user_id: str, test_id: str, variants: list[str]
    ) -> str:
        """Deterministically assign a variant based on user+test hash."""
        key = f"{user_id}:{test_id}"
        hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
        index = hash_val % len(variants)
        return variants[index]

    def record_outcome(
        self, test_id: str, variant: str, score: float
    ) -> None:
        """Record an A/B test outcome."""
        self.db.put_item(
            {
                "PK": f"ABTEST#{test_id}",
                "SK": f"VARIANT#{variant}#{int(time.time())}",
                "EntityType": "ABTestResult",
                "Version": 1,
                "Data": {
                    "variant": variant,
                    "score": score,
                    "timestamp": time.time(),
                },
            }
        )
