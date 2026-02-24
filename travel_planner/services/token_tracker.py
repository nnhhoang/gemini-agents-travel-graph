"""
Token usage tracking service.

Records input/output token counts per user per day in DynamoDB.
"""

from datetime import date
from typing import Any

from travel_planner.data.dynamodb import DynamoDBClient


class TokenTracker:
    """Tracks token usage per user per day."""

    def __init__(self, db: DynamoDBClient):
        self.db = db

    def track(
        self,
        user_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = "gemini-2.5-flash",
    ) -> None:
        """Record token usage for a user."""
        pk = f"USER#{user_id}#TOKEN_USAGE"
        sk = date.today().isoformat()

        existing = self.db.get_item(pk, sk)

        if existing:
            data = existing["Data"]
            self.db.update_item(
                pk,
                sk,
                {
                    "Data": {
                        "input_tokens": data["input_tokens"] + input_tokens,
                        "output_tokens": data["output_tokens"] + output_tokens,
                        "requests": data.get("requests", 0) + 1,
                        "model": model,
                    }
                },
            )
        else:
            self.db.put_item(
                {
                    "PK": pk,
                    "SK": sk,
                    "EntityType": "TokenUsage",
                    "Version": 1,
                    "Data": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "requests": 1,
                        "model": model,
                    },
                }
            )

    def get_usage(self, user_id: str, day: str | None = None) -> dict[str, Any]:
        """Get token usage for a user on a specific day."""
        pk = f"USER#{user_id}#TOKEN_USAGE"
        sk = day or date.today().isoformat()
        item = self.db.get_item(pk, sk)
        if not item:
            return {"input_tokens": 0, "output_tokens": 0, "requests": 0}
        return item["Data"]
