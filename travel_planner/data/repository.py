"""
DynamoDB repository implementing all access patterns.

Maps domain models to/from DynamoDB single-table items.
"""

from datetime import UTC, datetime
from typing import Any

from travel_planner.data.conversation_models import (
    Content,
    Conversation,
    Location,
    Message,
    Place,
    Session,
    User,
)
from travel_planner.data.dynamodb import DynamoDBClient
from travel_planner.data.preferences import UserPreferences
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


class DynamoDBRepository:
    """Repository for all DynamoDB operations across entity types."""

    def __init__(self, db: DynamoDBClient):
        self.db = db

    # --- Helpers ---

    def _to_item(
        self, entity: Any, entity_type: str, version: int = 1
    ) -> dict[str, Any]:
        """Convert a domain model to a DynamoDB item."""
        data = entity.model_dump(exclude={"pk", "sk", "gsi1pk", "gsi1sk"})
        now = datetime.now(UTC).isoformat()
        item: dict[str, Any] = {
            "PK": entity.pk,
            "SK": entity.sk,
            "EntityType": entity_type,
            "Version": version,
            "Data": data,
            "Metadata": {
                "createdAt": now,
                "updatedAt": now,
            },
        }
        if hasattr(entity, "gsi1pk"):
            item["GSI1PK"] = entity.gsi1pk
        if hasattr(entity, "gsi1sk"):
            item["GSI1SK"] = entity.gsi1sk
        if hasattr(entity, "ttl") and entity.ttl:
            item["TTL"] = entity.ttl
        return item

    # --- Users (AP1, AP2) ---

    def save_user(self, user: User) -> None:
        self.db.put_item(self._to_item(user, "User"))

    def get_user(self, user_id: str) -> User | None:
        item = self.db.get_item(f"USER#{user_id}", "PROFILE")
        if not item:
            return None
        return User.model_validate(item["Data"])

    def get_user_by_email(self, email: str) -> User | None:
        items = self.db.query_gsi1(f"EMAIL#{email}", limit=1)
        if not items:
            return None
        return User.model_validate(items[0]["Data"])

    # --- Sessions (AP3, AP4) ---

    def save_session(self, session: Session) -> None:
        self.db.put_item(self._to_item(session, "Session"))

    def get_session(self, session_id: str) -> Session | None:
        item = self.db.get_item(f"SESSION#{session_id}", "METADATA")
        if not item:
            return None
        return Session.model_validate(item["Data"])

    def get_user_sessions(self, user_id: str) -> list[Session]:
        items = self.db.query(pk=f"USER#{user_id}", sk_prefix="SESSION#")
        return [Session.model_validate(i["Data"]) for i in items]

    # --- Preferences (AP19) ---

    def save_preferences(self, user_id: str, prefs: UserPreferences) -> None:
        item = {
            "PK": f"USER#{user_id}",
            "SK": "PREFERENCES",
            "EntityType": "Preference",
            "Version": 1,
            "Data": prefs.model_dump(),
            "Metadata": {
                "updatedAt": datetime.now(UTC).isoformat(),
            },
        }
        self.db.put_item(item)

    def get_preferences(self, user_id: str) -> UserPreferences | None:
        item = self.db.get_item(f"USER#{user_id}", "PREFERENCES")
        if not item:
            return None
        return UserPreferences.model_validate(item["Data"])

    # --- Conversations (AP9) ---

    def save_conversation(self, conv: Conversation) -> None:
        self.db.put_item(self._to_item(conv, "Conversation"))

    def list_conversations(self, user_id: str) -> list[Conversation]:
        items = self.db.query(pk=f"USER#{user_id}#CONVERSATION")
        return [Conversation.model_validate(i["Data"]) for i in items]

    def get_conversation(
        self, user_id: str, conversation_id: str
    ) -> Conversation | None:
        items = self.db.query(
            pk=f"USER#{user_id}#CONVERSATION",
            sk_prefix=f"CONV#{conversation_id}",
        )
        if not items:
            return None
        return Conversation.model_validate(items[0]["Data"])

    # --- Messages (AP10) ---

    def save_message(self, msg: Message) -> None:
        self.db.put_item(self._to_item(msg, "Message"))

    def get_messages(
        self, conversation_id: str, limit: int | None = None
    ) -> list[Message]:
        kwargs: dict[str, Any] = {
            "pk": f"CONVERSATION#{conversation_id}#MESSAGE",
        }
        if limit is not None:
            kwargs["limit"] = limit
        items = self.db.query(**kwargs)
        return [Message.model_validate(i["Data"]) for i in items]

    # --- Locations (AP5) ---

    def save_location(self, location: Location) -> None:
        self.db.put_item(self._to_item(location, "Location"))

    def get_user_locations(
        self, user_id: str, start: str, end: str
    ) -> list[Location]:
        items = self.db.query(
            pk=f"USER#{user_id}#LOCATION",
            sk_between=(start, end),
        )
        return [Location.model_validate(i["Data"]) for i in items]

    # --- Places (AP6) ---

    def save_place(self, place: Place) -> None:
        self.db.put_item(self._to_item(place, "Place"))

    def get_place(self, place_id: str) -> Place | None:
        item = self.db.get_item(f"PLACE#{place_id}", "METADATA")
        if not item:
            return None
        return Place.model_validate(item["Data"])

    def get_places_by_geohash(self, geohash: str) -> list[Place]:
        items = self.db.query_gsi1(f"GEOHASH#{geohash}")
        return [
            Place.model_validate(i["Data"])
            for i in items
            if i.get("EntityType") == "Place"
        ]

    # --- Content (AP12, AP13, AP14) ---

    def save_content(self, content: Content) -> None:
        self.db.put_item(self._to_item(content, "Content"))

    def get_content(self, content_id: str) -> Content | None:
        item = self.db.get_item(f"CONTENT#{content_id}", "METADATA")
        if not item:
            return None
        return Content.model_validate(item["Data"])

    def get_org_content(
        self, org_id: str, status: str = "published"
    ) -> list[Content]:
        items = self.db.query_gsi1(f"ORG#{org_id}#CONTENT#{status}")
        return [Content.model_validate(i["Data"]) for i in items]

    def save_content_revision(
        self,
        content_id: str,
        version: int,
        snapshot: dict[str, Any],
        changed_by: str,
        reason: str = "",
    ) -> None:
        item = {
            "PK": f"CONTENT#{content_id}#REVISION",
            "SK": f"VERSION#{version:03d}",
            "EntityType": "ContentRevision",
            "Version": 1,
            "Data": {
                "snapshot": snapshot,
                "changedBy": changed_by,
                "reason": reason,
            },
            "Metadata": {"createdAt": datetime.now(UTC).isoformat()},
        }
        self.db.put_item(item)
