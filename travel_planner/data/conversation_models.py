"""
Conversation domain models for the AI conversation engine.

Each model includes DynamoDB key generation (pk, sk, gsi1pk, gsi1sk)
matching the single-table design access patterns.
"""

from datetime import UTC, datetime
from enum import StrEnum

import pygeohash as gh
from pydantic import BaseModel, Field, computed_field


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class EntityType(StrEnum):
    USER = "User"
    SESSION = "Session"
    LOCATION = "Location"
    PLACE = "Place"
    CHECKIN = "CheckIn"
    CONVERSATION = "Conversation"
    MESSAGE = "Message"
    CONTENT = "Content"
    CONTENT_REVISION = "ContentRevision"
    PROMPT_TEMPLATE = "PromptTemplate"
    PREFERENCE = "Preference"
    TOKEN_USAGE = "TokenUsage"
    ORGANIZATION = "Organization"


class User(BaseModel):
    """User entity. AP1: PK=USER#id, SK=PROFILE."""

    user_id: str
    email: str
    name: str
    provider: str = "google"
    account_type: str = "member"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field
    @property
    def pk(self) -> str:
        return f"USER#{self.user_id}"

    @computed_field
    @property
    def sk(self) -> str:
        return "PROFILE"

    @computed_field
    @property
    def gsi1pk(self) -> str:
        return f"EMAIL#{self.email}"

    @computed_field
    @property
    def gsi1sk(self) -> str:
        return f"USER#{self.user_id}"


class Session(BaseModel):
    """Session entity. AP4: PK=SESSION#id, SK=METADATA."""

    session_id: str
    user_id: str
    device_id: str | None = None
    ip_address: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ttl: int | None = None

    @computed_field
    @property
    def pk(self) -> str:
        return f"SESSION#{self.session_id}"

    @computed_field
    @property
    def sk(self) -> str:
        return "METADATA"

    @computed_field
    @property
    def gsi1pk(self) -> str:
        return f"USER#{self.user_id}#SESSION"

    @computed_field
    @property
    def gsi1sk(self) -> str:
        return self.created_at.isoformat()


class Conversation(BaseModel):
    """Conversation entity. AP9: PK=USER#id#CONVERSATION, SK=CONV#id."""

    conversation_id: str
    user_id: str
    title: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field
    @property
    def pk(self) -> str:
        return f"USER#{self.user_id}#CONVERSATION"

    @computed_field
    @property
    def sk(self) -> str:
        return f"CONV#{self.conversation_id}"


class Message(BaseModel):
    """Message entity. AP10: PK=CONVERSATION#id#MESSAGE, SK=seq (zero-padded)."""

    conversation_id: str
    sequence: int
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tokens_used: int | None = None

    @computed_field
    @property
    def pk(self) -> str:
        return f"CONVERSATION#{self.conversation_id}#MESSAGE"

    @computed_field
    @property
    def sk(self) -> str:
        return f"{self.sequence:06d}"


class Place(BaseModel):
    """Place entity. AP6: GSI1PK=GEOHASH#hash."""

    place_id: str
    name: str
    category: str
    lat: float
    lng: float
    description: str | None = None
    address: str | None = None

    @computed_field
    @property
    def pk(self) -> str:
        return f"PLACE#{self.place_id}"

    @computed_field
    @property
    def sk(self) -> str:
        return "METADATA"

    @computed_field
    @property
    def gsi1pk(self) -> str:
        return f"GEOHASH#{gh.encode(self.lat, self.lng, precision=7)}"

    @computed_field
    @property
    def gsi1sk(self) -> str:
        return f"PLACE#{self.place_id}"


class Location(BaseModel):
    """User location entity (time-series). AP5: PK=USER#id#LOCATION, SK=timestamp."""

    user_id: str
    lat: float
    lng: float
    accuracy: float | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field
    @property
    def pk(self) -> str:
        return f"USER#{self.user_id}#LOCATION"

    @computed_field
    @property
    def sk(self) -> str:
        return self.timestamp.isoformat()

    @computed_field
    @property
    def gsi1pk(self) -> str:
        return f"GEOHASH#{gh.encode(self.lat, self.lng, precision=7)}"


class Content(BaseModel):
    """CMS content entity. AP13: PK=CONTENT#id, SK=METADATA."""

    content_id: str
    org_id: str
    content_type: str = "government"
    title: str
    body: str | None = None
    status: str = "draft"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field
    @property
    def pk(self) -> str:
        return f"CONTENT#{self.content_id}"

    @computed_field
    @property
    def sk(self) -> str:
        return "METADATA"

    @computed_field
    @property
    def gsi1pk(self) -> str:
        return f"ORG#{self.org_id}#CONTENT#{self.status}"

    @computed_field
    @property
    def gsi1sk(self) -> str:
        return self.created_at.isoformat()
