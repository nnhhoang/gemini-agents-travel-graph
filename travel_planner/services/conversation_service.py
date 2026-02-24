"""
Conversation service orchestrating the chat flow.

Handles: load preferences, load history, build context,
call agent, save messages, return response.
"""

import uuid
from typing import Any

from travel_planner.data.conversation_models import (
    Conversation,
    Message,
    MessageRole,
)
from travel_planner.data.repository import DynamoDBRepository
from travel_planner.prompts.context import ContextBuilder
from travel_planner.prompts.moderation import moderate_input, moderate_output
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


class ConversationService:
    """Orchestrates the conversation flow."""

    def __init__(self, repo: DynamoDBRepository, agent: Any):
        self.repo = repo
        self.agent = agent
        self.context_builder = ContextBuilder()

    async def handle_chat(
        self,
        user_id: str,
        message: str,
        conversation_id: str | None = None,
        location: dict[str, float] | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Handle a chat message end-to-end."""
        # Validate input
        mod_result = moderate_input(message)
        if not mod_result.is_safe:
            return {
                "error": mod_result.reason,
                "conversation_id": conversation_id,
            }

        # Get or create conversation
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            conv = Conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                title=message[:50],
            )
            self.repo.save_conversation(conv)

        # Load preferences
        preferences = self.repo.get_preferences(user_id)

        # Load history
        messages = self.repo.get_messages(conversation_id)
        next_seq = len(messages) + 1

        # Save user message
        user_msg = Message(
            conversation_id=conversation_id,
            sequence=next_seq,
            role=MessageRole.USER,
            content=message,
        )
        self.repo.save_message(user_msg)

        # Build system prompt
        system_prompt = self.context_builder.build_system_prompt(
            preferences=preferences,
            location=location,
            timestamp=timestamp,
        )

        # Build history for agent
        history = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]

        # Call agent
        response_text = await self.agent.chat(
            message=message,
            system_prompt=system_prompt,
            history=history,
        )

        # Moderate output
        out_mod = moderate_output(response_text)
        if not out_mod.is_safe:
            response_text = (
                "I apologize, but I cannot provide that information. "
                "Can I help you with something else?"
            )

        # Save assistant message
        assistant_msg = Message(
            conversation_id=conversation_id,
            sequence=next_seq + 1,
            role=MessageRole.ASSISTANT,
            content=response_text,
        )
        self.repo.save_message(assistant_msg)

        return {
            "response": response_text,
            "conversation_id": conversation_id,
            "message_id": f"{next_seq + 1:06d}",
        }
