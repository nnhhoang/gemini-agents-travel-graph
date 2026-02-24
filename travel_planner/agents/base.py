"""
Base agent class for the travel planner system.

This module implements the foundational Agent class that all specialized
agents in the travel planner system will inherit from. It provides common
functionality and standardized interfaces for all agents.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

# Type variable for context
T = TypeVar("T")


class AgentContext(BaseModel):
    """Base class for agent context that can be passed between agents."""

    pass


class TravelPlannerAgentError(Exception):
    """Base exception for all agent-related errors."""

    pass


class InvalidConfigurationError(TravelPlannerAgentError):
    """Exception raised when agent configuration is invalid."""

    pass


# Alias for backward compatibility
InvalidConfigurationException = InvalidConfigurationError


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    name: str
    instructions: str
    model: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_tokens: int | None = None
    tools: list[Any] = field(default_factory=list)


class BaseAgent(Generic[T]):
    """
    Base class for all travel planner agents.

    This class provides the foundation for specialized agents that handle
    different aspects of travel planning, such as destination research,
    flight search, accommodation booking, etc.

    The BaseAgent implements common functionality like handling API clients,
    error management, and providing a standardized interface for all agents.
    """

    def __init__(
        self,
        config: AgentConfig,
        context_type: type[T] | None = None,
    ):
        """
        Initialize a base agent.

        Args:
            config: Configuration for the agent
            context_type: Type of context this agent handles (optional)
        """
        self.config = config
        self.client = genai.Client()
        self.context_type = context_type or AgentContext

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return self.config.name

    @property
    def instructions(self) -> str:
        """Get the instructions for the agent."""
        return self.config.instructions

    def invoke(self, state: Any) -> dict[str, Any]:
        """Synchronous bridge for LangGraph node calls.

        Extracts input from the workflow state and calls the async run()
        method via asyncio.run(). Safe because LangGraph runs sync nodes
        in a thread pool with no running event loop.

        Args:
            state: TravelPlanningState (or similar) passed by LangGraph

        Returns:
            Result dictionary suitable for node consumption
        """
        # Extract input from state
        if hasattr(state, "conversation_history") and state.conversation_history:
            input_data = state.conversation_history
        elif hasattr(state, "query") and state.query:
            input_data = getattr(state.query, "raw_query", "") or str(state.query)
        else:
            input_data = "Plan a trip"

        result = asyncio.run(self.run(input_data))

        # Flatten nested "result" key for node compatibility
        if (
            isinstance(result, dict)
            and "result" in result
            and isinstance(result["result"], dict)
        ):
            flat = dict(result["result"])
            flat.update({k: v for k, v in result.items() if k != "result"})
            return flat

        return result if isinstance(result, dict) else {"content": str(result)}

    async def run(
        self, input_data: str | list[dict[str, Any]], context: T | None = None
    ) -> Any:
        """
        Run the agent with the provided input and context.

        Args:
            input_data: User input or conversation history
            context: Optional context for the agent

        Returns:
            Agent response or result
        """
        # This would typically integrate with the Google Gemini SDK
        # For now, this is a simple implementation
        raise NotImplementedError("Subclasses must implement run method")

    async def process(self, *args, **kwargs) -> Any:
        """
        Process the input according to the agent's specialized function.

        This is the main method that specialized agents will implement
        to perform their specific tasks.
        """
        raise NotImplementedError("Subclasses must implement process method")

    def _validate_config(self) -> bool:
        """Validate the agent configuration."""
        if not self.config.name:
            raise InvalidConfigurationException("Agent name cannot be empty")
        if not self.config.instructions:
            raise InvalidConfigurationException("Agent instructions cannot be empty")
        return True

    def _convert_messages_for_gemini(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[types.Content], str | None]:
        """
        Convert chat-style messages to Gemini format.

        Extracts system messages into a system_instruction string,
        and maps remaining messages to types.Content objects.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Tuple of (contents list, system_instruction string or None)
        """
        system_parts = []
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_parts.append(content)
            else:
                # Map "assistant" role to "model" for Gemini
                gemini_role = "model" if role == "assistant" else "user"
                contents.append(
                    types.Content(
                        role=gemini_role,
                        parts=[types.Part.from_text(text=content)],
                    )
                )

        system_instruction = "\n\n".join(system_parts) if system_parts else None
        return contents, system_instruction

    def _prepare_messages(
        self, input_data: str | list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Prepare messages for the API call.

        Args:
            input_data: User input or conversation history

        Returns:
            List of messages formatted for the API
        """
        if isinstance(input_data, str):
            messages = [
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": input_data},
            ]
        # If input_data is already a list of messages, add system message if not present
        elif input_data and input_data[0].get("role") != "system":
            messages = [{"role": "system", "content": self.instructions}, *input_data]
        else:
            messages = input_data

        return messages
