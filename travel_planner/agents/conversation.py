"""
Conversation agent for the AI tourism guide.

Handles natural language conversation using Gemini,
incorporating user preferences and context.
"""

from typing import Any

from google.genai import types

from travel_planner.agents.base import AgentConfig, BaseAgent


class ConversationAgent(BaseAgent):
    """Main conversation agent for tourism chat."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        config = AgentConfig(
            name="Conversation Agent",
            instructions=(
                "You are a helpful Japanese tourism guide AI. "
                "Provide personalized, friendly recommendations. "
                "Consider the user's preferences, location, and time of day. "
                "Respond in the user's language."
            ),
            model=model,
            temperature=0.8,
        )
        super().__init__(config)

    async def chat(
        self,
        message: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Generate a conversational response.

        Args:
            message: User's message
            system_prompt: System prompt with context
            history: Previous messages as [{"role": "user/model", "content": "..."}]

        Returns:
            AI response text
        """
        contents = []

        # Add conversation history
        if history:
            for msg in history:
                role = "model" if msg["role"] == "assistant" else msg["role"]
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg["content"])],
                    )
                )

        # Add current message
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=message)],
            )
        )

        config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
            system_instruction=system_prompt or self.instructions,
        )

        response = await self.client.aio.models.generate_content(
            model=self.config.model,
            contents=contents,
            config=config,
        )

        return response.text

    async def chat_stream(
        self,
        message: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
    ):
        """
        Generate a streaming conversational response.

        Yields text chunks as they are generated.
        """
        contents = []

        if history:
            for msg in history:
                role = "model" if msg["role"] == "assistant" else msg["role"]
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg["content"])],
                    )
                )

        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=message)],
            )
        )

        config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
            system_instruction=system_prompt or self.instructions,
        )

        async for chunk in self.client.aio.models.generate_content_stream(
            model=self.config.model,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                yield chunk.text
