"""
Recommendation agent for spot/restaurant/activity suggestions.

Takes user preferences, location, and time context to generate
personalized recommendations via Gemini.
"""

from typing import Any

from google.genai import types

from travel_planner.agents.base import AgentConfig, BaseAgent
from travel_planner.data.preferences import UserPreferences
from travel_planner.prompts.context import ContextBuilder


class RecommendationAgent(BaseAgent):
    """Agent for generating personalized recommendations."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        config = AgentConfig(
            name="Recommendation Agent",
            instructions=(
                "You are a Japanese tourism recommendation engine. "
                "Generate specific, actionable recommendations based on "
                "user preferences and current context. "
                "Include place names, brief descriptions, and why they "
                "match the user's preferences. "
                "Respond in JSON format when possible."
            ),
            model=model,
            temperature=0.7,
        )
        super().__init__(config)
        self.context_builder = ContextBuilder()

    async def recommend(
        self,
        preferences: UserPreferences,
        location: dict[str, float] | None = None,
        category: str = "general",
        timestamp: str | None = None,
    ) -> str:
        """Generate recommendations based on preferences and context."""
        system_prompt = self.context_builder.build_system_prompt(
            preferences=preferences,
            location=location,
            timestamp=timestamp,
        )

        pref_text = preferences.to_prompt_context()
        message = (
            f"Recommend {category} options. "
            f"User preferences: {pref_text}"
        )
        if location:
            message += (
                f"\nNear: lat={location['lat']}, lng={location['lng']}"
            )

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=message)],
            )
        ]

        config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
            system_instruction=system_prompt,
        )

        response = await self.client.aio.models.generate_content(
            model=self.config.model,
            contents=contents,
            config=config,
        )

        return response.text
