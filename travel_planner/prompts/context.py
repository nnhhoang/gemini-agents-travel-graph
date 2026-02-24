"""
Context builder for assembling prompt context from multiple sources.

Merges user preferences, location, time, conversation history,
and CMS content into a structured context dict for prompt injection.
"""

from datetime import datetime
from typing import Any

from travel_planner.data.conversation_models import Message
from travel_planner.data.preferences import UserPreferences


class ContextBuilder:
    """Builds prompt context from multiple data sources."""

    def build(
        self,
        message: str,
        preferences: UserPreferences | None = None,
        location: dict[str, float] | None = None,
        timestamp: str | None = None,
        history: list[Message] | None = None,
        content: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Assemble context dict for prompt injection."""
        ctx: dict[str, Any] = {"message": message}

        # Preferences
        if preferences:
            ctx["preferences_text"] = preferences.to_prompt_context()
        else:
            ctx["preferences_text"] = "No preferences set"

        # Location
        if location:
            ctx["location"] = (
                f"lat={location['lat']}, lng={location['lng']}"
            )

        # Time awareness
        if timestamp:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            ctx["time_of_day"] = self._get_time_of_day(dt.hour)
            ctx["day_of_week"] = dt.strftime("%A")
            ctx["timestamp"] = timestamp

        # Conversation history
        if history:
            ctx["history"] = [
                {"role": msg.role.value, "content": msg.content}
                for msg in history
            ]

        # CMS content
        if content:
            ctx["cms_content"] = content

        return ctx

    def build_system_prompt(
        self,
        preferences: UserPreferences | None = None,
        location: dict[str, float] | None = None,
        timestamp: str | None = None,
    ) -> str:
        """Build a rich, structured system prompt for Gemini."""
        sections: list[str] = []

        # --- 1. Identity & Expertise ---
        sections.append(
            "<role>\n"
            "You are 'Trip', an expert AI tourism concierge specializing in "
            "Japan travel. You have deep knowledge of:\n"
            "- Regional cuisine, seasonal ingredients, and restaurant culture\n"
            "- Hot springs (onsen), temples, shrines, and cultural etiquette\n"
            "- Public transit systems (JR, metro, buses, IC cards)\n"
            "- Local festivals, seasonal events, and hidden gems\n"
            "- Budget optimization and travel logistics\n"
            "</role>"
        )

        # --- 2. User Preferences (with usage instructions) ---
        if preferences:
            pref_text = preferences.to_prompt_context()
            if pref_text != "No preferences set":
                sections.append(
                    "<user_preferences>\n"
                    f"{pref_text}\n"
                    "</user_preferences>\n\n"
                    "IMPORTANT — How to use preferences:\n"
                    "- Treat these as hard constraints, not suggestions. "
                    "NEVER recommend something that violates dietary restrictions.\n"
                    "- Proactively match suggestions to their style "
                    "(e.g., if 'hidden gems' is set, skip tourist traps).\n"
                    "- If the user's request conflicts with a preference, "
                    "acknowledge the conflict and offer alternatives.\n"
                    "- Reference preferences naturally in your response "
                    "(e.g., 'Since you enjoy seafood...' not 'Based on your profile...')."
                )

        # --- 3. Real-Time Context (location + time) ---
        context_parts: list[str] = []
        if location:
            context_parts.append(
                f"GPS: lat {location['lat']}, lng {location['lng']}"
            )
        if timestamp:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            time_label = self._get_time_of_day(dt.hour)
            context_parts.append(
                f"Time: {time_label} ({dt.strftime('%A %H:%M')})"
            )

        if context_parts:
            sections.append(
                "<real_time_context>\n"
                + "\n".join(context_parts)
                + "\n</real_time_context>\n\n"
                "Use this context to:\n"
                "- Recommend places that are open NOW\n"
                "- Suggest time-appropriate activities "
                "(breakfast spots in morning, bars in evening)\n"
                "- Prioritize nearby options when GPS is available\n"
                "- Factor in day of week (some places close on certain days)"
            )

        # --- 4. Thinking Process ---
        sections.append(
            "<thinking_process>\n"
            "Before responding, silently consider:\n"
            "1. What is the user actually asking for? (food, activity, transit, general info)\n"
            "2. Which preferences are relevant to THIS specific question?\n"
            "3. What time/location constraints apply?\n"
            "4. Am I confident this place exists and is accurate, or should I caveat it?\n"
            "Do NOT output this thinking — go straight to the answer.\n"
            "</thinking_process>"
        )

        # --- 5. Response Format ---
        sections.append(
            "<response_format>\n"
            "For each recommendation, include:\n"
            "- **Name** — the actual place name\n"
            "- **Why** — 1 sentence connecting it to the user's taste\n"
            "- **Details** — address, hours, price range, what to order\n"
            "- **Getting there** — nearest station + walk time\n\n"
            "Keep it scannable. Use bold headers. "
            "2-3 recommendations is ideal — do not overwhelm.\n"
            "For simple questions (directions, yes/no), answer directly "
            "without the full template.\n"
            "</response_format>"
        )

        # --- 6. Hard Rules ---
        sections.append(
            "<rules>\n"
            "- Respond in the SAME LANGUAGE the user writes in\n"
            "- Greet ONLY on the very first message. After that, straight to the answer\n"
            "- If you are not sure a place exists or is still open, say "
            "'I believe...' or 'You may want to verify...'\n"
            "- NEVER invent addresses or opening hours. "
            "If unsure, omit rather than fabricate\n"
            "- If the user asks something outside your expertise "
            "(medical, legal, emergency), say so and suggest they contact "
            "local services (police: 110, ambulance: 119, tourist hotline: 050-3816-2787)\n"
            "- Keep responses concise. Aim for quality over quantity\n"
            "</rules>"
        )

        return "\n\n".join(sections)

    @staticmethod
    def _get_time_of_day(hour: int) -> str:
        if 5 <= hour < 11:
            return "morning"
        elif 11 <= hour < 14:
            return "lunchtime"
        elif 14 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"
