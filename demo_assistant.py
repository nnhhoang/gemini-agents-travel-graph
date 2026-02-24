"""
Demo script — test the AI Tourism Assistant step by step in console.

Usage:
    python demo_assistant.py

Walks you through:
  1. Show your loaded preferences
  2. Show the system prompt Gemini receives
  3. Start chatting (with conversation history)
"""

import asyncio
import json
import os
import sys

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

from travel_planner.agents.conversation import ConversationAgent
from travel_planner.data.preferences import UserPreferences
from travel_planner.prompts.context import ContextBuilder

PREFS_FILE = "prefs.json"


def load_preferences() -> UserPreferences:
    """Step 1: Load and display preferences."""
    print("=" * 60)
    print("STEP 1: Loading preferences from", PREFS_FILE)
    print("=" * 60)

    with open(PREFS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    prefs = UserPreferences.model_validate(data)
    print("\nLoaded preferences:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return prefs


def build_prompt(prefs: UserPreferences) -> str:
    """Step 2: Build and display the system prompt."""
    print("\n" + "=" * 60)
    print("STEP 2: System prompt that Gemini receives")
    print("=" * 60)

    builder = ContextBuilder()
    prompt = builder.build_system_prompt(preferences=prefs)
    print(f"\n{prompt}")
    return prompt


async def chat_loop(system_prompt: str) -> None:
    """Step 3: Interactive chat with history."""
    print("\n" + "=" * 60)
    print("STEP 3: Start chatting! (type 'exit' to quit)")
    print("=" * 60)

    agent = ConversationAgent()
    history: list[dict[str, str]] = []

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("\nBye!")
            break

        try:
            response = await agent.chat(
                message=user_input,
                system_prompt=system_prompt,
                history=history,
            )

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})

            print(f"\nTrip: {response}")
            print(f"\n  [history: {len(history) // 2} turns]")

        except Exception as e:
            print(f"\nError: {e}")


def main():
    if not os.path.exists(PREFS_FILE):
        print(f"Error: {PREFS_FILE} not found. Create it first.")
        sys.exit(1)

    prefs = load_preferences()
    input("\n--- Press Enter to see the system prompt ---")

    system_prompt = build_prompt(prefs)
    input("\n--- Press Enter to start chatting ---")

    asyncio.run(chat_loop(system_prompt))


if __name__ == "__main__":
    main()
