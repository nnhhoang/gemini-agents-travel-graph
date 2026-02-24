"""
AWS Lambda handler for the AI conversation engine.

Entry point for Node.js backend calls via lambda.invokeWithResponseStream().
Routes events by "action" field to appropriate services.
"""

import json
import os
from typing import Any

from travel_planner.agents.conversation import ConversationAgent
from travel_planner.agents.recommendation import RecommendationAgent
from travel_planner.data.dynamodb import DynamoDBClient
from travel_planner.data.preferences import UserPreferences
from travel_planner.data.repository import DynamoDBRepository
from travel_planner.services.conversation_service import ConversationService
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


def _extract_user_id(user_id_raw: str) -> str:
    """Extract user ID from USER#123 format."""
    if user_id_raw.startswith("USER#"):
        return user_id_raw[5:]
    return user_id_raw


def _get_db() -> DynamoDBClient:
    return DynamoDBClient(
        table_name=os.environ.get("DYNAMODB_TABLE_NAME", "travel-planner"),
        endpoint_url=os.environ.get("DYNAMODB_ENDPOINT"),
        region=os.environ.get("AWS_REGION", "ap-northeast-1"),
    )


def _get_repo() -> DynamoDBRepository:
    return DynamoDBRepository(_get_db())


def route_event(event: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Parse event and extract action + parameters."""
    action = event.get("action", "unknown")
    params: dict[str, Any] = {}

    user_id_raw = event.get("userId", "")
    if user_id_raw:
        params["user_id"] = _extract_user_id(user_id_raw)

    params["message"] = event.get("message", "")
    params["conversation_id"] = event.get("conversationId")
    params["context"] = event.get("context", {})
    params["preferences"] = event.get("preferences", {})

    # Planner-specific fields
    params["query"] = event.get("query", "")
    params["origin"] = event.get("origin")
    params["destination"] = event.get("destination")
    params["budget"] = event.get("budget")

    return action, params


async def _handle_chat(params: dict[str, Any]) -> dict[str, Any]:
    repo = _get_repo()
    agent = ConversationAgent()
    service = ConversationService(repo=repo, agent=agent)

    location = params.get("context", {}).get("location")
    timestamp = params.get("context", {}).get("timestamp")

    return await service.handle_chat(
        user_id=params["user_id"],
        message=params["message"],
        conversation_id=params.get("conversation_id"),
        location=location,
        timestamp=timestamp,
    )


async def _handle_save_preferences(params: dict[str, Any]) -> dict[str, Any]:
    repo = _get_repo()
    prefs = UserPreferences.model_validate(params["preferences"])
    repo.save_preferences(params["user_id"], prefs)
    return {"status": "ok"}


async def _handle_get_preferences(params: dict[str, Any]) -> dict[str, Any]:
    repo = _get_repo()
    prefs = repo.get_preferences(params["user_id"])
    if prefs:
        return {"status": "ok", "data": prefs.model_dump()}
    return {"status": "ok", "data": None}


async def _handle_list_conversations(params: dict[str, Any]) -> dict[str, Any]:
    repo = _get_repo()
    convs = repo.list_conversations(params["user_id"])
    return {
        "status": "ok",
        "data": [c.model_dump() for c in convs],
    }


async def _handle_get_conversation(params: dict[str, Any]) -> dict[str, Any]:
    repo = _get_repo()
    msgs = repo.get_messages(params["conversation_id"])
    return {
        "status": "ok",
        "data": [m.model_dump() for m in msgs],
    }


async def _handle_plan_trip(params: dict[str, Any]) -> dict[str, Any]:
    """Handle a plan_trip action by running the full LangGraph workflow."""
    from travel_planner.orchestration.workflow import TravelWorkflow

    query = params.get("query") or params.get("message", "")
    if not query:
        return {"status": "error", "error": "No query provided"}

    # Append origin/destination/budget to query string if provided
    if params.get("origin"):
        query += f" from {params['origin']}"
    if params.get("destination"):
        query += f" to {params['destination']}"
    if params.get("budget"):
        query += f" budget {params['budget']}"

    workflow = TravelWorkflow()
    plan = await workflow.process_query_async(query)

    plan_data = plan.model_dump(mode="json")

    if plan.metadata and plan.metadata.get("status") == "failed":
        return {
            "status": "error",
            "error": plan.metadata.get("error", "Planning failed"),
            "plan": plan_data,
        }

    return {"status": "ok", "plan": plan_data}


# Action handlers map
_HANDLERS = {
    "chat": _handle_chat,
    "plan_trip": _handle_plan_trip,
    "save_preferences": _handle_save_preferences,
    "get_preferences": _handle_get_preferences,
    "list_conversations": _handle_list_conversations,
    "get_conversation": _handle_get_conversation,
}


async def async_handler(event: dict[str, Any]) -> dict[str, Any]:
    """Main async handler."""
    action, params = route_event(event)

    handler_fn = _HANDLERS.get(action)
    if not handler_fn:
        return {"status": "error", "error": f"Unknown action: {action}"}

    try:
        return await handler_fn(params)
    except Exception as e:
        logger.error(f"Error handling {action}: {e}")
        return {"status": "error", "error": str(e)}


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Lambda entry point (sync wrapper)."""
    import asyncio

    return asyncio.run(async_handler(event))
