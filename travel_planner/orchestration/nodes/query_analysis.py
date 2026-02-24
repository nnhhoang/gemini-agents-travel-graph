"""
Query analysis node implementation for the travel planning workflow.

This module defines the function that analyzes user queries to extract travel
requirements and preferences using the orchestrator agent.
"""

from travel_planner.agents.orchestrator import OrchestratorAgent
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


def _extract_destination(raw_query: str) -> str:
    """Extract a destination name from a raw travel query.

    Uses simple keyword-based parsing to pull out the destination.
    Looks for patterns like "Visit <place>", "trip to <place>", etc.

    Args:
        raw_query: The user's raw travel query string

    Returns:
        Extracted destination or empty string if not found
    """
    import re

    # Match common travel query patterns
    patterns = [
        r"(?:visit|go to|travel to|trip to|fly to|heading to|explore)\s+(.+?)(?:\s+for|\s+in|\s+from|\s*$)",
        r"(?:vacation|holiday)\s+(?:in|at|to)\s+(.+?)(?:\s+for|\s*$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw_query, re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip(".")

    return ""


def query_analysis(state: TravelPlanningState) -> TravelPlanningState:
    """
    Analyze the user query to understand requirements and preferences.

    Args:
        state: Current travel planning state

    Returns:
        Updated travel planning state
    """
    logger.info("Starting query analysis")

    orchestrator = OrchestratorAgent()
    result = orchestrator.invoke(state)

    # The agent returns LLM text, not structured query/preferences keys.
    # Parse destination from the raw query if not already set.
    if state.query and not state.query.destination:
        extracted = _extract_destination(state.query.raw_query or "")
        if extracted:
            state.query.destination = extracted

    state.update_stage(WorkflowStage.QUERY_ANALYZED)

    has_destination = state.query and state.query.destination
    destination = state.query.destination if has_destination else "Unknown"
    logger.info(f"Query analyzed. Destination: {destination}")

    # Add the result to conversation history for context
    state.conversation_history.append(
        {
            "role": "system",
            "content": (
                f"Query analyzed: {destination}"
                if destination != "Unknown"
                else "Query analyzed: Destination research needed"
            ),
        }
    )

    # Add the task result
    state.add_task_result("query_analysis", result)

    return state
