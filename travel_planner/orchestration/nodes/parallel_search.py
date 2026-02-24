"""
Parallel search node implementation for the travel planning workflow.

This module defines functions for running flight, accommodation, and
transportation searches, and combining their results.
"""

from travel_planner.orchestration.nodes.accommodation_search import accommodation_search
from travel_planner.orchestration.nodes.flight_search import flight_search
from travel_planner.orchestration.nodes.transportation_planning import (
    transportation_planning,
)
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


def parallel_search(state: TravelPlanningState) -> TravelPlanningState:
    """
    Run flight, accommodation, and transportation searches sequentially.

    Each search node updates the state with its results. Errors in individual
    searches are logged but don't stop the remaining searches.

    Args:
        state: Current travel planning state

    Returns:
        Updated state with search results
    """
    logger.info("Starting parallel search (sequential execution)")

    for name, search_fn in [
        ("flight_search", flight_search),
        ("accommodation_search", accommodation_search),
        ("transportation_planning", transportation_planning),
    ]:
        try:
            state = search_fn(state)
        except Exception as e:
            logger.error(f"Error in {name}: {e!s}")
            state.conversation_history.append(
                {"role": "system", "content": f"Warning: {name} failed: {e!s}"}
            )

    return state


def combine_search_results(state: TravelPlanningState) -> TravelPlanningState:
    """
    Combine the results from the parallel search branch.

    Args:
        state: Current travel planning state

    Returns:
        Updated state with combined results from parallel search
    """
    logger.info("Combining results from parallel search")

    # Update the stage to indicate completion of parallel search
    state.update_stage(WorkflowStage.PARALLEL_SEARCH_COMPLETED)

    # Log combined results
    has_plan = state.plan is not None
    # Get counts of each type of result
    flights = state.plan.flights if has_plan else None
    flight_count = len(flights) if flights else 0

    accom = state.plan.accommodation if has_plan else None
    accom_count = len(accom) if accom else 0

    transport = state.plan.transportation if has_plan else None
    transport_count = len(transport) if transport else 0

    state.conversation_history.append(
        {
            "role": "system",
            "content": (
                f"Completed parallel search: {flight_count} flights, "
                f"{accom_count} accommodations, "
                f"{transport_count} transportation options"
            ),
        }
    )

    return state
