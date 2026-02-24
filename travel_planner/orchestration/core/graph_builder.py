"""
Graph builder for the travel planner workflow.

This module defines the functions for building the workflow state graph
using LangGraph. It connects the various nodes and defines the edges
and conditions for workflow transitions.
"""

from langgraph.graph import END, START, StateGraph

from travel_planner.orchestration.nodes.activity_planning import activity_planning
from travel_planner.orchestration.nodes.budget_management import budget_management
from travel_planner.orchestration.nodes.destination_research import destination_research
from travel_planner.orchestration.nodes.final_plan import generate_final_plan
from travel_planner.orchestration.nodes.parallel_search import (
    combine_search_results,
    parallel_search,
)
from travel_planner.orchestration.nodes.query_analysis import query_analysis
from travel_planner.orchestration.routing.conditions import query_research_needed
from travel_planner.orchestration.routing.error_recovery import handle_error
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


def create_planning_graph() -> StateGraph:
    """
    Create a state graph for travel planning.

    The graph implements this flow:
        START -> analyze_query -> [conditional] -> research_destination OR parallel_search
        research_destination -> parallel_search
        parallel_search -> combine_search_results -> plan_activities
        plan_activities -> manage_budget -> generate_final_plan -> END

    Returns:
        Compiled StateGraph instance that orchestrates the travel planning workflow
    """
    logger.info("Creating planning graph")

    # Create a new state graph
    workflow = StateGraph(TravelPlanningState)

    # Define the nodes in the graph
    workflow.add_node("analyze_query", query_analysis)
    workflow.add_node("research_destination", destination_research)
    workflow.add_node("parallel_search", parallel_search)
    workflow.add_node("combine_search_results", combine_search_results)
    workflow.add_node("plan_activities", activity_planning)
    workflow.add_node("manage_budget", budget_management)
    workflow.add_node("generate_final_plan", generate_final_plan)
    workflow.add_node("handle_error", handle_error)

    # Define edges

    # Start with query analysis
    workflow.add_edge(START, "analyze_query")

    # Conditional: Need destination research or can move directly to search?
    workflow.add_conditional_edges(
        "analyze_query",
        query_research_needed,
        {
            "research_destination": "research_destination",
            "flight_search": "parallel_search",
        },
    )

    # After destination research, move to parallel search
    workflow.add_edge("research_destination", "parallel_search")

    # After parallel search, combine results
    workflow.add_edge("parallel_search", "combine_search_results")

    # After combining search results, move to activity planning
    workflow.add_edge("combine_search_results", "plan_activities")

    # After activity planning, move to budget management
    workflow.add_edge("plan_activities", "manage_budget")

    # After budget management, generate the final plan
    workflow.add_edge("manage_budget", "generate_final_plan")

    # After generating the final plan, end the workflow
    workflow.add_edge("generate_final_plan", END)

    # Error handling ends the workflow
    workflow.add_edge("handle_error", END)

    # Compile the graph
    logger.info("Planning graph created and compiled")
    return workflow.compile()
