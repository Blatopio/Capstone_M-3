from typing import Literal
from langgraph.graph import StateGraph, END

from JobStation_app.graph.state import State
from JobStation_app.graph.agents import (
    supervisor_node,
    jobseeker_agent_node,
    company_agent_node,
    jobseeker_tool_node,
    company_tool_node,
)


# routing config for supervisor and agents
def supervisor_router(state: State) -> Literal["jobseeker_agent", "company_agent", "__end__"]:
    """Routes from supervisor to the correct agent or ends the conversation."""
    if state["next"] == "FINISH":
        return "__end__"
    if state["next"] == "jobseeker_agent":
        return "jobseeker_agent"
    return "company_agent"


def jobseeker_router(state: State) -> Literal["jobseeker_tools", "supervisor"]:
    """
    After jobseeker agent responds:
    - If it wants to call a tool → go to tool node
    - Otherwise → return to supervisor
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "jobseeker_tools"
    return "supervisor"


def company_router(state: State) -> Literal["company_tools", "supervisor"]:
    """
    After company agent responds:
    - If it wants to call a tool → go to tool node
    - Otherwise → return to supervisor
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "company_tools"
    return "supervisor"


# graph construction
def build_graph():
    graph = StateGraph(State)

    # Add nodes
    graph.add_node("supervisor",       supervisor_node)
    graph.add_node("jobseeker_agent",  jobseeker_agent_node)
    graph.add_node("company_agent",    company_agent_node)
    graph.add_node("jobseeker_tools",  jobseeker_tool_node)
    graph.add_node("company_tools",    company_tool_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Supervisor routes to agents or END
    graph.add_conditional_edges("supervisor", supervisor_router)

    # Agents route to tools or back to supervisor
    graph.add_conditional_edges("jobseeker_agent", jobseeker_router)
    graph.add_conditional_edges("company_agent",   company_router)

    # Tools always return to their agent
    graph.add_edge("jobseeker_tools", "jobseeker_agent")
    graph.add_edge("company_tools",   "company_agent")

    return graph.compile()


# Compiled graph
app = build_graph()