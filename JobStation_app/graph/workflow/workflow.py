from typing import Literal
from langgraph.graph import StateGraph, END

from JobStation_app.graph.state import State
from JobStation_app.graph.agents import (
    supervisor_node,
    jobseeker_agent_node,
    company_agent_node,
    general_agent_node,
    jobseeker_tool_node,
    company_tool_node,
)


# ─── ROUTERS ──────────────────────────────────────────────────────────────────
def supervisor_router(state: State) -> Literal[
    "jobseeker_agent", "company_agent", "general_agent", "__end__"
]:
    next_node = state.get("next", "general_agent")
    if next_node == "FINISH":
        return "__end__"
    if next_node == "jobseeker_agent":
        return "jobseeker_agent"
    if next_node == "company_agent":
        return "company_agent"
    return "general_agent"


def jobseeker_router(state: State) -> Literal["jobseeker_tools", "supervisor"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "jobseeker_tools"
    return "supervisor"


def company_router(state: State) -> Literal["company_tools", "supervisor"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "company_tools"
    return "supervisor"


# ─── BUILD GRAPH ──────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(State)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    graph.add_node("supervisor",      supervisor_node)
    graph.add_node("jobseeker_agent", jobseeker_agent_node)
    graph.add_node("company_agent",   company_agent_node)
    graph.add_node("general_agent",   general_agent_node)
    graph.add_node("jobseeker_tools", jobseeker_tool_node)
    graph.add_node("company_tools",   company_tool_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("supervisor")

    # ── Supervisor routes to agents or END ────────────────────────────────────
    graph.add_conditional_edges("supervisor", supervisor_router)

    # ── Role agents route to tools or back to supervisor ──────────────────────
    graph.add_conditional_edges("jobseeker_agent", jobseeker_router)
    graph.add_conditional_edges("company_agent",   company_router)

    # ── FIX: general_agent goes DIRECTLY to END — not back to supervisor ──────
    # This is the primary fix for the infinite loop:
    # Before: general_agent → supervisor → general_agent → supervisor → ...
    # After:  general_agent → END
    graph.add_edge("general_agent", END)

    # ── Tools return to their agent (which then re-routes via supervisor) ──────
    graph.add_edge("jobseeker_tools", "jobseeker_agent")
    graph.add_edge("company_tools",   "company_agent")

    return graph.compile()


# Compiled graph — import this in main.py
app = build_graph()