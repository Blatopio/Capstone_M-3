from typing import Literal, BaseModel
from pydantic import BaseModel

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode

from JobStation_app.config import *
from JobStation_app.tools import *
from JobStation_app.graph.state import State


# structured output for supervisor decisions
class RouteDecision(BaseModel):
    next: Literal["jobseeker_agent", "company_agent", "FINISH"]


supervisor_llm = llm.with_structured_output(RouteDecision)


# tool lists for agents
jobseeker_tools = [upload_cv_tool, get_recommendations_tool]
company_tools   = [search_candidates_tool]

jobseeker_tool_node = ToolNode(jobseeker_tools)
company_tool_node   = ToolNode(company_tools)


# bind tools to LLMs for each agent
jobseeker_llm = llm.bind_tools(jobseeker_tools)
company_llm   = llm.bind_tools(company_tools)


# Supervisor agent node — routes to the correct agent based on role and intent
def supervisor_node(state: State) -> dict:
    """
    Routes conversation to the correct agent based on role and intent.
    Uses structured output — always returns a clean RouteDecision object.
    No parsing, no safety fallback needed.
    """
    system_prompt = SystemMessage(content="""
You are a supervisor managing a job placement platform called JobStation.

You have two agents available:
- jobseeker_agent : handles jobseekers — CV upload and job recommendations
- company_agent   : handles company HR teams — searching for candidates

Based on the conversation and the user's role, decide who should respond next.
If the user's request has been fully answered, return FINISH.

The user's role is available in the conversation context.
    """)

    messages = [system_prompt] + state["messages"]
    decision = supervisor_llm.invoke(messages)

    return {"next": decision.next}


# Jobseeker agent node — handles CV uploads and job recommendations
def jobseeker_agent_node(state: State) -> dict:
    """
    Handles jobseeker requests.
    Uses bind_tools — can dynamically call upload_cv_tool or get_recommendations_tool.
    """
    system_prompt = SystemMessage(content=f"""
You are a helpful career assistant for JobStation, a job placement platform.

You are currently helping: {state['username']} (jobseeker)

You can help with:
1. Uploading their CV to the platform so companies can find them
2. Getting job recommendations based on their CV

When uploading a CV, always pass the username '{state['username']}' to the tool.
Be friendly, encouraging, and professional.
After completing a task, summarize what was done clearly.
    """)

    messages = [system_prompt] + state["messages"]
    response = jobseeker_llm.invoke(messages)

    return {"messages": [response]}


# Company agent node — handles candidate search requests from company HR teams
def company_agent_node(state: State) -> dict:
    """
    Handles company/HR requests.
    Uses bind_tools — can dynamically call search_candidates_tool.
    """
    system_prompt = SystemMessage(content=f"""
You are a professional recruitment assistant for JobStation, a job placement platform.

You are currently helping: {state['username']} (company HR)

You can help with:
1. Searching for candidates by job category and experience level
2. Finding the best semantic matches for specific role requirements

When searching, always ask for:
- Job category (e.g. HR, ENGINEERING, FINANCE)
- Experience level (junior, senior, or specialist)
- A description of what they are looking for

Present candidate results clearly and professionally.
Remind the HR team that state changes (interviewed, placed) are done
through the admin panel, not through this chat.
    """)

    messages = [system_prompt] + state["messages"]
    response = company_llm.invoke(messages)

    return {"messages": [response]}