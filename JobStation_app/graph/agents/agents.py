from typing import Literal
from pydantic import BaseModel

from langchain_core.messages import SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode

from JobStation_app.config import llm, langfuse_handler
from JobStation_app.tools import (
    search_candidates_tool,
    upload_cv_tool,
    get_recommendations_tool,
)
from JobStation_app.graph.state import State


# ─── STRUCTURED OUTPUT FOR SUPERVISOR ─────────────────────────────────────────
class RouteDecision(BaseModel):
    next: Literal["jobseeker_agent", "company_agent", "general_agent", "FINISH"]


supervisor_llm = llm.with_structured_output(RouteDecision)


# ─── TOOL NODES ────────────────────────────────────────────────────────────────
jobseeker_tools     = [upload_cv_tool, get_recommendations_tool]
company_tools       = [search_candidates_tool]

jobseeker_tool_node = ToolNode(jobseeker_tools)
company_tool_node   = ToolNode(company_tools)


# ─── LLM WITH TOOLS BOUND ─────────────────────────────────────────────────────
jobseeker_llm = llm.bind_tools(jobseeker_tools)
company_llm   = llm.bind_tools(company_tools)

# ─── SAFETY CAP ───────────────────────────────────────────────────────────────
MAX_TURNS = 6   # supervisor will force FINISH after this many re-routes


# ─── SUPERVISOR ───────────────────────────────────────────────────────────────
def supervisor_node(state: State) -> dict:
    """
    Routes conversation to the correct agent.
    Uses structured output — always returns a clean RouteDecision object.
    Includes a turn_count guard to break infinite loops.

    Routing logic:
    - Platform actions (CV upload, candidate search) → role-specific agent
    - Career/job/platform questions                  → general_agent
    - Fully answered or turn cap hit                 → FINISH
    """
    # ── infinite-loop guard ──────────────────────────────────────────────────
    turn_count = state.get("turn_count", 0) + 1
    if turn_count > MAX_TURNS:
        return {"next": "FINISH", "turn_count": turn_count}

    # ── check if the last message is already a plain AI response ────────────
    # If so, no need to re-route — just finish.
    messages = state["messages"]
    if messages:
        last = messages[-1]
        is_ai_text = (
            hasattr(last, "type")
            and last.type == "ai"
            and not getattr(last, "tool_calls", None)
        )
        if is_ai_text and state.get("next") in ("general_agent",):
            # general_agent just replied — we're done
            return {"next": "FINISH", "turn_count": turn_count}

    system_prompt = SystemMessage(content=f"""
You are a supervisor managing JobStation, a job placement platform.

Current user: {state['username']} | Role: {state['role']}

You have three agents available:
- jobseeker_agent : handles jobseeker actions — CV upload, job recommendations
                    use when role=jobseeker AND user wants to DO something
- company_agent   : handles company HR actions — searching for candidates
                    use when role=company AND user wants to search candidates
- general_agent   : handles ANY career/job/platform related questions
                    use for greetings, tips, platform info, career advice,
                    company suggestions, CV tips — for BOTH roles
                    also use when topic is unrelated to jobs (to politely decline)

Routing rules:
1. Greetings, small talk → general_agent
2. Career advice, job tips → general_agent
3. Questions about JobStation platform → general_agent
4. Jobseeker wants to upload CV or get recommendations → jobseeker_agent
5. Company HR wants to search candidates → company_agent
6. Request fully answered OR last message is already an AI reply → FINISH
    """)

    invoke_messages = [system_prompt] + state["messages"]
    decision = supervisor_llm.invoke(
        invoke_messages,
        config={"callbacks": [langfuse_handler]},   # ← Langfuse trace
    )

    return {"next": decision.next, "turn_count": turn_count}


# ─── JOBSEEKER AGENT ──────────────────────────────────────────────────────────
def jobseeker_agent_node(state: State) -> dict:
    """
    Handles jobseeker platform actions.
    Can call upload_cv_tool or get_recommendations_tool.
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
    response = jobseeker_llm.invoke(
        messages,
        config={"callbacks": [langfuse_handler]},   # ← Langfuse trace
    )

    return {"messages": [response]}


# ─── COMPANY AGENT ────────────────────────────────────────────────────────────
def company_agent_node(state: State) -> dict:
    """
    Handles company/HR platform actions.
    Can call search_candidates_tool.
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
    response = company_llm.invoke(
        messages,
        config={"callbacks": [langfuse_handler]},   # ← Langfuse trace
    )

    return {"messages": [response]}


# ─── GENERAL AGENT ────────────────────────────────────────────────────────────
def general_agent_node(state: State) -> dict:
    """
    Handles general career, job, and platform questions for both roles.
    No tools needed — answers from platform knowledge in system prompt.

    IMPORTANT: After responding, the workflow ends at the supervisor because
    the supervisor detects a plain AI reply and returns FINISH.
    This breaks the general_agent → supervisor → general_agent loop.
    """
    system_prompt = SystemMessage(content=f"""
You are a friendly and knowledgeable assistant for JobStation,
a job placement platform in Indonesia.

You are talking to: {state['username']} (role: {state['role']})

PLATFORM INFORMATION:
- Mission: connect companies directly with candidates — no outsourcing middlemen,
  no salary deductions
- Placement fee: one-time fee paid by the jobseeker upon successful placement.
  No ongoing salary cuts.
- If placed, the jobseeker's data is cleared for privacy and space efficiency.
  They can re-register with a new slot if needed.
- Available job categories:
  HR, ENGINEERING, FINANCE, INFORMATION-TECHNOLOGY, SALES, HEALTHCARE,
  BANKING, CONSULTANT, DESIGNER, CHEF, ARTS, AVIATION, FITNESS, ADVOCATE,
  ACCOUNTANT, BUSINESS-DEVELOPMENT, CONSTRUCTION, DIGITAL-MEDIA,
  AGRICULTURE, AUTOMOBILE, APPAREL, BPO, PUBLIC-RELATIONS, TEACHER
- Professionalism levels:
  junior (≤2 years experience), senior (≤3 years), specialist (>3 years)
- CV upload: jobseekers upload once and become visible to all companies
- Companies search by category, level, and job description
- State changes (interviewed, placed, inactive) are managed by company HR
  through the admin panel

YOUR SCOPE:
- Answer greetings warmly and professionally
- Give career advice, CV tips, interview tips
- Suggest companies or industries based on user's background
- Explain how JobStation works
- Help users understand the platform

OUT OF SCOPE:
- Anything completely unrelated to careers, jobs, or the platform
  (food recipes, weather, entertainment, etc.)
- For out-of-scope questions, politely say you're focused on career topics
  and offer to help with something job-related instead

Be warm, encouraging, and professional.
    """)

    messages = [system_prompt] + state["messages"]
    response = llm.invoke(
        messages,
        config={"callbacks": [langfuse_handler]},   # ← Langfuse trace
    )

    return {"messages": [response]}