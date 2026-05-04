import streamlit as st
import pdfplumber
import io

from JobStation_app.graph.workflow import app
from JobStation_app.tools.utils import *
from JobStation_app.config import langfuse_handler

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JobStation",
    page_icon="💼",
    layout="centered"
)

# ─── SESSION STATE INIT ────────────────────────────────────────────────────────
if "logged_in"  not in st.session_state:
    st.session_state.logged_in  = False
if "username"   not in st.session_state:
    st.session_state.username   = ""
if "role"       not in st.session_state:
    st.session_state.role       = ""
if "messages"   not in st.session_state:
    st.session_state.messages   = []
if "chat_meta"  not in st.session_state:
    # Parallel list to AI messages only.
    # Each entry: {"tool_results": [...], "input_tokens": int, "output_tokens": int}
    st.session_state.chat_meta  = []


# ─── HELPERS ───────────────────────────────────────────────────────────────────
def verify_login(username: str, password: str) -> dict | None:
    try:
        conn   = get_mysql_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM users WHERE username = %s AND password = %s",
            (username, password)
        )
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        return user
    except Exception as e:
        st.error(f"Database error: {e}")
        return None


def extract_pdf_text(uploaded_file) -> str:
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages).strip()


def run_agent(user_input: str) -> dict:
    """
    Invoke the LangGraph app.
    Returns:
        response      : str   - final AI text reply
        tool_results  : list  - raw content from ToolMessage nodes (RAG sources)
        input_tokens  : int
        output_tokens : int
    """
    langfuse_handler.session_id = f"jobstation-{st.session_state.username}"
    langfuse_handler.user_id    = st.session_state.username

    state = {
        "messages":   st.session_state.messages + [HumanMessage(content=user_input)],
        "next":       "",
        "role":       st.session_state.role,
        "username":   st.session_state.username,
        "turn_count": 0,
    }

    result = app.invoke(
        state,
        config={"callbacks": [langfuse_handler]},
    )

    all_messages = result.get("messages", [])

    # Final AI text
    response_text = "I could not generate a response. Please try again."
    for msg in reversed(all_messages):
        if hasattr(msg, "type") and msg.type == "ai":
            if not getattr(msg, "tool_calls", None):
                response_text = msg.content
                break

    # Tool results (RAG documents / tool outputs)
    tool_results = []
    for msg in all_messages:
        if isinstance(msg, ToolMessage):
            tool_results.append(msg.content)

    # Token usage
    input_tokens  = 0
    output_tokens = 0
    for msg in all_messages:
        if hasattr(msg, "type") and msg.type == "ai":
            usage = getattr(msg, "usage_metadata", None) or getattr(msg, "response_metadata", {})
            if isinstance(usage, dict):
                input_tokens  += usage.get("input_tokens",  0) or usage.get("prompt_tokens",    0)
                output_tokens += usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)

    return {
        "response":      response_text,
        "tool_results":  tool_results,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
    }


def get_all_candidates():
    conn   = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT id, category, prof_level, state, updated_at
        FROM candidates ORDER BY updated_at DESC
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def update_candidate_state(candidate_id: int, new_state: str):
    conn   = get_mysql_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE candidates SET state = %s WHERE id = %s",
        (new_state, candidate_id)
    )
    conn.commit()
    cursor.close()
    conn.close()


# ─── RENDER ASSISTANT TURN WITH EXPANDERS ──────────────────────────────────────
def render_assistant_turn(response: str, meta: dict):
    """
    Renders one assistant chat bubble with three collapsible expanders:
      Tool Calls    - only shown when tools were used (RAG sources)
      History Chat  - full conversation so far
      Usage Details - input / output token counts
    """
    with st.chat_message("assistant"):
        st.markdown(response)

        # 1. Tool Calls - only when tools were actually invoked
        if meta.get("tool_results"):
            with st.expander("🔧 Tool Calls:"):
                for tr in meta["tool_results"]:
                    st.code(tr, language="text")

        # 2. Chat History
        with st.expander("🕘 History Chat:"):
            lines = []
            for msg in st.session_state.messages:
                role = "Human" if msg.type == "human" else "AI"
                lines.append(f"{role}: {msg.content}")
            lines.append(f"AI: {response}")
            st.code("\n".join(lines), language="text")

        # 3. Usage Details
        with st.expander("📊 Usage Details:"):
            st.code(
                f"input token  : {meta['input_tokens']}\n"
                f"output token : {meta['output_tokens']}",
                language="text"
            )


# ─── LOGIN SCREEN ──────────────────────────────────────────────────────────────
def show_login():
    st.title("💼 JobStation")
    st.caption("Direct placement. No salary deductions.")
    st.divider()

    with st.form("login_form"):
        username  = st.text_input("Username")
        password  = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)

    if submitted:
        if not username or not password:
            st.warning("Please enter both username and password.")
            return
        user = verify_login(username, password)
        if user:
            st.session_state.logged_in = True
            st.session_state.username  = user["username"]
            st.session_state.role      = user["role"]
            st.session_state.messages  = []
            st.session_state.chat_meta = []
            st.rerun()
        else:
            st.error("Invalid username or password.")


# ─── CHAT INTERFACE ────────────────────────────────────────────────────────────
def show_chat():
    # Replay stored history with metadata expanders
    ai_turn_index = 0
    for msg in st.session_state.messages:
        if msg.type == "human":
            with st.chat_message("user"):
                st.markdown(msg.content)
        else:
            meta = (
                st.session_state.chat_meta[ai_turn_index]
                if ai_turn_index < len(st.session_state.chat_meta)
                else {"tool_results": [], "input_tokens": 0, "output_tokens": 0}
            )
            render_assistant_turn(msg.content, meta)
            ai_turn_index += 1

    # CV upload widget (jobseekers only)
    if st.session_state.role == "jobseeker":
        with st.expander("📄 Upload your CV"):
            col1, col2 = st.columns([3, 1])
            with col1:
                uploaded_file = st.file_uploader(
                    "Upload PDF", type=["pdf"], label_visibility="collapsed"
                )
            with col2:
                category = st.selectbox("Category", [
                    "HR", "ENGINEERING", "FINANCE", "INFORMATION-TECHNOLOGY",
                    "SALES", "MARKETING", "HEALTHCARE", "EDUCATION",
                    "BANKING", "CONSULTANT", "DESIGNER", "CHEF",
                    "ARTS", "AVIATION", "FITNESS", "ADVOCATE",
                    "ACCOUNTANT", "BUSINESS-DEVELOPMENT", "CONSTRUCTION",
                    "DIGITAL-MEDIA", "AGRICULTURE", "AUTOMOBILE",
                    "APPAREL", "BPO", "PUBLIC-RELATIONS", "TEACHER"
                ])

            if uploaded_file and st.button("Upload CV", use_container_width=True):
                with st.spinner("Processing your CV..."):
                    cv_text = extract_pdf_text(uploaded_file)
                    if cv_text:
                        prompt = (
                            f"Please upload my CV to the platform. "
                            f"My category is {category}. "
                            f"Here is my CV text:\n\n{cv_text[:3000]}"
                        )
                        result = run_agent(prompt)
                        st.session_state.messages.append(
                            HumanMessage(content=f"[CV Upload - {category}]")
                        )
                        st.session_state.messages.append(
                            AIMessage(content=result["response"])
                        )
                        st.session_state.chat_meta.append({
                            "tool_results":  result["tool_results"],
                            "input_tokens":  result["input_tokens"],
                            "output_tokens": result["output_tokens"],
                        })
                        st.rerun()
                    else:
                        st.error("Could not extract text from PDF.")

    # Chat input
    if prompt := st.chat_input("Type your message..."):
        st.session_state.messages.append(HumanMessage(content=prompt))

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            result = run_agent(prompt)

        st.session_state.messages.append(AIMessage(content=result["response"]))
        st.session_state.chat_meta.append({
            "tool_results":  result["tool_results"],
            "input_tokens":  result["input_tokens"],
            "output_tokens": result["output_tokens"],
        })

        # st.rerun() replays show_chat() which renders the message via history loop
        # DO NOT call render_assistant_turn() here — it would duplicate the bubble
        st.rerun()


# ─── ADMIN PANEL ───────────────────────────────────────────────────────────────
def show_admin():
    st.subheader("Candidate Management")
    st.caption("Update candidate states here. Changes are saved immediately.")

    col1, col2, col3 = st.columns(3)
    with col1:
        filter_category = st.selectbox("Filter by category", ["All"] + [
            "HR", "ENGINEERING", "FINANCE", "INFORMATION-TECHNOLOGY",
            "SALES", "HEALTHCARE", "BANKING", "CONSULTANT"
        ])
    with col2:
        filter_level = st.selectbox("Filter by level",
            ["All", "junior", "senior", "specialist"])
    with col3:
        filter_state = st.selectbox("Filter by state",
            ["All", "available", "interviewed", "placed", "inactive"])

    if st.button("Refresh"):
        st.rerun()

    candidates = get_all_candidates()
    if filter_category != "All":
        candidates = [c for c in candidates if c["category"] == filter_category]
    if filter_level != "All":
        candidates = [c for c in candidates if c["prof_level"] == filter_level]
    if filter_state != "All":
        candidates = [c for c in candidates if c["state"] == filter_state]

    st.caption(f"Showing {len(candidates)} candidates")
    st.divider()

    if not candidates:
        st.info("No candidates match the selected filters.")
        return

    STATE_OPTIONS = ["available", "interviewed", "placed", "inactive"]
    for candidate in candidates:
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        with col1:
            st.caption("ID")
            st.write(str(candidate["id"]))
        with col2:
            st.caption("Category / Level")
            st.write(f"{candidate['category']} - {candidate['prof_level']}")
        with col3:
            st.caption("Current state")
            state_color = {"available": "🟢", "interviewed": "🟡", "placed": "🔵", "inactive": "⚫"}
            st.write(f"{state_color.get(candidate['state'], '')} {candidate['state']}")
        with col4:
            st.caption("Update state")
            new_state = st.selectbox(
                label="state",
                options=STATE_OPTIONS,
                index=STATE_OPTIONS.index(candidate["state"]),
                key=f"state_{candidate['id']}",
                label_visibility="collapsed"
            )
            if new_state != candidate["state"]:
                update_candidate_state(candidate["id"], new_state)
                st.success("Updated!")
                st.rerun()
        st.divider()


# ─── MAIN APP ──────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.logged_in:
        show_login()
        return

    with st.sidebar:
        st.title("💼 JobStation")
        st.caption(f"Logged in as **{st.session_state.username}**")
        st.caption(f"Role: `{st.session_state.role}`")
        st.divider()
        if st.button("Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    if st.session_state.role == "company":
        chat_tab, admin_tab = st.tabs(["💬 Chat", "🗂️ Candidates"])
        with chat_tab:
            show_chat()
        with admin_tab:
            show_admin()
    else:
        st.header(f"Welcome, {st.session_state.username}! 👋")
        show_chat()


if __name__ == "__main__":
    main()